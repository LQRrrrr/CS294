"""Scoring-only reference trial; no proprietary data or foundation model inference."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


QUANTILES = tuple(index / 20 for index in range(1, 20))


def gaussian_crps(mean, sigma, outcome):
    if sigma < 0 or not all(math.isfinite(value) for value in (mean, sigma, outcome)):
        raise ValueError("Expected finite values and nonnegative standard deviation")
    if sigma == 0:
        return abs(outcome - mean)
    residual = (outcome - mean) / sigma
    density = math.exp(-residual * residual / 2) / math.sqrt(2 * math.pi)
    cumulative = (1 + math.erf(residual / math.sqrt(2))) / 2
    return sigma * (residual * (2 * cumulative - 1) + 2 * density - 1 / math.sqrt(math.pi))


def empirical_crps(samples, outcome):
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("At least one sample is required")
    count = len(ordered)
    absolute_error = statistics.fmean(abs(value - outcome) for value in ordered)
    dispersion = sum((2 * index - count + 1) * value for index, value in enumerate(ordered))
    return absolute_error - dispersion / (count * count)


def quantile_score(forecasts, outcome):
    if len(forecasts) != len(QUANTILES):
        raise ValueError("Expected 19 quantiles, from 0.05 to 0.95")
    if not all(math.isfinite(value) for value in (*forecasts, outcome)):
        raise ValueError("Quantiles and outcome must be finite")
    if any(first > second for first, second in zip(forecasts, forecasts[1:])):
        raise ValueError("Quantile forecasts must be nondecreasing")
    return 2 * statistics.fmean(
        (level - (outcome < forecast)) * (outcome - forecast)
        for level, forecast in zip(QUANTILES, forecasts)
    )


def fit_linear(rows, horizon):
    features = [row["sentiment"] for row in rows]
    outcomes = [row["outcomes"][horizon] for row in rows]
    feature_mean = statistics.fmean(features)
    outcome_mean = statistics.fmean(outcomes)
    feature_variance = sum((value - feature_mean) ** 2 for value in features)
    slope = sum(
        (feature - feature_mean) * (outcome - outcome_mean)
        for feature, outcome in zip(features, outcomes)
    ) / feature_variance
    intercept = outcome_mean - slope * feature_mean
    sigma = math.sqrt(statistics.fmean(
        (outcome - intercept - slope * feature) ** 2
        for feature, outcome in zip(features, outcomes)
    ))
    return intercept, slope, sigma


def synthetic_trial():
    generator = random.Random(294)
    slopes = (0.004, 0.002, 0.001)
    sigmas = (0.010, 0.014, 0.018)
    rows = []
    for origin in range(1200):
        sentiment = generator.choice((-1, 0, 1))
        outcomes = tuple(
            slope * sentiment + generator.gauss(0, sigma)
            for slope, sigma in zip(slopes, sigmas)
        )
        rows.append({"origin": origin, "sentiment": sentiment, "outcomes": outcomes})
    train, validation, test = rows[:720], rows[720:960], rows[960:]
    fitted = [fit_linear(train, horizon) for horizon in range(3)]
    scales = [statistics.pstdev(row["outcomes"][horizon] for row in train) for horizon in range(3)]
    calibration = min(
        (0.5, 0.75, 1.0, 1.25, 1.5),
        key=lambda factor: statistics.fmean(
            gaussian_crps(intercept + slope * row["sentiment"], sigma * factor, row["outcomes"][horizon]) / scales[horizon]
            for horizon, (intercept, slope, sigma) in enumerate(fitted)
            for row in validation
        ),
    )
    groups = {
        (horizon, sentiment): [row["outcomes"][horizon] for row in test if row["sentiment"] == sentiment]
        for horizon in range(3) for sentiment in (-1, 0, 1)
    }
    results = []
    for horizon in range(3):
        intercept, slope, sigma = fitted[horizon]
        scale = scales[horizon]
        outcomes = [row["outcomes"][horizon] for row in test]
        baseline_sigma = math.sqrt(statistics.fmean(row["outcomes"][horizon] ** 2 for row in train))
        predictors = {
            "zero_mean_gaussian": [(0.0, baseline_sigma) for row in test],
            "trained_linear_gaussian": [(intercept + slope * row["sentiment"], sigma * calibration) for row in test],
            "known_bayes_oracle": [(slopes[horizon] * row["sentiment"], sigmas[horizon]) for row in test],
        }
        scores = {
            name: {
                "normalized_mse": statistics.fmean((mean - outcome) ** 2 / scale ** 2 for (mean, spread), outcome in zip(predictions, outcomes)),
                "normalized_crps": statistics.fmean(gaussian_crps(mean, spread, outcome) / scale for (mean, spread), outcome in zip(predictions, outcomes)),
            }
            for name, predictions in predictors.items()
        }
        scores["retrospective_feature_cell_oracle"] = {
            "normalized_mse": statistics.fmean(
                (statistics.fmean(groups[horizon, row["sentiment"]]) - row["outcomes"][horizon]) ** 2 / scale ** 2 for row in test
            ),
            "normalized_crps": statistics.fmean(
                empirical_crps(groups[horizon, row["sentiment"]], row["outcomes"][horizon]) / scale for row in test
            ),
        }
        scores["outcome_lookup_oracle"] = {"normalized_mse": 0.0, "normalized_crps": 0.0}
        results.append({
            "horizon": horizon + 1,
            "training_scale": scale,
            "expected_bayes_normalized_mse": sigmas[horizon] ** 2 / scale ** 2,
            "expected_bayes_normalized_crps": sigmas[horizon] / (math.sqrt(math.pi) * scale),
            "scores": scores,
        })
    return {
        "status": "Synthetic scoring demonstration, NOT RavenPack/TAQ or foundation-model results",
        "seed": 294,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation_sigma_multiplier": calibration,
        "results": results,
    }


def score_csv(path, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be positive, finite, and fixed using training data")
    grouped = {}
    keys = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"series_id", "origin", "horizon", "target", "mean"} | {f"q{index:02d}" for index in range(5, 100, 5)}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required CSV fields: {sorted(required)}")
        for row in reader:
            key = (row["series_id"], row["origin"], row["horizon"])
            if key in keys:
                raise ValueError(f"Duplicate forecast key: {key}")
            keys.add(key)
            outcome, mean = float(row["target"]), float(row["mean"])
            if not math.isfinite(mean):
                raise ValueError("Forecast mean must be finite")
            forecasts = [float(row[f"q{index:02d}"]) for index in range(5, 100, 5)]
            score = quantile_score(forecasts, outcome) / scale
            grouped.setdefault(row["horizon"], []).append(((mean - outcome) ** 2 / scale ** 2, score))
    if not grouped:
        raise ValueError("Prediction file must contain at least one row")
    return {
        horizon: {"count": len(scores), "normalized_mse": statistics.fmean(score[0] for score in scores),
                  "normalized_quantile_score": statistics.fmean(score[1] for score in scores)}
        for horizon, scores in sorted(grouped.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--scale", type=float, help="Training-only target standard deviation for CSV scoring")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.predictions and arguments.scale is None:
        parser.error("--predictions requires --scale")
    result = score_csv(arguments.predictions, arguments.scale) if arguments.predictions else synthetic_trial()
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
