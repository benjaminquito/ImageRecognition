"""Create and summarize the paper's proposed AV safety-study protocol."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence


WEATHER = {
    "S": "sunny",
    "C": "cloudy",
    "R": "rain (table-only code)",
    "F": "foggy",
    "DR": "daytime rainy",
    "NR": "nighttime rainy",
    "SW": "snowy",
}
TECHNOLOGIES = ("lidar", "camera")
DRIVER_MODES = ("human", "autonomous")
FIELDS = (
    "driver_mode",
    "technology",
    "weather_code",
    "weather",
    "run",
    "expected_label",
    "predicted_label",
    "confidence",
    "latency_ms",
    "obstacle_detected",
    "stopping_distance_m",
    "notes",
)


def create_protocol(path: Path, runs: int = 25) -> int:
    """Write a blank long-form experimental protocol and return its row count."""
    if runs < 1:
        raise ValueError("runs must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for driver_mode in DRIVER_MODES:
            for technology in TECHNOLOGIES:
                for code, weather in WEATHER.items():
                    for run_number in range(1, runs + 1):
                        writer.writerow(
                            {
                                "driver_mode": driver_mode,
                                "technology": technology,
                                "weather_code": code,
                                "weather": weather,
                                "run": run_number,
                            }
                        )
                        count += 1
    return count


def _optional_float(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def _optional_correct(expected: str, predicted: str) -> float | None:
    expected, predicted = expected.strip(), predicted.strip()
    if not expected or not predicted:
        return None
    return float(expected == predicted)


def _mean(values: list[float]) -> str:
    return f"{statistics.fmean(values):.6f}" if values else ""


def summarize(source: Path, destination: Path) -> int:
    """Aggregate populated measurements and return the number of groups."""
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = (row["driver_mode"], row["technology"], row["weather_code"])
            values = {
                "accuracy": _optional_correct(
                    row["expected_label"], row["predicted_label"]
                ),
                "confidence": _optional_float(row["confidence"]),
                "latency_ms": _optional_float(row["latency_ms"]),
                "stopping_distance_m": _optional_float(row["stopping_distance_m"]),
            }
            for name, value in values.items():
                if value is not None:
                    grouped[key][name].append(value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    output_fields = (
        "driver_mode",
        "technology",
        "weather_code",
        "weather",
        "labeled_samples",
        "accuracy",
        "mean_confidence",
        "mean_latency_ms",
        "mean_stopping_distance_m",
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for key in sorted(grouped):
            driver_mode, technology, code = key
            values = grouped[key]
            writer.writerow(
                {
                    "driver_mode": driver_mode,
                    "technology": technology,
                    "weather_code": code,
                    "weather": WEATHER.get(code, "unknown"),
                    "labeled_samples": len(values["accuracy"]),
                    "accuracy": _mean(values["accuracy"]),
                    "mean_confidence": _mean(values["confidence"]),
                    "mean_latency_ms": _mean(values["latency_ms"]),
                    "mean_stopping_distance_m": _mean(
                        values["stopping_distance_m"]
                    ),
                }
            )
    return len(grouped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="create a blank study protocol")
    initialize.add_argument("output", type=Path)
    initialize.add_argument("--runs", type=int, default=25)
    aggregate = commands.add_parser("summarize", help="aggregate completed runs")
    aggregate.add_argument("source", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        rows = create_protocol(args.output, args.runs)
        print(f"Created {rows} protocol rows at {args.output}")
    else:
        groups = summarize(args.source, args.output)
        print(f"Wrote {groups} populated groups to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

