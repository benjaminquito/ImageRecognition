import csv
import tempfile
import unittest
from pathlib import Path

from safety_framework import create_protocol, summarize


class SafetyFrameworkTests(unittest.TestCase):
    def test_protocol_has_all_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.csv"
            self.assertEqual(create_protocol(output, runs=2), 2 * 2 * 7 * 2)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 56)
            self.assertEqual(
                {row["technology"] for row in rows}, {"lidar", "camera"}
            )
            self.assertEqual(
                {row["driver_mode"] for row in rows}, {"human", "autonomous"}
            )

    def test_summary_uses_only_populated_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.csv"
            summary = Path(directory) / "summary.csv"
            create_protocol(protocol, runs=2)

            with protocol.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

            selected = [
                row
                for row in rows
                if row["driver_mode"] == "human"
                and row["technology"] == "lidar"
                and row["weather_code"] == "S"
            ]
            selected[0].update(
                expected_label="car",
                predicted_label="car",
                confidence="0.9",
                latency_ms="10",
            )
            selected[1].update(
                expected_label="car",
                predicted_label="truck",
                confidence="0.7",
                latency_ms="14",
            )
            with protocol.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            self.assertEqual(summarize(protocol, summary), 1)
            with summary.open(newline="", encoding="utf-8") as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["labeled_samples"], "2")
            self.assertAlmostEqual(float(result["accuracy"]), 0.5)
            self.assertAlmostEqual(float(result["mean_confidence"]), 0.8)
            self.assertAlmostEqual(float(result["mean_latency_ms"]), 12.0)


if __name__ == "__main__":
    unittest.main()
