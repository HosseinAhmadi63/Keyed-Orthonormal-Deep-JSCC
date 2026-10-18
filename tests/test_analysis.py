from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keyed_jscc.analysis import load_matched_images, matched_estimates, paired_seed_contrasts, pooled_psnr, read_rows, verify_reference, wilson_interval


class AnalysisTests(unittest.TestCase):
    def test_psnr_uses_pooled_distortion(self):
        mse = np.asarray([0.001, 0.1])
        self.assertAlmostEqual(pooled_psnr(mse), -10 * math.log10(0.0505), places=13)
        self.assertGreater(abs(pooled_psnr(mse) - float((-10 * np.log10(mse)).mean())), 1)
        self.assertEqual(pooled_psnr(np.zeros(3)), float("inf"))
        with self.assertRaises(ValueError):
            pooled_psnr([-0.1, 0.2])

    def test_actual_holdout_matches_six_codec_metrics(self):
        arrays = load_matched_images(ROOT / "data")
        estimates = matched_estimates(arrays)
        expected = [r for r in read_rows(ROOT / "data/reference_matched_values.csv") if r["case"] in estimates and "__ci_" not in r["metric"]]
        actual = {(case, metric): value for case, metrics in estimates.items() for metric, value in metrics.items()}
        error = verify_reference(actual, expected, ("case", "metric"))
        self.assertLessEqual(error, 1e-12)
        self.assertEqual(len(arrays), 6)
        self.assertEqual(len(expected), 90)

    def test_training_seed_is_replication_unit(self):
        estimates = matched_estimates(load_matched_images(ROOT / "data"))
        contrasts = paired_seed_contrasts(estimates)
        for metric in {r["metric"] for r in contrasts}:
            rows = [r for r in contrasts if r["metric"] == metric]
            self.assertEqual(len(rows), 4)
            values = np.asarray([r["difference"] for r in rows[:-1]])
            mean = rows[-1]
            self.assertAlmostEqual(mean["difference"], float(values.mean()), places=14)
            expected_half_width = 4.302652729911275 * float(values.std(ddof=1)) / math.sqrt(3)
            self.assertAlmostEqual(mean["ci_high"] - mean["difference"], expected_half_width, places=14)

    def test_zero_error_header_interval(self):
        low, high = wilson_interval(0, 10000)
        self.assertAlmostEqual(low, 0, places=15)
        self.assertAlmostEqual(high, 0.00038399837067659573, places=15)
        with self.assertRaises(ValueError):
            wilson_interval(10001, 10000)


if __name__ == "__main__":
    unittest.main()
