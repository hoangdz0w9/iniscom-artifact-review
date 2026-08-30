import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_sensitivity import (aligned_unique_mask, feature_hash, joint_patterns,
                                    paired_row_bootstrap, weighted_f1_from_patterns)


class EvaluationSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        self.p = np.array([[[0, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 1]],
                           [[0, 1, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1, 1]]])
        self.contrasts = (("A_minus_B", 0, 1),)

    def test_compressed_counts_match_explicit_shared_row_resamples(self):
        y, pred, counts, inverse = joint_patterns(self.y, self.p)
        rng = np.random.default_rng(42)
        for _ in range(20):
            indices = rng.integers(0, len(self.y), len(self.y))
            weights = np.bincount(inverse[indices], minlength=len(counts))
            actual = weighted_f1_from_patterns(y, pred, weights).ravel()
            expected = [100 * f1_score(self.y[indices], p[indices], average="weighted", zero_division=0)
                        for p in self.p.reshape(-1, len(self.y))]
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_identity_contrast_is_exact_zero(self):
        p = self.p.copy()
        p[1] = p[0]
        report, draws = paired_row_bootstrap(self.y, p, 300, 5, self.contrasts)
        self.assertEqual(report.iloc[0].mean_difference_pp, 0)
        np.testing.assert_array_equal(draws.to_numpy(), 0)

    def test_sign_reversal_uses_identical_draws(self):
        c = (("forward", 0, 1), ("backward", 1, 0))
        _, draws = paired_row_bootstrap(self.y, self.p, 200, 6, c)
        np.testing.assert_array_equal(draws.forward, -draws.backward)

    def test_reproducible_and_correct_full_mean(self):
        a, x = paired_row_bootstrap(self.y, self.p, 200, 7, self.contrasts)
        b, z = paired_row_bootstrap(self.y, self.p, 200, 7, self.contrasts)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(x, z)
        expected = np.mean([100 * (f1_score(self.y, self.p[0, s], average="weighted") -
                                  f1_score(self.y, self.p[1, s], average="weighted")) for s in range(2)])
        self.assertAlmostEqual(a.iloc[0].mean_difference_pp, expected)

    def test_bad_dimensions_and_nonbinary_rejected(self):
        with self.assertRaises(ValueError):
            joint_patterns(self.y, self.p[:, :, :-1])
        with self.assertRaises(ValueError):
            joint_patterns(self.y, self.p + 1)

    def test_feature_mask_excludes_labels_and_preserves_permuted_order(self):
        train = pd.DataFrame({"id": [1, 2], "label": [0, 1], "attack_cat": ["Normal", "X"],
                              **{f"f{i}": [0., 1.] for i in range(42)}})
        test = train.copy()
        test.loc[0, "label"] = 1  # conflicting label is still feature overlap
        test.loc[1, "f1"] = 2.
        order = np.array([1, 0])
        mask, columns = aligned_unique_mask(train, test, order, np.array([1, 1]))
        np.testing.assert_array_equal(mask, [True, False])
        self.assertEqual(len(columns), 42)
        with self.assertRaises(ValueError):
            aligned_unique_mask(train, test, order, np.array([0, 0]))
        with self.assertRaises(ValueError):
            aligned_unique_mask(train, test, np.array([0, 0]), np.array([1, 1]))

    def test_feature_hash_trims_strings_but_does_not_round_numbers(self):
        x = pd.DataFrame({"s": [" a ", "a", "a"], "n": [1., 1., 1.000001]})
        h = feature_hash(x, ["s", "n"])
        self.assertEqual(h[0], h[1])
        self.assertNotEqual(h[1], h[2])


if __name__ == "__main__":
    unittest.main()
