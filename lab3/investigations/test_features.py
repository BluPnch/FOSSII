"""Checks that additional predictors cannot fit preprocessing on future data."""

import unittest
from unittest.mock import patch

import numpy as np

import features


class FeaturePreparationTests(unittest.TestCase):
    def test_future_numeric_values_do_not_change_train_transform(self):
        original = features.study.prepared
        clean, _, clean_info = features.prepared("numeric")

        def changed(config):
            datasets, frame, info = original(config)
            frame.loc[datasets["train"].stop:, features.GROUPS["numeric"]] = 999999.0
            return datasets, frame, info

        with patch.object(features.study, "prepared", side_effect=changed):
            modified, _, modified_info = features.prepared("numeric")
        self.assertEqual(clean_info["additional_transform"], modified_info["additional_transform"])
        stop = clean["train"].stop
        np.testing.assert_array_equal(clean["train"].features[:stop], modified["train"].features[:stop])
        self.assertFalse(np.array_equal(clean["val"].features[stop:], modified["val"].features[stop:]))

    def test_unknown_future_category_does_not_expand_train_vocabulary(self):
        original = features.study.prepared

        def changed(config):
            datasets, frame, info = original(config)
            frame.loc[datasets["val"].start, "Load_Type"] = "Unseen_Test_Category"
            return datasets, frame, info

        with patch.object(features.study, "prepared", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Unseen validation/test category"):
                features.prepared("load_type")

    def test_window_ends_strictly_before_target(self):
        datasets, _, _ = features.prepared("all")
        for ds in datasets.values():
            index = 1
            target_time = ds.start + index
            ds.features[target_time - 1, -1] = 12345.0
            ds.features[target_time, -1] = -12345.0
            x, y = ds[index]
            self.assertEqual(len(x), 24)
            self.assertEqual(float(x[-1, -1]), 12345.0)
            self.assertFalse(bool((x[:, -1] == -12345.0).any()))
            self.assertEqual(float(y), float(ds.targets[target_time]))


if __name__ == "__main__":
    unittest.main()
