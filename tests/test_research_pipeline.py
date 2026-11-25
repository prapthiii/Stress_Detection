import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments.grouped_experiments import _extract_grouped_xy


class GroupedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "feature_0": [0.1, 0.2, 0.3, 0.4],
                "label": [0, 1, 0, 1],
                "filename": [
                    "Actor_01/recording-01.wav",
                    "Actor_01/recording-02.wav",
                    "Actor_02/recording-01.wav",
                    "Actor_02/recording-02.wav",
                ],
                "actor": [1, 1, 2, 2],
                "actor_id": [1, 1, 2, 2],
                "sample_rate": [22050, 22050, 22050, 22050],
                "mfcc_dimensions": [13, 13, 13, 13],
            }
        )

    def test_actor_metadata_is_not_a_predictor(self):
        features, labels, groups = _extract_grouped_xy(self.frame, "label", "filename")

        np.testing.assert_array_equal(features[:, 0], self.frame["feature_0"])
        np.testing.assert_array_equal(labels, self.frame["label"])
        np.testing.assert_array_equal(groups, [1, 1, 2, 2])
        self.assertEqual(features.shape[1], 1)

    def test_grouped_splits_never_share_actors(self):
        features, labels, groups = _extract_grouped_xy(self.frame, "label", "filename")

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
        train_idx, test_idx = next(splitter.split(features, labels, groups))
        self.assertFalse(set(groups[train_idx]) & set(groups[test_idx]))

        for train_idx, test_idx in GroupKFold(n_splits=2).split(features, labels, groups):
            self.assertFalse(set(groups[train_idx]) & set(groups[test_idx]))


if __name__ == "__main__":
    unittest.main()