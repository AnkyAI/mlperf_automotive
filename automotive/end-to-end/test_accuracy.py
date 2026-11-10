import unittest

import numpy as np

from accuracy import PlanningL2Metric


class PlanningL2MetricTest(unittest.TestCase):
    def test_masked_l2_average(self):
        metric = PlanningL2Metric(n_future=2)
        prediction = np.array([[[3.0, 4.0], [2.0, 8.0]]])
        ground_truth = np.zeros_like(prediction)
        mask = np.array([[[1.0, 1.0], [1.0, 0.0]]])

        metric.update(prediction, ground_truth, mask)
        metric.update(prediction, ground_truth, mask)

        np.testing.assert_allclose(metric.compute(), np.array([5.0, 2.0]))

    def test_rejects_shape_mismatch(self):
        metric = PlanningL2Metric(n_future=2)
        with self.assertRaisesRegex(ValueError, "Trajectory shape mismatch"):
            metric.update(
                np.zeros((1, 2, 2)),
                np.zeros((1, 1, 2)),
                np.zeros((1, 1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
