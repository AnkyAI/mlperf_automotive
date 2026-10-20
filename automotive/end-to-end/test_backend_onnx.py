import unittest

import numpy as np

from backend_onnx import BackendUniADOnnx


class ValueInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, stateful=True):
        self.input_values = [ValueInfo("img", [1, 6, 3, 2, 2])]
        self.output_values = [ValueInfo("sdc_traj", [1, 6, 2])]
        if stateful:
            self.input_values.extend([
                ValueInfo("prev_bev", [4, 1, 2]),
                ValueInfo("use_prev_bev", [1]),
            ])
            self.output_values.append(ValueInfo("next_bev", [4, 1, 2]))
        self.feeds = []

    def get_inputs(self):
        return self.input_values

    def get_outputs(self):
        return self.output_values

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        next_bev = feeds["prev_bev"] + 1
        outputs = {
            "sdc_traj": np.zeros((1, 6, 2), dtype=np.float32),
            "next_bev": next_bev,
        }
        return [outputs[name] for name in output_names]


def sample(scene_token):
    return {
        "img": np.zeros((1, 6, 3, 2, 2), dtype=np.float32),
        "img_metas": [{"scene_token": scene_token}],
    }


class BackendUniADOnnxTest(unittest.TestCase):
    def test_rejects_first_frame_only_graph(self):
        with self.assertRaisesRegex(ValueError, "first-frame-only"):
            BackendUniADOnnx("model.onnx", session=FakeSession(False)).load()

    def test_carries_state_within_scene_and_resets_between_scenes(self):
        session = FakeSession()
        backend = BackendUniADOnnx("model.onnx", session=session).load()

        first = backend.predict(sample("scene-a"))
        backend.predict(sample("scene-a"))
        backend.predict(sample("scene-b"))

        self.assertEqual(first[0]["planning"]["result_planning"]["sdc_traj"].shape,
                         (1, 6, 2))
        self.assertFalse(session.feeds[0]["use_prev_bev"][0])
        self.assertTrue(session.feeds[1]["use_prev_bev"][0])
        self.assertFalse(session.feeds[2]["use_prev_bev"][0])
        np.testing.assert_array_equal(session.feeds[0]["prev_bev"], 0)
        np.testing.assert_array_equal(session.feeds[1]["prev_bev"], 1)
        np.testing.assert_array_equal(session.feeds[2]["prev_bev"], 0)


if __name__ == "__main__":
    unittest.main()
