import unittest

import numpy as np

from backend_onnx import BackendUniADOnnx


class ValueInfo:
    def __init__(self, name, shape, type_name="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_name


class FakeSession:
    def __init__(self, stateful=True):
        self.input_values = [ValueInfo("img", [1, 6, 3, 2, 2])]
        self.output_values = [ValueInfo("sdc_traj", [1, 6, 2])]
        if stateful:
            self.input_values.append(ValueInfo("use_prev_bev", [1], "tensor(bool)"))
            for input_name, output_name in BackendUniADOnnx.STATE_PAIRS.items():
                if input_name.startswith("track_"):
                    shape = ["num_queries", 2]
                else:
                    shape = [1]
                self.input_values.append(ValueInfo(input_name, shape))
                self.output_values.append(ValueInfo(output_name, shape))
        self.feeds = []

    def get_inputs(self):
        return self.input_values

    def get_outputs(self):
        return self.output_values

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        outputs = {
            "sdc_traj": np.zeros((1, 6, 2), dtype=np.float32),
        }
        for input_name, output_name in BackendUniADOnnx.STATE_PAIRS.items():
            outputs[output_name] = feeds[input_name] + 1
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
        self.assertEqual(session.feeds[0]["track_query"].shape[0], 901)


if __name__ == "__main__":
    unittest.main()
