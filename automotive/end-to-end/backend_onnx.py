import numpy as np


class BackendUniADOnnx:
    """ONNX Runtime backend for the stateful UniAD 2.0 planning graph."""

    IMAGE_INPUT = "img"
    USE_PREV_BEV_INPUT = "use_prev_bev"
    PLANNING_OUTPUT = "sdc_traj"
    INITIAL_TRACK_QUERIES = 901
    STATE_PAIRS = {
        "prev_bev": "next_bev",
        "prev_timestamp": "next_timestamp",
        "prev_l2g_r_mat": "next_l2g_r_mat",
        "prev_l2g_t": "next_l2g_t",
        "max_obj_id": "next_max_obj_id",
        "track_query": "next_track_query",
        "track_ref_pts": "next_track_ref_pts",
        "track_obj_idxes": "next_track_obj_idxes",
        "track_matched_gt_idxes": "next_track_matched_gt_idxes",
        "track_disappear_time": "next_track_disappear_time",
        "track_iou": "next_track_iou",
        "track_scores": "next_track_scores",
        "track_pred_boxes": "next_track_pred_boxes",
        "track_mem_bank": "next_track_mem_bank",
        "track_mem_padding_mask": "next_track_mem_padding_mask",
        "track_save_period": "next_track_save_period",
    }

    def __init__(self, model_path, providers=None, session=None):
        self.model_path = model_path
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = session
        self.scene_token = None
        self.state = {}

    def load(self):
        if self.session is None:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                self.model_path,
                providers=self.providers,
            )

        self.inputs = {value.name: value for value in self.session.get_inputs()}
        self.outputs = {value.name: value for value in self.session.get_outputs()}
        required_inputs = {
            self.IMAGE_INPUT,
            self.USE_PREV_BEV_INPUT,
            *self.STATE_PAIRS.keys(),
        }
        required_outputs = {self.PLANNING_OUTPUT, *self.STATE_PAIRS.values()}
        missing_inputs = sorted(required_inputs - self.inputs.keys())
        missing_outputs = sorted(required_outputs - self.outputs.keys())
        if missing_inputs or missing_outputs:
            raise ValueError(
                "UniAD MLPerf requires a stateful ONNX graph; "
                f"missing inputs={missing_inputs}, missing outputs={missing_outputs}. "
                "The first-frame-only export is not benchmark compatible."
            )
        return self

    @staticmethod
    def _as_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _metadata(sample):
        metadata = sample.get("img_metas", [])
        while isinstance(metadata, (list, tuple)) and len(metadata) == 1:
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            raise ValueError("Each preprocessed sample must contain one img_metas dict")
        return metadata

    @staticmethod
    def _dtype_for(value):
        type_name = getattr(value, "type", "tensor(float)")
        return {
            "tensor(bool)": np.bool_,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }.get(type_name, np.float32)

    def _zeros_for(self, input_name):
        value = self.inputs[input_name]
        shape = list(value.shape)
        if input_name.startswith("track_") and (
            not isinstance(shape[0], int) or shape[0] <= 0
        ):
            shape[0] = self.INITIAL_TRACK_QUERIES
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(
                f"Cannot initialize dynamic state input {input_name}: {value.shape}"
            )
        return np.zeros(shape, dtype=self._dtype_for(value))

    def reset(self):
        self.scene_token = None
        self.state = {}

    def predict(self, sample):
        metadata = self._metadata(sample)
        scene_token = metadata.get("scene_token")
        if scene_token is None:
            raise ValueError("img_metas is missing scene_token")

        first_frame = scene_token != self.scene_token
        if first_frame:
            self.state = {
                input_name: self._zeros_for(input_name)
                for input_name in self.STATE_PAIRS
            }

        feeds = {
            self.IMAGE_INPUT: self._as_numpy(sample[self.IMAGE_INPUT]),
            self.USE_PREV_BEV_INPUT: np.asarray([not first_frame], dtype=np.bool_),
            **self.state,
        }
        for name in self.inputs:
            if name in feeds:
                continue
            if name in sample:
                feeds[name] = self._as_numpy(sample[name])
            elif name in metadata:
                feeds[name] = self._as_numpy(metadata[name])
            else:
                raise ValueError(f"No value available for ONNX input {name}")

        output_names = list(self.outputs)
        values = self.session.run(output_names, feeds)
        result = dict(zip(output_names, values))
        self.state = {
            input_name: result[output_name]
            for input_name, output_name in self.STATE_PAIRS.items()
        }
        self.scene_token = scene_token

        return [{
            "planning": {
                "result_planning": {
                    "sdc_traj": result[self.PLANNING_OUTPUT],
                }
            }
        }]
