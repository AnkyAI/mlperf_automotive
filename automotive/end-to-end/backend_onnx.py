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
        "track_output_embedding": "next_track_output_embedding",
        "track_obj_idxes": "next_track_obj_idxes",
        "track_matched_gt_idxes": "next_track_matched_gt_idxes",
        "track_disappear_time": "next_track_disappear_time",
        "track_iou": "next_track_iou",
        "track_scores": "next_track_scores",
        "track_track_scores": "next_track_track_scores",
        "track_pred_boxes": "next_track_pred_boxes",
        "track_pred_logits": "next_track_pred_logits",
        "track_mem_bank": "next_track_mem_bank",
        "track_mem_padding_mask": "next_track_mem_padding_mask",
        "track_save_period": "next_track_save_period",
    }

    def __init__(self, model_path, initial_state_path=None, providers=None, session=None):
        self.model_path = model_path
        self.initial_state_path = initial_state_path
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = session
        self.scene_token = None
        self.state = {}
        self.initial_state = {}

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
        if not self.initial_state_path:
            raise ValueError(
                "--onnx-initial-state is required for the learned UniAD tracker "
                "queries and deterministic scene resets"
            )
        with np.load(self.initial_state_path) as archive:
            missing_state = sorted(self.STATE_PAIRS.keys() - archive.files)
            if missing_state:
                raise ValueError(
                    f"Initial-state artifact is missing tensors: {missing_state}"
                )
            self.initial_state = {
                name: np.asarray(archive[name], dtype=self._dtype_for(self.inputs[name]))
                for name in self.STATE_PAIRS
            }
        for name, value in self.initial_state.items():
            expected = self.inputs[name].shape
            if len(expected) != value.ndim or any(
                isinstance(dim, int) and dim > 0 and dim != actual
                for dim, actual in zip(expected, value.shape)
            ):
                raise ValueError(
                    f"Initial state {name} has shape {value.shape}; expected {expected}"
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
                input_name: value.copy()
                for input_name, value in self.initial_state.items()
            }

        feeds = {
            self.IMAGE_INPUT: self._as_numpy(sample[self.IMAGE_INPUT]),
            self.USE_PREV_BEV_INPUT: np.asarray(
                [not first_frame],
                dtype=self._dtype_for(self.inputs[self.USE_PREV_BEV_INPUT]),
            ),
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
