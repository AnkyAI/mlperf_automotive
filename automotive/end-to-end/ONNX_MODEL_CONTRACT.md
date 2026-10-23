# UniAD 2.0 ONNX model contract

The MLPerf backend requires a stateful UniAD 2.0 ONNX graph. A graph exported
only for the first frame is not sufficient for benchmark accuracy because
UniAD uses information from earlier frames.

The exporter must also produce an NPZ initial-state artifact containing every
state input below. This preserves the checkpoint's learned `track_query` and
`track_ref_pts` tensors; initializing those values to zero changes the model.

## Required inputs

- `img`: current six-camera image tensor
- `prev_bev`: BEV features returned by the previous frame
- `use_prev_bev`: false for the first frame of a scene, true otherwise
- `prev_timestamp`, `prev_l2g_r_mat`, and `prev_l2g_t`: prior ego state
- `max_obj_id`: tracker identifier counter
- `track_query`, `track_ref_pts`, `track_obj_idxes`
- `track_matched_gt_idxes`, `track_disappear_time`, `track_iou`
- `track_track_scores`, `track_pred_boxes`
- `track_mem_bank`, `track_mem_padding_mask`
- `track_save_period`

`output_embedding`, classification `scores`, and `pred_logits` are transient
per-frame values. UniAD overwrites them before reading the incoming tracker
state, so the ONNX exporter prunes those inputs and the backend does not carry
them between frames.

Any additional graph inputs, including ego pose and timestamp tensors, must use
the same names as keys in the preprocessed sample or its `img_metas` dictionary.

## Required outputs

- `sdc_traj`: planned ego trajectory
- `next_bev`: BEV features to feed into the next frame
- `next_timestamp`, `next_l2g_r_mat`, `next_l2g_t`, and `next_max_obj_id`
- A `next_` output corresponding to every `track_` input listed above

The backend reloads the complete initial-state artifact when `scene_token`
changes. LoadGen samples must
therefore be issued in dataset order for a temporal accuracy run. Performance
runs need sequence-aware grouping or independent state per active scene; random
sample ordering with one global state is invalid.

The tracker state uses a dynamic first dimension. A new scene starts with 901
queries; later frames may append active tracks.
