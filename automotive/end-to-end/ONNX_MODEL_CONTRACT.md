# UniAD 2.0 ONNX model contract

The MLPerf backend requires a stateful UniAD 2.0 ONNX graph. A graph exported
only for the first frame is not sufficient for benchmark accuracy because
UniAD uses information from earlier frames.

## Required inputs

- `img`: current six-camera image tensor
- `prev_bev`: BEV features returned by the previous frame
- `use_prev_bev`: false for the first frame of a scene, true otherwise

Any additional graph inputs, including ego pose and timestamp tensors, must use
the same names as keys in the preprocessed sample or its `img_metas` dictionary.

## Required outputs

- `sdc_traj`: planned ego trajectory
- `next_bev`: BEV features to feed into the next frame

The backend resets `prev_bev` when `scene_token` changes. LoadGen samples must
therefore be issued in dataset order for a temporal accuracy run. Performance
runs need sequence-aware grouping or independent state per active scene; random
sample ordering with one global state is invalid.

UniAD 2.0 also propagates tracker instances between frames. The final model
contract must expose every tracker tensor that affects the planning output, or
the exporter must prove that the selected benchmark path does not consume it.
