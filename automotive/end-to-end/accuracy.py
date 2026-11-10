import argparse
import gzip
import json
import pickle
import numpy as np
import os


class PlanningL2Metric:
    """Accumulate UniAD's masked per-timestep planning L2 metric."""

    def __init__(self, n_future=6):
        self.l2 = np.zeros(n_future, dtype=np.float64)
        self.total = 0

    def update(self, trajectories, ground_truth, ground_truth_mask):
        if trajectories.shape != ground_truth.shape:
            raise ValueError(
                f"Trajectory shape mismatch: {tuple(trajectories.shape)} != "
                f"{tuple(ground_truth.shape)}"
            )
        if ground_truth_mask.shape != ground_truth.shape:
            raise ValueError(
                f"Mask shape mismatch: {tuple(ground_truth_mask.shape)} != "
                f"{tuple(ground_truth.shape)}"
            )
        values = np.sqrt(
            (((trajectories - ground_truth) ** 2) * ground_truth_mask).sum(axis=-1)
        )
        self.l2 += values.astype(np.float64).sum(axis=0)
        self.total += len(trajectories)

    def compute(self):
        if self.total == 0:
            raise RuntimeError("No planning samples were evaluated")
        return self.l2 / self.total

def get_args():
    parser = argparse.ArgumentParser(description="UniAD Accuracy Checker (Planning Metric)")
    parser.add_argument("--log-file", type=str, default="results/mlperf_log_accuracy.json", help="Path to MLPerf accuracy log")
    parser.add_argument("--dataset-path", help="Path to preprocessed NuScenes dataset (.pkl files)", required=True)
    parser.add_argument(
        "--ground-truth-path",
        default=None,
        help="Optional directory containing compressed planning ground-truth sidecars",
    )
    return parser.parse_args()


def load_sample(root, index):
    gzip_path = os.path.join(root, f"{index}.pkl.gz")
    pickle_path = os.path.join(root, f"{index}.pkl")
    if os.path.exists(gzip_path):
        path = gzip_path
        open_sample = gzip.open
    elif os.path.exists(pickle_path):
        path = pickle_path
        open_sample = open
    else:
        raise FileNotFoundError(f"No sample {index} found in {root}")
    with open_sample(path, "rb") as handle:
        return pickle.load(handle)

def unwrap(obj):
    if isinstance(obj, dict):
        return {k: unwrap(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [unwrap(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(unwrap(x) for x in obj)
    return obj

def to_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def prepare_ground_truth(sample):
    sdc_planning = to_array(sample["sdc_planning"])
    sdc_planning_mask = to_array(sample["sdc_planning_mask"])
    segmentation = to_array(sample["gt_segmentation"])

    if tuple(sdc_planning.shape) != (1, 1, 6, 3):
        raise ValueError(f"Unexpected sdc_planning shape: {tuple(sdc_planning.shape)}")
    if tuple(sdc_planning_mask.shape) != (1, 1, 6, 2):
        raise ValueError(
            "Unexpected sdc_planning_mask shape: "
            f"{tuple(sdc_planning_mask.shape)}"
        )
    if tuple(segmentation.shape) != (1, 7, 200, 200):
        raise ValueError(
            f"Unexpected gt_segmentation shape: {tuple(segmentation.shape)}"
        )

    return (
        sdc_planning[0, :, :6, :2],
        sdc_planning_mask[0, :, :6, :2],
        segmentation[:, 1:7],
    )

def main():
    args = get_args()
    ground_truth_path = args.ground_truth_path or args.dataset_path
    
    print(f"Parsing accuracy log: {args.log_file}")
    try:
        with open(args.log_file, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.log_file} not found. Did you run main.py with --accuracy?")
        return
        
    raw_outputs_dict = {}
    for entry in log_data:
        qsl_idx = entry["qsl_idx"]
        data_hex = entry["data"]
        prediction_orig = pickle.loads(bytes.fromhex(data_hex))
        raw_outputs_dict[qsl_idx] = prediction_orig
        
    ordered_keys = sorted(raw_outputs_dict.keys())
    
    planning_metric = PlanningL2Metric()

    print("\n" + "="*50)
    print(" UniAD Planning Accuracy Results")
    print("="*50)

    failures = []
    evaluated = 0
    for qsl_idx in ordered_keys:
        prediction = unwrap(raw_outputs_dict[qsl_idx])
        
        if 'planning' not in prediction:
            continue
            
        try:
            # 1. Load Ground Truth data from the preprocessed offline dataset
            gt_data = load_sample(ground_truth_path, qsl_idx)
            
            # 2. Extract Model predictions
            res_plan = prediction['planning']['result_planning']
            pred_sdc_traj = to_array(res_plan['sdc_traj'])
            
            # 3. Extract Ground Truth directly from the dataset item
            gt_sliced, mask_sliced, _ = prepare_ground_truth(gt_data)
            
            # Apply the exact shape slices from the test.py script
            pred_sliced = pred_sdc_traj[:, :6, :2]
            planning_metric.update(pred_sliced, gt_sliced, mask_sliced)
            evaluated += 1
            
        except Exception as e:
            print(f"Warning: Failed to process QSL Index {qsl_idx}. Error: {e}")
            failures.append((qsl_idx, str(e)))

    if failures:
        raise RuntimeError(
            f"Accuracy evaluation failed for {len(failures)} of "
            f"{len(ordered_keys)} samples"
        )
    if evaluated != len(ordered_keys):
        raise RuntimeError(
            f"Evaluated {evaluated} of {len(ordered_keys)} accuracy samples"
        )
            
    l2_list = planning_metric.compute().tolist()
    avg_l2 = sum(l2_list) / len(l2_list)

    time_labels = [f"{(i+1)*0.5:.1f}s" for i in range(len(l2_list))]
    header = "        " + "".join(f"{label:>8}" for label in time_labels)
    row = " L2:    " + "".join(f"{val:>8.4f}" for val in l2_list)

    print(header)
    print(row)
    print(f"Avg: {avg_l2}")

if __name__ == "__main__":
    main()
