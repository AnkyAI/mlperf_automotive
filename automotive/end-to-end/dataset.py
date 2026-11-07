import os
import glob
import gzip
import pickle


class _PreprocessedSampleUnpickler(pickle.Unpickler):
    """Load legacy samples without importing the bundled CUDA training ops."""

    def find_class(self, module, name):
        if module.startswith("third_party.uniad_mmdet3d."):
            if name == "Box3DMode":
                from mmdet3d.core.bbox import Box3DMode

                return Box3DMode
            if name == "LiDARInstance3DBoxes":
                from mmdet3d.core.bbox import LiDARInstance3DBoxes

                return LiDARInstance3DBoxes
        return super().find_class(module, name)

class Nuscenes:
    def __init__(self, dataset_path, config, length):
        # dataset_path points to a directory containing .pkl or .pkl.gz files.
        self.dataset_path = dataset_path
        self.preloaded = {}
        
        # Determine dataset size based on file count if not explicitly provided
        if length > 0:
            self.length = length
        else:
            indices = {
                os.path.basename(path).split(".", 1)[0]
                for pattern in ("*.pkl", "*.pkl.gz")
                for path in glob.glob(os.path.join(self.dataset_path, pattern))
            }
            self.length = len(indices)
            
        if self.length == 0:
            print(
                "Warning: No preprocessed .pkl or .pkl.gz files found in "
                f"{self.dataset_path}!"
            )

    def __len__(self):
        return self.length

    def get_item_count(self):
        return self.length

    def load_item(self, index):
        gzip_path = os.path.join(self.dataset_path, f"{index}.pkl.gz")
        pickle_path = os.path.join(self.dataset_path, f"{index}.pkl")
        if os.path.exists(gzip_path):
            filepath = gzip_path
            open_sample = gzip.open
        elif os.path.exists(pickle_path):
            filepath = pickle_path
            open_sample = open
        else:
            raise FileNotFoundError(
                f"No preprocessed sample found for index {index} in {self.dataset_path}"
            )
        with open_sample(filepath, "rb") as f:
            item = _PreprocessedSampleUnpickler(f).load()
        return item

    def get_samples(self, id_list):
        data = []
        for id in id_list:
            if id in self.preloaded:
                item = self.preloaded[id]
            else:
                item = self.load_item(id)
            data.append(item)
            
        return data

    def load_query_samples(self, sample_list):
        for sample in sample_list:
            self.preloaded[sample] = self.load_item(sample)

    def unload_query_samples(self, sample_list):
        if sample_list is None:
            self.preloaded.clear()
            return
            
        for sample in sample_list:
            if sample in self.preloaded:
                del self.preloaded[sample]
