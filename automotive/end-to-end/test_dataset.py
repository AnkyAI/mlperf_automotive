import gzip
import os
import pickle
import tempfile
import unittest

from dataset import Nuscenes


class NuscenesTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def write_sample(self, index, value, compressed=False):
        suffix = ".pkl.gz" if compressed else ".pkl"
        path = os.path.join(self.tempdir.name, f"{index}{suffix}")
        open_sample = gzip.open if compressed else open
        with open_sample(path, "wb") as handle:
            pickle.dump({"value": value}, handle)

    def test_loads_compressed_and_uncompressed_samples(self):
        self.write_sample(0, "plain")
        self.write_sample(1, "gzip", compressed=True)
        dataset = Nuscenes(self.tempdir.name, config=None, length=0)

        self.assertEqual(dataset.get_item_count(), 2)
        self.assertEqual(dataset.load_item(0)["value"], "plain")
        self.assertEqual(dataset.load_item(1)["value"], "gzip")

    def test_prefers_compressed_copy_without_double_counting(self):
        self.write_sample(0, "plain")
        self.write_sample(0, "gzip", compressed=True)
        dataset = Nuscenes(self.tempdir.name, config=None, length=0)

        self.assertEqual(dataset.get_item_count(), 1)
        self.assertEqual(dataset.load_item(0)["value"], "gzip")


if __name__ == "__main__":
    unittest.main()
