"""Exercise the persisted dataset objects through the normal reader and writer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.base_serialization import SerializedBaseInstanceStore
from experiments.manifests.base import BaseBenchmarkRecord


class SavedDataTests(unittest.TestCase):
    def test_all_saved_base_objects_read_and_write_without_old_package(self):
        root = Path(__file__).resolve().parents[2]
        paths = sorted((root / "data").rglob("objects/*.npy"))
        self.assertTrue(paths)
        with TemporaryDirectory() as directory:
            for index, path in enumerate(paths):
                with self.subTest(path=path.relative_to(root)):
                    metadata = json.loads(path.with_suffix(".yaml").read_text())
                    record = BaseBenchmarkRecord.from_dict(metadata["record"]).with_manifest_path(
                        path.parent.parent / "manifest.json")
                    instance = SerializedBaseInstanceStore.load_instance(record)
                    self.assertIsNotNone(instance)
                    output_record = record.with_manifest_path(Path(directory) / str(index) / "manifest.json")
                    SerializedBaseInstanceStore.write_instance(output_record, instance)
                    loaded = SerializedBaseInstanceStore.load_instance(output_record)
                    self.assertEqual(instance.name, loaded.name)
                    self.assertEqual(list(instance.stgcs.G.nodes), list(loaded.stgcs.G.nodes))
                    self.assertEqual(list(instance.stgcs.G.edges), list(loaded.stgcs.G.edges))
                    for name in instance.stgcs.G.nodes:
                        before = instance.stgcs.get_vertex(name).st_hpoly
                        after = loaded.stgcs.get_vertex(name).st_hpoly
                        np.testing.assert_array_equal(before.A(), after.A())
                        np.testing.assert_array_equal(before.b(), after.b())
                    saved_metadata = json.loads(SerializedBaseInstanceStore.metadata_path(output_record).read_text())
                    self.assertEqual(metadata, saved_metadata)


if __name__ == "__main__":
    unittest.main()
