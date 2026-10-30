"""Direct NumPy collation preserves training/parallel tensors and ONNX feed contracts.

Run: python tests/test_runtime_adoption.py. No weights or optional runtimes.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import collate_items  # noqa: E402
from laya.onnx_agent import _collate_items_numpy  # noqa: E402


class CollationContract(unittest.TestCase):
    def test_mixed_width_training_rows_and_parallel_layout(self):
        rows = [[{"ids": [101, 9, 7], "markers": [1, 2], "qtype": 0,
                  "target": [0.25, 0.75], "label": 1, "tag": {"user": "first"},
                  "layout": {"position_ids": [0, 1, 1], "option_ids": [0, 1, 2]}},
                 {"ids": [101, 8], "markers": [1], "qtype": 1,
                  "layout": {"position_ids": [0, 1], "option_ids": [0, 1]}}]]
        batch = collate_items(rows, 99)
        expected = {"input_ids": [[101, 9, 7], [101, 8, 99]],
                    "attention_mask": [[1, 1, 1], [1, 1, 0]],
                    "marker_pos": [[1, 2], [1, 0]],
                    "marker_mask": [[True, True], [True, False]],
                    "target": [[0.25, 0.75], [0.0, 0.0]], "qtype": [0, 1], "label": [1, -1],
                    "position_ids": [[0, 1, 1], [0, 1, 0]], "option_ids": [[0, 1, 2], [0, 1, 0]]}
        for key, values in expected.items():
            self.assertEqual(batch[key].tolist(), values)
            dtype = torch.float32 if key == "target" else torch.bool if key == "marker_mask" else torch.int64
            self.assertEqual(batch[key].dtype, dtype)
            self.assertTrue(batch[key].is_contiguous())
        self.assertEqual(batch["meta"][0]["tag"], {"user": "first"})
        self.assertNotIn("layout", batch["meta"][0])
        batch["input_ids"][0, 0] = 17
        self.assertEqual(rows[0][0]["ids"][0], 101)

    def test_validation_and_empty_contract(self):
        self.assertIsNone(collate_items([], 0))
        self.assertEqual(_collate_items_numpy([], 0), {})
        row = {"ids": [1, 2], "markers": [1], "qtype": 0, "target": [1, 0]}
        with self.assertRaisesRegex(ValueError, "only 1 marker positions"):
            collate_items([[row, {"ids": [1, 2], "markers": [0, 1], "qtype": 0}]], 0)
        with self.assertRaisesRegex(ValueError, "some items carry a parallel layout"):
            collate_items([[{"ids": [1], "markers": [0], "qtype": 0,
                             "layout": {"position_ids": [0], "option_ids": [0]}},
                            {"ids": [1], "markers": [0], "qtype": 0}]], 0)

    def test_onnx_single_option_and_mixed_rows(self):
        for widths in ((1, 1), (1, 3)):
            rows = [[{"ids": [1, 2, 3], "markers": list(range(widths[0])), "qtype": 0}],
                    [{"ids": [4, 5], "markers": list(range(widths[1])), "qtype": 1}]]
            batch = _collate_items_numpy(rows, 99)
            self.assertEqual(set(batch), {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"})
            self.assertEqual(batch["marker_pos"].shape, (2, max(2, max(widths))))
            self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [4, 5, 99]])
            self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
            for key, array in batch.items():
                self.assertEqual(array.dtype, np.bool_ if key == "marker_mask" else np.int64)
                self.assertTrue(array.flags.c_contiguous)
            for i, width in enumerate(widths):
                self.assertEqual(batch["marker_mask"][i].tolist(), [True] * width + [False] * (batch["marker_pos"].shape[1] - width))


if __name__ == "__main__":
    unittest.main()
