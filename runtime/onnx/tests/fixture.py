"""Create an offline runtime contract fixture; no Torch or Transformers required."""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers


def create_fixture(directory):
    directory = Path(directory)
    tokenizer_dir = directory / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "billing", "help"]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(tokens[:5])
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    special = dict(zip(("pad_token", "unk_token", "cls_token", "sep_token", "mask_token"), tokens[:5]))
    (tokenizer_dir / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", **special}), encoding="utf-8")
    (directory / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "offline-fixture", "head_layers": 1, "max_len": 128, "head_max_len": 64,
        "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
    }), encoding="utf-8")
    inputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
        ("input_ids", TensorProto.INT64, ["N", "L"]),
        ("attention_mask", TensorProto.INT64, ["N", "L"]),
        ("marker_pos", TensorProto.INT64, ["N", "K"]),
        ("marker_mask", TensorProto.BOOL, ["N", "K"]),
        ("qtype", TensorProto.INT64, ["N"]),
    )]
    initializers = [numpy_helper.from_array(value, name) for name, value in (
        ("scale", np.array(-0.1, dtype=np.float32)),
        ("masked", np.array(-1e9, dtype=np.float32)),
        ("two", np.array([2], dtype=np.int64)),
    )]
    nodes = [helper.make_node("Cast", ["marker_pos"], ["positions"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["positions", "scale"], ["scores"]),
             helper.make_node("Where", ["marker_mask", "scores", "masked"], ["logits"]),
             helper.make_node("Shape", ["qtype"], ["rows"]),
             helper.make_node("Concat", ["rows", "two"], ["act_shape"], axis=0),
             helper.make_node("ConstantOfShape", ["act_shape"], ["act_logits"],
                              value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]))]
    graph = helper.make_graph(nodes, "offline-contract", inputs, [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", "K"]),
        helper.make_tensor_value_info("act_logits", TensorProto.FLOAT, ["N", 2]),
    ], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    return directory


if __name__ == "__main__":
    import sys

    create_fixture(sys.argv[1])
