import json
import os
from dataclasses import asdict
from pathlib import Path

import torch


class GenericSequenceClassifierONNXExportMixin:
    def export_onnx(self, path_to_package):
        package_dir = Path(path_to_package)
        package_dir.mkdir(exist_ok=True)
        with (package_dir / "label_mapping.json").open("w") as f:
            json.dump(asdict(self.model.config), f)
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "token_type_ids": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        }
        dummy_inputs = (
            self.tokenizer(["dummy", "inputs"], return_tensors="pt", padding=True)
            .to(next(self.model.parameters()).device)
            .data
        )
        self.model.eval()
        torch.onnx.export(
            model=self.model,
            args=dummy_inputs,
            f=os.path.join(path_to_package, "model.onnx"),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=19,
        )
