"""
Example: text classification with OOD detection.
"""

import torch

from open_autonlu.auto_classes import (
    TextClassificationTrainingPipeline,
)
from open_autonlu.methods.data_types import OodMethod

TRAIN_PATH = "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_train.csv"
TEST_PATH = "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_test.csv"


def main():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("Warning: no GPU; training may be slow.")

    # Example: train with OOD as a separate class (LOGIT)
    pipeline = TextClassificationTrainingPipeline(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        config_overrides={
            "language": "en",
            "ood_method": OodMethod.LOGIT,
        },
    )
    _ = pipeline.diagnose()
    metrics = pipeline.train()
    print("Method:", pipeline.training_method_cls.__name__)
    if metrics.test_metrics:
        print("Test F1:", metrics.test_metrics.f1)
    test_data = pipeline.splits.get("test")
    if test_data is not None and hasattr(pipeline.method_manager, "test"):
        metrics.test_metrics_inscope = pipeline.method_manager.test(
            test_data, inscope_only=True
        )
        if metrics.test_metrics_inscope:
            print("Test F1 (in-scope only):", metrics.test_metrics_inscope.f1)

    # Optional: save and run inference
    # pipeline.save("saved_ood_torch", SaveFormat.TORCH)
    # pipeline.save("saved_ood_onnx", SaveFormat.ONNX)
    # manager = TextClassificationInferenceManager("saved_ood_torch")
    # test_df = pd.read_csv(TEST_PATH)
    # outputs = manager.predict(test_df.text.tolist(), 32)
    # labels_hat = [o.most_probable.label for o in outputs]
    # print("Inference F1:", f1_score(test_df.label, labels_hat, average="macro"))


if __name__ == "__main__":
    main()
