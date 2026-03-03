"""
Example: text classification pipeline.
"""

import torch

from open_autonlu.auto_classes import (
    TextClassificationTrainingPipeline,
)

TRAIN_PATH = "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_train.csv"
TEST_PATH = "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_test.csv"


def main():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("Warning: no GPU; training may be slow.")

    pipeline = TextClassificationTrainingPipeline(
        TRAIN_PATH,
        TEST_PATH,
        config_overrides={"language": "en"},
    )
    _ = pipeline.diagnose()
    training_artifact_info = pipeline.train()
    print("Selected method:", pipeline.training_method_cls.__name__)
    if training_artifact_info.test_metrics:
        print("Test F1:", training_artifact_info.test_metrics.f1)
    test_data = pipeline.splits.get("test")
    if test_data is not None and hasattr(pipeline.method_manager, "test"):
        training_artifact_info.test_metrics_inscope = pipeline.method_manager.test(
            test_data, inscope_only=True
        )
        if training_artifact_info.test_metrics_inscope:
            print(
                "Test F1 (in-scope only):",
                training_artifact_info.test_metrics_inscope.f1,
            )

    # Optional: save model and run inference
    # pipeline.save("saved_text_clf_torch", SaveFormat.TORCH)
    # pipeline.save("saved_text_clf_onnx", SaveFormat.ONNX)
    # manager = TextClassificationInferenceManager("saved_text_clf_torch")
    # outputs = manager.predict(["example sentence"], batch_size=32)
    # for out in outputs:
    #     print(out.most_probable.label, out.most_probable.score)


if __name__ == "__main__":
    main()
