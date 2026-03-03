"""
Example: NER pipeline.
"""

import torch

from open_autonlu.auto_classes import (
    TokenClassificationTrainingPipeline,
)

TRAIN_PATH = "examples/test_data/noise_n_shot_data_ner/train.json"
TEST_PATH = "examples/test_data/noise_n_shot_data_ner/test.json"


def main():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("Warning: no GPU; training may be slow.")

    pipeline = TokenClassificationTrainingPipeline(
        TRAIN_PATH,
        TEST_PATH,
        config_overrides={"language": "en"},
    )
    _ = pipeline.diagnose()
    training_artifact_info = pipeline.train()
    print("Selected method:", pipeline.training_method_cls.__name__)
    if training_artifact_info.test_metrics:
        print("Test F1:", training_artifact_info.test_metrics.f1)

    # Optional: save and run inference
    # pipeline.save("saved_ner_torch", SaveFormat.TORCH)
    # pipeline.save("saved_ner_onnx", SaveFormat.ONNX)
    # manager = TokenClassificationInferenceManager("saved_ner_torch")
    # sample_texts = ["Sample query one.", "Another example text."]
    # results = manager.predict(sample_texts, batch_size=32)
    # for r in results:
    #     print(r.text, [(e.label, e.text) for e in r.labels])


if __name__ == "__main__":
    main()
