"""
Example: text classification with LLM-based augmentation for low-resource classes.
Requires MODEL_API_KEY and BASE_URL (and optionally MODEL_ID) in the environment.
"""

import os
import torch

from open_autonlu.auto_classes import TextClassificationTrainingPipeline

TRAIN_PATH = "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_train.csv"
TEST_PATH = "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_test.csv"


def main():
    api_key = os.environ.get("MODEL_API_KEY", "")
    base_url = os.environ.get("BASE_URL", "").strip()
    model_id = os.environ.get("MODEL_ID", "").strip()
    if not api_key or not base_url:
        print(
            "Set MODEL_API_KEY and BASE_URL (and optionally MODEL_ID) "
            "in the environment to run LLM augmentation."
        )
        return
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("Warning: no GPU; training may be slow.")

    pipeline = TextClassificationTrainingPipeline(
        TRAIN_PATH,
        TEST_PATH,
        config_overrides={
            "language": "en",
            "llm_augmentation": {
                "enabled": True,
                "use_domain_analysis": True,
                "max_attempts": 20,
                "config_overrides": {
                    "LlmClientConfig": {
                        "api_key": api_key,
                        "base_url": base_url,
                        "model_id": model_id,
                    }
                },
            },
        },
    )
    _ = pipeline.diagnose()
    training_artifact_info = pipeline.train()
    print("Method:", pipeline.training_method_cls.__name__)
    if training_artifact_info.test_metrics:
        print("Test F1:", training_artifact_info.test_metrics.f1)


if __name__ == "__main__":
    main()
