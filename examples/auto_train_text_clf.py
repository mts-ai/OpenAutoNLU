from open_autonlu.auto_classes import TextClassificationTrainingPipeline
import logging

logging.basicConfig()

pipeline = TextClassificationTrainingPipeline(
    "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_train.csv",
    "examples/test_data/noise_n_shot_data/snips_200_300_0.01_42_test.csv",
    config_overrides={"language": "en"},
)
dq_output = pipeline.diagnose()
print(dq_output)
training_artifact_info = pipeline.train()
print(training_artifact_info)
# pipeline.save("setfit_test_model_torch", SaveFormat.TORCH)
