MAX_ANC_SET_FIT_SIZE = 5
MAX_SET_FIT_SIZE = 80
MIN_ANC_SET_FIT_SIZE = 2
LOWER_OPTIMAL_SETFIT_SIZE = 70
LOW_RES_CLS_FRACT_THRESHOLD = 0.3
SETFIT_NAME = "setfit"
ANCSETFIT_NAME = "ancsetfit"
FINETUNING_NAME = "finetuning"
MODEL_SAVE_PATH = "models"
TOKEN_CLF_FINETUNING = "token_clf_finetuning"
OOD_METHOD_MAP = {
    SETFIT_NAME: "SetFitOOD",
    ANCSETFIT_NAME: "AncSetFitMethod",
    FINETUNING_NAME: "FinetunerWithOOD",
    TOKEN_CLF_FINETUNING: "TokenClassificationFinetuner",
}
NO_OOD_METHOD_MAP = {
    SETFIT_NAME: "SetFitMethod",
    ANCSETFIT_NAME: "AncSetFitMethod",
    FINETUNING_NAME: "Finetuner",
    TOKEN_CLF_FINETUNING: "TokenClassificationFinetuner",
}

ALL_AVAILABLE_METHODS = [
    "AncSetFitMethod",
    "Finetuner",
    "SetFitMethod",
    "TokenClassificationFinetuner",
    "FinetunerWithOOD",
    "SetFitOOD",
    "AncSetFitOOD",
]

METHOD_HUMAN_READABLE_NAME_MAP = {
    "AncSetFitMethod": "AncSetFit",
    "AncSetFitOOD": "AncSetFit",
    "SetFitMethod": "SetFit",
    "SetFitOOD": "SetFit",
    "Finetuner": "Finetuning",
    "TokenClassificationFinetuner": "Finetuning",
    "FinetunerWithOOD": "Finetuning",
}

TRAIN_KEY = "train"
DEV_KEY = "dev"
TEST_KEY = "test"
UPSAMPLE_AUGMENTEX = "augmentex"
UPSAMPLE_LLM = "llm"
LLM_AUGMENTATION_THRESHOLD = 81

DATA_LOADING_STAGE = "loading data"
DATA_PROCESSING_STAGE = "processing data"
DATA_EVALUATION_STAGE = "evaluating data"
HPO_STAGE = "searching optimal hyperparameters"
PRETRAINING_STAGE = "pretraining"
TRAINING_STAGE = "training"
DEV_EVALUATION_STAGE = "evaluating the model"
SAVING_STAGE = "saving the model"
SPAN_DETECTION_ADAPTATION_STAGE = "span detection adaptation"
DQ_TRAIN = "scanning data"
