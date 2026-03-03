DEFAULT_MODEL_BY_LANGUAGE: dict[str, str] = {
    "ru": "ai-forever/ruBert-base",
    "en": "bert-base-uncased",
}

DEFAULT_TEMPLATE_BY_LANGUAGE: dict[str, str] = {
    "ru": "Пользователь просит бота выполнить запрос с помощью навыка: ",
    "en": "User asks the bot to perform a request using the skill: ",
}
MARGIN = 0.25
NUM_ITERATION = 20
MAX_SEQ_LENGTH = 512
ADD_NORMALIZATION_LAYER = False
BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-4
SEED = 1223
OOS_LABEL = "outOfScope"
