from types import SimpleNamespace

import outlines

# ---------------------------------------------------------------------------
# English prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in creating diverse synthetic texts that accurately "
    "imitate the style, structure, and semantic features of original examples, "
    "while demonstrating a wide range of variations in structure and vocabulary. "
    "When generating, you should primarily follow the detailed linguistic profile "
    "of the label from the CATEGORY CHARACTERISTICS section, using examples only "
    "as additional illustrations of this profile, not as templates to copy. "
    "Your goal is to create a set of texts that reflect the full richness and "
    "diversity of possible expressions within the given category. "
    "You ALWAYS strictly follow instructions regarding the number of texts to generate."
)

ANALYZER_SYSTEM_PROMPT = (
    "You are a highly qualified linguistic analyst specializing in deep analysis "
    "of text corpora. Your task is to create a comprehensive and structured "
    "description of the text domain, which will be used for subsequent synthetic "
    "data generation. Remember that the quality and detail of your analysis will "
    "directly affect the AI model's ability to create diverse yet stylistically "
    "accurate texts."
)


@outlines.prompt
def generate_artificial_data(topic, texts, data_size):
    """Your task is to make up a dataset of textual requests, that would resemble user requests to a virtual assistant's skill "{{ topic }}", that look like the following:
    {% for example in texts %}
    {{ example }}
    {% endfor %}

    Respond only with a table with one column "text".
    Each text should not be longer than 10 words.
    You should provide {{ data_size }} texts."""


@outlines.prompt
def generate_texts(topic, texts, data_size, domain_desc=None, label_desc=None):
    """### TASK
    Generate EXACTLY {{ data_size }} new unique texts that belong to the category "{{ topic }}", preserve the characteristic linguistic features of this category, but contain new, original content.
    {% if domain_desc %}
    ### CONTEXT
    {{ domain_desc }}
    {% endif %}
    {% if label_desc %}
    ### CATEGORY CHARACTERISTICS "{{ topic }}"
    {{ label_desc }}
    {% endif %}
    ### EXAMPLE TEXTS OF CATEGORY "{{ topic }}"
    (Examples serve as additional general stylistic and structural guidance for generation)

    {% for example in texts %}
    - {{ example }}
    {% endfor %}

    ### REQUIREMENTS:
    1. Create texts that follow the GENERAL PATTERNS of the examples and category characteristics, but DO NOT copy the specific structure of individual examples
    2. Ensure MAXIMUM diversity in topics, structure, and vocabulary of texts. Avoid obvious literal repetition of the same constructions and words. Use alternative formulations and synonyms, while always following the general lexical and stylistic features of the examples and category description. Diversity is important but should not contradict the characteristic features and style of the original texts.
    3. Use syntactic constructions and parts of speech characteristic of the category
    4. Create NEW content that is contextually close to the category, but do not copy phrases from examples directly
    5. Preserve the stylistic features and emotional tone typical of the category
    6. Preserve the punctuation and formatting features characteristic of the category (for example, if punctuation is absent in the examples, this indicates that the generated texts should also follow this punctuation pattern)
    7. Generate EXACTLY {{ data_size }} texts — no more and no fewer

    IMPORTANT: Your goal is not to mechanically change individual words in examples, but to create new texts that could organically fit into the corpus of texts of this category, preserving their stylistic and structural features.

    ### OUTPUT FORMAT

    Present STRICTLY {{ data_size }} texts in table format:

    | text |
    | --- |
    | generated text1 |
    | generated text2 |
    | generated text3 |
    ...
    | generated text{{ data_size }} |

    IMPORTANT: The table must contain STRICTLY {{ data_size }} rows of texts, not counting the header and separator."""


@outlines.prompt
def analyze_domain(examples_by_label, label_names):
    """### TASK
    Analyze the text examples presented below by category and create a detailed structured description of the nature of the entire text corpus.

    ### TEXT EXAMPLES BY CATEGORY:

    {{ examples_by_label }}

    ### TASK CONTEXT:
    Based on your description, new synthetic texts will subsequently be generated that closely resemble the originals in style, topic, structure, and usage context. Your analysis should identify key linguistic patterns characteristic of each category, enabling the AI model to create diverse texts while preserving the specifics of the originals.

    ### INSTRUCTIONS:
    1. Identify and explicitly specify the general text type for the entire corpus.
    2. Identify and describe the main purpose of the texts and the context of their use.
    3. Describe the general topics covered by the presented texts.
    4. Highlight the general stylistic and syntactic features of the entire text corpus.
    5. For each text category, additionally compose a separate structured description according to the following plan:
       - Typical structure: characteristic sentence construction, average text length, types of constructions
       - Characteristic vocabulary: typical word groups, terminology, common parts of speech
       - Stylistic features: formality/informality, emotional tone, presence of humor or irony
       - Punctuation and formatting: features of punctuation usage (if there is no punctuation, explicitly state this), paragraph formats

    IMPORTANT: When describing vocabulary, try to reflect as fully as possible the diversity of possible words and thematic groups characteristic of each category. Do not limit yourself to only the text examples, but list all likely keywords and topics that may be relevant to the category based on your understanding of its essence.

    ### RESPONSE FORMAT (strictly follow the format):

    DOMAIN DESCRIPTION:
    Data type: [clear definition of the text type across the entire corpus]
    Text purpose: [briefly about the purpose and usage context]
    Main topics: [list the main thematic groups of texts identified during analysis]
    Stylistic and syntactic features: [briefly describe the style, tone, emotionality of texts, punctuation features, predominance of certain parts of speech, etc.]

    LABEL DESCRIPTIONS:
    {{ label_names }}:
    Structure: [description of typical text structure]
    Vocabulary: [description of characteristic vocabulary]
    Style: [description of stylistic features]
    Punctuation and formatting: [description of punctuation and formatting features]"""


# ---------------------------------------------------------------------------
# Russian prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT_RU = (
    "Вы — эксперт по созданию разнообразных синтетических текстов, которые точно "
    "имитируют стиль, структуру и семантические особенности оригинальных примеров, "
    "при этом демонстрируя широкий спектр вариаций в структуре и словарном запасе. "
    "При генерации вы должны в первую очередь следовать подробному лингвистическому профилю "
    "метки из раздела ХАРАКТЕРИСТИКИ КАТЕГОРИИ, используя примеры лишь "
    "как дополнительные иллюстрации этого профиля, а не как шаблоны для копирования. "
    "Ваша цель — создать набор текстов, отражающих всё богатство и "
    "разнообразие возможных выражений в рамках данной категории. "
    "Вы ВСЕГДА строго следуете инструкциям относительно количества текстов для генерации."
)

ANALYZER_SYSTEM_PROMPT_RU = (
    "Вы — высококвалифицированный лингвистический аналитик, специализирующийся на глубоком анализе "
    "текстовых корпусов. Ваша задача — создать всестороннее и структурированное "
    "описание текстового домена, которое будет использоваться для последующей генерации "
    "синтетических данных. Помните, что качество и детальность вашего анализа напрямую "
    "повлияют на способность ИИ-модели создавать разнообразные, но стилистически "
    "точные тексты."
)


@outlines.prompt
def generate_artificial_data_ru(topic, texts, data_size):
    """Ваша задача — составить набор данных из текстовых запросов, которые будут напоминать пользовательские запросы к навыку виртуального ассистента "{{ topic }}", похожие на следующие:
    {% for example in texts %}
    {{ example }}
    {% endfor %}

    Отвечайте только таблицей с одним столбцом "text".
    Каждый текст должен быть не длиннее 10 слов.
    Вы должны предоставить {{ data_size }} текстов."""


@outlines.prompt
def generate_texts_ru(topic, texts, data_size, domain_desc=None, label_desc=None):
    """### ЗАДАЧА
    Сгенерируйте РОВНО {{ data_size }} новых уникальных текстов, принадлежащих категории "{{ topic }}", сохраняя характерные лингвистические особенности данной категории, но содержащих новое, оригинальное содержание.
    {% if domain_desc %}
    ### КОНТЕКСТ
    {{ domain_desc }}
    {% endif %}
    {% if label_desc %}
    ### ХАРАКТЕРИСТИКИ КАТЕГОРИИ "{{ topic }}"
    {{ label_desc }}
    {% endif %}
    ### ПРИМЕРЫ ТЕКСТОВ КАТЕГОРИИ "{{ topic }}"
    (Примеры служат дополнительным общим стилистическим и структурным ориентиром для генерации)

    {% for example in texts %}
    - {{ example }}
    {% endfor %}

    ### ТРЕБОВАНИЯ:
    1. Создавайте тексты, следующие ОБЩИМ ЗАКОНОМЕРНОСТЯМ примеров и характеристикам категории, но НЕ копируйте конкретную структуру отдельных примеров
    2. Обеспечьте МАКСИМАЛЬНОЕ разнообразие тем, структуры и словарного запаса текстов. Избегайте очевидного буквального повторения одних и тех же конструкций и слов. Используйте альтернативные формулировки и синонимы, всегда следуя при этом общим лексическим и стилистическим особенностям примеров и описания категории. Разнообразие важно, но не должно противоречить характерным особенностям и стилю оригинальных текстов.
    3. Используйте синтаксические конструкции и части речи, характерные для данной категории
    4. Создавайте НОВОЕ содержание, контекстуально близкое к категории, но не копируйте фразы из примеров напрямую
    5. Сохраняйте стилистические особенности и эмоциональный тон, типичные для данной категории
    6. Сохраняйте особенности пунктуации и форматирования, характерные для данной категории (например, если в примерах отсутствует пунктуация, это означает, что сгенерированные тексты также должны следовать этому паттерну)
    7. Сгенерируйте РОВНО {{ data_size }} текстов — не больше и не меньше

    ВАЖНО: Ваша цель — не механически заменять отдельные слова в примерах, а создавать новые тексты, которые могли бы органично вписаться в корпус текстов данной категории, сохраняя их стилистические и структурные особенности.

    ### ФОРМАТ ВЫВОДА

    Представьте СТРОГО {{ data_size }} текстов в формате таблицы:

    | text |
    | --- |
    | сгенерированный текст1 |
    | сгенерированный текст2 |
    | сгенерированный текст3 |
    ...
    | сгенерированный текст{{ data_size }} |

    ВАЖНО: Таблица должна содержать СТРОГО {{ data_size }} строк текстов, не считая заголовка и разделителя."""


@outlines.prompt
def analyze_domain_ru(examples_by_label, label_names):
    """### ЗАДАЧА
    Проанализируйте представленные ниже примеры текстов по категориям и создайте подробное структурированное описание характера всего текстового корпуса.

    ### ПРИМЕРЫ ТЕКСТОВ ПО КАТЕГОРИЯМ:

    {{ examples_by_label }}

    ### КОНТЕКСТ ЗАДАЧИ:
    На основе вашего описания впоследствии будут сгенерированы новые синтетические тексты, максимально похожие на оригиналы по стилю, тематике, структуре и контексту использования. Ваш анализ должен выявить ключевые лингвистические паттерны, характерные для каждой категории, что позволит ИИ-модели создавать разнообразные тексты, сохраняя специфику оригиналов.

    ### ИНСТРУКЦИИ:
    1. Определите и явно укажите общий тип текста для всего корпуса.
    2. Определите и опишите основное назначение текстов и контекст их использования.
    3. Опишите общие темы, затрагиваемые в представленных текстах.
    4. Выделите общие стилистические и синтаксические особенности всего текстового корпуса.
    5. Для каждой категории текстов дополнительно составьте отдельное структурированное описание по следующему плану:
       - Типичная структура: характерное построение предложений, средняя длина текста, типы конструкций
       - Характерная лексика: типичные группы слов, терминология, часто встречающиеся части речи
       - Стилистические особенности: формальность/неформальность, эмоциональный тон, наличие юмора или иронии
       - Пунктуация и форматирование: особенности использования пунктуации (если пунктуация отсутствует, явно укажите это), форматы абзацев

    ВАЖНО: При описании лексики постарайтесь максимально полно отразить разнообразие возможных слов и тематических групп, характерных для каждой категории. Не ограничивайтесь только примерами текстов, а перечислите все вероятные ключевые слова и темы, которые могут быть релевантны для категории, исходя из вашего понимания её сути.

    ### ФОРМАТ ОТВЕТА (строго следуйте формату):

    ОПИСАНИЕ ДОМЕНА:
    Тип данных: [чёткое определение типа текста по всему корпусу]
    Назначение текстов: [кратко о назначении и контексте использования]
    Основные темы: [перечислите основные тематические группы текстов, выявленные при анализе]
    Стилистические и синтаксические особенности: [кратко опишите стиль, тон, эмоциональность текстов, особенности пунктуации, преобладание определённых частей речи и т.д.]

    ОПИСАНИЯ МЕТОК:
    {{ label_names }}:
    Структура: [описание типичной структуры текста]
    Лексика: [описание характерной лексики]
    Стиль: [описание стилистических особенностей]
    Пунктуация и форматирование: [описание особенностей пунктуации и форматирования]"""


# ---------------------------------------------------------------------------
# Language dispatcher
# ---------------------------------------------------------------------------


def get_prompts(language: str = "en") -> SimpleNamespace:
    """Return a namespace of prompts for the given language.

    Args:
        language: Language code ("en" or "ru").

    Returns:
        SimpleNamespace with attributes:
            default_system_prompt, analyzer_system_prompt,
            generate_texts, analyze_domain, generate_artificial_data,
            label_prefix, default_label_desc_template,
            domain_description_header, label_descriptions_header
    """
    if language == "ru":
        return SimpleNamespace(
            default_system_prompt=DEFAULT_SYSTEM_PROMPT_RU,
            analyzer_system_prompt=ANALYZER_SYSTEM_PROMPT_RU,
            generate_texts=generate_texts_ru,
            analyze_domain=analyze_domain_ru,
            generate_artificial_data=generate_artificial_data_ru,
            label_prefix="МЕТКА",
            default_label_desc_template=(
                "Тексты категории '{label}' соответствуют общим стилистическим и "
                "тематическим особенностям данного текстового корпуса, при этом имеют "
                "характерное содержание, связанное с темой {label}."
            ),
            domain_description_header="ОПИСАНИЕ ДОМЕНА:",
            label_descriptions_header="ОПИСАНИЯ МЕТОК:",
        )

    return SimpleNamespace(
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
        analyzer_system_prompt=ANALYZER_SYSTEM_PROMPT,
        generate_texts=generate_texts,
        analyze_domain=analyze_domain,
        generate_artificial_data=generate_artificial_data,
        label_prefix="LABEL",
        default_label_desc_template=(
            "Texts of category '{label}' correspond to the general stylistic and "
            "thematic features of this text corpus, while having "
            "characteristic content related to the topic {label}."
        ),
        domain_description_header="DOMAIN DESCRIPTION:",
        label_descriptions_header="LABEL DESCRIPTIONS:",
    )
