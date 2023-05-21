import os
import logging
from typing import List

import data.classification_dataset as sentiment_dataset

from skllm.config import SKLLMConfig
from skllm import ZeroShotGPTClassifier

logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {lineno} - {message}", style='{')
logger = logging.getLogger("sci-kit-llm")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)
OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY", default=None)

def  get_sentiments(feedback: List)-> List[str]:
    try:
        clf = ZeroShotGPTClassifier(openai_model = "gpt-3.5-turbo")
        clf.fit(None, ['positive', 'negative', 'neutral'])
        labels = clf.predict(feedback)

        return labels
    except Exception as err:
        raise err

try:
    logger.info("Setting up openai keys")
    SKLLMConfig.set_openai_key(OPENAI_API_KEY)
    SKLLMConfig.set_openai_org(OPENAI_ORG_KEY)

    logger.info("Starting sentiment analysis")

    X, _ = sentiment_dataset.get_classification_dataset()
    labels = get_sentiments(X)

    logger.info(f"\n{labels}\n")


except Exception as err:
    logger.error(f"{err}")