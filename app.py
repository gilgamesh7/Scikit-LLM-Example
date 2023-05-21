import os
import logging
from typing import List

import data.classification_dataset as sentiment_dataset
import data.multi_label_dataset as multi_label_dataset

from skllm.config import SKLLMConfig
from skllm import ZeroShotGPTClassifier, MultiLabelZeroShotGPTClassifier

logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {lineno} - {message}", style='{')
logger = logging.getLogger("sci-kit-llm")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)
OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY", default=None)

def  get_sentiments(feedback: List, training_labels: List)-> List[str]:
    # passed labelled training dataset to the classifier. This is done solely for making the API scikit-learn compatible. In fact, X is not used during training at all. Moreover, for y it is sufficient to provide candidate labels in an arbitrary order. 
    try:
        clf = ZeroShotGPTClassifier(openai_model = "gpt-3.5-turbo")
        # even if no labelled training data is available, the model can still be built (as shown below)
        # clf.fit(None, ['positive', 'negative', 'neutral'])
        clf.fit(feedback, training_labels)
        labels = clf.predict(feedback)

        return labels
    except Exception as err:
        raise err

def get_labels(descriptions: List, training_labels: List)-> List[str]:
    try:
        clf = MultiLabelZeroShotGPTClassifier(max_labels=3)
        clf.fit(descriptions, training_labels)
        labels = clf.predict(descriptions)

        return labels
    except Exception as err:
        raise err
try:
    logger.info("Setting up openai keys")
    SKLLMConfig.set_openai_key(OPENAI_API_KEY)
    SKLLMConfig.set_openai_org(OPENAI_ORG_KEY)

    logger.info("Starting sentiment analysis")
    X, y = sentiment_dataset.get_classification_dataset()
    labels = get_sentiments(X, y)
    logger.info(f"Sentiment Analysis labels : \n{labels}\n")

    logger.info("Starting multi-label analysis")
    X, y = multi_label_dataset.get_multilabel_classification_dataset()
    labels = get_labels(X, y)
    logger.info(f"Multi labels : \n{labels}\n")

except Exception as err:
    logger.error(f"{err}")