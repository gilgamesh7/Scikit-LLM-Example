# Scikit-LLM-Example
<br>
Scikit-LLM is an easy and efficient way to build ChatGPT-based text classification models using conventional scikit-learn compatible estimators without having to manually interact with OpenAI APIs.
<br><br>
Classification and labelling are common tasks in natural language processing (NLP). In traditional machine learning workflows these tasks would involve collecting labeled data, training a model, deploying it in the cloud, and making inferences. However, this process can be time-consuming, requiring separate models for each task, and not always yielding optimal results.
<br> <br>
With recent advancements in the area of large language models, such as ChatGPT, we now have a new way to approach NLP tasks. Rather than training and deploying separate models for each task, we can use a single model to perform a wide range of NLP tasks simply by providing it with a prompt.
<br><br>
In this article we will explore how to build the models for multiclass and multi-label text classification using ChatGPT as a backbone. To achieve this, we will use the scikit-LLM library, which provides a scikit-learn compatible wrapper around OpenAI REST API. Hence, allowing to build the model in the same way as you would do with any other scikit-learn model.

# Links
- [Scikit-LLM: NLP with ChatGPT in Scikit-Learn](https://medium.com/@iryna230520/scikit-llm-nlp-with-chatgpt-in-scikit-learn-733b92ab74b1)


# Installations
## To Install Poetry
- pip install poetry
- pip3 install --upgrade pip
- poetry init

## Recreate environment
- poetry config virtualenvs.in-project true
- poetry install
- Add/Remove
    - poetry add (package name)
    - poetry remove (package name)
    - List active venv : poetry env list
- Set up keys :
    - export OPENAI_API_KEY=<api key>
    - export OPENAI_ORG_KEY=<org id>