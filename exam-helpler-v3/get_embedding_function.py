from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODELS = ['llama3:70b', 'llama3.1:70b',
                 'mistral-large', 'nomic-embed-text']


def get_embedding_function():
    embeddings = OpenAIEmbeddings()
    # for code, model in enumerate(OLLAMA_MODELS):
    #     print(f'[{code}] - {model}')
    # choice = int(input('digite o código do modelo que sera utilizado: '))
    # embeddings = OllamaEmbeddings(model=OLLAMA_MODELS[choice])
    return embeddings
