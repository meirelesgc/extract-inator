"""
Objetivo:
Analisar um PDF utilizando um LLM para extrair e formatar dados de forma que 
possam ser facilmente utilizados no Python. 

Funcionalidades:

**Análise de PDF:** 
O script utiliza um LLM para processar o conteúdo textual do PDF, extraindo 
informações relevantes e relevantes.

**Extração de dados:**
O script identifica e extrai dados estruturados do PDF, como tabelas, listas e 
valores numéricos.

**Formatação de dados:**
Os dados extraídos são formatados em um formato compatível com o Python, 
facilitando sua manipulação e análise.

Observações:

* A precisão da extração de dados pode variar dependendo da qualidade e da formatação do PDF.
* O script pode exigir personalização para lidar com diferentes tipos de PDFs e estruturas de dados.
* É recomendável revisar os dados extraídos manualmente para garantir a precisão e a consistência.
"""

from dotenv import load_dotenv

load_dotenv(override=True)

## Entrada com os dados de um documento escolhido


def select_document(document: str = "extract_inator/documents/"):
    import os

    files = os.listdir(document)
    for index, file in enumerate(files):
        print(f"[{index}] - {file}")
    choise = int(input("Digite o código do documento escolhido: "))
    return f"{document}{files[choise]}"


def document_loader(document: str):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(document)
    return loader.load()


def create_vector_store(document):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    return Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())


def select_model():
    from langchain_openai import ChatOpenAI

    models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
    for index, model in enumerate(models):
        print(f"[{index}] - {model}")
    choise = int(input("Digite o código do modelo escolhido: "))
    return ChatOpenAI(model=models[choise])


def create_few_shot_prompt():
    from langchain_core.prompts.few_shot import FewShotPromptTemplate
    from langchain_core.prompts.prompt import PromptTemplate

    exemples = [
        {
            "question": """
        Take the following information from the following article:
        list with the names of researchers.
        list of the article's keywords.
        D.O.I (Digital Object Identifier) of the article.
        """,
            "context": document_loader("extract_inator/documents/Artigo_1.pdf"),
            "answer": """
        Mauricio Lima Barreto
        Erika Aragão
        Luis Eugenio Portela Fernandes de Sousa
        Táris Maria Santana
        Rita Barradas Barata
        """,
        }
    ]


if __name__ == "__main__":

    print("Selecione um documento")
    document = select_document()

    print("Construindo os objetos...")
    doc = document_loader(document)

    print("Convertendo em vetores...")
    retriver = create_vector_store(doc)

    print("Selecione o modelo para gerar as respostas")
    model = select_model()

    # print("Dando contexto para o modelo...")
    # from langchain.chains.retrieval import create_retrieval_chain
    # from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

    # create_stuff_documents_chain(
    #     model,
    # )
    # retrieval_chain = create_retrieval_chain(retriver)
