import pandas as pd
from dao import connection


from dotenv import load_dotenv

load_dotenv()


def create_embeddings(doc):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()


def create_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = (
        "Você é um assistente de extração de dados."
        "Use o seguinte documento, e somente ele, para me devolver o que for"
        "solicitado."
        "Se você não sabe ou não consegue encontrar a informação, me avise."
        "Ser conciso."
        "\n\n"
        "{format_instructions}"
        "\n\n"
        "{context}"
    )
    setup = [("system", system_prompt), ("user", "{input}")]
    prompt = ChatPromptTemplate.from_messages(setup)
    return prompt


def create_parser():
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.output_parsers import JsonOutputParser

    class document_metadata(BaseModel):
        for term in desc:
            term: str = Field(description=f"{term}")
    return JsonOutputParser(pydantic_object=document_metadata)


def document_loader(doc):
    from langchain_community.document_loaders import PyPDFLoader
    print(doc)
    loader = PyPDFLoader(
        f"documents/TNPGE-documents/{doc}")
    return loader.load()


if __name__ == "__main__":
    import sys

    choice = "gpt-4o-mini"

    from pprint import pprint
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    prompt = create_prompt()

    parser = create_parser()

    doc = sys.argv[1]
    doc = document_loader(doc)

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=choice)
    question_answer_chain = create_stuff_documents_chain(
        llm=model, prompt=prompt, output_parser=parser
    )
    vectorstore = create_embeddings(doc)
    rag_chain = create_retrieval_chain(vectorstore, question_answer_chain)

    desc = []

    sql = """
            SELECT
                descricao
            FROM
                exames"""

    reg = connection.consultar_db(sql)

    df_bd = pd.DataFrame(reg, columns=["descricao"])

    for i, infos in df_bd.iterrows():
        desc.append(infos.descricao)

    prompt_terms = ""
    for term in desc:
        prompt_terms = " Qual o resultado:'%s' " % term + "? \n" + prompt_terms

    result = rag_chain.invoke(
        {
            "input": f"""
            Analise o texto em português brasileiro,
            extraia os seguintes dados:
            {prompt_terms}
            """,
            "format_instructions": parser.get_format_instructions(),
        },
    )
    pprint(result["answer"])
