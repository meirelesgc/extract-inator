from dotenv import load_dotenv

load_dotenv()


def create_embeddings(doc):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()


def create_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = (
        "Você é um assistente de extração de dados."
        "Use o seguinte documento, e somente ele, para me devolver o que for solicitado."
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
        data: list[str] = Field(
            description="""
            Data de referencia do documento, apenas o mês e o ano.
            Essa informação fica logo abaixo de  "REF:MÊS/ANO"
            """
        )
        saldo_total: list[int] = Field(
            description="""
                Saldo  total de credito para o proximo faturamento.
                Essa informação fica na sessão de "INFORMAÇÕES IMPORTANTES"
                """
        )
        energia_injetada: list[int] = Field(
            description="""
                Energia injetada na unidade de microgeração durante o mês em kWh.
                Essa informação fica na sessão de "INFORMAÇÕES IMPORTANTES
                """
        )
        CAT: list[int] = Field(
            description="""
                No valor do consumo faturado está incluído o ajuste na(s)  função(ões) CAT de:
                Essa informação fica na sessão de "INFORMAÇÕES IMPORTANTES
                """
        )

    return JsonOutputParser(pydantic_object=document_metadata)


if __name__ == "__main__":
    import sys

    choice = "gpt-4-turbo"
    from langchain.chains import create_retrieval_chain
    from pprint import pprint

    import os
    from langchain_community.document_loaders import PyPDFLoader

    doc = sys.argv[1]
    print(doc)
    loader = PyPDFLoader(f"documents/graph-documents/{doc}")
    docs = loader.load()
    from langchain.chains.combine_documents import create_stuff_documents_chain

    prompt = create_prompt()

    parser = create_parser()

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=choice)
    question_answer_chain = create_stuff_documents_chain(
        llm=model, prompt=prompt, output_parser=parser
    )
    vectorstore = create_embeddings(docs)
    rag_chain = create_retrieval_chain(vectorstore, question_answer_chain)
    result = rag_chain.invoke(
        {
            "input": """
            Analise o texto em português brasileiro e extraia os seguintes dados:
            
            Qual o mês e o ano que esse documento se refere, esta localizado em "REF:MÊS/ANO"?
            Quantos kWh de energia foram injetados por mês pela unidade de microgeração?
            Saldo  total de credito para o proximo faturamento?
            Complete: No valor do consumo faturado está incluído o ajuste na(s)  função(ões) CAT de:
            
            Todos os documentos possuem essas informações.
            """,
            "format_instructions": parser.get_format_instructions(),
        },
    )
    pprint(result["answer"])
