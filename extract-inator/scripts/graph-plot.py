from dotenv import load_dotenv

load_dotenv()


def document_loader():
    import os
    from langchain_community.document_loaders import PyPDFLoader

    docs = list()
    for doc in os.listdir("documents/graph-documents"):
        if doc.endswith(".pdf"):
            loader = PyPDFLoader(f"documents/graph-documents/{doc}")
            docs.append(loader.load())
    return docs


def create_embeddings(doc):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()


def select_model():
    from langchain_openai import ChatOpenAI

    models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
    for options, model in enumerate(models):
        print(f"[{options}] - {model.title()}")
    choice = int(input("Digite o código do modelo utilizado: "))
    return ChatOpenAI(model=models[choice])


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
        saldo_total: list[int] = Field(
            description="""
                Saldo  total de credito para o proximo faturamento.
                Essa informação fica na sessão de "INFORMAÇÕES IMPORTANTES"
                """
        )
        energia_injetada: list[int] = Field(
            description="""
                Energia injetada na unidade de microgeração durante o mês em kWh
                Essa informação fica na sessão de "INFORMAÇÕES IMPORTANTES
                """
        )

    return JsonOutputParser(pydantic_object=document_metadata)


if __name__ == "__main__":

    from langchain.chains.combine_documents import create_stuff_documents_chain

    prompt = create_prompt()

    parser = create_parser()

    model = select_model()

    question_answer_chain = create_stuff_documents_chain(
        llm=model, prompt=prompt, output_parser=parser
    )

    from langchain.chains import create_retrieval_chain
    from pprint import pprint

    results = list()
    docs = document_loader()
    print("Analisando os documentos: ")
    for index, doc in enumerate(docs):
        vectorstore = create_embeddings(doc)
        rag_chain = create_retrieval_chain(vectorstore, question_answer_chain)
        print(f"\ncarregando documento [{index}]")
        result = rag_chain.invoke(
            {
                "input": """
                Analise o texto em português brasileiro e extraia os seguintes dados:
                Quantos kWh de energia foram injetados por mês pela unidade de microgeração?
                Saldo  total de credito para o proximo faturamento?
                """,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        pprint(result["answer"])

    import matplotlib.pyplot as plt

    print("Montando o gráfico")
    T = range(len(result["answer"]["saldo_total"]))

    plt.figure(figsize=(10, 5))
    plt.plot(T, result["answer"]["saldo_total"], label="Saldo Total", marker="o")
    plt.plot(
        T, result["answer"]["energia_injetada"], label="Energia Injetada", marker="o"
    )

    plt.xlabel("Tempo (T)")
    plt.ylabel("Valores")
    plt.title("Gráfico de Saldo Total e Energia Injetada ao longo do Tempo")
    plt.legend()

    plt.grid(True)
    plt.savefig("graph.png")
