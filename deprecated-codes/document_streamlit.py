import os
import streamlit as st

directory = "extract_inator/documents"


def save_uploaded_file(uploaded_file):
    with open(os.path.join(directory, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Arquivo {uploaded_file.name} salvo com sucesso!")


files = os.listdir(directory)

st.title("extract-inator")
st.header("protótipo para extração de metadados de artigos")


uploaded_file = st.file_uploader("faça upload de um PDF", type=["pdf"])

if uploaded_file is not None:
    save_uploaded_file(uploaded_file)
    files = os.listdir(directory)

selected_file = st.selectbox("arquivos disponiveis", files)

file_path = os.path.join(directory, selected_file)

with st.expander("Perguntar ao documento"):
    query = st.text_input("O que deseja buscar no arquivo selecionado?")
    if st.button("Enviar pergunta"):
        from langchain_community.document_loaders import PyPDFLoader

        # Load OS Vars
        from dotenv import load_dotenv

        load_dotenv()

        # Load document
        print(f"Arquivo analisado {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-3.5-turbo")

        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )

        retriever = vectorstore.as_retriever()

        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = (
            "Você é um assistente para extração de dados. "
            "Use o documento a seguir, e somente ele para me retornar o que for pedido. "
            "Se você não souber, ou não encontrar a informação me diga. "
            "Seja conciso."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        results = rag_chain.invoke({"input": query})

        st.write(results["answer"])


with st.expander("Extração de dados e estruturação"):
    exemple = {
        "authors": ["Alice Smith", "Bob Johnson"],
        "doi": "10.1234/example.doi.5678",
        "keywords": ["science", "research", "pydantic"],
    }
    st.write("dados que seram extraidos")
    st.json(exemple)

    if st.button("Enviar"):
        from langchain_community.document_loaders import PyPDFLoader

        # Load OS Vars
        from dotenv import load_dotenv

        load_dotenv()

        # Load document
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4-turbo")

        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )

        retriever = vectorstore.as_retriever()

        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.pydantic_v1 import BaseModel, Field

        class ArticleMetaData(BaseModel):
            authors: list[str] = Field(
                description="list of researchers of the article investigated"
            )
            doi: str = Field(
                description="""DOI, Digital Object Identifier, registration for any type
                    of digital file, scientific works, magazines, books, images and even
                    music that, when cataloged, have a permanent link to the published 
                    digital document."""
            )
            keywords: list[str] = Field(
                description="are a tool that helps indexers and search engines find relevant articles."
            )

        parser = JsonOutputParser(pydantic_object=ArticleMetaData)

        system_prompt = (
            "Você é um assistente para extração de dados. "
            "Use o documento a seguir, e somente ele para me retornar o que for pedido. "
            "Se você não souber, ou não encontrar a informação me diga. "
            "Seja conciso."
            "\n\n"
            "{format_instructions}"
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm=model, prompt=prompt, output_parser=parser
        )
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        results = rag_chain.invoke(
            {
                "input": "I would like to know who the authors are, the DOI, and the keywords of the content",
                "format_instructions": parser.get_format_instructions(),
            }
        )

        st.json(results["answer"])
