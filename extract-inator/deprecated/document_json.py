from langchain_community.document_loaders import PyPDFLoader

# Load OS Vars
from dotenv import load_dotenv

load_dotenv()

# Load document
file_path = "documents/Manuscrito I^0S.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
# model = ChatOpenAI(model="gpt-4o")
# model = ChatOpenAI(model="gpt-4-turbo")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

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
    "Você esta analisando artigos cientificos."
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
        "input": "Gostaria de saber quem são os autores, o DOI e as palavras chave do artigo, descritores equivalem a palavras chave",
        "format_instructions": parser.get_format_instructions(),
    }
)
from pprint import pprint

pprint(results["answer"])
