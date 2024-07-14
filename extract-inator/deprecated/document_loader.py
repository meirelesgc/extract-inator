from langchain_community.document_loaders import PyPDFLoader

# Load OS Vars
from dotenv import load_dotenv

load_dotenv()

# Load document
file_path = "extract_inator/documents/Artigo_1.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(docs[0].page_content, end="\n\n")
print(docs[0].metadata, end="\n\n")
print(len(docs), end="\n\n")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

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
query = str(input("User: "))
results = rag_chain.invoke({"input": query})

print("\n\n\n", results["answer"])
