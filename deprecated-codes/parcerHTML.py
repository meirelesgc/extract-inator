from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", temperature=0)

loader = PyPDFLoader("extract_inator/documents/Manuscrito I^0S.pdf")
docs = loader.load()

embeddings = OpenAIEmbeddings()
VectorStore = DocArrayInMemorySearch.from_documents(docs, embeddings)


qa_article = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=VectorStore.as_retriever(search_kwargs={"k": 1}),
    verbose=True,
)

result = qa_article.run("me retorne os autores")

print(result)
