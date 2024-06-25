from langchain.evaluation.qa import QAGenerateChain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA


load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0)


chain = QAGenerateChain.from_llm(model)


loader = PyPDFLoader("extract_inator/documents/I-56670.pdf")
docs = loader.load()

exemples_to_evaluation = chain([{"doc": t} for t in docs[:5]])


embeddings = OpenAIEmbeddings()
VectorStore = DocArrayInMemorySearch.from_documents(docs, embeddings)


qa_article = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=VectorStore.as_retriever(search_kwargs={"k": 1}),
)

predictions = qa_article.apply(exemples_to_evaluation)


eval_chain = QAGenerateChain.from_llm(model)

graded_outputs = eval_chain.validate(exemples_to_evaluation, predictions)

for i, eg in enumerate(exemples_to_evaluation):
    print(f"Example {i}:")
    print("Question: " + predictions[i]["query"])
    print("Real Answer: " + predictions[i]["answer"])
    print("Predicted Answer: " + predictions[i]["result"])
    print("Predicted Grade: " + graded_outputs[i]["text"])
    print()
