from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model="gpt-3.5-turbo")


# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# config = {"configurable": {"session_id": "abc2"}}
# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm xpto")],
#     config=config,
# )

# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config,
# )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm xpto")], "language": "Spanish"}
)

response.content
