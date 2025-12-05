from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)


messages = [
    SystemMessage('You are a helpful assistant'),
    HumanMessage('Tell me about LangChain')
]
result = model.invoke(messages)
messages.append(AIMessage(result.content))
print(messages)