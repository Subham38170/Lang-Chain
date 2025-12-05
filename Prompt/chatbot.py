from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage('You are a helpful AI assistant.')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print('AI : ',result.content)



