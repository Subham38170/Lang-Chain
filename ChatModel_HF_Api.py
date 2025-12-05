from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from dataclasses import dataclass


# python -m streamlit run "Gen AI/Lang_Chain/ChatModel_HF_Api.py"

@dataclass
class Chat:
    isUser: bool
    time: str
    msg: str

MESSAGES = "messages"
CHAT_INPUT = 'chat_input'

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", #google/gemma-2-2b-it
    task="conversational"
)

model = ChatHuggingFace(llm=llm)
st.set_page_config(page_title="DeepSeek-AI Chat", page_icon="🤖")
st.title("🤖 DeepSeek-AI with LangChain")

if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = []

user_input = st.chat_input(key=CHAT_INPUT, placeholder="Write your query here ..")

if user_input:
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state[MESSAGES].append(Chat(isUser=True, time=timestamp, msg=user_input))
    st.session_state[MESSAGES].append(Chat(isUser=False, time="", msg="Typing..."))

for i, chat in enumerate(st.session_state[MESSAGES]):
    if chat.isUser:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 12px 0;'>
                <div style='max-width: 70%;'>
                    <div style='display: flex; justify-content: flex-end; align-items: center; margin-bottom: 4px;'>
                        <span style='font-weight: bold; color: #0078FF; margin-right: 6px;'>🧑 User</span>
                        <span style='font-size: 11px; color: gray;'>{chat.time}</span>
                    </div>
                    <div style='background-color: #0078FF; color: white; padding: 12px 16px;
                                border-radius: 20px 20px 0 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                                word-wrap: break-word; font-size: 14px; line-height: 1.4;'>
                        {chat.msg}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if chat.msg == "Typing...":
            try:
                result = model.invoke(st.session_state[MESSAGES][i-1].msg)
                st.session_state[MESSAGES][i].msg = result.content
                st.session_state[MESSAGES][i].time = datetime.now().strftime("%H:%M")
            except Exception as e:
                st.session_state[MESSAGES][i].msg = f"Error: {e}"
                st.session_state[MESSAGES][i].time = datetime.now().strftime("%H:%M")

        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin: 12px 0;'>
                <div style='max-width: 70%;'>
                    <div style='display: flex; justify-content: flex-start; align-items: center; margin-bottom: 4px;'>
                        <span style='font-weight: bold; color: #333; margin-right: 6px;'>🤖 Chatbot</span>
                        <span style='font-size: 11px; color: gray;'>{chat.time}</span>
                    </div>
                    <div style='background-color: #E5E5EA; color: black; padding: 12px 16px;
                                border-radius: 20px 20px 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                                word-wrap: break-word; font-size: 14px; line-height: 1.4;'>
                        {chat.msg}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
