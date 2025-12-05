from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import PromptTemplate,load_prompt

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# python -m streamlit run "Gen AI/Lang_Chain/Prompt/Prompt_UI.py"


st.header('Reasearch Tool')

paper_input = st.selectbox(
    'Select Research Paper Name',
    ['Attention Is All You Need', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'GPT-3: Language Models are Few-Shot Learners','Diffusion Models Beat GANs on Image Synthesis' ]
    )

style_input = st.selectbox(
    'Select Explanation style',
    ['Beginner-Friendly','Technical','Code-Oriented','Mathematical']
)

length_input = st.selectbox(
    'Select Explanation Length',
    ['Short (1-2 paragraphs)','Medium (3-5 paragraphs)','Long (detailed explanation)']
)


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)


template = load_prompt("D:/AI ML/Gen AI/Lang_Chain/Prompt/template.json")

if st.button('Summarize'):
    prompt = template.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    }
    )
    result = model.invoke(prompt)

    result = model.invoke(paper_input + "Explain this paper in a " + style_input + " style and " + length_input + " length")
    st.write(result.content)
