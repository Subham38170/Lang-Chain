from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("Gen AI/Lang_Chain/Document_Loader/cricket.txt",encoding='utf-8')

documents = loader.load()




llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", #google/gemma-2-2b-it
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({'poem': documents[0].page_content})

print(result)

# print(type(documents))

# print(type(documents[0]))

# print(documents[0])