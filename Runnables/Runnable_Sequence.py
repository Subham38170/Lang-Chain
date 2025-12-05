from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace




llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", #google/gemma-2-2b-it
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a short joke about {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Explain the  following joke {joke}',
    input_variables=['joke']
)
parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)


print(chain.invoke({'topic': 'AI'}))