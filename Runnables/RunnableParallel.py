from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace




llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", #google/gemma-2-2b-it
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {joke}',
    input_variables=['joke']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1,model,parser),
        'linkedin': RunnableSequence(prompt2,model,parser)
    }
)

print(parallel_chain.invoke({'topic': 'AI'}))