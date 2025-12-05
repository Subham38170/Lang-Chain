from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal


class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback ')





load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",  #openai/gpt-oss-20b
    task='conversational'
)
model = ChatHuggingFace(llm = llm)

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the follwing feedback text is  positive or negative \n {feedback}\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']

)
classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative',prompt3 | model | parser1),
    RunnableLambda(    lambda x: 'Could not find sentiment')
)

chain = classifier_chain | branch_chain


result = chain.invoke({'feedback':'This is a amazing phone'})

print(result)
chain.get_graph().print_ascii()