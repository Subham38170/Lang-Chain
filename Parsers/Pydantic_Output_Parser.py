from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


class Person(BaseModel):
    name: str = Field(description='Name of the personn')
    age: int = Field(description='Age of the person')
    city: str = Field(description='City of the person the person belongs to')


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",  #google/gemma-2-2b-it
    task="conversational"
)
model = ChatHuggingFace(llm =llm)


parser  = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name,age and city of fiction {place} person\n{format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': 'Indian'})
print(template)

print(result)



