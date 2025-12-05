from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",  #google/gemma-2-2b-it
    task="conversational"
)
model = ChatHuggingFace(llm =llm)

schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()} 
)

# prompt = template.invoke({'topic': 'Albert Einstein'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)


chain = template | model | parser
result = chain.invoke({'topic','Albert Einstein'})
print(result)

