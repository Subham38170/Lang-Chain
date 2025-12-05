from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnableBranch,RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace




llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", #google/gemma-2-2b-it
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']

)

prompt2 = PromptTemplate(
    template='Summarize the flowwing text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()


report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain,branch_chain)


print(final_chain.invoke({'topic': 'Black hole'}))


final_chain.get_graph().print_ascii()