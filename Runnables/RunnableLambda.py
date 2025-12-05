from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace



llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it", #openai/gpt-oss-20b
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)



parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)




parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(lambda x: len(x.split()))
    }
)



final_chain = RunnableSequence(joke_gen_chain,parallel_chain)



print(final_chain.invoke({'topic': 'America'}))

final_chain.get_graph().print_ascii()