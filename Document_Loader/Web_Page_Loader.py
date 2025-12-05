from langchain_community.document_loaders import WebBaseLoader


from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", #google/gemma-2-2b-it
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

loader = WebBaseLoader()

url = "https://www.flipkart.com/apple-macbook-air-m4-16-gb-256-gb-ssd-macos-sequoia-mw123hn-a/p/itm08069ed2395aa?pid=COMH9ZWQDGMTF3HA&lid=LSTCOMH9ZWQDGMTF3HAIAWW11&marketplace=FLIPKART&sattr[]=color&sattr[]=system_memory&sattr[]=ssd_capacity&sattr[]=screen_size&st=system_memory"
#Here you can pass list of url's also
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

result= chain.invoke({'question':'Tell me about the specification of this laptop','text': docs[0].page_content})


print(result)