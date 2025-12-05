from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""


loader = PyPDFLoader("Gen AI/Lang_Chain/Text_Splitters/dl-curriculum.pdf")

docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size= 100,  #How many characters should contain in each chunk
    chunk_overlap=0,  #How many characters should overlap in each chunk
)
#result = splitter.split_text(text=text)
result = splitter.split_documents(docs)
print(result)
#print(result[0].page_content)

