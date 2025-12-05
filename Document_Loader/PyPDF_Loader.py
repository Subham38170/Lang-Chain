from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader("Gen AI/Lang_Chain/Document_Loader/dl-curriculum.pdf",)


docs = loader.load()

print(type(docs))
print(type(docs[0]))
print(docs[0].metadata)
print(docs[0].page_content)