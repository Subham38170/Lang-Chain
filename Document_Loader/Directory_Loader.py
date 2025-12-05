from langchain_community.document_loaders import DirectoryLoader,PyPDFDirectoryLoader,PyPDFLoader


loader = DirectoryLoader(
    path="Gen AI/Lang_Chain/Document_Loader/Books",
    glob='*.pdf',  #To load which files
    loader_cls = PyPDFLoader
    )

docs = loader.load()


print("Total PDF's : ",len(set(doc.metadata['source'] for doc in docs)))


print(len(docs))