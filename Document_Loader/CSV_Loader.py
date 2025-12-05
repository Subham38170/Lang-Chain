from langchain_community.document_loaders import CSVLoader



loader = CSVLoader("Gen AI/Lang_Chain/Document_Loader/Social_Network_Ads.csv")
docs = loader.load()

print(len(docs))
print(docs[0].page_content)

