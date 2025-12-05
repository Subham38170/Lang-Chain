from langchain_huggingface import HuggingFaceEmbeddings

#all-MiniLM-L6-v2
#This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = 'Delhi is the capital of India'
documents = [
    'Delhi is the captial of India',
    'Kolkata is the capital of West Bengal',
    'Paris is the capital of France'
]
vector = embedding.embed_query(text)
print(vector)
vectors = embedding.embed_documents(documents)
print(vector)