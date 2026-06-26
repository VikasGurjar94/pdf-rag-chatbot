# -------------------- text embed model --------------------------
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
load_dotenv()

embeddings = MistralAIEmbeddings(
    model = "mistral-embed" , 
)

# v1 = embeddings.embed_query("Vikas is a good boy")

# print(v1)

#--------------------------------------------------------------------------------


#------------- in memory vector store example ------------------

from langchain_core.vectorstores import InMemoryVectorStore


vector_store = InMemoryVectorStore(embeddings)
# Add Document objects (holding page_content and optional metadata) like so:
# vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])

#delete documen
# vector_store.delete(ids=["id1"])
similar_docs = vector_store.similarity_search("Does chat gpt understands like a human ? ")

for doc in similar_docs: 
    print(doc.page_content)

# similar_docs = vector_store.similarity_search("your query here" , optional some k value also can be given(ex. k = 2))