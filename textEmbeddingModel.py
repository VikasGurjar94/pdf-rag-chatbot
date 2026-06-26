from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
load_dotenv()

embeddings = MistralAIEmbeddings(
    model = "mistral-embed" , 
)

v1 = embeddings.embed_query("Vikas is a good boy")

print(v1)

