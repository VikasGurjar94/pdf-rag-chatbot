# from dotenv import load_dotenv 
# load_dotenv()
# print("vikas")
from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = "data/data.pdf"
loader = PyPDFLoader(file_path=FILE_PATH)

# Load all documents
docs = loader.load()

# print(type(documents))

for doc in docs:
    print( doc.page_content)