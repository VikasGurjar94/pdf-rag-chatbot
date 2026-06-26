#--------------documet loader ------------

from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = "data/data.pdf"
loader = PyPDFLoader(file_path=FILE_PATH)

# Load all documents
document = loader.load()



#-------------- text splitter -------------
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2)

print(document[0].page_content)

texts = text_splitter.split_text(document[0].page_content)
print(texts[0:20])
# texts = text_splitter.split_text(document[0].page_content)

# print(texts[:10])