import os

from langchain_pymupdf4llm import PyMuPDF4LLMLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data/"
DB_PATH = "chroma_db/"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyMuPDF4LLMLoader(
            os.path.join(DATA_PATH, file)
        )
        documents.extend(loader.load())

print(f"Loaded documents: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 130
)

chunks = splitter.split_documents(documents)
print(f"chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("ChromaDB created Succesfully")