from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Step 1: Load the lecture notes PDF
loader = PyPDFLoader("data/ctse_notes.pdf")
documents = loader.load()

# Step 2: Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

print(f"✅ Total chunks: {len(docs)}")
print("🧩 Sample chunk content:\n")
print(docs[0].page_content)

# Step 3: Initialize HuggingFace Embedding Model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Test embedding
sample_text = docs[0].page_content
vector = embedding.embed_query(sample_text)
print(f"✅ Embedding length: {len(vector)}")
print(f"🧪 First 5 values of the vector: {vector[:5]}")

# Step 4: Store embeddings using Chroma
persist_directory = "./ctse_db"
vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)

# No need for vectordb.persist() as Chroma persists automatically in newer versions

print("📦 Vector store created and saved to disk!")

