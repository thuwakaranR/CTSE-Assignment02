from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Load the lecture notes PDF
loader = PyPDFLoader("data/ctse_notes.pdf")  # Ensure the file path is correct
documents = loader.load()

# Step 2: Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Step 3: Check results
print(f"✅ Total chunks: {len(docs)}")
print("🧩 Sample chunk content:\n")
print(docs[0].page_content)

# Step 4: Initialize the embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 5: Test embedding on the first chunk
sample_text = docs[0].page_content
vector = embedding.embed_query(sample_text)

print(f"✅ Embedding length: {len(vector)}")
print(f"🧪 First 5 values of the vector: {vector[:5]}")
