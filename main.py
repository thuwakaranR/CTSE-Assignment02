from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load the lecture notes PDF
loader = PyPDFLoader("data/ctse_notes.pdf")  # Make sure the file path is correct
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
