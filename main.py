from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline

# ✅ Step 1: Load PDF documents
print("📘 Loading PDF...")
loader = PyPDFLoader("data/ctse_notes.pdf")
documents = loader.load()
print(f"📄 Loaded {len(documents)} pages")

# ✅ Step 2: Split text into manageable chunks
print("✂️ Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(docs)}")

# ✅ Step 3: Generate embeddings using HuggingFace
print("🔍 Generating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embedding model loaded successfully")

# ✅ Step 4: Store vectors in Chroma DB
persist_directory = "./ctse_db"
print("💾 Creating Chroma vector store...")
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)
print("📦 Vector store created and saved")

# ✅ Step 5: Load open-source LLM using HuggingFace Pipeline
print("🤖 Loading open-source model (google/flan-t5-base)...")
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=512,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("✅ LLM pipeline ready")

# ✅ Step 6: Set up Retrieval QA chain
print("🧠 Setting up retrieval-based QA chain...")
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print("🔗 RetrievalQA chain is ready")

# ✅ Step 7: Ask questions in a loop
while True:
    query = input("\n❓ Ask a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("👋 Exiting...")
        break
    answer = qa_chain.invoke({"query": query})
    print("💡 Answer:", answer["result"])
