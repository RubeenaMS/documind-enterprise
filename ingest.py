import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# 1. Load environment variables
load_dotenv()

# Configuration Variables
PDF_PATH = "data/sample_policy.pdf"  
INDEX_NAME = "documind-index"

def run_ingestion():
    # 2. Extract Text from PDF
    if not os.path.exists(PDF_PATH):
        print(f"[-] Error: Please place a sample PDF file at: {PDF_PATH}")
        return

    print(f"[+] Loading PDF document from: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"[+] Successfully loaded {len(raw_documents)} pages.")

    # 3. Chunk Text Smartly
    print("[+] Chunking document text into overlapping segments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"[+] Generated {len(chunks)} text chunks.")

    # 4. Initialize OpenAI Embeddings Model
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

    # 5. Connect and Provision Pinecone Vector Database
    print("[+] Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Create the index if it doesn't exist yet
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"[+] Index '{INDEX_NAME}' not found. Creating a new one...")
        # Look for where your script creates the index and ensure dimension is 384:
        pc.create_index(
        name=INDEX_NAME,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
        print("[+] Pinecone index provisioned successfully.")
    else:
        print(f"[+] Index '{INDEX_NAME}' already exists.")

    # 6. Upload Chunks & Metadata to Pinecone
    print(f"[+] Upserting text segments into index '{INDEX_NAME}'...")
    
    index = pc.Index(INDEX_NAME)
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        vector_val = embeddings.embed_query(chunk.page_content)
        
        metadata = {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page", 0) + 1 
        }
        vectors_to_upsert.append((f"chunk_{i}", vector_val, metadata))

    index.upsert(vectors=vectors_to_upsert)
    print("[+++] Ingestion pipeline execution complete! Codebase synced cleanly.")

if __name__ == "__main__":
    run_ingestion()
