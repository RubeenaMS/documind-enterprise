import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone

# 1. Load environment variables
load_dotenv()

INDEX_NAME = "documind-index"

def query_knowledge_base(query_text: str):
    print(f"\n🚀 Initializing Search for: '{query_text}'...")

    # 2. Re-initialize the exact same 384-dimension HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    # 3. Connect to Pinecone index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    # 4. Vectorize the user's question
    print("[+] Vectorizing query text...")
    query_vector = embeddings.embed_query(query_text)

    # 5. Search Pinecone for top 3 matching chunks
    print("[+] Querying Pinecone vector database...")
    search_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # 6. Extract context chunks
    context_chunks = []
    for match in search_results.get('matches', []):
        if 'metadata' in match and 'text' in match['metadata']:
            context_chunks.append(match['metadata']['text'])
    
    if not context_chunks:
        print("[-] No matching context found in database.")
        return

    context = "\n---\n".join(context_chunks)
    print(f"[+] Retrieved {len(context_chunks)} relevant source document segments.")

    # 7. Setup the Base LLM structure (Swapped to Llama 3 for better stability)
    print("[+] Connecting to Hugging Face serverless chat system...")
    base_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=512,
        temperature=0.2,
    )
    # Wrap it to explicitly support the 'conversational' task mapping
    chat_llm = ChatHuggingFace(llm=base_llm)

    # 8. Structure the query as standard Chat Messages
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Answer the user's question using ONLY the provided document context. "
            "If you do not know the answer based on the context, say exactly 'I cannot find the answer in the provided document.'\n\n"
            f"Context:\n{context}"
        )),
        HumanMessage(content=query_text)
    ]

    # 9. Invoke the chat model and print response
    print("[+] Generating answer...")
    response = chat_llm.invoke(messages)
    
    print("\n==================== ANSWER ====================")
    print(response.content.strip())
    print("================================================")

if __name__ == "__main__":
    print("🤖 DocuMind Chatbot Initialized! Type 'exit' or 'quit' to stop.")
    print("-" * 50)
    
    while True:
        # This pauses the script and waits for keyboard input from the user
        user_question = input("\n💬 Ask a question about your documents: ")
        
        # Check if the user wants to leave the chat
        if user_question.strip().lower() in ['exit', 'quit']:
            print("Goodbye! Shutting down system.")
            break
            
        # Skip empty inputs
        if not user_question.strip():
            continue
            
        # Run the RAG pipeline with whatever the user just typed
        query_knowledge_base(user_question)