import streamlit as st
from app import query_knowledge_base  # This pulls in your working RAG logic!

# 1. Set up the web browser window configuration
st.set_page_config(page_title="DocuMind AI", page_icon="🤖", layout="centered")
st.title("🤖 DocuMind Enterprise Assistant")
st.markdown("Ask any question regarding your custom loaded knowledge-base documents.")
st.divider()

# 2. Initialize a clean Chat History tracking state in the browser session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready. Ask me anything about your uploaded documents."}
    ]

# 3. Keep past chat bubbles visible on screen when typing new questions
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle a user's input when they press Enter or Send
if user_input := st.chat_input("Type your question here..."):
    
    # Render user's message immediately on screen
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Connect with your Pinecone/Llama 3 backend script to get the text response
    with st.chat_message("assistant"):
        with st.spinner("Searching Pinecone index & generating answer..."):
            try:
                # Triggers the return statement you just added!
                answer = query_knowledge_base(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                