import streamlit as st
import pdfplumber
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import uuid
# import base64


# Configure Google Gemini API Key
API_KEY = "***********************************"
genai.configure(api_key=API_KEY)


# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS Index
dim = 384  # Embedding size for MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(dim)
document_store = {}

# Process uploaded PDF documents
def process_uploaded_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        if content:
            docs.append({"text": content, "id": str(uuid.uuid4())})
    return docs

# Create VectorDB Index
def create_vector_index(docs):
    global faiss_index
    for doc in docs:
        embedding = embedding_model.encode(doc["text"]).astype(np.float32)
        faiss_index.add(np.array([embedding]))
        document_store[len(document_store)] = doc["text"]  # Store doc in dictionary

# Retrieve relevant documents
def retrieve_documents(query, k=3):
    if faiss_index.ntotal == 0:
        return "No documents available."
    
    query_embedding = embedding_model.encode(query).astype(np.float32)
    D, I = faiss_index.search(np.array([query_embedding]), k)
    retrieved_texts = [document_store[i] for i in I[0] if i in document_store]
    return "\n".join(retrieved_texts) if retrieved_texts else "No relevant documents found."

# Generate AI Response using Gemini 1.5 Flash
def query_gemini_flash(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Voice Input (Speech-to-Text)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write("🎤 Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError:
            return "Speech recognition service is unavailable."

# Streamlit UI
st.set_page_config(page_title="Friendly AI Chatbot", layout="wide")
st.title("🤖 AI CHATBOT")

st.sidebar.write("💡 **QUERY BUDDY**")
st.sidebar.write("📖 **Upload PDF documents to enhance AI knowledge**")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    docs = process_uploaded_files(uploaded_files)
    if docs:
        create_vector_index(docs)
        st.sidebar.success("Documents processed and indexed successfully!")

# Voice Input Button
if st.sidebar.button("🎙️ Speak Your Query"):
    user_input = recognize_speech()
    st.sidebar.write(f"🗣️ You said: {user_input}")
else:
    user_input = st.chat_input("Ask me anything!")

if user_input:
    # Handle simple greetings
    if user_input.lower() in ["hi", "hello", "hey"]:
        response = "Hi! How can I help you today?"
    else:
        # Retrieve relevant documents
        context = retrieve_documents(user_input)
        
        # Generate final prompt for Gemini
        final_prompt = f"Context:\n{context}\n\nUser Query:\n{user_input}"
        response = query_gemini_flash(final_prompt)
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
