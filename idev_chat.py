import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading

# Initialize FastAPI app
api_app = FastAPI()

# Set up Streamlit configurations
st.set_page_config(page_title="Chat with the KB chatbot AI, powered by iDev", page_icon="🦙", layout="centered")
st.title("Chat with the KB chatbot AI, powered by iDev")

# Load OpenAI API key
openai.api_key = st.secrets["openai_key"]

# Initialize chat history in Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the KB uploaded."}
    ]

# Define a FastAPI model for input data
class ChatRequest(BaseModel):
    question: str

# Load the document index with caching
@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index

# Load the index
index = load_data()

# Initialize chat engine if not set
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

# Chat input and response handling in Streamlit
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat history and generate response
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is from the user, generate an assistant response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        response_content = response_stream.response
        for content in response_stream.response_gen:
            st.write(content)
        # Add the response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

# FastAPI endpoint to handle external chat requests
@api_app.post("/chat")
async def chat(request: ChatRequest):
    prompt = request.question
    response_stream = st.session_state.chat_engine.stream_chat(prompt)
    response_content = response_stream.response
    return {"response": response_content}

# Function to run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# Start FastAPI in a new thread
threading.Thread(target=run_fastapi, daemon=True).start()
