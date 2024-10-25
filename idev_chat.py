import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Set up Streamlit app configuration
st.set_page_config(page_title="Chat with the KB chatbot AI, powered by iDev", page_icon="🦙", layout="centered")
st.title("Chat with the KB chatbot AI, powered by iDev")

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_key"]

# Initialize chat messages history in Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the KB uploaded."}
    ]

# Load document index with Streamlit caching
@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index

# Load the index and set up the chat engine
index = load_data()
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

# User input for the chatbot
prompt = st.chat_input("Ask a question")

# Handle chat input and responses within Streamlit
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response from the chat engine
    response_stream = st.session_state.chat_engine.stream_chat(prompt)
    response_content = response_stream.response
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
