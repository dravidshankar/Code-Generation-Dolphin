import streamlit as st
import json
from llama_cpp import LLaMA

# Initialize LLaMA model
llama_model = LLaMA("llama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request data
    request_data = {
        "prompt": prompt
    }

    # Send request to LLaMA model
    response = llama_model.generate(request_data)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
