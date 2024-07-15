import streamlit as st
import requests
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize model and tokenizer
model_name = "llm-llama-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def stream_content(url, data):
    """Make a streaming POST request and yield the 'content' part of each JSON response as they arrive."""
    headers = {'Content-Type': 'application/json'}
    
    # If data is a dictionary, convert it to a JSON string
    if isinstance(data, dict):
        data = json.dumps(data)
    
    # Make a streaming POST request
    with requests.post(url, data=data, headers=headers, stream=True) as response:
        # Raise an error for bad responses
        response.raise_for_status()
        
        # Process the stream
        for chunk in response.iter_lines():
            if chunk:
                # Decode chunk from bytes to string
                decoded_chunk = chunk.decode('utf-8')
                
                # Convert string to JSON
                json_chunk = json.loads(decoded_chunk)
                
                # if the model is done generating, return
                if json_chunk["done"] == True:
                    return
                # Yield the 'content' part
                yield json_chunk["message"]["content"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

request_data = {
    "model": "llm-llama-base",
    "messages": []
}

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare input for the model
    input_text = prompt
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate response using the model
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
