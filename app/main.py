import google.generativeai as genai
import os
from dotenv import load_dotenv
from process import read_embeddings, retrieve_relevant_resources, prompt_formatter
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize embeddings and resources
embeddings, pages_and_chunks = read_embeddings()

# Function to process the query and generate a response
def generate_response(query):
    # Retrieve relevant resources
    score, indices, time_taken = retrieve_relevant_resources(query, embeddings)
    context_items = [pages_and_chunks[i] for i in indices]
    prompt = prompt_formatter(query, context_items)
    
    # Configure API
    api_key = os.getenv('GEMINI_KEY')
    genai.configure(api_key=api_key)
    
    # Safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # Initialize model
    model = genai.GenerativeModel(model_name='gemini-1.5-flash', safety_settings=safety_settings)
    
    # Generate content
    response = model.generate_content(prompt)
    return response.text, time_taken

# Streamlit UI setup
st.title("RAG Query Generation")
st.subheader("Powered by GEMINI (Generative AI Embedding Model)")
url = "https://pressbooks.oer.hawaii.edu/humannutrition2/"
st.write("Source PDF: %s" % url)

# User input
query = st.text_input("Enter your query:", "How often should I eat protein?")

if st.button("Generate Response"):
    if query:
        response, time_taken = generate_response(query)
        st.subheader("Response:")
        st.write(response)
        st.subheader("Time Taken to serach from source:")
        st.write(str(time_taken), "seconds")  # Format to remove microseconds
    else:
        st.warning("Please enter a query to generate a response.")

