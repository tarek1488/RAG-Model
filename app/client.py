import streamlit as st
import requests

# Define the FastAPI URL where your RAG chain is running
FASTAPI_URL = "http://127.0.0.1:8000/query"

# Streamlit app
st.title("RAG Chain Query Interface")

# Input box for user to ask a question
query = st.text_input("Enter your question:")

if st.button("Submit Query"):
    if query:
        # Call the FastAPI server to get the response from the RAG chain
        try:
            with st.spinner('Fetching response...'):
                response = requests.post(FASTAPI_URL, json={"query": query})
                response.raise_for_status()  # Ensure the request was successful
                result = response.json()

                # Display the response
                st.success("Response from RAG Chain:")
                st.write(result.get("answer", "No answer found"))
                
                # If your RAG chain also returns sources, show them
                sources = result.get("sources", [])
                if sources:
                    st.write("Sources:")
                    for source in sources:
                        st.write(f"- {source}")

        except requests.HTTPError as e:
            st.error(f"Error fetching response: {e}")
    else:
        st.warning("Please enter a query.")
