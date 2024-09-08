import streamlit as st
import requests

# FastAPI endpoints
FASTAPI_QUERY_URL = "http://127.0.0.1:8000/query"
FASTAPI_SESSION_URL = "http://127.0.0.1:8000/get-session"

# Check if session_id exists in Streamlit session state, if not, request one
if 'session_id' not in st.session_state:
    response = requests.get(FASTAPI_SESSION_URL)
    if response.status_code == 200:
        session_data = response.json()
        session_id = session_data.get("session_id")
        # Store the session ID in Streamlit's session state
        st.session_state['session_id'] = session_id
        #st.success(f"Session ID: {session_id}")
else:
    session_id = st.session_state['session_id']

# Define the RAG config
config = {"configurable": {"session_id": session_id}}

# Streamlit app interface
st.title("RAG Chain Query Interface")

# Input box for user query
query = st.text_input("Enter your question:")

# Submit button for the query
if st.button("Submit Query"):
    if query:
        try:
            # Show a spinner while waiting for the result
            with st.spinner('Fetching response...'):
                # Make the request to the FastAPI endpoint
                response = requests.post(FASTAPI_QUERY_URL, json={"query": query, "config": config})
                response.raise_for_status()  # Ensure the request was successful
                result = response.json()

                # Display the result from the RAG chain
                st.success("Response from RAG Chain:")
                st.write(result.get("answer", "No answer found"))

                # If there are sources, display them
                sources = result.get("sources", [])
                if sources:
                    st.write("Sources:")
                    for source in sources:
                        st.write(f"- {source}")

        except requests.HTTPError as e:
            st.error(f"Error fetching response: {e}")
    else:
        st.warning("Please enter a query.")
