# Q/A RAG Model Using PDF Documents

This project is a Question/Answer Retrieval-Augmented Generation (RAG) model that uses PDF documents as the data source. The RAG model is developed using [LangChain](https://github.com/hwchase17/langchain) and [Ollama](https://ollama.com/) with the local `llama3.1` model. The system is deployed using FastAPI and connected to a Streamlit UI, supporting chat history for a more conversational experience.

## Features
- **RAG Model**: Retrieves information from PDF documents and provides enhanced generation through LangChain and Ollama's `llama3.1`.
- **PDF Datasources**: The model answers questions based on PDF document inputs.
- **Chat History**: Maintains the context across interactions for a smooth conversational experience.
- **FastAPI**: The backend is served using FastAPI for quick, asynchronous interactions.
- **Streamlit UI**: Provides an easy-to-use web interface for interaction with the model.

## Requirements
- **Python 3.12** or above
- **Virtual Environment** (`.venv`)
- **PowerShell** (For running the application)
- **FastAPI**
- **Streamlit**
- **LangChain**
- **Ollama**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install llama3.1:
   ```bash
   ollama pull llama3.1
   ```
## Usage
To run the project, follow these steps:
1. Start the FastAPI server
   ```bash
   uvicorn app.server:app --reload
   ```
This will start FastAPI server, enabling the backend for the Q/A system

2. Start the Streamlit client
   ```bash
   streamlit run app/client.py
   ```
This will launch the Streamlit Web interface, Where you can interact with the Q/A RAG model
## License
This project is licensed under the UST License


