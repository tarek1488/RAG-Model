#first importing needed packages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .prompts import contextualize_q_prompt, qa_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
#from langchain.retrievers.multi_query import MultiQueryRetriever
#-------------------------------------------------------------------------

#Adding chat history parameters
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#DATA PREPROCESSING:
#--------------

#defining data path
DATA_PATHS = r'data'

#defining path to the vectorstore
VECTOR_STORE_DIR = 'vectorstoreDB'

def load_db():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(VECTOR_STORE_DIR):
        #Intializing Loader
        loader = PyPDFDirectoryLoader(DATA_PATHS)
        
        #loading pdf
        docs = loader.load() #each page in the pdf loaded as Documnet object so docs will have number of Docmenst equal tp pages num in the pdf
                
        #Intializing the text splitter to split data to chuncks to fit sequence window in chat models
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        #splitting data
        splits =  text_splitter.split_documents(docs)
        
        #Intializing chroma vectore store
        vector_store =  Chroma(embedding_function=embedding,
                               persist_directory=VECTOR_STORE_DIR)
        
        #Adding splits
        vector_store.add_documents(splits)

    else:
        vector_store = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)
    return vector_store
        


    

vector_store = load_db()
#-------------------------------------------------------------------------------
#BUILDING THE RAG MODEL
#-----------------------

# Laoding ollama chat model
chat_model = ChatOllama(model="llama3.1",
                        temperature=0,
                        )

#intializing a retriever 
retriever =  vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs={"k": 5}
)

#creating a history aware retriever
history_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_q_prompt
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

#converting normal rag_chain to converstional rag chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
