#first importing needed packages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompts import prompt, agent_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from notes import note_tool
import os
#from langchain.retrievers.multi_query import MultiQueryRetriever
#-------------------------------------------------------------------------
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

retriever_tool = create_retriever_tool(
    retriever,
    "search docs",
    "Search for information about countries. For any questions about countries, you must use this tool!"
)

tools = [retriever_tool, note_tool]

agent = create_tool_calling_agent(chat_model, tools, agent_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


while (question := input("Ask a question about github issues (q to quit): ")) != "q":
    result = agent_executor.invoke({"input": question})
    print(result["output"])

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# #Forming the chain
# rag_chain = (
#     {'context' : retriever | format_docs, 'input' : RunnablePassthrough()}
#     | prompt  
#     | chat_model
#     | StrOutputParser()
# )
