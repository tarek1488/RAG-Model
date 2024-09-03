#first importing needed packages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompts import prompt 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
#-------------------------------------------------------------------------
#DATA PREPROCESSING:
#--------------

#defining data path
DATA_PATHS = r'data'

#Intializing Loader
loader = PyPDFDirectoryLoader(DATA_PATHS)

#loading pdf
docs = loader.load() #each page in the pdf loaded as Documnet object so docs will have number of Docmenst equal tp pages num in the pdf

print(f'the docs len is : ----------> {len(docs)}')

#Intializing the text splitter to split data to chuncks to fit sequence window in chat models
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#splitting data
splits =  text_splitter.split_documents(docs)

#saving data in chroma vector store
vector_store = Chroma.from_documents(
    splits,
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)
#-------------------------------------------------------------------------------
#BUILDING THE RAG MODEL
#-----------------------

#intializing a retriever 
retriever =  vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs={"k": 3}
)

# #Laoding ollama chat model
chat_model = ChatOllama(model="llama3",
                        temperature=0,
                        )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Forming the chain
rag_chain = (
    {'context' : retriever | format_docs, 'input' : RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

for chunk in rag_chain.stream("what is the capital of egypt"):
    print(chunk, end="", flush=True)


# results = retriever.invoke('which is bigger canda or egypt')
# with open('out.txt', 'w', encoding='utf-8') as f:
#     f.write(str(results))