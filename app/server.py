
from fastapi import FastAPI, HTTPException
from llm_chain import rag_chain
from pydantic import BaseModel
import uvicorn

class Query(BaseModel):
    query : str 


app =  FastAPI(
    title='RAG API',
    version='1.0',
    description='simple API for my RAG app'
)

@app.post("/query")
async def get_results(input_query : Query):
    try:
        query_answer = rag_chain.invoke(input_query.query)
        return {'answer' : query_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
