from fastapi import FastAPI, HTTPException, Response, Request
from .openai_rag  import conversational_rag_chain
from pydantic import BaseModel
import uuid

class Query(BaseModel):
    query : str 
    config: dict


app =  FastAPI(
    title='RAG API',
    version='1.0',
    description='simple API for my RAG app'
)

#adding the session endpoint
@app.get("/get-session")
async def get_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:  # If there's no session ID, generate a new one
        session_id = str(uuid.uuid4())  # Generate a new session ID
        response.set_cookie(key="session_id", value=session_id)  # Set the session ID in the cookie
    return {"session_id": session_id}

#adding the query end point
@app.post("/query")
async def get_results(input_query : Query):
    try:
        query_answer = conversational_rag_chain.invoke({"input": input_query.query}, config=input_query.config)
        return {'answer' : query_answer['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

