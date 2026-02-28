from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag import RAGClient
import time
from datetime import datetime
import uvicorn
from typing import Optional
import conf

# Initialize FastAPI application
app = FastAPI(title=conf.API_TITLE, description=conf.API_DESCRIPTION)

# Initialize RAG client
rag_client = RAGClient()

# Define request model
class QueryRequest(BaseModel):
    query: str = Field(..., description="Nursing record text content")
    model_name: str = Field('deepseek', description="Model name")

# Define response model
class QueryResponse(BaseModel):
    result: str = Field(..., description="Coding result")
    processing_time: float = Field(..., description="Processing time (seconds)")
    timestamp: str = Field(..., description="Processing timestamp")

@app.post("/query", response_model=QueryResponse, summary="Query nursing record codes")
async def query(request: QueryRequest):
    """
    Query corresponding CCC codes based on nursing record text
    
    - **query**: Nursing record text content
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query content cannot be empty")
        
    # Record start time
    start_time = time.time()
    
    # Execute query
    result = rag_client.query(request.query, model_name=request.model_name)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Build response
    return QueryResponse(
        result=result,
        processing_time=round(processing_time, 2),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/health", summary="Health check")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "rag-api",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host=conf.API_HOST, port=conf.API_PORT, workers=conf.API_WORKERS, reload=conf.API_RELOAD)
