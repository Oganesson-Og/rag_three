from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ..rag.pipeline import Pipeline
from ..rag.models import Document, GenerationResult, ContentModality

app = FastAPI()
pipeline = Pipeline(config_path="config.yaml")

class QueryRequest(BaseModel):
    query: str
    context: Optional[dict] = None

class DocumentRequest(BaseModel):
    content: str
    modality: ContentModality
    metadata: Optional[dict] = None

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Endpoint for querying the RAG system."""
    try:
        response = await pipeline.generate_response(
            query=request.query,
            context=request.context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: DocumentRequest):
    """Endpoint for ingesting single documents."""
    try:
        document = await pipeline.process_document(
            content=request.content,
            modality=request.modality,
            metadata=request.metadata
        )
        return {"document_id": document.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 