"""
# Note:
# - This application entry point (backend/app/main.py) is used for production.
# - A separate main (knowledge_bank/rag_two/src/api/main.py) exists for isolated testing of the RAG pipeline.
#   For production, all RAG endpoints are included via the rag_router.



This file aggregates routers for the entire API, including:

1. RAG-specific endpoints (e.g. document upload, search)
2. Educational endpoints (e.g. audio transcription, document processing)
3. Default endpoints (e.g. /query, /ingest)

The main application sets up CORS, logging, and any additional middleware,
and combines routers defined in other modules into one cohesive FastAPI instance.

Author: Keith Satuku
Version: 2.2.0
License: MIT
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from the API directory.
# Note: If the modules (e.g., education_api.py and routes.py) do not currently
# define an APIRouter, consider refactoring them so that their endpoints are added
# to routers that can be included here.
from src.rag_router import rag_router  # Existing RAG endpoints
from src.education_api import education_router  # Educational API endpoints (assumes a router is defined)
from .routes import router as default_router  # Additional endpoints (e.g., query, ingest)

app = FastAPI(
    title="Enhanced RAG API",
    description="API for Educational and RAG processing with integrated chunking, embedding, and retrieval.",
    version="2.2.0"
)

# Setup CORS middleware and any additional middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the RAG-specific router
app.include_router(rag_router)

# Include the Educational router
app.include_router(education_router)

# Include the default/query router
app.include_router(default_router)

# Additional setup or event handlers can be added below