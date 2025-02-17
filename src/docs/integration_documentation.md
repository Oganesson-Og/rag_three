# Enhanced RAG Pipeline Integration Documentation

## Overview

This document details the integration between the document ingestion system and the multi-stage processing pipeline in the RAG system. The pipeline is responsible for reading and processing input documents, chunking their contents, generating vector embeddings, caching results, and interfacing with Qdrant for vector storage and search.

## Components and Their Roles

### 1. Document Ingestion

- **Location:** `knowledge_bank/rag_two/src/scripts/ingest_documents.py`
- **Role:**  
  - Scans an input directory for supported document types.
  - Processes syllabus files (used to structure document content).
  - Passes each document to the processing pipeline via:
    ```python
    await pipeline.process_document(
        source=str(file_path),
        modality=ContentModality.TEXT,
        options={...}
    )
    ```
  - Integrates a `ProgressTracker` instance to update real-time progress and performance statistics while documents are being processed.

### 2. Processing Pipeline

- **Location:** `knowledge_bank/rag_two/src/rag/pipeline.py`
- **Role:**  
  Implements a modular, asynchronous processing architecture composed of several stages:

  #### a. Extraction Stage

  - **Component:** `ExtractionStage`
  - **Function:**  
    - Uses modality-specific extractors (e.g., `TextExtractor` for text or `AudioExtractor` for audio content) to convert raw document data into a structured format.
    - Records processing metrics (like extraction time and token count).

  #### b. Chunking Stage

  - **Component:** `ChunkingStage`
  - **Function:**  
    - Cleans the extracted text using helper functions like `clean_text`.
    - Splits the cleaned text into sentences with `split_into_sentences`.
    - Groups sentences into coherent chunks by evaluating boundary conditions with `validate_chunk` and checking for complete sentences.
    - Populates the `document.chunks` property with a list of `Chunk` objects containing the chunk text.

  #### c. Embedding Generation and Caching

  - **Integration Detail:**  
    - Within various stages or as a post-process, the pipeline generates vector embeddings for document chunks.
    - The method `_get_embedding(text: str)`:
      - **Step 1:** Computes a cache key (e.g., using a hash of the text).
      - **Step 2:** Checks for an existing embedding in `VectorCache`.
      - **Step 3:** If no cache entry exists, calls the configured embedding model (configured via `embedding_config.py`) to obtain a vector representation.
      - **Step 4:** Caches the generated embedding for future use.
    - This enables efficient reuse of embeddings, reducing redundant computations when processing similar text.

  #### d. Additional Stages

  - The pipeline supports other specialized processing stages (e.g., `DiagramAnalysisStage`) to handle diagrams or other document features.
  - Each stage records its own processing metrics and appends a `ProcessingEvent` to the document.

### 3. Storage & Qdrant Integration

- **Location:** `knowledge_bank/rag_two/src/api/main.py`
- **Role:**  
  - Exposes API endpoints (for example, `/rag/documents/upload` and `/rag/search`) that interact with the processed document data.
  - The search endpoint utilizes the vector embeddings stored (and cached) by the pipeline to perform semantic or keyword-based searches via Qdrant.

### 4. Caching Mechanisms

- **Vector Caching:**  
  - **Module:** `knowledge_bank/rag_two/src/cache/vector_cache.py`
  - **Role:**  
    Provides a caching layer for vector embeddings, minimizing re-computation and speeding up both ingestion and search queries.
  
- **Multi-Modal Caching:**  
  - **Module:** `knowledge_bank/rag_two/src/cache/advanced_cache.py`
  - **Role:**  
    Offers additional caching support for various intermediate processing results.

## Overall Processing Workflow

1. **Document Ingestion:**
   - The `ingest_documents.py` script scans the input directory, processes files (including syllabus extraction), and calls the pipeline for document processing.

2. **Pipeline Processing:**
   - **Extraction:**  
     The document is parsed and its content extracted into a structured format.
   - **Chunking:**  
     The extracted text is cleaned and segmented into coherent chunks.
   - **Embedding Generation:**  
     Each chunk is transformed into a vector using the embedding model. Already computed embeddings are retrieved from `VectorCache` where possible.
   - **Metric Collection & Error Handling:**  
     Each step records performance metrics and includes robust error handling.
  
3. **Storage & Retrieval via API:**
   - Embeddings are eventually stored in Qdrant and can be queried via the `/rag/search` endpoint.
   - This enables fast, semantically relevant retrieval of document chunks based on user queries.

## Summary

The integration of document ingestion with the enhanced pipeline ensures:
- **Efficiency:**  
  Asynchronous processing and caching reduce processing times.
- **Modularity:**  
  Each processing stage (extraction, chunking, embedding) is designed to be independent, allowing for easy updates or modifications.
- **Scalability:**  
  Support for large document collections and diverse content modalities.
- **Robustness:**  
  Comprehensive metric tracking and error handling improve system reliability.
- **Retrievability:**  
  Integration with Qdrant ensures that processed vector embeddings are available for fast and accurate semantic searches.

---

**Version:** 1.0.0  
**Author:** Keith Satuku  
**License:** MIT 