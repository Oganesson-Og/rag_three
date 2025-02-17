"""
Compatibility Tests
----------------

Verify backwards compatibility with old code.

Author: Keith Satuku
Version: 1.0.0
"""

import pytest
import numpy as np
from src.nlp.tokenizer import rag_tokenizer
from src.nlp.query import query
from src.nlp.search import SearchEngine

def test_tokenizer_compatibility():
    # Test old-style tokenization
    text = "This is a test sentence"
    result = rag_tokenizer.tokenize(text)
    assert isinstance(result, str)
    assert "test" in result
    
    # Test fine_grained_tokenize
    result = rag_tokenizer.fine_grained_tokenize(text)
    assert isinstance(result, str)
    assert "test" in result

def test_query_compatibility():
    # Test old-style query processing
    query_text = "How does this work?"
    query_obj = query.FulltextQueryer()
    
    # Test question method
    result, keywords = query_obj.question(query_text)
    assert isinstance(result, dict)
    assert isinstance(keywords, list)
    
    # Test similarity methods
    query_vec = np.random.rand(768)
    doc_vecs = [np.random.rand(768) for _ in range(3)]
    query_tks = "test query"
    doc_tks = ["test doc one", "test doc two", "test doc three"]
    
    hybrid_sim, token_sim, vec_sim = query_obj.hybrid_similarity(
        query_vec, doc_vecs, query_tks, doc_tks
    )
    assert len(hybrid_sim) == 3
    assert len(token_sim) == 3
    assert len(vec_sim) == 3

def test_search_integration():
    engine = SearchEngine()
    query_text = "test query"
    doc_text = "test document"
    
    # Test tokenization integration
    tokens = rag_tokenizer.tokenize(query_text)
    assert isinstance(tokens, str)
    
    # Test query integration
    result = engine.qryr.token_similarity(query_text, [doc_text])
    assert isinstance(result, list)
    assert len(result) == 1 