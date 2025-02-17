"""
Settings Configuration Module
--------------------------

Central settings management for the RAG pipeline.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import os
from pathlib import Path
from ..nlp.tokenizer import UnifiedTokenizer, default_tokenizer  # Fixed import
#from ..nlp.search import SearchEngine
#from ..rag.utils import num_tokens_from_string
from datetime import date
from enum import IntEnum, Enum
from .. import rag
from ..nlp import search
from typing import Union, Dict, List, Optional
from pydantic import BaseModel

LIGHTEN = int(os.environ.get('LIGHTEN', "0"))

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
CACHE_DIR = BASE_DIR / ".cache"

# Initialize tokenizer
tokenizer = UnifiedTokenizer()

# # Data paths
# DATA_DIR = PROJECT_ROOT / "Form 4"
# CACHE_DIR = PROJECT_ROOT / ".cache"
# SYLLABUS_PATH = PROJECT_ROOT / "syllabus"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SYLLABUS_PATH = CONFIG_DIR / "syllabus"
# Create directories if they don't exist
for directory in [CACHE_DIR, SYLLABUS_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# Model settings
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "phi"

# API settings
OLLAMA_BASE_URL = "http://localhost:11434/api"

# Token settings
MAX_TOKENS = 2048
def get_max_chunk_size(text: str) -> int:
    """Calculate maximum chunk size based on token count."""
    return min(CHUNK_SIZE, MAX_TOKENS - tokenizer.count_tokens(text))

LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
IMAGE2TEXT_MDL = ""
API_KEY = None
PARSERS = None
HOST_IP = None
HOST_PORT = None
SECRET_KEY = None


# authentication
AUTHENTICATION_CONF = None

# client
CLIENT_AUTHENTICATION = None
HTTP_APP_KEY = None
GITHUB_OAUTH = None
FEISHU_OAUTH = None

DOC_ENGINE = None
docStoreConn = None

retrievaler = None
kg_retrievaler = None

# Embedding settings
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "cache_dir": str(CACHE_DIR / "embeddings"),
    "dimension": 384,
    "batch_size": 32
}

def init_settings():
    global LLM, LLM_FACTORY, LLM_BASE_URL, LIGHTEN, DATABASE_TYPE, DATABASE
    LIGHTEN = int(os.environ.get('LIGHTEN', "0"))
    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    if not LIGHTEN:
        default_llm = {
            "Tongyi-Qianwen": {
                "chat_model": "qwen-plus",
                "embedding_model": "text-embedding-v2",
                "image2text_model": "qwen-vl-max",
                "asr_model": "paraformer-realtime-8k-v1",
            },
            "OpenAI": {
                "chat_model": "gpt-3.5-turbo",
                "embedding_model": "text-embedding-ada-002",
                "image2text_model": "gpt-4-vision-preview",
                "asr_model": "whisper-1",
            },
            "Azure-OpenAI": {
                "chat_model": "gpt-35-turbo",
                "embedding_model": "text-embedding-ada-002",
                "image2text_model": "gpt-4-vision-preview",
                "asr_model": "whisper-1",
            },
            "ZHIPU-AI": {
                "chat_model": "glm-3-turbo",
                "embedding_model": "embedding-2",
                "image2text_model": "glm-4v",
                "asr_model": "",
            },
            "Ollama": {
                "chat_model": "qwen-14B-chat",
                "embedding_model": "flag-embedding",
                "image2text_model": "",
                "asr_model": "",
            },
            "Moonshot": {
                "chat_model": "moonshot-v1-8k",
                "embedding_model": "",
                "image2text_model": "",
                "asr_model": "",
            },
            "DeepSeek": {
                "chat_model": "deepseek-chat",
                "embedding_model": "",
                "image2text_model": "",
                "asr_model": "",
            },
            "VolcEngine": {
                "chat_model": "",
                "embedding_model": "",
                "image2text_model": "",
                "asr_model": "",
            },
            "BAAI": {
                "chat_model": "",
                "embedding_model": "BAAI/bge-large-zh-v1.5",
                "image2text_model": "",
                "asr_model": "",
                "rerank_model": "BAAI/bge-reranker-v2-m3",
            }
        }

        if LLM_FACTORY:
            CHAT_MDL = default_llm[LLM_FACTORY]["chat_model"] + f"@{LLM_FACTORY}"
            ASR_MDL = default_llm[LLM_FACTORY]["asr_model"] + f"@{LLM_FACTORY}"
            IMAGE2TEXT_MDL = default_llm[LLM_FACTORY]["image2text_model"] + f"@{LLM_FACTORY}"
        EMBEDDING_MDL = default_llm["BAAI"]["embedding_model"] + "@BAAI"
        RERANK_MDL = default_llm["BAAI"]["rerank_model"] + "@BAAI"

    global API_KEY, PARSERS, HOST_IP, HOST_PORT, SECRET_KEY
    API_KEY = LLM.get("api_key", "")
    PARSERS = LLM.get(
        "parsers",
        "naive:General,qa:Q&A,resume:Resume,manual:Manual,table:Table,paper:Paper,book:Book,laws:Laws,presentation:Presentation,picture:Picture,one:One,audio:Audio,knowledge_graph:Knowledge Graph,email:Email,tag:Tag")


    global DOC_ENGINE, docStoreConn, retrievaler, kg_retrievaler
    DOC_ENGINE = os.environ.get('DOC_ENGINE', "elasticsearch")
    lower_case_doc_engine = DOC_ENGINE.lower()
    if lower_case_doc_engine == "elasticsearch":
        docStoreConn = rag.utils.es_conn.ESConnection()
    elif lower_case_doc_engine == "infinity":
        docStoreConn = rag.utils.infinity_conn.InfinityConnection()
    else:
        raise Exception(f"Not supported doc engine: {DOC_ENGINE}")

    retrievaler = search.Dealer(docStoreConn)

class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except BaseException:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]


class RetCode(IntEnum, CustomEnum):
    SUCCESS = 0
    NOT_EFFECTIVE = 10
    EXCEPTION_ERROR = 100
    ARGUMENT_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    CONNECTION_ERROR = 105
    RUNNING = 106
    PERMISSION_ERROR = 108
    AUTHENTICATION_ERROR = 109
    UNAUTHORIZED = 401
    SERVER_ERROR = 500
    FORBIDDEN = 403
    NOT_FOUND = 404

class Settings(BaseModel):
    embedding_model: str
    vector_dimension: int
    batch_size: Optional[int] = 32
    device: str = "cpu"
    cache_size: Union[int, str] = "1GB"
    
    class Config:
        arbitrary_types_allowed = True

QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "prefer_grpc": True,
    "collections": {
        "education": {
            "vector_size": 384,  
            "distance": "Cosine",
            "optimizers_config": {
                "default_segment_number": 2,
                "memmap_threshold": 20000
            }
        }
    }
}
