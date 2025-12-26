"""
Configuration settings for local Ollama implementation
"""
import os
import torch

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "llama3:8b"

# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Vector Store Configuration
PERSIST_DIRECTORY = "./chroma_db_local"

# RAG Configuration
K_RETRIEVED = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Parameters
TEMPERATURE = 0

