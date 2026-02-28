# -*- coding: utf-8 -*-
"""
Configuration file for CNTRAM project
"""

import os

# 
# Data Directory Configuration
# 
DATA_DIR = "data"

# 
# DeepSeek LLM Service Configuration
# 
DEEPSEEK_MODEL_NAME = "deepseek-reasoner"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_SYSTEM_PROMPT = "You are a senior nursing expert, skilled in judging patient conditions, knowledgeable in various medical knowledge, and always answer in Chinese."
DEEPSEEK_TEMPERATURE = 0.0

# 
# Ollama LLM Service Configuration
# 
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = "3h"
OLLAMA_MODEL_NAME = "llama3.2:3b"
OLLAMA_SYSTEM_PROMPT = "You are a senior nursing expert, skilled in judging patient conditions, knowledgeable in various medical knowledge, and always answer in Chinese."
OLLAMA_TEMPERATURE = 0.0
OLLAMA_SEED = 41
OLLAMA_REPEAT_PENALTY = 1.1

# 
# Reranking Service Configuration
# 
RERANKING_MODEL_NAME = "BAAI/bge-m3"
RERANKING_TOKENIZER_MODEL = "BAAI/bge-small-zh"
RERANKING_EMBEDDING_MODEL = "BAAI/bge-small-zh"
RERANKING_USE_FP16 = True

# 
# Vector Store Configuration
# 
VECTOR_STORE_INDEX_NAME = "faiss_index"

# 
# BM25 Retriever Configuration
# 
BM25_INDEX_FILE = "bm25.pkl"

# 
# Document Processor Configuration
# 
DOC_FINGERPRINTS_FILE = "doc_fingerprints.pkl"
SUPPORTED_FILE_TYPES = [".txt"]

# 
# File Paths Configuration
# 
NURSING_INTERVENTIONS_FILE = "./files/nursing_interventions.xlsx"
NURSING_DIAGNOSES_FILE = "./files/nursing_diagnoses.xlsx"
RULE_FILES_PATTERN = "./files/rule_*.txt"
EXAMPLE_FILE = "./files/example.txt"

# 
# RAG Template Configuration
# 
RAG_TEMPLATE = """
1. Return result in JSON format, example:
{{   'codes':
    ["diagnosis-K30.0", "diagnosis-K30.1", "treatment-L36.0", ...]
}}
2. Multiple codes separated by English commas
3. If no matching codes, return empty array: []
4. Same code should only be returned once
5. Only return code JSON, do not include other explanatory text
6. Do not miss any related codes.

Nursing report content:
{question}

<Reference content>
{context}
</Reference content>
""".replace(' ', '')

# 
# API Server Configuration
# 
API_HOST = "0.0.0.0"
API_PORT = 5000
API_WORKERS = 4
API_RELOAD = False
API_TITLE = "CCC Coding Assistant API"
API_DESCRIPTION = "Nursing Record CCC Coding Query Service"

# 
# Search Configuration
# 
DEFAULT_SEARCH_K = 5
VECTOR_SEARCH_K = 40
BM25_SEARCH_K = 40

# 
# Cache Configuration
# 
LRU_CACHE_MAXSIZE = 1000

# Api base url
BASE_URL = 'http://localhost:8881'
