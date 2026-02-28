# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from pydantic import BaseModel
from typing import List
import json
from functools import lru_cache
import concurrent.futures
from modules.llm_service_ollama import LLMService_Ollama
from modules.llm_service_openai import LLMService_Deepseek
from modules.vector_store import VectorStoreService
from modules.bm25_retriever import BM25RetrieverService
from modules.document_processor import DocumentProcessor
from modules.reranking import RerankingService
from modules.reranking_zh import RerankingZhService
import conf


def extract_last_json(s):
    s = s.replace('\\n', '').replace('\n', '').replace(' ', '')
    # Use regex to match all JSON data
    matches = re.findall(r'\{.*?\}', s)
    if matches:
        # Get the last matched JSON string
        last_match = matches[-1]
        # Replace single quotes with double quotes
        last_match = last_match.replace("'", '"')
        try:
            # Try to parse JSON
            json.loads(last_match)
            return last_match
        except json.JSONDecodeError:
            return None
    return None


class CccCode(BaseModel):
    codes: list[str]

class RAGClient:
    def __init__(self, 
                 keep_alive=conf.OLLAMA_KEEP_ALIVE,
                 data_dir=conf.DATA_DIR):
        # Initialize data directory
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        
        # Initialize service components
        self.deepseek_service = LLMService_Deepseek()
        self.ollama_service = LLMService_Ollama()
        self.reranker = RerankingZhService()
        
        # Load CCC code data
        self.treatment_df = pd.read_excel(conf.NURSING_INTERVENTIONS_FILE, dtype=str)
        self.diagnosis_df = pd.read_excel(conf.NURSING_DIAGNOSES_FILE, dtype=str)
        
        # Set RAG template
        self.rag_template = conf.RAG_TEMPLATE
    
    def add_document(self, file_path: str) -> bool:
        """Add document to knowledge base
        
        Args:
            file_path: Document path
            
        Returns:
            bool: Whether added successfully
        """
        try:
            return self.doc_processor.add_document(file_path)
        except Exception as e:
            print(f"Failed to add document: {str(e)}")
            return False


    @lru_cache(maxsize=conf.LRU_CACHE_MAXSIZE)
    def _get_code_info(self, code_type: str, cla_code: str, code_value: str) -> str:
        """Cache code info query results"""
        type_config = {
            'diagnosis': {'df': self.diagnosis_df, 'type_name': 'Nursing Diagnosis'},
            'treatment': {'df': self.treatment_df, 'type_name': 'Nursing Intervention'}
        }
        
        try:
            config = type_config[code_type]
            df = config['df'].loc[(config['df']['classification_code'] == cla_code) & (config['df']['code'] == code_value), :]
            return (
                f"[Type] {config['type_name']} | [Classification] {df['classification'].values[0]} | "
                f"[Classification Code] {df['classification_code'].values[0]} | [Code] {df['code'].values[0]} | "
                f"[Name] {df['name'].values[0]}"
            )
        except:
            return ""

    def _parallel_search(self, question: str, k: int = conf.VECTOR_SEARCH_K):
        """Execute vector search and BM25 search in parallel"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_faiss = executor.submit(self.vector_store.similarity_search, question, k)
            future_bm25 = executor.submit(self.bm25_retriever.search, question, k)
            
            faiss_docs = future_faiss.result()
            bm25_docs = future_bm25.result()
            
        return faiss_docs, bm25_docs



    def query(self, question: str, k: int = conf.DEFAULT_SEARCH_K, model_name='deepseek') -> str:
        """Query processing"""
        # Reranking
        rule_docs = self.reranker.rerank_documents(question, k)
        exam_docs = self.reranker.rerank_documents(question, k, file_path=conf.EXAMPLE_FILE)
        all_docs = rule_docs + exam_docs

        
        # Merge document content
        docs_content = "\n".join(doc for doc in all_docs)
        
        # Generate prompt
        prompt = self.rag_template.format(context=docs_content, question=question)
        
        # print(prompt)
        # Call LLM to generate response
        if model_name=='deepseek':
            response = self.deepseek_service.generate(prompt=prompt)
        else:
            response = self.ollama_service.generate(prompt=prompt, model_name=model_name)

        print(response)
        resp_json = extract_last_json(response)
        print(resp_json)
        # Parse codes
        ccc_codes = json.loads(resp_json)
        
        # Format output
        try:
            result = []
            for code in ccc_codes['codes']:
                code_type, cla_code, code_value = code.split('-')
                code_type, cla_code, code_value = str(code_type), str(cla_code), str(code_value)
                code_info = self._get_code_info(code_type, cla_code, code_value)
                if code_info:
                    result.append(code_info)
        except:
            result = []
        
        return "\n".join(result)

# Test code
if __name__ == "__main__":
    import platform
    # Initialize client
    client = RAGClient()
    # Test query
    test_question = """The patient passed grass-green loose stools, given basic care and skin care."""
    response = client.query(test_question)
    print(response)
