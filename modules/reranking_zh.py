# -*- coding: utf-8 -*-
from FlagEmbedding import FlagReranker
from typing import List
from modelscope import AutoTokenizer, AutoModel
import glob
import torch
import re
import jieba
from langchain_core.documents import Document
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf


class RerankingZhService:
    """
    Reranking service, responsible for document reranking
    """
    def __init__(self, model_name: str = None, use_fp16: bool = None):
        """
        Initialize reranking service
        
        Args:
            model_name: Reranking model name
            use_fp16: Whether to use half precision
        """
        self.tokenizer = AutoTokenizer.from_pretrained(conf.RERANKING_TOKENIZER_MODEL)
        self.model = AutoModel.from_pretrained(conf.RERANKING_EMBEDDING_MODEL)
        self.model.eval()
        
        

    def rerank_documents(self, query: str, k: int, file_path: str = None):
        """
        Rerank documents
        
        Args:
            query: Query text
            k: Number of documents to return
            file_path: File path pattern
            
        Returns:
            List[Document]: Reranked document list
        """
        file_path = file_path or conf.RULE_FILES_PATTERN
        txt_paths = glob.glob(file_path)
        doc_ls = [line for i in txt_paths for line in open(i, 'r', encoding='utf-8')]
        doc_ls = [i.strip() for i in doc_ls if i.strip()!='']
        try:
            # Split query text using Chinese and English commas
            query_parts = [q.strip() for q in re.split(r'[，,。]', query) if q.strip()]
            if len(query_parts) < 3:
                query_parts += jieba.lcut(query)
                k = k-1
            query_parts = [q for q in query_parts if q] + [query]
            print('query_parts: ', query_parts)
            # Batch process all sub-queries
            query_tokens = self.tokenizer(query_parts, padding=True, truncation=True, return_tensors='pt')
            doc_tokens = self.tokenizer(doc_ls, padding=True, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                # Get query and document embeddings
                query_output = self.model(**query_tokens)
                query_embeddings = query_output[0][:, 0]  # [num_queries, hidden_size]
                doc_output = self.model(**doc_tokens)
                doc_embeddings = doc_output[0][:, 0]  # [num_docs, hidden_size]

            # Normalize
            query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
            # Calculate similarity scores between all queries and documents
            scores = torch.matmul(query_embeddings, doc_embeddings.T)

            # Get top-k documents for each query
            top_k_scores, top_k_indices = torch.topk(scores, k=k, dim=1)
            # Collect top-k documents from all queries
            all_docs = []
            for indices in top_k_indices:
                docs = [doc_ls[idx] for idx in indices]
                all_docs.extend(docs)
            
            # Deduplicate and return top k documents
            unique_docs = list(dict.fromkeys(all_docs))
            return unique_docs
            
        except Exception as e:
            print(f"Error during reranking: {str(e)}")
            return []

if __name__ == '__main__':
    # Initialize reranking service
    reranking_service = RerankingZhService()

    # Test reranking service
    query = 'Doctor ordered continuous ice blanket cooling'
    k = 3
    docs = reranking_service.rerank_documents(query, k)
    print('/n'.join(docs))
    print(len(docs))
