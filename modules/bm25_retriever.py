# -*- coding: utf-8 -*-
import os
import pickle
import jieba
import numpy as np
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf

class BM25RetrieverService:
    """
    BM25 retrieval service, responsible for keyword retrieval
    """
    def __init__(self, data_dir: str = None):
        """
        Initialize BM25 retrieval service
        
        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir or conf.DATA_DIR
        self.bm25_path = os.path.join(self.data_dir, conf.BM25_INDEX_FILE)
        
        # Initialize BM25 retriever
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data['bm25']
                self.doc_texts = bm25_data['doc_texts']
        else:
            self.bm25 = None
            self.doc_texts = []
    
    def update_index(self, documents: List[Document]) -> bool:
        """
        Update BM25 index
        
        Args:
            documents: Document list
            
        Returns:
            bool: Whether updated successfully
        """
        try:
            new_texts = [[doc.page_content, doc.metadata] for doc in documents]
            if self.bm25 is None:
                # First time adding documents, initialize BM25
                tokenized_texts = [list(jieba.cut(text[0])) for text in new_texts]
                self.bm25 = BM25Okapi(tokenized_texts)
                self.doc_texts = new_texts
            else:
                # Already have documents, update BM25
                self.doc_texts.extend(new_texts)
                tokenized_texts = [list(jieba.cut(text[0])) for text in self.doc_texts]
                self.bm25 = BM25Okapi(tokenized_texts)
            
            # Save BM25 index
            self.save()
            return True
        except Exception as e:
            print(f"Failed to update BM25 index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        BM25 search
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List[Document]: Relevant document list
        """
        try:
            if self.bm25 is None:
                return []
            
            tokenized_query = list(jieba.cut(query))
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_k = np.argsort(bm25_scores)
            return [Document(page_content=self.doc_texts[i][0], metadata=self.doc_texts[i][-1]) for i in bm25_top_k]
        except Exception as e:
            print(f"BM25 search failed: {str(e)}")
            return []
    
    def save(self) -> bool:
        """
        Save BM25 index
        
        Returns:
            bool: Whether saved successfully
        """
        try:
            with open(self.bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'doc_texts': self.doc_texts
                }, f)
            print(f"BM25 index saved successfully, path: {self.bm25_path}")
            return True
        except Exception as e:
            print(f"Failed to save BM25 index: {str(e)}")
            return False
