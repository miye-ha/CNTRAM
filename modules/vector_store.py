# -*- coding: utf-8 -*-
import os
import faiss
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf

class VectorStoreService:
    """
    Vector store service, responsible for managing FAISS vector database
    """
    def __init__(self, embeddings, data_dir: str = None):
        """
        Initialize vector store service
        
        Args:
            embeddings: Embedding model
            data_dir: Data directory
        """
        self.embeddings = embeddings
        self.data_dir = data_dir or conf.DATA_DIR
        self.index_path = os.path.join(self.data_dir, conf.VECTOR_STORE_INDEX_NAME)
        
        # Initialize vector store
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            # First get embedding dimension
            dimension = len(self.embeddings.embed_query("test"))
            # Create empty vector store
            index = faiss.IndexFlatL2(dimension)
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to vector store
        
        Args:
            documents: Document list
            
        Returns:
            bool: Whether added successfully
        """
        self.vector_store.add_documents(documents=documents)
        self.save()
        return True
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Similarity search
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List[Document]: Similar document list
        """
        try:
            return self.vector_store.similarity_search(query, k=k or conf.VECTOR_SEARCH_K)
        except Exception as e:
            print(f"Similarity search failed: {str(e)}")
            return []
    
    def save(self) -> bool:
        """
        Save vector store
        
        Returns:
            bool: Whether saved successfully
        """
        try:
            print("Saving vector store, path:", self.index_path)
            self.vector_store.save_local(self.index_path)
            print("Vector store saved successfully, path:", self.index_path)
            return True
        except Exception as e:
            print(f"Failed to save vector store: {str(e)}")
            return False
