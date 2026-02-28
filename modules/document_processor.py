# -*- coding: utf-8 -*-
import os
import glob
import hashlib
import pickle
from typing import List
from langchain_core.documents import Document
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
import conf

class DocumentProcessor:
    """
    Document processor, responsible for document loading, splitting and fingerprint calculation
    """
    def __init__(self, vector_store=None, bm25_retriever=None, data_dir: str = None):
        """
        Initialize document processor
        
        Args:
            vector_store: Vector store service
            bm25_retriever: BM25 retrieval service
            data_dir: Data directory
        """
        self.data_dir = data_dir or conf.DATA_DIR
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        
        # Initialize document fingerprint related attributes
        self.fingerprints_path = os.path.join(self.data_dir, conf.DOC_FINGERPRINTS_FILE)
        self.doc_fingerprints = set()
        if os.path.exists(self.fingerprints_path):
            try:
                with open(self.fingerprints_path, 'rb') as f:
                    self.doc_fingerprints = pickle.load(f)
            except Exception as e:
                print(f"Failed to load document fingerprints: {str(e)}")
                self.doc_fingerprints = set()
    
    def calculate_doc_fingerprint(self, file_path: str) -> str:
        """
        Calculate document fingerprint
        
        Args:
            file_path: Document path
            
        Returns:
            str: Document fingerprint
        """
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    
    def load_txt_file(self, file_path: str) -> List[Document]:
        """
        Load txt file and split into Document list by line
        
        Args:
            file_path: txt file path
            
        Returns:
            List[Document]: Document object list
        """
        docs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        docs.append(Document(
                            page_content=line,
                            metadata={"source": file_path}
                        ))
            return docs
        except Exception as e:
            print(f"Failed to load document: {str(e)}")
            return []
    
    def load_and_split(self, file_path: str) -> List[Document]:
        """
        Load and split document
        
        Args:
            file_path: Document path
            
        Returns:
            List[Document]: Document object list
        """
        # Choose different loading methods based on file extension
        if file_path.endswith('.txt'):
            return self.load_txt_file(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return []
    
    def add_document(self, file_path: str) -> bool:
        """
        Add document to knowledge base
        
        Args:
            file_path: Document path
            
        Returns:
            bool: Whether added successfully
        """
        try:
            # Calculate document fingerprint, check if already exists
            doc_fingerprint = self.calculate_doc_fingerprint(file_path)
            if doc_fingerprint in self.doc_fingerprints:
                print(f"Document already exists: {file_path}")
                return True
            
            # Load and process document
            print(f"Adding document: {file_path}")
            documents = self.load_and_split(file_path)
            print(f"Document split complete, total {len(documents)} segments")

            # Add to vector store
            if self.vector_store:
                print(f"Adding to vector store...")
                if not self.vector_store.add_documents(documents):
                    return False
            print(f"Added to vector store complete")
            # Update BM25 index
            if self.bm25_retriever:
                print(f"Updating BM25 index...")
                if not self.bm25_retriever.update_index(documents):
                    return False
            print(f"BM25 index update complete")

            # Update document fingerprint
            self.doc_fingerprints.add(doc_fingerprint)
            with open(self.fingerprints_path, 'wb') as f:
                pickle.dump(self.doc_fingerprints, f)
            print(f"Document fingerprint updated")

            return True
        except Exception as e:
            print(f"Failed to add document: {str(e)}")
            return False
    
    def add_documents_from_dir(self, dir_path: str, pattern: str = "*.txt") -> bool:
        """
        Batch add documents from directory
        
        Args:
            dir_path: Document directory
            pattern: File matching pattern
            
        Returns:
            bool: Whether all added successfully
        """
        files_path = glob.glob(f"{dir_path}/{pattern}")
        success = True
        for path in files_path:
            if not self.add_document(path):
                success = False
        return success
