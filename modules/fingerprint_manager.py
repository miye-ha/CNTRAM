# -*- coding: utf-8 -*-
import os
import pickle
from typing import Set
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf

class FingerprintManager:
    """
    Document fingerprint manager, responsible for managing document fingerprints to avoid duplicate document additions
    """
    def __init__(self, data_dir: str = None):
        """
        Initialize document fingerprint manager
        
        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir or conf.DATA_DIR
        self.fingerprints_path = os.path.join(self.data_dir, conf.DOC_FINGERPRINTS_FILE)
        
        # Initialize document fingerprint storage
        if os.path.exists(self.fingerprints_path):
            with open(self.fingerprints_path, 'rb') as f:
                self.doc_fingerprints = pickle.load(f)
        else:
            self.doc_fingerprints = set()
    
    def exists(self, fingerprint: str) -> bool:
        """
        Check if document fingerprint exists
        
        Args:
            fingerprint: Document fingerprint
            
        Returns:
            bool: Whether it exists
        """
        return fingerprint in self.doc_fingerprints
    
    def add(self, fingerprint: str) -> None:
        """
        Add document fingerprint
        
        Args:
            fingerprint: Document fingerprint
        """
        self.doc_fingerprints.add(fingerprint)
        self.save()
    
    def add_document(self, file_path: str) -> None:
        """
        Add document fingerprint
        
        Args:
            file_path: Document path
        """
        from modules.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        fingerprint = doc_processor.calculate_doc_fingerprint(file_path)
        self.add(fingerprint)
    
    def get_all(self) -> Set[str]:
        """
        Get all document fingerprints
        
        Returns:
            Set[str]: All document fingerprints
        """
        return self.doc_fingerprints
    
    def is_document_exists(self, file_path: str) -> bool:
        """
        Check if document already exists
        
        Args:
            file_path: Document path
            
        Returns:
            bool: Whether it exists
        """
        from modules.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        fingerprint = doc_processor.calculate_doc_fingerprint(file_path)
        return self.exists(fingerprint)
    
    def save(self) -> bool:
        """
        Save document fingerprints
        
        Returns:
            bool: Whether saved successfully
        """
        try:
            with open(self.fingerprints_path, 'wb') as f:
                pickle.dump(self.doc_fingerprints, f)
            return True
        except Exception as e:
            print(f"Failed to save document fingerprints: {str(e)}")
            return False
