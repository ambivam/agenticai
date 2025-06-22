
import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manage FAISS vector store for document embeddings"""
    
    def __init__(self, openai_api_key: str, vector_db_path: str = "./vector_store"):
        self.openai_api_key = openai_api_key
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.vector_store = None
        self.document_metadata = {}
        
        # Create directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Load existing vector store if available
        self.load_vector_store()
    
    def add_documents(self, documents: List[Document], show_progress: bool = True) -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add to vector store")
                return False
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            if show_progress:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Create or update vector store
            if self.vector_store is None:
                if show_progress:
                    status_text.text("Creating new vector store...")
                    progress_bar.progress(0.3)
                
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                if show_progress:
                    progress_bar.progress(0.7)
            else:
                if show_progress:
                    status_text.text("Adding documents to existing vector store...")
                    progress_bar.progress(0.3)
                
                # Add documents to existing vector store
                self.vector_store.add_documents(documents)
                
                if show_progress:
                    progress_bar.progress(0.7)
            
            # Update metadata
            for doc in documents:
                doc_id = len(self.document_metadata)
                self.document_metadata[doc_id] = doc.metadata
            
            # Save vector store
            self.save_vector_store()
            
            if show_progress:
                status_text.text("Vector store updated successfully!")
                progress_bar.progress(1.0)
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            if show_progress:
                st.error(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores"""
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold if specified
            if score_threshold > 0:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            logger.info(f"Similarity search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def save_vector_store(self) -> bool:
        """Save vector store to disk"""
        try:
            if self.vector_store is None:
                logger.warning("No vector store to save")
                return False
            
            # Save FAISS index
            faiss_path = os.path.join(self.vector_db_path, "faiss_index")
            self.vector_store.save_local(faiss_path)
            
            # Save metadata
            metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            logger.info(f"Vector store saved to {self.vector_db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load_vector_store(self) -> bool:
        """Load vector store from disk"""
        try:
            faiss_path = os.path.join(self.vector_db_path, "faiss_index")
            metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
            
            if os.path.exists(faiss_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.vector_store = FAISS.load_local(
                    faiss_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                
                logger.info(f"Vector store loaded from {self.vector_db_path}")
                return True
            else:
                logger.info("No existing vector store found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def clear_vector_store(self) -> bool:
        """Clear all documents from vector store"""
        try:
            # Remove files
            faiss_path = os.path.join(self.vector_db_path, "faiss_index")
            metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
            
            if os.path.exists(faiss_path + ".faiss"):
                os.remove(faiss_path + ".faiss")
            if os.path.exists(faiss_path + ".pkl"):
                os.remove(faiss_path + ".pkl")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Reset in-memory objects
            self.vector_store = None
            self.document_metadata = {}
            
            logger.info("Vector store cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        try:
            if self.vector_store is None:
                return {
                    "initialized": False,
                    "total_documents": 0,
                    "total_embeddings": 0
                }
            
            # Get vector store statistics
            index = self.vector_store.index
            total_vectors = index.ntotal if hasattr(index, 'ntotal') else 0
            
            # Get unique files
            unique_files = set()
            for metadata in self.document_metadata.values():
                if 'filename' in metadata:
                    unique_files.add(metadata['filename'])
            
            return {
                "initialized": True,
                "total_documents": len(self.document_metadata),
                "total_embeddings": total_vectors,
                "unique_files": len(unique_files),
                "files": list(unique_files)
            }
            
        except Exception as e:
            logger.error(f"Error getting store info: {str(e)}")
            return {"error": str(e)}
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID"""
        return self.document_metadata.get(doc_id)
    
    def search_documents_by_metadata(self, **kwargs) -> List[Dict[str, Any]]:
        """Search documents by metadata fields"""
        matching_docs = []
        
        for doc_id, metadata in self.document_metadata.items():
            match = True
            for key, value in kwargs.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                matching_docs.append({
                    "doc_id": doc_id,
                    "metadata": metadata
                })
        
        return matching_docs
