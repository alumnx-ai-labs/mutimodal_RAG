# app/vector_store.py
import os
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("INDEX_NAME", "crop-disease-index")
        
        if not self.pinecone_api_key:
            raise ValueError("Missing required API key: PINECONE_API_KEY")
        
        # Initialize Pinecone with new SDK
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize index (will create if doesn't exist)
        asyncio.create_task(self._ensure_index_exists())
    
    async def _ensure_index_exists(self):
        """Ensure Pinecone index exists, create if not."""
        try:
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=512,  # CLIP ViT-B-32 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD", "aws"),
                        region=os.getenv("PINECONE_REGION", "us-east-1")
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    await asyncio.sleep(1)
                    
                logger.info(f"Index {self.index_name} created and ready.")
            else:
                logger.info(f"Index {self.index_name} already exists.")
                
            # Get index instance
            self.index = self.pc.Index(self.index_name)
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise
    
    async def upsert_data(self, vectors: List[Dict[str, Any]]):
        """
        Upsert vectors to Pinecone.
        
        vectors format:
        [
            {
                'id': 'unique_id',
                'values': [0.1, 0.2, ...],  # embedding (512-dim for CLIP)
                'metadata': {
                    'type': 'image' or 'text',
                    'filename': 'image.jpg',
                    'disease': 'leaf_blight',
                    'description': 'Disease description',
                    'file_path': 'path/to/file'
                }
            }
        ]
        """
        try:
            if not hasattr(self, 'index'):
                await self._ensure_index_exists()
            
            # Upsert vectors using new SDK
            await asyncio.get_event_loop().run_in_executor(
                None, self.index.upsert, vectors
            )
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            
        except Exception as e:
            logger.error(f"Error upserting data: {e}")
            raise
    
    async def search(self, query_vector: List[float], top_k: int = 5, filter_dict: Dict = None):
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Search results from Pinecone
        """
        try:
            if not hasattr(self, 'index'):
                await self._ensure_index_exists()
            
            # Search using new SDK
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=filter_dict if filter_dict else None,
                    include_metadata=True
                )
            )
            
            logger.info(f"Search returned {len(search_results.matches)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise
    
    async def delete_vectors(self, ids: List[str]):
        """Delete vectors by IDs."""
        try:
            if not hasattr(self, 'index'):
                await self._ensure_index_exists()
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.index.delete, ids
            )
            logger.info(f"Successfully deleted {len(ids)} vectors")
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    async def get_index_stats(self):
        """Get index statistics."""
        try:
            if not hasattr(self, 'index'):
                await self._ensure_index_exists()
            
            stats = await asyncio.get_event_loop().run_in_executor(
                None, self.index.describe_index_stats
            )
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            raise
    
    async def clear_index(self):
        """Clear all vectors from the index."""
        try:
            if not hasattr(self, 'index'):
                await self._ensure_index_exists()
            
            # Delete all vectors
            await asyncio.get_event_loop().run_in_executor(
                None, self.index.delete, delete_all=True
            )
            logger.info("Successfully cleared all vectors from index")
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise