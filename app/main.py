# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

from .models import MultimodalEmbedding
from .vector_store import VectorStore
from .utils import save_upload_file, create_context_from_results

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Crop Disease RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components - will be initialized on startup
embedding_model = None
vector_store = None
gemini_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global embedding_model, vector_store, gemini_model
    
    try:
        logger.info("Initializing components...")
        
        # Initialize embedding model
        embedding_model = MultimodalEmbedding()
        logger.info("✓ Embedding model initialized")
        
        # Initialize vector store
        vector_store = VectorStore()
        logger.info("✓ Vector store initialized")
        
        # Initialize Gemini
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✓ Gemini model initialized")
        
        logger.info("🚀 All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Crop Disease RAG API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if components are initialized
        if not all([embedding_model, vector_store, gemini_model]):
            return {"status": "unhealthy", "message": "Components not initialized"}
        
        # Get index stats
        stats = await vector_store.get_index_stats()
        
        return {
            "status": "healthy",
            "components": {
                "embedding_model": "✓ Ready",
                "vector_store": "✓ Ready", 
                "gemini_model": "✓ Ready"
            },
            "index_stats": {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...),
    disease_name: str = Form(...),
    description: str = Form(...),
    crop_type: str = Form(default="unknown")
):
    """Upload and process crop disease images"""
    try:
        if not all([embedding_model, vector_store]):
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        os.makedirs("data/images", exist_ok=True)
        file_path = f"data/images/{file.filename}"
        await save_upload_file(file, file_path)
        logger.info(f"Saved file: {file_path}")
        
        # Generate embedding
        embedding = embedding_model.encode_image(file_path)
        logger.info(f"Generated embedding with dimension: {len(embedding)}")
        
        # Prepare data for vector store
        vector_data = [{
            'id': f"img_{disease_name}_{file.filename}_{crop_type}",
            'values': embedding,
            'metadata': {
                'type': 'image',
                'filename': file.filename,
                'disease': disease_name,
                'description': description,
                'file_path': file_path,
                'crop_type': crop_type,
                'upload_timestamp': str(asyncio.get_event_loop().time())
            }
        }]
        
        # Store in vector database
        await vector_store.upsert_data(vector_data)
        logger.info(f"Successfully stored vector for: {file.filename}")
        
        return {
            "message": "Data uploaded successfully",
            "filename": file.filename,
            "disease": disease_name,
            "crop_type": crop_type,
            "vector_id": vector_data[0]['id']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query")
async def query_rag(
    text_query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(default=5)
):
    """Query the RAG system with text and/or image"""
    try:
        if not all([embedding_model, vector_store, gemini_model]):
            raise HTTPException(status_code=503, detail="Service not ready")
        
        if not text_query and not image:
            raise HTTPException(status_code=400, detail="Please provide either text query or image")
        
        # Process query based on input type
        if image:
            # Validate image
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")
            
            # Save temporary image
            os.makedirs("uploads", exist_ok=True)
            temp_path = f"uploads/{image.filename}"
            await save_upload_file(image, temp_path)
            query_embedding = embedding_model.encode_image(temp_path)
            query_type = "image"
            logger.info(f"Generated query embedding from image: {image.filename}")
        else:
            query_embedding = embedding_model.encode_text(text_query)
            query_type = "text"
            logger.info(f"Generated query embedding from text: {text_query}")
        
        # Search vector store
        search_results = await vector_store.search(query_embedding, top_k=top_k)
        logger.info(f"Found {len(search_results.matches)} similar results")
        
        if not search_results.matches:
            return {
                "response": "I couldn't find any similar cases in the database. Please try uploading more data or rephrasing your query.",
                "similar_cases": 0,
                "confidence": 0.0,
                "query_type": query_type
            }
        
        # Create context for Gemini
        context = create_context_from_results(search_results)
        
        # Generate response with Gemini
        prompt = f"""
You are an experienced agricultural expert helping farmers identify crop diseases and provide practical solutions.

Context from similar cases in database:
{context}

Farmer's query: {text_query or "The farmer has uploaded an image for analysis"}
Query type: {query_type}

Based on the similar cases found, please provide helpful advice covering:

1. **Disease Identification**: What disease or condition this likely represents
2. **Symptoms**: Key symptoms to look for
3. **Treatment**: Specific, actionable treatment recommendations
4. **Prevention**: How to prevent this in the future
5. **Urgency**: How quickly the farmer should act

Keep your response:
- Practical and easy to understand
- Focused on actionable steps
- Appropriate for farmers with varying levels of technical knowledge
- Based on the evidence from similar cases found

If you're uncertain about the diagnosis, clearly state this and recommend consulting a local agricultural extension office.
"""
        
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None, gemini_model.generate_content, prompt
        )
        
        # Calculate confidence based on top match
        confidence = round(search_results.matches[0].score, 3) if search_results.matches else 0.0
        
        # Prepare similar cases info
        similar_cases_info = []
        for match in search_results.matches[:3]:  # Top 3 matches
            similar_cases_info.append({
                "disease": match.metadata.get('disease', 'Unknown'),
                "confidence": round(match.score, 3),
                "crop_type": match.metadata.get('crop_type', 'Unknown')
            })
        
        return {
            "response": response.text,
            "similar_cases": len(search_results.matches),
            "confidence": confidence,
            "query_type": query_type,
            "top_matches": similar_cases_info
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not ready")
        
        stats = await vector_store.get_index_stats()
        
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": getattr(stats, 'index_fullness', 0.0)
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.delete("/clear-database")
async def clear_database():
    """Clear all data from the database (use with caution!)"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not ready")
        
        await vector_store.clear_index()
        
        return {"message": "Database cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")