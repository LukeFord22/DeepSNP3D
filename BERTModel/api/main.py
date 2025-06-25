import os
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertConfig
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from BERTModel.model.Model import DNABERTClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model components
model = None
tokenizer = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup, cleanup on shutdown"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading model and tokenizer...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model components
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "model", "DNABERT-2-117M"
        )
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model path not found: {local_path}")
        
        # Load tokenizer and model configuration
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        config = BertConfig.from_pretrained(local_path, trust_remote_code=True)
        base_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Initialize custom classifier
        model = DNABERTClassifier(base_model)
        model.to(device)
        
        # Load trained weights if available
        model_path = r"C:\Users\lhfor\AppData\Local\Programs\Python\Python38-32\DeepSNP3D.pth"
        if os.path.exists(model_path):
            logger.info(f"Loading trained weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
            logger.warning(f"No trained weights found at {model_path}, using untrained model")
        
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield  # Server runs here
    
    # Cleanup (if needed)
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="DeepSNP3D API",
    description="API for DNA sequence analysis using BERT-based deep learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SequenceInput(BaseModel):
    sequence: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,  # Adjust based on your model's limits
        description="DNA sequence to analyze"
    )
    
    @validator('sequence')
    def validate_dna_sequence(cls, v):
        """Validate that sequence contains only valid DNA bases"""
        valid_bases = set('ATCGN')  # Include N for unknown bases
        if not all(c.upper() in valid_bases for c in v):
            raise ValueError("Sequence must contain only valid DNA bases (A, T, C, G, N)")
        return v.upper()

class PredictionResponse(BaseModel):
    features: Dict[str, Any] = Field(..., description="Model prediction features")
    sequence_length: int = Field(..., description="Length of input sequence")
    model_version: str = Field(..., description="Model version used for prediction")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(error="Invalid input", detail=str(exc)).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="Internal server error").dict()
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown"
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DeepSNP3D API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: SequenceInput):
    """
    Predict features for a DNA sequence using the DeepSNP3D model
    
    - **sequence**: DNA sequence string (A, T, C, G, N allowed)
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        logger.info(f"Processing sequence of length {len(input_data.sequence)}")
        
        # Tokenize input sequence
        inputs = tokenizer(
            input_data.sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=384  # Adjust based on your model's requirements
        )
        
        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Model inference
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
        
        # Convert tensors to JSON-serializable format
        features = {}
        for key, tensor in outputs.items():
            # Convert to numpy and handle different tensor shapes
            arr = tensor.cpu().numpy()
            
            # Remove batch dimension if present
            if arr.shape[0] == 1:
                arr = arr[0]
            
            # Convert to list for JSON serialization
            features[key] = arr.tolist()
        
        logger.info("Prediction completed successfully")
        
        return PredictionResponse(
            features=features,
            sequence_length=len(input_data.sequence),
            model_version="1.0.0"
        )
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Sequence too long for available GPU memory"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Additional utility endpoints
@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "DNABERTClassifier",
        "device": str(device),
        "max_sequence_length": 512,
        "features": list(model(
            torch.zeros((1, 10), dtype=torch.long).to(device),
            torch.ones((1, 10), dtype=torch.long).to(device)
        ).keys()) if model else []
    }

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )