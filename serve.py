#!/usr/bin/env python3
"""
RA-CoT Recommendation Service API
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from loguru import logger

from config.config import config
from src.ra_cot_model import RACoTModel, RACoTTrainer
from src.data_processor import DataProcessor


# API model definitions
class UserBehavior(BaseModel):
    item_id: str
    category: str
    action: str  # click, purchase
    timestamp: Optional[int] = None


class RecommendationRequest(BaseModel):
    user_id: str
    user_history: List[UserBehavior]
    candidate_items: List[Dict[str, str]]  # [{"item_id": "xxx", "category": "xxx"}]
    top_k: int = 10
    explain: bool = True


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict]
    explanations: Optional[List[Dict]] = None
    performance_stats: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    performance_stats: Dict


# Global variables
app = FastAPI(title="RA-CoT Recommendation API", version="1.0.0")
model = None
trainer = None
data_processor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, trainer, data_processor
    
    logger.info("Loading RA-CoT model...")
    
    try:
        # Initialize data processor (for encoders)
        data_processor = DataProcessor(config)
        
        # Load model
        model_path = "./models/ra_cot_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            vocab_sizes = checkpoint.get('vocab_sizes', {'n_items': 5000, 'n_categories': 50, 'n_users': 1000})
            
            # Create model
            model = RACoTModel(config, vocab_sizes)
            trainer = RACoTTrainer(model, config)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Creating new model for demonstration")
            
            # Create demo model
            vocab_sizes = {'n_items': 5000, 'n_categories': 50, 'n_users': 1000}
            model = RACoTModel(config, vocab_sizes)
            trainer = RACoTTrainer(model, config)
            model.eval()
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        performance_stats=model.get_statistics() if model else {}
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Recommendation endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request data to model input format
        batch = _convert_request_to_batch(request)
        
        # Model inference
        with torch.no_grad():
            model_output = model(batch, explain=request.explain)
            predictions = model_output['predictions'].cpu().numpy()
        
        # Sort and select top-k
        candidate_scores = list(zip(request.candidate_items, predictions))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = candidate_scores[:request.top_k]
        
        # Build recommendation results
        recommendations = []
        for item, score in top_recommendations:
            recommendations.append({
                "item_id": item["item_id"],
                "category": item["category"],
                "score": float(score),
                "rank": len(recommendations) + 1
            })
        
        # Generate explanations (if needed)
        explanations = None
        if request.explain:
            explanations = model_output.get('explanations', [])
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            explanations=explanations,
            performance_stats=model.get_statistics()
        )
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_recommendation(request: RecommendationRequest):
    """Explain recommendation results"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request data
        batch = _convert_request_to_batch(request)
        
        # Generate explanations
        with torch.no_grad():
            model_output = model(batch, explain=True)
            explanations = model_output.get('explanations', [])
        
        return {
            "user_id": request.user_id,
            "explanations": explanations
        }
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model.get_statistics()


@app.post("/reset_stats")
async def reset_stats():
    """Reset performance statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model.reset_statistics()
    return {"message": "Stats reset successfully"}


def _convert_request_to_batch(request: RecommendationRequest) -> Dict:
    """Convert API request to model input batch"""
    batch_size = len(request.candidate_items)
    
    # Build user sequence
    user_items = []
    user_categories = []
    user_actions = []
    
    for behavior in request.user_history[-config.MAX_SEQUENCE_LENGTH:]:
        user_items.append(hash(behavior.item_id) % 5000)  # Simplified item encoding
        user_categories.append(hash(behavior.category) % 50)  # Simplified category encoding
        user_actions.append(1 if behavior.action == 'click' else 2)
    
    # Pad sequence
    seq_length = len(user_items)
    if seq_length < config.MAX_SEQUENCE_LENGTH:
        padding_length = config.MAX_SEQUENCE_LENGTH - seq_length
        user_items = [0] * padding_length + user_items
        user_categories = [0] * padding_length + user_categories
        user_actions = [0] * padding_length + user_actions
    
    # Build batch data
    batch = {
        'user_id': [request.user_id] * batch_size,
        'user_sequence': {
            'items': torch.tensor([user_items] * batch_size, dtype=torch.long),
            'categories': torch.tensor([user_categories] * batch_size, dtype=torch.long),
            'actions': torch.tensor([user_actions] * batch_size, dtype=torch.long),
            'length': torch.tensor([seq_length] * batch_size, dtype=torch.long)
        },
        'candidate_item': torch.tensor([
            hash(item['item_id']) % 5000 for item in request.candidate_items
        ], dtype=torch.long),
        'candidate_category': torch.tensor([
            hash(item['category']) % 50 for item in request.candidate_items
        ], dtype=torch.long),
        'label': torch.zeros(batch_size, dtype=torch.float32)  # No real labels needed for inference
    }
    
    return batch


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("./logs/service.log", rotation="10 MB", level="DEBUG")
    
    # Create log directory
    os.makedirs("./logs", exist_ok=True)
    
    # Start service
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        workers=1,  # Use single process due to model state
        log_level="info"
    )
