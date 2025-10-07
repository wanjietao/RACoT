#!/usr/bin/env python3
"""
RA-CoT Usage Examples
Demonstrates how to use the RA-CoT model for recommendations and explanations
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from loguru import logger
import requests
import json
from typing import List, Dict

from config.config import config
from src.data_processor import DataProcessor
from src.ra_cot_model import RACoTModel, RACoTTrainer


def demo_model_training():
    """Demonstrate model training process"""
    logger.info("=== RA-CoT Model Training Demo ===")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Data processing
    logger.info("1. Data Processing")
    data_processor = DataProcessor(config)
    train_loader, val_loader, test_loader = data_processor.load_and_preprocess_data("./data/train.csv")
    vocab_sizes = data_processor.get_vocab_sizes()
    
    logger.info(f"Vocabulary sizes: {vocab_sizes}")
    logger.info(f"Training batches: {len(train_loader)}")
    
    # 2. Create model
    logger.info("2. Create RA-CoT Model")
    model = RACoTModel(config, vocab_sizes)
    trainer = RACoTTrainer(model, config)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train one epoch (demo)
    logger.info("3. Training Demo (1 epoch)")
    model.train()
    
    # Take a small batch for demo
    sample_batch = next(iter(train_loader))
    sample_batch = trainer._move_batch_to_device(sample_batch)
    
    # Forward pass
    model_output = model(sample_batch, explain=False)
    
    # Compute loss
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(model_output['predictions'], sample_batch['label'])
    
    logger.info(f"Loss value: {loss.item():.4f}")
    
    # 4. Generate explanations
    logger.info("4. Generate Recommendation Explanations")
    model.eval()
    with torch.no_grad():
        # Take first 3 samples
        small_batch = {}
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                small_batch[key] = value[:3]
            elif isinstance(value, list):
                small_batch[key] = value[:3]
            elif isinstance(value, dict):
                small_batch[key] = {k: v[:3] for k, v in value.items()}
            else:
                small_batch[key] = value
        
        explanations_output = model(small_batch, explain=True)
        explanations = explanations_output.get('explanations', [])
        
        for i, exp in enumerate(explanations):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Explanation: {exp}")
    
    # 5. Performance statistics
    logger.info("\n5. Performance Statistics")
    stats = model.get_statistics()
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"LLM calls: {stats['llm_calls']}")
    logger.info(f"Cache hits: {stats['cache_hits']}")
    logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    logger.info(f"Average latency: {stats['avg_latency']:.4f}s")
    
    return model, trainer


def demo_api_usage():
    """Demonstrate API usage"""
    logger.info("\n=== API Usage Demo ===")
    
    # API base URL
    base_url = "http://localhost:8000"
    
    # 1. Health check
    logger.info("1. Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"Service status: {health_data['status']}")
            logger.info(f"Model loaded: {health_data['model_loaded']}")
        else:
            logger.warning(f"Health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to API service: {e}")
        logger.info("Please start the service first: python serve.py")
        return
    
    # 2. Recommendation request
    logger.info("\n2. Recommendation Request")
    
    request_data = {
        "user_id": "demo_user_001",
        "user_history": [
            {"item_id": "smartphone_001", "category": "electronics", "action": "click"},
            {"item_id": "laptop_002", "category": "electronics", "action": "click"},
            {"item_id": "headphones_003", "category": "electronics", "action": "purchase"},
            {"item_id": "book_004", "category": "books", "action": "click"},
            {"item_id": "tablet_005", "category": "electronics", "action": "click"}
        ],
        "candidate_items": [
            {"item_id": "smartphone_006", "category": "electronics"},
            {"item_id": "book_007", "category": "books"},
            {"item_id": "camera_008", "category": "electronics"},
            {"item_id": "clothing_009", "category": "fashion"},
            {"item_id": "watch_010", "category": "electronics"}
        ],
        "top_k": 3,
        "explain": True
    }
    
    try:
        response = requests.post(f"{base_url}/recommend", json=request_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            
            logger.info(f"User ID: {result['user_id']}")
            logger.info("\nRecommendation Results:")
            for rec in result['recommendations']:
                logger.info(f"  Rank {rec['rank']}: {rec['item_id']} ({rec['category']}) - Score: {rec['score']:.4f}")
            
            if result.get('explanations'):
                logger.info("\nRecommendation Explanations:")
                for i, exp in enumerate(result['explanations']):
                    logger.info(f"\n  Item {i+1}: {exp}")
            
            logger.info("\nPerformance Statistics:")
            perf_stats = result['performance_stats']
            logger.info(f"  Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.2%}")
            logger.info(f"  Average latency: {perf_stats.get('avg_latency', 0):.4f}s")
            
        else:
            logger.error(f"Recommendation request failed: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Recommendation request exception: {e}")
    
    # 3. Get performance statistics
    logger.info("\n3. Performance Statistics")
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            logger.info(f"Total requests: {stats.get('total_requests', 0)}")
            logger.info(f"LLM calls: {stats.get('llm_calls', 0)}")
            logger.info(f"Cache hits: {stats.get('cache_hits', 0)}")
            logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get statistics: {e}")


def demo_batch_recommendation():
    """Demonstrate batch recommendation"""
    logger.info("\n=== Batch Recommendation Demo ===")
    
    # Simulate recommendation requests for multiple users
    users = [
        {
            "user_id": "tech_enthusiast_001",
            "history": [
                {"item_id": "iphone_14", "category": "electronics", "action": "purchase"},
                {"item_id": "macbook_pro", "category": "electronics", "action": "click"},
                {"item_id": "airpods", "category": "electronics", "action": "purchase"}
            ]
        },
        {
            "user_id": "book_lover_002",
            "history": [
                {"item_id": "python_book", "category": "books", "action": "purchase"},
                {"item_id": "ai_book", "category": "books", "action": "click"},
                {"item_id": "kindle", "category": "electronics", "action": "click"}
            ]
        },
        {
            "user_id": "fashion_fan_003",
            "history": [
                {"item_id": "nike_shoes", "category": "fashion", "action": "purchase"},
                {"item_id": "adidas_shirt", "category": "fashion", "action": "click"},
                {"item_id": "watch", "category": "accessories", "action": "click"}
            ]
        }
    ]
    
    candidate_items = [
        {"item_id": "new_smartphone", "category": "electronics"},
        {"item_id": "programming_book", "category": "books"},
        {"item_id": "running_shoes", "category": "fashion"},
        {"item_id": "wireless_earbuds", "category": "electronics"},
        {"item_id": "fitness_tracker", "category": "electronics"}
    ]
    
    base_url = "http://localhost:8000"
    
    for user in users:
        logger.info(f"\nGenerating recommendations for user {user['user_id']}:")
        
        request_data = {
            "user_id": user["user_id"],
            "user_history": user["history"],
            "candidate_items": candidate_items,
            "top_k": 3,
            "explain": True
        }
        
        try:
            response = requests.post(f"{base_url}/recommend", json=request_data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                
                logger.info("Recommendation Results:")
                for rec in result['recommendations']:
                    logger.info(f"  {rec['rank']}. {rec['item_id']} - Score: {rec['score']:.4f}")
                
                # Show explanation for first recommendation
                if result.get('explanations') and len(result['explanations']) > 0:
                    exp = result['explanations'][0]
                    logger.info(f"Explanation (Item: {exp})")
            else:
                logger.error(f"Request failed: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            break


def main():
    """Main function"""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    logger.info("RA-CoT Recommendation System Usage Demo")
    logger.info(f"Configuration: Device={config.DEVICE}, Batch Size={config.BATCH_SIZE}")
    
    # Create necessary directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    try:
        # 1. Model training demo
        model, trainer = demo_model_training()
        
        # 2. API usage demo (requires service to be started first)
        demo_api_usage()
        
        # 3. Batch recommendation demo
        demo_batch_recommendation()
        
        logger.info("\n=== Demo Complete ===")
        logger.info("To start full training, run: python train.py")
        logger.info("To start API service, run: python serve.py")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise


if __name__ == "__main__":
    main()
