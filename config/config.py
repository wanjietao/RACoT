"""
RA-CoT Recommendation System Configuration File
"""
from pydantic import BaseSettings
from typing import List, Dict, Any
import os


class RACoTConfig(BaseSettings):
    """RA-CoT System Configuration"""
    
    # Model configuration
    TRADITIONAL_MODEL_TYPE: str = "dien"  # dien, bst, sim
    LLM_MODEL_NAME: str = "gpt-4.1-mini"
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    
    # Reasoning configuration
    MAX_SEQUENCE_LENGTH: int = 50
    COT_DIMENSIONS: List[str] = ["preference", "evolution", "attribution"]
    RETRIEVAL_TOP_K: int = 5
    CACHE_TTL: int = 3600  # Cache expiration time (seconds)
    
    # Training configuration
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 10
    DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
    
    # Data configuration
    DATA_PATH: str = "./data"
    MODEL_SAVE_PATH: str = "./models"
    LOG_PATH: str = "./logs"
    
    # API configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Service configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_WORKERS: int = 4
    
    # Experiment configuration
    RANDOM_SEED: int = 42
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    
    class Config:
        env_file = ".env"


# Global configuration instance
config = RACoTConfig()
