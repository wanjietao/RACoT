"""
RA-CoT Recommendation System
Retrieval-Augmented Chain-of-Thought for Scalable and Explainable Industrial Recommendation
"""

__version__ = "1.0.0"
__author__ = "RA-CoT Team"
__email__ = "racot@example.com"

from .ra_cot_model import RACoTModel, RACoTTrainer
from .data_processor import DataProcessor
from .traditional_models import TraditionalModelFactory
from .llm_reasoning import LLMReasoningEngine
from .fusion_network import GatedAttentionResidualFusionNetwork

__all__ = [
    "RACoTModel",
    "RACoTTrainer", 
    "DataProcessor",
    "TraditionalModelFactory",
    "LLMReasoningEngine",
    "GatedAttentionResidualFusionNetwork"
]
