"""
Gated Attention Residual Fusion Network
Implements intelligent fusion of traditional model predictions and LLM reasoning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AdaptiveGatingModule(nn.Module):
    """Adaptive gating module"""
    
    def __init__(self, traditional_dim: int, reasoning_dim: int, hidden_dim: int):
        super().__init__()
        self.traditional_dim = traditional_dim
        self.reasoning_dim = reasoning_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection layers
        self.traditional_proj = nn.Linear(traditional_dim, hidden_dim)
        self.reasoning_proj = nn.Linear(reasoning_dim, hidden_dim)
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, traditional_features: torch.Tensor, 
                reasoning_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of adaptive gating
        Args:
            traditional_features: Traditional model features [batch_size, traditional_dim]
            reasoning_features: LLM reasoning features [batch_size, reasoning_dim]
        Returns:
            gated_traditional: Gated traditional features
            gated_reasoning: Gated reasoning features
        """
        # Project features to same dimension
        proj_traditional = self.traditional_proj(traditional_features)
        proj_reasoning = self.reasoning_proj(reasoning_features)
        
        # Compute gate weights
        combined_features = torch.cat([proj_traditional, proj_reasoning], dim=-1)
        gate_weight = self.gate_network(combined_features)  # [batch_size, 1]
        
        # Apply gating
        gated_traditional = proj_traditional * (1 - gate_weight)
        gated_reasoning = proj_reasoning * gate_weight
        
        return gated_traditional, gated_reasoning


class ContextAwareAttention(nn.Module):
    """Context-aware attention mechanism"""
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, traditional_features: torch.Tensor, 
                reasoning_features: torch.Tensor) -> torch.Tensor:
        """
        Context-aware attention fusion
        Args:
            traditional_features: [batch_size, feature_dim]
            reasoning_features: [batch_size, feature_dim]
        Returns:
            attended_features: [batch_size, feature_dim]
        """
        batch_size = traditional_features.size(0)
        
        # Stack features for attention
        features = torch.stack([traditional_features, reasoning_features], dim=1)  # [batch_size, 2, feature_dim]
        
        # Multi-head attention
        queries = self.query_proj(features).view(batch_size, 2, self.num_heads, self.head_dim)
        keys = self.key_proj(features).view(batch_size, 2, self.num_heads, self.head_dim)
        values = self.value_proj(features).view(batch_size, 2, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # [batch_size, num_heads, 2, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_values = torch.matmul(attention_weights, values)
        
        # Concatenate heads and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, 2, self.feature_dim
        )
        attended_features = self.output_proj(attended_values)
        
        # Aggregate attended features
        final_features = torch.mean(attended_features, dim=1)  # [batch_size, feature_dim]
        
        return final_features


class GatedAttentionResidualFusionNetwork(nn.Module):
    """Gated attention residual fusion network"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.traditional_dim = config.HIDDEN_DIM
        self.reasoning_dim = config.EMBEDDING_DIM + config.EMBEDDING_DIM // 4  # reasoning + confidence
        self.fusion_dim = config.HIDDEN_DIM
        
        # Core components
        self.adaptive_gating = AdaptiveGatingModule(
            self.traditional_dim, self.reasoning_dim, self.fusion_dim
        )
        
        self.context_attention = ContextAwareAttention(
            self.fusion_dim, num_heads=4
        )
        
        # Residual connection and normalization
        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        self.residual_proj = nn.Linear(self.traditional_dim, self.fusion_dim)
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 2, 1)
        )
        
    def forward(self, traditional_prediction: torch.Tensor, 
                reasoning_vector: torch.Tensor, 
                availability_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of fusion network
        Args:
            traditional_prediction: Traditional model output [batch_size, traditional_dim]
            reasoning_vector: LLM reasoning vector [batch_size, reasoning_dim]
            availability_mask: Mask for reasoning availability [batch_size, 1]
        Returns:
            Dictionary containing predictions and intermediate results
        """
        batch_size = traditional_prediction.size(0)
        
        # Handle missing reasoning vectors
        if availability_mask is not None:
            # Use zero vectors where reasoning is not available
            reasoning_vector = reasoning_vector * availability_mask.unsqueeze(-1)
        
        # Adaptive gating
        gated_traditional, gated_reasoning = self.adaptive_gating(
            traditional_prediction, reasoning_vector
        )
        
        # Context-aware attention fusion
        attended_features = self.context_attention(gated_traditional, gated_reasoning)
        
        # Residual connection
        residual = self.residual_proj(traditional_prediction)
        fused_features = self.layer_norm(attended_features + residual)
        
        # Final prediction
        final_prediction = self.predictor(fused_features)
        
        return {
            'prediction': final_prediction.squeeze(-1),
            'fused_features': fused_features,
            'gated_traditional': gated_traditional,
            'gated_reasoning': gated_reasoning,
            'attended_features': attended_features
        }
    
    def get_fusion_weights(self, traditional_prediction: torch.Tensor, 
                          reasoning_vector: torch.Tensor) -> torch.Tensor:
        """Get fusion weights for interpretability"""
        with torch.no_grad():
            gated_traditional, gated_reasoning = self.adaptive_gating(
                traditional_prediction, reasoning_vector
            )
            
            # Compute relative importance
            traditional_norm = torch.norm(gated_traditional, dim=-1, keepdim=True)
            reasoning_norm = torch.norm(gated_reasoning, dim=-1, keepdim=True)
            
            total_norm = traditional_norm + reasoning_norm + 1e-8
            traditional_weight = traditional_norm / total_norm
            reasoning_weight = reasoning_norm / total_norm
            
            return torch.cat([traditional_weight, reasoning_weight], dim=-1)
