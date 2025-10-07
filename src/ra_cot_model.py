"""
RA-CoT Main Model
Integrates traditional recommendation models, LLM reasoning, and fusion networks
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from loguru import logger
import time

from .traditional_models import TraditionalModelFactory
from .llm_reasoning import LLMReasoningEngine, ReasoningVectorEncoder, CategoryPreferenceLibrary
from .fusion_network import GatedAttentionResidualFusionNetwork


class RACoTModel(nn.Module):
    """RA-CoT recommendation model"""
    
    def __init__(self, config, vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = config
        self.vocab_sizes = vocab_sizes
        
        # Traditional recommendation model
        self.traditional_model = TraditionalModelFactory.create_model(
            config.TRADITIONAL_MODEL_TYPE,
            vocab_sizes,
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            config.MAX_SEQUENCE_LENGTH
        )
        
        # LLM reasoning components
        self.llm_engine = LLMReasoningEngine(config)
        self.reasoning_encoder = ReasoningVectorEncoder(config)
        self.preference_library = CategoryPreferenceLibrary(config)
        
        # Fusion network
        self.fusion_network = GatedAttentionResidualFusionNetwork(config)
        
        # Performance statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'llm_calls': 0,
            'avg_latency': 0.0
        }
        
    def forward(self, batch: Dict, explain: bool = False) -> Dict:
        """
        Forward pass of RA-CoT model
        Args:
            batch: Input batch containing user sequences and candidate items
            explain: Whether to generate explanations
        Returns:
            Dictionary containing predictions and optional explanations
        """
        start_time = time.time()
        
        # Traditional model prediction
        traditional_pred = self.traditional_model(batch)
        
        # Generate reasoning if requested
        reasoning_vectors = []
        explanations = []
        availability_mask = []
        
        for i in range(len(batch['user_id'])):
            user_id = batch['user_id'][i]
            item_id = batch['candidate_item'][i].item()
            category = batch['candidate_category'][i].item()
            
            # Try to retrieve from preference library first
            cached_reasoning = self.preference_library.retrieve_reasoning(
                str(user_id), str(category)
            )
            
            if cached_reasoning is not None:
                reasoning_vectors.append(cached_reasoning)
                availability_mask.append(1.0)
                if explain:
                    explanations.append({"cached": True, "reasoning": "Retrieved from preference library"})
                self.stats['cache_hits'] += 1
            else:
                # Generate new reasoning
                try:
                    user_sequence = {
                        'items': batch['user_sequence']['items'][i],
                        'categories': batch['user_sequence']['categories'][i],
                        'actions': batch['user_sequence']['actions'][i]
                    }
                    
                    reasoning_dict = self.llm_engine.generate_reasoning(
                        user_sequence, str(item_id), str(category)
                    )
                    
                    # Encode reasoning to vector
                    reasoning_vector = self.reasoning_encoder.encode_reasoning(reasoning_dict)
                    reasoning_vectors.append(reasoning_vector)
                    
                    # Store in preference library
                    self.preference_library.store_reasoning(
                        str(user_id), str(category), reasoning_vector
                    )
                    
                    availability_mask.append(1.0)
                    if explain:
                        explanations.append(reasoning_dict)
                    
                    self.stats['llm_calls'] += 1
                    
                except Exception as e:
                    logger.warning(f"Reasoning generation failed: {e}")
                    # Use zero vector as fallback
                    reasoning_vectors.append(torch.zeros(self.reasoning_encoder.config.EMBEDDING_DIM + 
                                                       self.reasoning_encoder.config.EMBEDDING_DIM // 4))
                    availability_mask.append(0.0)
                    if explain:
                        explanations.append({"error": str(e)})
        
        # Stack reasoning vectors
        reasoning_batch = torch.stack(reasoning_vectors).to(traditional_pred.device)
        availability_tensor = torch.tensor(availability_mask).to(traditional_pred.device)
        
        # Fusion network
        fusion_output = self.fusion_network(
            traditional_pred.unsqueeze(-1) if traditional_pred.dim() == 1 else traditional_pred,
            reasoning_batch,
            availability_tensor.unsqueeze(-1)
        )
        
        # Update statistics
        self.stats['total_requests'] += len(batch['user_id'])
        latency = time.time() - start_time
        self.stats['avg_latency'] = (self.stats['avg_latency'] * (self.stats['total_requests'] - len(batch['user_id'])) + 
                                   latency) / self.stats['total_requests']
        
        result = {
            'predictions': fusion_output['prediction'],
            'traditional_predictions': traditional_pred,
            'fusion_features': fusion_output['fused_features']
        }
        
        if explain:
            result['explanations'] = explanations
            result['fusion_weights'] = self.fusion_network.get_fusion_weights(
                traditional_pred.unsqueeze(-1) if traditional_pred.dim() == 1 else traditional_pred,
                reasoning_batch
            )
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get model performance statistics"""
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_requests'], 1)
        llm_call_rate = self.stats['llm_calls'] / max(self.stats['total_requests'], 1)
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'llm_call_rate': llm_call_rate,
            'preference_library_stats': self.preference_library.get_stats()
        }
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'llm_calls': 0,
            'avg_latency': 0.0
        }


class RACoTTrainer:
    """RA-CoT model trainer"""
    
    def __init__(self, model: RACoTModel, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.8
        )
        
    def train_epoch(self, train_loader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            output = self.model(batch, explain=False)
            predictions = output['predictions']
            
            # Compute loss
            loss = self.criterion(predictions, batch['label'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        # Update learning rate
        self.scheduler.step()
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                output = self.model(batch, explain=False)
                predictions = output['predictions']
                
                loss = self.criterion(predictions, batch['label'])
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (torch.sigmoid(predictions) > 0.5).float()
                correct += (predicted == batch['label']).sum().item()
                total += batch['label'].size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        device_batch = {}
        for key, value in batch.items():
            if key == 'user_sequence':
                device_batch[key] = {
                    sub_key: sub_value.to(self.device) if isinstance(sub_value, torch.Tensor) else sub_value
                    for sub_key, sub_value in value.items()
                }
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch
