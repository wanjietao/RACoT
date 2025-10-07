"""
Traditional Recommendation Model Implementation
Including classic models like DIEN, BST, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class AttentionLayer(nn.Module):
    """Attention layer"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: [batch_size, seq_len, hidden_dim]
            lengths: [batch_size]
        Returns:
            weighted_output: [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = sequences.shape
        
        # Calculate attention weights
        attention_scores = self.attention(sequences)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Create mask
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(sequences.device)
        mask = mask < lengths.unsqueeze(1)
        
        # Apply mask
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum
        weighted_output = torch.sum(sequences * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_output


class DIENModel(nn.Module):
    """DIEN model implementation"""
    
    def __init__(self, n_items: int, n_categories: int, embedding_dim: int, 
                 hidden_dim: int, max_seq_length: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        self.action_embedding = nn.Embedding(3, embedding_dim)  # 0: pad, 1: click, 2: purchase
        
        # Interest extraction layer
        self.interest_extractor = nn.GRU(
            embedding_dim * 3, hidden_dim, batch_first=True
        )
        
        # Interest evolution layer
        self.interest_evolution = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Batch data containing user sequences and candidate items
        Returns:
            predictions: [batch_size, 1]
        """
        user_sequence = batch['user_sequence']
        candidate_item = batch['candidate_item']
        candidate_category = batch['candidate_category']
        seq_lengths = batch['user_sequence']['length']
        
        # Sequence embedding
        item_emb = self.item_embedding(user_sequence['items'])  # [batch_size, seq_len, emb_dim]
        cat_emb = self.category_embedding(user_sequence['categories'])
        action_emb = self.action_embedding(user_sequence['actions'])
        
        # Concatenate features
        sequence_emb = torch.cat([item_emb, cat_emb, action_emb], dim=-1)
        
        # Interest extraction
        interest_states, _ = self.interest_extractor(sequence_emb)
        
        # Interest evolution
        evolved_interests, _ = self.interest_evolution(interest_states)
        
        # Attention aggregation
        user_representation = self.attention(evolved_interests, seq_lengths)
        
        # Candidate item embedding
        candidate_item_emb = self.item_embedding(candidate_item)
        candidate_cat_emb = self.category_embedding(candidate_category)
        
        # Feature concatenation
        features = torch.cat([
            user_representation, 
            candidate_item_emb, 
            candidate_cat_emb
        ], dim=-1)
        
        # Prediction
        predictions = self.predictor(features)
        
        return predictions.squeeze(-1)


class BSTModel(nn.Module):
    """BST (Behavior Sequence Transformer) model implementation"""
    
    def __init__(self, n_items: int, n_categories: int, embedding_dim: int,
                 hidden_dim: int, max_seq_length: int, n_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        self.action_embedding = nn.Embedding(3, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer layers
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 3,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2
        )
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3 + embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Batch data containing user sequences and candidate items
        Returns:
            predictions: [batch_size, 1]
        """
        user_sequence = batch['user_sequence']
        candidate_item = batch['candidate_item']
        candidate_category = batch['candidate_category']
        seq_lengths = batch['user_sequence']['length']
        
        batch_size, seq_len = user_sequence['items'].shape
        
        # Sequence embedding
        item_emb = self.item_embedding(user_sequence['items'])
        cat_emb = self.category_embedding(user_sequence['categories'])
        action_emb = self.action_embedding(user_sequence['actions'])
        
        # Position embedding
        positions = torch.arange(seq_len).expand(batch_size, seq_len).to(item_emb.device)
        pos_emb = self.position_embedding(positions)
        
        # Concatenate features
        sequence_emb = torch.cat([item_emb, cat_emb, action_emb], dim=-1)
        sequence_emb = sequence_emb + pos_emb.unsqueeze(-1).expand(-1, -1, sequence_emb.size(-1))
        
        # Create padding mask
        padding_mask = torch.arange(seq_len).expand(batch_size, seq_len).to(item_emb.device)
        padding_mask = padding_mask >= seq_lengths.unsqueeze(1)
        
        # Transformer encoding
        transformer_output = self.transformer(sequence_emb, src_key_padding_mask=padding_mask)
        
        # Get output from the last valid position
        last_indices = (seq_lengths - 1).clamp(min=0)
        user_representation = transformer_output[torch.arange(batch_size), last_indices]
        
        # Candidate item embedding
        candidate_item_emb = self.item_embedding(candidate_item)
        candidate_cat_emb = self.category_embedding(candidate_category)
        
        # Feature concatenation
        features = torch.cat([
            user_representation,
            candidate_item_emb,
            candidate_cat_emb
        ], dim=-1)
        
        # Prediction
        predictions = self.predictor(features)
        
        return predictions.squeeze(-1)


class TraditionalModelFactory:
    """Traditional model factory"""
    
    @staticmethod
    def create_model(model_type: str, vocab_sizes: Dict[str, int], 
                    embedding_dim: int, hidden_dim: int, max_seq_length: int) -> nn.Module:
        """Create traditional recommendation model"""
        
        if model_type.lower() == "dien":
            return DIENModel(
                n_items=vocab_sizes['n_items'],
                n_categories=vocab_sizes['n_categories'],
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                max_seq_length=max_seq_length
            )
        elif model_type.lower() == "bst":
            return BSTModel(
                n_items=vocab_sizes['n_items'],
                n_categories=vocab_sizes['n_categories'],
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                max_seq_length=max_seq_length
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class TraditionalModelTrainer:
    """Traditional model trainer"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            predictions = self.model(batch)
            loss = self.criterion(predictions, batch['label'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
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
                
                predictions = self.model(batch)
                loss = self.criterion(predictions, batch['label'])
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (predictions > 0.5).float()
                correct += (predicted == batch['label']).sum().item()
                total += batch['label'].size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to specified device"""
        device_batch = {}
        for key, value in batch.items():
            if key == 'user_sequence':
                device_batch[key] = {
                    k: v.to(self.device) for k, v in value.items()
                }
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
