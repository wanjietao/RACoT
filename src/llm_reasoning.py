"""
LLM Reasoning Module
Implements Chain-of-Thought reasoning and retrieval-augmented mechanisms
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
import hashlib
import redis
from openai import OpenAI
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer


class CoTPromptTemplate:
    """Chain-of-Thought prompt template"""
    
    SYSTEM_PROMPT = """You are a professional recommendation system analyst. Please analyze user preferences for candidate items from three dimensions based on user historical behavior sequences:

1. Preference Analysis: Analyze user's long-term preference patterns, explain why users might like or dislike this item
2. Interest Evolution: Analyze user's recent interest change trajectory, explain the conversion logic in user behavior sequences
3. Conversion Attribution: Identify key factors that might drive users from browsing to purchasing

Please answer in concise and clear language, no more than 50 words per dimension."""
    
    USER_PROMPT_TEMPLATE = """User Historical Behavior Sequence:
{user_sequence}

Candidate Item:
- Item ID: {item_id}
- Category: {category}
- Item Features: {item_features}

Please analyze user's possible reactions to this candidate item from three dimensions: preference analysis, interest evolution, and conversion attribution.

Output Format:
{{
    "preference": "Preference analysis result",
    "evolution": "Interest evolution analysis",
    "attribution": "Conversion attribution analysis",
    "confidence": 0.8
}}"""
    
    @classmethod
    def format_user_sequence(cls, sequence_data: Dict) -> str:
        """Format user behavior sequence"""
        items = sequence_data.get('items', [])
        categories = sequence_data.get('categories', [])
        actions = sequence_data.get('actions', [])
        
        sequence_str = []
        for i, (item, cat, action) in enumerate(zip(items, categories, actions)):
            if item != 0:  # Skip padding
                sequence_str.append(f"{i+1}. Item{item} (Category{cat}) - {action}")
        
        return "\n".join(sequence_str[-10:])  # Show only the last 10 behaviors
    
    @classmethod
    def create_prompt(cls, user_sequence: Dict, item_id: str, 
                     category: str, item_features: str = "") -> str:
        """Create complete reasoning prompt"""
        formatted_sequence = cls.format_user_sequence(user_sequence)
        
        return cls.USER_PROMPT_TEMPLATE.format(
            user_sequence=formatted_sequence,
            item_id=item_id,
            category=category,
            item_features=item_features or "No detailed features available"
        )


class LLMReasoningEngine:
    """LLM reasoning engine"""
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI()  # Use API key from environment variables
        self.prompt_template = CoTPromptTemplate()
        
        # Initialize Redis cache
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
            self.memory_cache = {}
    
    def generate_reasoning(self, user_sequence: Dict, item_id: str, 
                          category: str, item_features: str = "") -> Dict:
        """Generate reasoning chain"""
        # Create cache key
        cache_key = self._create_cache_key(user_sequence, item_id, category)
        
        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_result
        
        # Generate new reasoning
        try:
            prompt = self.prompt_template.create_prompt(
                user_sequence, item_id, category, item_features
            )
            
            response = self.client.chat.completions.create(
                model=self.config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.prompt_template.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Parse response
            reasoning_text = response.choices[0].message.content
            reasoning_dict = self._parse_reasoning_response(reasoning_text)
            
            # Cache result
            self._save_to_cache(cache_key, reasoning_dict)
            
            logger.debug(f"Generated new reasoning for key: {cache_key}")
            return reasoning_dict
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            # Return default reasoning
            return self._get_default_reasoning()
    
    def _create_cache_key(self, user_sequence: Dict, item_id: str, category: str) -> str:
        """Create cache key"""
        # Use the last few items from user sequence and candidate item to create key
        recent_items = user_sequence.get('items', [])[-5:]  # Last 5 items
        key_data = {
            'recent_items': recent_items.tolist() if hasattr(recent_items, 'tolist') else recent_items,
            'item_id': item_id,
            'category': category
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache"""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                return self.memory_cache.get(cache_key)
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, reasoning_dict: Dict):
        """Save result to cache"""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    cache_key, 
                    self.config.CACHE_TTL, 
                    json.dumps(reasoning_dict)
                )
            else:
                self.memory_cache[cache_key] = reasoning_dict
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
    
    def _parse_reasoning_response(self, response_text: str) -> Dict:
        """Parse LLM response"""
        try:
            # Try to parse JSON directly
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        
        # If JSON parsing fails, use rule-based parsing
        return self._rule_based_parse(response_text)
    
    def _rule_based_parse(self, text: str) -> Dict:
        """Rule-based response parsing"""
        reasoning = {
            "preference": "",
            "evolution": "",
            "attribution": "",
            "confidence": 0.5
        }
        
        lines = text.split('\n')
        current_key = None
        
        for line in lines:
            line = line.strip()
            if 'preference' in line.lower():
                current_key = 'preference'
            elif 'evolution' in line.lower():
                current_key = 'evolution'
            elif 'attribution' in line.lower():
                current_key = 'attribution'
            elif current_key and line and not line.startswith('{') and not line.startswith('}'):
                reasoning[current_key] = line
                current_key = None
        
        return reasoning
    
    def _get_default_reasoning(self) -> Dict:
        """Get default reasoning result"""
        return {
            "preference": "Based on historical behavior, user has some interest in this type of item",
            "evolution": "User interest is relatively stable, consistent with historical preference patterns",
            "attribution": "Item features have moderate matching with user needs",
            "confidence": 0.3
        }


class ReasoningVectorEncoder(nn.Module):
    """Reasoning vector encoder"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use pre-trained sentence encoder
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Reasoning vector projection layer
        self.reasoning_projector = nn.Sequential(
            nn.Linear(sentence_dim * 3, config.HIDDEN_DIM),  # 3 dimensions of reasoning
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HIDDEN_DIM, config.EMBEDDING_DIM)
        )
        
        # Confidence encoder
        self.confidence_encoder = nn.Linear(1, config.EMBEDDING_DIM // 4)
        
    def encode_reasoning(self, reasoning_dict: Dict) -> torch.Tensor:
        """Encode reasoning dictionary to vector"""
        # Encode three dimensions of reasoning text
        preference_emb = self.sentence_encoder.encode(reasoning_dict.get('preference', ''))
        evolution_emb = self.sentence_encoder.encode(reasoning_dict.get('evolution', ''))
        attribution_emb = self.sentence_encoder.encode(reasoning_dict.get('attribution', ''))
        
        # Concatenate reasoning vectors
        reasoning_vector = np.concatenate([preference_emb, evolution_emb, attribution_emb])
        reasoning_tensor = torch.tensor(reasoning_vector, dtype=torch.float32)
        
        # Project to target dimension
        projected_reasoning = self.reasoning_projector(reasoning_tensor.unsqueeze(0))
        
        # Encode confidence
        confidence = torch.tensor([reasoning_dict.get('confidence', 0.5)], dtype=torch.float32)
        confidence_emb = self.confidence_encoder(confidence.unsqueeze(0))
        
        # Concatenate reasoning vector and confidence
        final_vector = torch.cat([projected_reasoning, confidence_emb], dim=-1)
        
        return final_vector.squeeze(0)
    
    def forward(self, reasoning_batch: List[Dict]) -> torch.Tensor:
        """Batch encode reasoning"""
        batch_vectors = []
        for reasoning_dict in reasoning_batch:
            vector = self.encode_reasoning(reasoning_dict)
            batch_vectors.append(vector)
        
        return torch.stack(batch_vectors)


class CategoryPreferenceLibrary:
    """Category preference CoT attribution library"""
    
    def __init__(self, config):
        self.config = config
        self.library = {}  # {(user_id, category): reasoning_vector}
        self.access_count = {}  # Access count
        self.max_size = 10000  # Maximum storage size
        
    def store_reasoning(self, user_id: str, category: str, 
                       reasoning_vector: torch.Tensor):
        """Store reasoning vector"""
        key = (user_id, category)
        self.library[key] = reasoning_vector.detach().clone()
        self.access_count[key] = self.access_count.get(key, 0) + 1
        
        # If exceeds maximum capacity, remove least used entries
        if len(self.library) > self.max_size:
            self._evict_least_used()
    
    def retrieve_reasoning(self, user_id: str, category: str) -> Optional[torch.Tensor]:
        """Retrieve reasoning vector"""
        key = (user_id, category)
        if key in self.library:
            self.access_count[key] += 1
            return self.library[key].clone()
        return None
    
    def _evict_least_used(self):
        """Remove least used entries"""
        if not self.access_count:
            return
        
        # Find the key with least access count
        least_used_key = min(self.access_count.keys(), 
                           key=lambda k: self.access_count[k])
        
        # Remove entry
        del self.library[least_used_key]
        del self.access_count[least_used_key]
        
        logger.debug(f"Evicted reasoning for key: {least_used_key}")
    
    def get_stats(self) -> Dict:
        """Get library statistics"""
        return {
            'total_entries': len(self.library),
            'total_accesses': sum(self.access_count.values()),
            'avg_access_per_entry': sum(self.access_count.values()) / len(self.access_count) if self.access_count else 0
        }
