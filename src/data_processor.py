"""
Data Processing Module
Handles user behavior sequences, item features, and recommendation data
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class UserBehaviorSequence:
    """User behavior sequence data structure"""
    
    def __init__(self, user_id: str, items: List[str], 
                 categories: List[str], timestamps: List[int],
                 actions: List[str]):
        self.user_id = user_id
        self.items = items
        self.categories = categories
        self.timestamps = timestamps
        self.actions = actions
        
    def get_recent_sequence(self, max_length: int = 50) -> Dict:
        """Get recent behavior sequence"""
        if len(self.items) <= max_length:
            return {
                'items': self.items,
                'categories': self.categories,
                'timestamps': self.timestamps,
                'actions': self.actions
            }
        else:
            return {
                'items': self.items[-max_length:],
                'categories': self.categories[-max_length:],
                'timestamps': self.timestamps[-max_length:],
                'actions': self.actions[-max_length:]
            }


class RecommendationDataset(Dataset):
    """Recommendation system dataset"""
    
    def __init__(self, data: pd.DataFrame, item_encoder: LabelEncoder,
                 category_encoder: LabelEncoder, max_seq_length: int = 50):
        self.data = data
        self.item_encoder = item_encoder
        self.category_encoder = category_encoder
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # User historical sequence
        user_sequence = self._parse_sequence(row['user_sequence'])
        
        # Candidate item
        candidate_item = self.item_encoder.transform([row['item_id']])[0]
        candidate_category = self.category_encoder.transform([row['category']])[0]
        
        # Label
        label = float(row['label'])
        
        return {
            'user_id': row['user_id'],
            'user_sequence': user_sequence,
            'candidate_item': candidate_item,
            'candidate_category': candidate_category,
            'label': label
        }
    
    def _parse_sequence(self, sequence_str: str) -> Dict:
        """Parse user behavior sequence string"""
        # Assume sequence format: "item1:cat1:action1,item2:cat2:action2,..."
        if pd.isna(sequence_str) or sequence_str == "":
            return {
                'items': torch.zeros(self.max_seq_length, dtype=torch.long),
                'categories': torch.zeros(self.max_seq_length, dtype=torch.long),
                'actions': torch.zeros(self.max_seq_length, dtype=torch.long),
                'length': 0
            }
        
        items, categories, actions = [], [], []
        for item_info in sequence_str.split(','):
            parts = item_info.split(':')
            if len(parts) >= 3:
                items.append(parts[0])
                categories.append(parts[1])
                actions.append(parts[2])
        
        # Encoding
        encoded_items = self.item_encoder.transform(items)
        encoded_categories = self.category_encoder.transform(categories)
        encoded_actions = [1 if action == 'click' else 2 if action == 'purchase' else 0 
                          for action in actions]
        
        # Truncate or pad
        seq_length = min(len(encoded_items), self.max_seq_length)
        
        item_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        category_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        action_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        
        if seq_length > 0:
            item_tensor[:seq_length] = torch.tensor(encoded_items[-seq_length:])
            category_tensor[:seq_length] = torch.tensor(encoded_categories[-seq_length:])
            action_tensor[:seq_length] = torch.tensor(encoded_actions[-seq_length:])
        
        return {
            'items': item_tensor,
            'categories': category_tensor,
            'actions': action_tensor,
            'length': seq_length
        }


class DataProcessor:
    """Data processor"""
    
    def __init__(self, config):
        self.config = config
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and preprocess data"""
        logger.info(f"Loading data from {data_path}")
        
        # Assume there's a CSV file containing user behavior data
        # In practice, adjust according to specific data format
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            # Generate mock data for demonstration
            data = self._generate_mock_data()
        
        # Data preprocessing
        data = self._preprocess_data(data)
        
        # Fit encoders
        self._fit_encoders(data)
        
        # Data splitting
        train_data, val_data, test_data = self._split_data(data)
        
        # Create datasets and data loaders
        train_dataset = RecommendationDataset(
            train_data, self.item_encoder, self.category_encoder, 
            self.config.MAX_SEQUENCE_LENGTH
        )
        val_dataset = RecommendationDataset(
            val_data, self.item_encoder, self.category_encoder,
            self.config.MAX_SEQUENCE_LENGTH
        )
        test_dataset = RecommendationDataset(
            test_data, self.item_encoder, self.category_encoder,
            self.config.MAX_SEQUENCE_LENGTH
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, 
            shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE,
            shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.BATCH_SIZE,
            shuffle=False, num_workers=2
        )
        
        logger.info(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_loader, val_loader, test_loader
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock data for demonstration"""
        np.random.seed(self.config.RANDOM_SEED)
        
        n_users = 1000
        n_items = 5000
        n_categories = 50
        n_samples = 10000
        
        data = []
        for _ in range(n_samples):
            user_id = f"user_{np.random.randint(0, n_users)}"
            item_id = f"item_{np.random.randint(0, n_items)}"
            category = f"cat_{np.random.randint(0, n_categories)}"
            
            # Generate user historical sequence
            seq_length = np.random.randint(5, 20)
            sequence_items = [f"item_{np.random.randint(0, n_items)}" for _ in range(seq_length)]
            sequence_cats = [f"cat_{np.random.randint(0, n_categories)}" for _ in range(seq_length)]
            sequence_actions = np.random.choice(['click', 'purchase'], seq_length, p=[0.8, 0.2])
            
            user_sequence = ','.join([f"{item}:{cat}:{action}" 
                                    for item, cat, action in zip(sequence_items, sequence_cats, sequence_actions)])
            
            # Generate labels (click/conversion)
            label = np.random.choice([0, 1], p=[0.7, 0.3])
            
            data.append({
                'user_id': user_id,
                'item_id': item_id,
                'category': category,
                'user_sequence': user_sequence,
                'label': label
            })
        
        return pd.DataFrame(data)
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data preprocessing"""
        # Remove missing values
        data = data.dropna()
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        logger.info(f"Data preprocessing completed. Shape: {data.shape}")
        return data
    
    def _fit_encoders(self, data: pd.DataFrame):
        """Fit encoders"""
        # Collect all items and categories
        all_items = set(data['item_id'].unique())
        all_categories = set(data['category'].unique())
        all_users = set(data['user_id'].unique())
        
        # Extract items and categories from sequences
        for sequence in data['user_sequence'].dropna():
            for item_info in sequence.split(','):
                parts = item_info.split(':')
                if len(parts) >= 2:
                    all_items.add(parts[0])
                    all_categories.add(parts[1])
        
        # Fit encoders
        self.item_encoder.fit(list(all_items))
        self.category_encoder.fit(list(all_categories))
        self.user_encoder.fit(list(all_users))
        
        logger.info(f"Encoders fitted: Items={len(all_items)}, Categories={len(all_categories)}, Users={len(all_users)}")
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Data splitting"""
        data = data.sample(frac=1, random_state=self.config.RANDOM_SEED).reset_index(drop=True)
        
        n_total = len(data)
        n_train = int(n_total * self.config.TRAIN_RATIO)
        n_val = int(n_total * self.config.VAL_RATIO)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes"""
        return {
            'n_items': len(self.item_encoder.classes_),
            'n_categories': len(self.category_encoder.classes_),
            'n_users': len(self.user_encoder.classes_)
        }
