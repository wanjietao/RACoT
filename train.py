#!/usr/bin/env python3
"""
RA-CoT Model Training Script
"""
import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from loguru import logger
from tqdm import tqdm

from config.config import config
from src.data_processor import DataProcessor
from src.ra_cot_model import RACoTModel, RACoTTrainer


def set_random_seed(seed: int):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train RA-CoT Recommendation Model")
    parser.add_argument("--data_path", type=str, default="./data/train.csv",
                       help="Path to training data")
    parser.add_argument("--model_save_path", type=str, default="./models/ra_cot_model.pth",
                       help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add("./logs/training.log", rotation="10 MB", level="DEBUG")
    
    # Create necessary directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    logger.info("Starting RA-CoT model training")
    logger.info(f"Configuration: {config.__dict__}")
    
    # Set random seed
    set_random_seed(config.RANDOM_SEED)
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Data processing
        logger.info("Loading and preprocessing data...")
        data_processor = DataProcessor(config)
        train_loader, val_loader, test_loader = data_processor.load_and_preprocess_data(args.data_path)
        vocab_sizes = data_processor.get_vocab_sizes()
        
        logger.info(f"Vocabulary sizes: {vocab_sizes}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # 2. Create model
        logger.info("Creating RA-CoT model...")
        model = RACoTModel(config, vocab_sizes)
        trainer = RACoTTrainer(model, config)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 3. Training loop
        best_val_accuracy = 0.0
        patience = 3
        patience_counter = 0
        
        for epoch in range(args.epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
            
            # Training
            train_loss = trainer.train_epoch(train_loader)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = trainer.evaluate(val_loader)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Performance statistics
            perf_stats = model.get_statistics()
            logger.info(f"Performance stats: {perf_stats}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_val_accuracy,
                    'config': config.__dict__
                }, args.model_save_path)
                logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
            
            # Reset statistics
            model.reset_statistics()
        
        # 4. Final testing
        logger.info("\n=== Final Testing ===")
        checkpoint = torch.load(args.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")
        
        # 5. Generate sample explanations
        logger.info("\n=== Sample Explanations ===")
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(test_loader))
            sample_batch = trainer._move_batch_to_device(sample_batch)
            
            # Take only first 3 samples
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
            
            output = model(small_batch, explain=True)
            explanations = output.get('explanations', [])
            
            for i, exp in enumerate(explanations):
                logger.info(f"\nSample {i+1}:")
                logger.info(f"Explanation: {exp}")
        
        logger.info("\nTraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
