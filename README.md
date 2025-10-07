# RA-CoT: Retrieval-Augmented Chain-of-Thought for Scalable and Explainable Industrial Recommendation

This project implements the RA-CoT recommendation system framework proposed in the paper "Beyond End-to-End: A Low-Latency, Retrieval-Augmented Chain-of-Thought for Scalable and Explainable Industrial Recommendation".

## Overview

RA-CoT is an innovative recommendation system framework that cleverly combines the efficiency of traditional recommendation models with the reasoning capabilities of Large Language Models (LLMs). The framework uses a dual-pathway parallel architecture to provide high-quality explainable recommendations while maintaining millisecond-level response times.

### Key Features

- **Dual-pathway Parallel Architecture**: Traditional recommendation models as backbone, LLM as reasoning co-processor
- **Tri-dimensional Reasoning Framework**: Preference analysis, interest evolution, conversion attribution
- **Retrieval-Augmented Mechanism**: Intelligent caching and reuse of reasoning results to control latency
- **Residual Fusion Network**: Gated attention mechanism for heterogeneous feature fusion
- **Industrial-grade Performance**: Millisecond-level response time, supporting large-scale deployment

## Project Structure

```
ra_cot_project/
├── src/                          # Source code directory
│   ├── data_processor.py         # Data processing module
│   ├── traditional_models.py     # Traditional recommendation models (DIEN, BST, etc.)
│   ├── llm_reasoning.py          # LLM reasoning engine
│   ├── fusion_network.py         # Gated attention residual fusion network
│   └── ra_cot_model.py          # RA-CoT main model
├── config/                       # Configuration files
│   └── config.py                # Main configuration file
├── data/                        # Data directory
├── models/                      # Model save directory
├── logs/                        # Log directory
├── tests/                       # Test files
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation directory
├── train.py                     # Training script
├── serve.py                     # Inference service API
├── requirements.txt             # Dependency list
└── README.md                    # Project documentation
```

## Installation and Configuration

### 1. Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### 2. Install Dependencies

```bash
cd ra_cot_project
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

Create a `.env` file and configure the following environment variables:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# CUDA Configuration (optional)
CUDA_AVAILABLE=true
```

## Usage

### 1. Data Preparation

Prepare training data in CSV format with the following fields:
- `user_id`: User ID
- `item_id`: Item ID
- `category`: Item category
- `user_sequence`: User historical behavior sequence (format: "item1:cat1:action1,item2:cat2:action2,...")
- `label`: Label (0 or 1)

### 2. Model Training

```bash
python train.py --data_path ./data/train.csv --epochs 10 --batch_size 256
```

Training parameters:
- `--data_path`: Training data path
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--device`: Device (cpu/cuda)
- `--model_save_path`: Model save path

### 3. Start Inference Service

```bash
python serve.py
```

The service will start at `http://localhost:8000` with the following API endpoints:

- `GET /health`: Health check
- `POST /recommend`: Get recommendation results
- `POST /explain`: Get recommendation explanations
- `GET /stats`: Get performance statistics
- `POST /reset_stats`: Reset performance statistics

### 4. API Usage Example

```python
import requests

# Recommendation request
request_data = {
    "user_id": "user_123",
    "user_history": [
        {"item_id": "item_1", "category": "electronics", "action": "click"},
        {"item_id": "item_2", "category": "electronics", "action": "purchase"}
    ],
    "candidate_items": [
        {"item_id": "item_3", "category": "electronics"},
        {"item_id": "item_4", "category": "books"}
    ],
    "top_k": 5,
    "explain": True
}

response = requests.post("http://localhost:8000/recommend", json=request_data)
result = response.json()

print("Recommendations:", result["recommendations"])
print("Explanations:", result["explanations"])
```

## Core Components

### 1. Traditional Recommendation Models

Supports multiple classic recommendation models:
- **DIEN**: Deep Interest Evolution Network
- **BST**: Behavior Sequence Transformer

### 2. LLM Reasoning Engine

- **Tri-dimensional Reasoning**: Preference analysis, interest evolution, conversion attribution
- **Intelligent Caching**: Redis/memory caching mechanism
- **Prompt Engineering**: Carefully designed CoT prompt templates

### 3. Fusion Network

- **Adaptive Gating**: Dynamic adjustment of reasoning signal contribution
- **Context Attention**: Context-aware feature fusion
- **Residual Compensation**: Dynamic compensation mechanism for traditional predictions

## Performance Characteristics

### Accuracy Improvement
- CTR prediction AUC improvement: 3-5%
- CVR prediction AUC improvement: 2-4%
- Significant improvement over traditional models

### Latency Control
- Average response time: <50ms
- Cache hit rate: >80%
- LLM inference time: <200ms

### Explainability
- High-quality natural language explanations
- Tri-dimensional reasoning covers complete decision process
- Significant improvement in user trust

## Configuration

Main configuration parameters (`config/config.py`):

```python
# Model Configuration
TRADITIONAL_MODEL_TYPE = "dien"  # Traditional model type
LLM_MODEL_NAME = "gpt-4.1-mini"  # LLM model name
EMBEDDING_DIM = 128              # Embedding dimension
HIDDEN_DIM = 256                 # Hidden layer dimension

# Reasoning Configuration
MAX_SEQUENCE_LENGTH = 50         # Maximum sequence length
RETRIEVAL_TOP_K = 5             # Retrieval top-k
CACHE_TTL = 3600                # Cache expiration time

# Training Configuration
BATCH_SIZE = 256                # Batch size
LEARNING_RATE = 0.001           # Learning rate
EPOCHS = 10                     # Training epochs
```

## Extension and Customization

### 1. Adding New Traditional Models

Implement new model classes in `traditional_models.py` and register them in `TraditionalModelFactory`.

### 2. Custom Reasoning Dimensions

Modify `CoTPromptTemplate` in `llm_reasoning.py` to add new reasoning dimensions.

### 3. Optimize Fusion Strategy

Adjust gating mechanisms and attention computation in `fusion_network.py`.

## Monitoring and Debugging

### 1. Logging System

- Training logs: `./logs/training.log`
- Service logs: `./logs/service.log`
- Multi-level logging support

### 2. Performance Monitoring

```python
# Get performance statistics
stats = model.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Average inference time: {stats['avg_total_inference_time_ms']:.2f}ms")
```

### 3. Model Explanation

```python
# Get recommendation explanations
explanations = model.get_explanations(batch)
for exp in explanations:
    print(f"User: {exp['user_id']}")
    print(f"Preference analysis: {exp['explanation']['preference']}")
    print(f"Interest evolution: {exp['explanation']['evolution']}")
    print(f"Conversion attribution: {exp['explanation']['attribution']}")
```

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite the original paper:

```bibtex
@inproceedings{racot2026,
  title={Beyond End-to-End: A Low-Latency, Retrieval-Augmented Chain-of-Thought for Scalable and Explainable Industrial Recommendation},
  author={Author Names},
  booktitle={Proceedings of the Web Conference 2026},
  year={2026}
}
```

## Contact

For questions or suggestions, please contact us through:
- Submit an Issue
- Send email to: [wanjietao@gmail.com]

---

**Note**: This project is for academic research and educational purposes only. Please ensure thorough testing and optimization before deploying in production environments.
