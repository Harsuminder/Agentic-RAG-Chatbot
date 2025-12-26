# Local RAG Implementation - No Chunking Evaluation

This implementation evaluates a RAG (Retrieval-Augmented Generation) pipeline using a local Ollama setup with a no-chunking strategy. Each context from the SQuAD v2 dataset is stored as a single document chunk, and retrieval is evaluated by checking if the golden context ID appears in the top K retrieved contexts.

## Setup

### Requirements

- Python 3.8+
- Ollama installed and running locally
- Llama3:8b model available in Ollama
- NVIDIA GPU (optional, for faster embeddings)

### Installation

```bash
pip install -r requirements_local.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Configuration

Edit `config_local.py` to adjust settings:
- `OLLAMA_BASE_URL`: URL of your Ollama instance (default: http://localhost:11434)
- `LLM_MODEL`: Model name in Ollama (default: llama3:8b)
- `EMBEDDING_MODEL`: Embedding model (default: BAAI/bge-base-en-v1.5)
- `K_RETRIEVED`: Number of contexts to retrieve (default: 3)

## Files

- `rag_pipeline_tutorial_2.ipynb`: Main notebook with step-by-step RAG pipeline implementation and evaluation
- `config_local.py`: Configuration settings
- `dataset_loader.py`: SQuAD v2 dataset loading utilities
- `metrics.py`: Basic evaluation metrics (F1, Exact Match)
- `metrics_extended.py`: Extended metrics (ordered word recall)
- `requirements_local.txt`: Python dependencies

## Evaluation Approach

### No Chunking Strategy

Unlike traditional chunking approaches, each context from SQuAD v2 is stored as a single document. This simplifies evaluation by ensuring that:
- Each context has a unique ID
- Golden context ID exactly matches retrieved chunk ID if correct context was found
- No ambiguity from chunk boundaries

### Evaluation Metrics

**Retrieval Evaluation:**
- **Recall@K**: Percentage of questions where the golden context ID appears in the top K retrieved context IDs

**Answer Quality Evaluation:**
- **Recall (Ordered)**: Binary metric checking if all ground truth words appear in the same order in the generated answer
- **Exact Match**: Whether the normalized prediction exactly matches the normalized ground truth

## Usage

1. Ensure Ollama is running with llama3:8b model
2. Open `rag_pipeline_tutorial_2.ipynb`
3. Run cells sequentially to:
   - Load and prepare the dataset
   - Create or load the vector store
   - Evaluate retriever accuracy
   - Build and test the RAG chain
   - Evaluate answer quality on multiple examples
   - Save results

## Results

The notebook saves evaluation results to JSON files containing:
- Retriever metrics (Recall@K)
- Answer quality metrics (Recall, Exact Match)
- Configuration used
- Sample predictions and scores

