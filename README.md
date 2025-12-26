# RAG v1

Initial RAG implementation for question answering on SQuAD v2 dataset using Groq API.

## What This Branch Contains

This branch implements a standard RAG pipeline with the following components:

- **Document Processing**: SQuAD v2 contexts are chunked (1000 chars, 200 overlap) and stored in ChromaDB
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic search
- **LLM**: Groq API with `openai/gpt-oss-20b` model
- **Chatbot Interface**: Streamlit app with conversational RAG and session history
- **Evaluation**: Batch evaluation script with F1 and Exact Match metrics

## Files

- `app.py` - Streamlit chatbot interface
- `notebook.ipynb` - Evaluation notebook that runs RAG pipeline on SQuAD v2
- `dataset_loader.py` - SQuAD v2 dataset loading and chunking utilities
- `metrics.py` - Evaluation metrics (F1, Exact Match, unanswerable detection)
- `evaluation_results_1_50.json` - Results from evaluating on 50 samples

## Setup

```bash
pip install -r requirements.txt
```

Set `GROQ_API_KEY` in `.env` file. Run `notebook.ipynb` to build vector store and evaluate, or `streamlit run app.py` for the chatbot interface.

## Results

Evaluation on 50 SQuAD v2 samples: F1=0.128, EM=0.08, Unanswerable Detection=0.55

