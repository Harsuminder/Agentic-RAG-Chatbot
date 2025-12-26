# eval-metrics

Evaluation framework and metrics implementation for RAG systems on SQuAD v2.

## What This Branch Contains

This branch focuses on building a comprehensive evaluation framework with metrics for assessing RAG system performance. The work includes implementing standard QA evaluation metrics, creating evaluation pipelines, and establishing result tracking.

Key components:

- **Evaluation Metrics**: F1 score, Exact Match, and unanswerable question detection
- **Batch Evaluation**: Automated evaluation pipeline for processing multiple examples
- **Result Tracking**: Structured output format for evaluation results with individual and aggregate metrics
- **SQuAD v2 Integration**: Evaluation workflow designed for SQuAD v2 dataset format

## Files

- `metrics.py` - Core evaluation metrics (F1, Exact Match, unanswerable detection)
- `notebook.ipynb` - Evaluation notebook that runs the complete evaluation pipeline
- `dataset_loader.py` - SQuAD v2 dataset loading utilities
- `evaluation_results_1_50.json` - Example evaluation results showing metrics structure
- `app.py` - Streamlit interface for interactive testing
- `requirements.txt` - Dependencies

## Evaluation Metrics

The evaluation framework includes:

- **F1 Score**: Token-level overlap between predictions and ground truth
- **Exact Match**: Binary score for exact string matches after normalization
- **Unanswerable Detection**: Measures how well the system identifies unanswerable questions using keyword-based detection

All metrics handle multiple ground truth answers by taking the best score across all valid answers.

## Usage

Run the evaluation notebook to process examples and generate metrics:

```bash
jupyter notebook notebook.ipynb
```

Results are saved in JSON format with aggregate metrics and individual predictions for detailed analysis.

