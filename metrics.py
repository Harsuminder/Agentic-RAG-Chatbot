import re
from typing import List, Dict
import numpy as np

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('.,!?;:')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = len(common) / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_single(prediction: str, ground_truths: List[str]) -> Dict:
    """
    Evaluate a single prediction against multiple ground truth answers.
    Returns best F1 and EM scores.
    """
    # FIX: Handle None, empty list, or empty strings
    if not ground_truths or ground_truths == [] or ground_truths == [None]:
        ground_truths = [""]
    
    # Ensure all are strings
    ground_truths = [str(gt) if gt is not None else "" for gt in ground_truths]
    
    # Calculate scores
    f1_scores = [f1_score(prediction, gt) for gt in ground_truths]
    em_scores = [exact_match(prediction, gt) for gt in ground_truths]
    
    # FIX: Safety check - if somehow still empty, return zeros
    if not f1_scores or not em_scores:
        return {'f1': 0.0, 'em': 0.0}
    
    return {
        'f1': max(f1_scores),
        'em': max(em_scores)
    }

def evaluate_batch(predictions: List[str], ground_truths_list: List[List[str]]) -> Dict:
    """
    Evaluate a batch of predictions.
    """
    results = [evaluate_single(pred, gt) for pred, gt in zip(predictions, ground_truths_list)]
    
    return {
        'f1': np.mean([r['f1'] for r in results]),
        'em': np.mean([r['em'] for r in results]),
        'individual_results': results
    }

def evaluate_unanswerable(predictions: List[str], is_impossible: List[bool], 
                         threshold_keywords: List[str] = None) -> Dict:
    """
    Evaluate how well the model handles unanswerable questions.
    """
    if threshold_keywords is None:
        threshold_keywords = [
            "don't know", "don't have", "cannot", "not in", 
            "not available", "no information", "unable to"
        ]
    
    unanswerable_predictions = [pred.lower() for pred, impossible in 
                                zip(predictions, is_impossible) if impossible]
    
    # Check if predictions contain unanswerable indicators
    detected_unanswerable = []
    for pred in unanswerable_predictions:
        detected = any(keyword in pred for keyword in threshold_keywords)
        detected_unanswerable.append(detected)
    
    accuracy = np.mean(detected_unanswerable) if detected_unanswerable else 0.0
    
    return {
        'unanswerable_detection_accuracy': accuracy,
        'total_unanswerable': len(is_impossible) - sum(is_impossible),
        'total_answerable': sum(is_impossible)
    }