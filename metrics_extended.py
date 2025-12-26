"""
Extended metrics with recall and exact match calculations
"""
import re
from typing import List, Dict, Tuple
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

def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def recall_ordered(prediction: str, ground_truth: str) -> float:
    """
    Calculate binary recall: Check if all words from ground truth appear 
    in the same order in the prediction.
    
    Returns 1.0 if all ground truth words found in order, 0.0 otherwise.
    Uses normalized text (lowercase, no punctuation, no articles).
    """
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)
    
    gt_words = normalized_gt.split()
    pred_words = normalized_pred.split()
    
    if len(gt_words) == 0:
        return 0.0
    if len(pred_words) == 0:
        return 0.0
    
    # Check if all ground truth words appear in order (subsequence check)
    gt_idx = 0
    for pred_word in pred_words:
        if gt_idx < len(gt_words) and pred_word == gt_words[gt_idx]:
            gt_idx += 1
    
    return 1.0 if gt_idx == len(gt_words) else 0.0

