"""
src/evaluation/evaluate.py
==========================
Deterministic evaluation computing PCC (Pearson), RMSE, and SRC (Spearman)
at phoneme, word, and sentence levels.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from typing import Dict

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes RMSE, PCC, and SRC for a given set of targets and predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # PCC and SRC can be undefined if variance is 0 (all predictions or targets are the same)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pcc, _ = pearsonr(y_true, y_pred)
        src, _ = spearmanr(y_true, y_pred)
    else:
        pcc = float('nan')
        src = float('nan')
        
    return {
        "rmse": float(rmse),
        "pcc": float(pcc),
        "src": float(src)
    }

def evaluate_all_metrics(targets: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all metrics present in both targets and predictions.
    
    Args:
        targets: Dictionary mapping metric name to 1D array of ground truth scores.
        predictions: Dictionary mapping metric name to 1D array of predicted scores.
        
    Returns:
        Dictionary mapping metric name to a dictionary of (rmse, pcc, src).
    """
    results = {}
    for metric_name in targets.keys():
        if metric_name in predictions:
            results[metric_name] = compute_metrics(targets[metric_name], predictions[metric_name])
    return results
