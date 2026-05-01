"""
src/evaluation/fairness.py
==========================
Stratified fairness analysis by demographic groups (child vs. adult).
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import logging

from src.evaluation.evaluate import compute_metrics

log = logging.getLogger(__name__)

def load_speaker_info(spk_info_path: Path) -> pd.DataFrame:
    """
    Load speaker demographic info.
    Expected format is tab or comma separated with columns: SPEAKER, AGE, GENDER, etc.
    """
    if not spk_info_path.exists():
        raise FileNotFoundError(f"Speaker info file not found: {spk_info_path}")
        
    # Attempt to read as txt/csv
    try:
        df = pd.read_csv(spk_info_path, sep='\t')
        if 'SPEAKER' not in df.columns:
            df = pd.read_csv(spk_info_path, sep=',')
    except Exception as e:
        df = pd.read_csv(spk_info_path)
        
    # Standardize columns
    df.columns = [col.strip().upper() for col in df.columns]
    
    if 'SPEAKER' not in df.columns or 'AGE' not in df.columns:
        raise ValueError(f"Expected SPEAKER and AGE columns in {spk_info_path}")
        
    # Convert SPEAKER to string to match speaker_id
    df['SPEAKER'] = df['SPEAKER'].astype(str)
    
    # Classify as child (<= 14) vs adult (> 14)
    # Note: Speechocean762 speakers typically have age in years
    df['DEMO_GROUP'] = df['AGE'].apply(lambda age: 'child' if age <= 14 else 'adult')
    
    return df

def evaluate_fairness(
    targets: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    speaker_ids: List[str],
    spk_info_path: Path
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Stratifies evaluation metrics by demographic groups (child vs adult).
    
    Args:
        targets: Dictionary mapping metric name to 1D array of ground truth scores.
        predictions: Dictionary mapping metric name to 1D array of predicted scores.
        speaker_ids: List of speaker IDs corresponding to each item in the targets/predictions arrays.
        spk_info_path: Path to the dataset's speaker information file.
        
    Returns:
        Dict mapping metric_name -> group_name -> {pcc, rmse, src}
    """
    df_spk = load_speaker_info(spk_info_path)
    spk_to_group = dict(zip(df_spk['SPEAKER'], df_spk['DEMO_GROUP']))
    
    # Assign group to each prediction based on speaker_id
    groups = np.array([spk_to_group.get(str(spk), 'unknown') for spk in speaker_ids])
    
    unique_groups = np.unique(groups)
    log.info(f"Fairness evaluation groups found: {unique_groups}")
    
    fairness_results = {}
    
    for metric_name in targets.keys():
        if metric_name not in predictions:
            continue
            
        metric_targets = np.asarray(targets[metric_name])
        metric_preds = np.asarray(predictions[metric_name])
        
        # Ensure lengths match
        if len(metric_targets) != len(groups):
            log.warning(
                f"Length mismatch for metric {metric_name}. "
                f"targets: {len(metric_targets)}, groups: {len(groups)}"
            )
            continue
            
        metric_res = {}
        for group in unique_groups:
            mask = (groups == group)
            if np.sum(mask) == 0:
                continue
                
            group_targets = metric_targets[mask]
            group_preds = metric_preds[mask]
            
            metric_res[group] = compute_metrics(group_targets, group_preds)
            
        fairness_results[metric_name] = metric_res
        
    return fairness_results
