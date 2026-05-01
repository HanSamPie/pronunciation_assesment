"""
src/evaluation/cache.py
=======================
SQLite MD5 result cache.
Hashes the config, model weights, and test dataset to determine if evaluation can be skipped.
"""

import hashlib
import json
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def hash_file(filepath: Path) -> str:
    """Computes MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_cache_key(cfg: DictConfig, model_weights_path: Path, test_manifest_path: Path) -> str:
    """
    Generates a unique hash key based on config, model weights, and test dataset.
    """
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_hash = compute_md5(cfg_str.encode('utf-8'))
    
    model_hash = hash_file(model_weights_path) if model_weights_path.exists() else "no_model"
    data_hash = hash_file(test_manifest_path) if test_manifest_path.exists() else "no_data"
    
    combined = f"{cfg_hash}_{model_hash}_{data_hash}"
    return compute_md5(combined.encode('utf-8'))

class EvaluationCache:
    """
    SQLite-backed result cache for evaluations.
    """
    def __init__(self, db_path: str = "results.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initializes the database and table if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_cache (
                    hash_key TEXT PRIMARY KEY,
                    results_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            
    def get(self, hash_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached results if they exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT results_json FROM evaluation_cache WHERE hash_key = ?', (hash_key,))
            row = cursor.fetchone()
            if row:
                log.info(f"Cache hit for hash {hash_key}")
                return json.loads(row[0])
        log.info(f"Cache miss for hash {hash_key}")
        return None

    def put(self, hash_key: str, results: Dict[str, Any]):
        """Saves evaluation results to the cache."""
        results_json = json.dumps(results)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO evaluation_cache (hash_key, results_json)
                VALUES (?, ?)
            ''', (hash_key, results_json))
            conn.commit()
        log.info(f"Cached results for hash {hash_key}")
