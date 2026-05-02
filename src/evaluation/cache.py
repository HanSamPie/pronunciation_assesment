"""
src/evaluation/cache.py
=======================
SQLite MD5 result cache with model/split identifiability.

Hashes the config, model weights, and dataset to determine if evaluation
can be skipped.  Each cached entry stores ``model_name`` and ``split`` so
results are human-queryable via SQL.
"""

import hashlib
import json
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
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


def generate_cache_key(
    cfg: DictConfig,
    model_weights_path: Path,
    data_manifest_path: Path,
    model_name: str,
    split: str,
) -> str:
    """
    Generates a unique hash key based on config, model weights, dataset,
    model name, and split.
    """
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_hash = compute_md5(cfg_str.encode('utf-8'))

    model_hash = hash_file(model_weights_path) if model_weights_path.exists() else "no_model"
    data_hash = hash_file(data_manifest_path) if data_manifest_path.exists() else "no_data"

    combined = f"{cfg_hash}_{model_hash}_{data_hash}_{model_name}_{split}"
    return compute_md5(combined.encode('utf-8'))


class EvaluationCache:
    """
    SQLite-backed result cache for evaluations.

    The schema stores ``model_name`` and ``split`` alongside each hash key
    so that cached results are human-identifiable via direct SQL queries::

        SELECT model_name, split, created_at FROM evaluation_cache;
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
                    model_name TEXT NOT NULL,
                    split TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def get(self, hash_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached results if they exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT results_json FROM evaluation_cache WHERE hash_key = ?',
                (hash_key,),
            )
            row = cursor.fetchone()
            if row:
                log.info("Cache hit for hash %s", hash_key)
                return json.loads(row[0])
        log.info("Cache miss for hash %s", hash_key)
        return None

    def put(
        self,
        hash_key: str,
        results: Dict[str, Any],
        model_name: str,
        split: str,
    ):
        """Saves evaluation results to the cache."""
        results_json = json.dumps(results)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO evaluation_cache
                    (hash_key, model_name, split, results_json)
                VALUES (?, ?, ?, ?)
            ''', (hash_key, model_name, split, results_json))
            conn.commit()
        log.info("Cached results for %s / %s (hash %s)", model_name, split, hash_key)

    def has_result(self, model_name: str, split: str) -> bool:
        """Check if any cached result exists for a model/split combination."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT 1 FROM evaluation_cache WHERE model_name = ? AND split = ? LIMIT 1',
                (model_name, split),
            )
            return cursor.fetchone() is not None

    def get_by_model_split(
        self, model_name: str, split: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached results by model name and split."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT results_json FROM evaluation_cache '
                'WHERE model_name = ? AND split = ? '
                'ORDER BY created_at DESC LIMIT 1',
                (model_name, split),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def list_cached(self) -> List[Tuple[str, str, str]]:
        """Returns all cached entries as (model_name, split, created_at)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT model_name, split, created_at FROM evaluation_cache '
                'ORDER BY created_at DESC'
            )
            return cursor.fetchall()
