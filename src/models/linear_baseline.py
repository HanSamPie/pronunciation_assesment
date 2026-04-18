"""
Baseline 1 — Linear Regression.

Input:  statically pooled eGeMAPS features (mean-pooled across phonemes
        within a sentence), shape (N_sentences, 88).
Output: per-sentence pronunciation score predictions.

The model wraps ``sklearn.linear_model.LinearRegression`` and exposes:
  * ``fit``    — trains on the training partition.
  * ``predict``— produces predictions for any partition.
  * ``save``   — persists model + metadata via ``joblib``.
  * ``load``   — class-method for restoring a persisted model.

MLflow logging is handled by the caller (``trainer.py``).  This module
intentionally has no MLflow dependency.

Global random seeds are set via ``src.utils.seed`` before any call to
ensure reproducibility, as required by project rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class LinearBaseline:
    """Scikit-learn Linear Regression baseline for pronunciation assessment.

    Args:
        cfg: Hydra DictConfig (currently unused; kept for API symmetry
             with ``TreeBaseline`` and future hyperparameter hooks).
    """

    def __init__(self, cfg: Optional[DictConfig] = None) -> None:
        self.cfg = cfg
        self._model = LinearRegression()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "LinearBaseline":
        """Fit the linear model on the training partition.

        Args:
            X_train: shape (N_sentences, n_features) — pooled features.
            y_train: shape (N_sentences,)             — target scores.

        Returns:
            self
        """
        logger.info(
            "Fitting LinearBaseline on %d samples, %d features.",
            X_train.shape[0],
            X_train.shape[1],
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict pronunciation scores.

        Args:
            X:  shape (N, n_features).

        Returns:
            y_pred: shape (N,).
        """
        if not self._is_fitted:
            raise RuntimeError("LinearBaseline must be fitted before calling predict.")
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist the fitted model to disk using joblib.

        Args:
            path: Destination file path (e.g. ``models/linear_baseline.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(payload, path)
        logger.info("LinearBaseline saved to %s", path)

    @classmethod
    def load(cls, path: Path | str, cfg: Optional[DictConfig] = None) -> "LinearBaseline":
        """Restore a previously saved LinearBaseline.

        Args:
            path: Path to the joblib file created by ``save``.
            cfg:  Optional config (passed through to constructor).

        Returns:
            Restored ``LinearBaseline`` instance.
        """
        payload = joblib.load(path)
        instance = cls(cfg=cfg)
        instance._model = payload["model"]
        instance._is_fitted = payload["is_fitted"]
        logger.info("LinearBaseline loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def params_dict(self) -> dict:
        """Return a flat dict of hyperparameters for ``mlflow.log_params``."""
        return {"baseline_model": "linear_regression"}
