"""
Baseline 2 — Regression Tree / XGBoost.

Input:  statically pooled eGeMAPS features (mean-pooled across phonemes
        within a sentence), shape (N_sentences, 88).
Output: per-sentence pronunciation score predictions.

The model backend is selected via ``cfg.model_type``:
  * ``"xgboost"``       — uses ``xgboost.XGBRegressor`` (default).
  * ``"decision_tree"`` — uses ``sklearn.tree.DecisionTreeRegressor``.

Hyperparameters come from ``configs/model/tree.yaml``:

    name: tree
    model_type: xgboost
    max_depth: 4
    min_samples_leaf: 5   # only for decision_tree backend

Max-depth and min-samples-leaf are deliberately constrained to prevent
overfitting on the Speechocean762 5000-sample dataset.

Global random seeds are applied inside each constructor call using the
seed from the Hydra config to satisfy the project's reproducibility rule.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def _build_sklearn_tree(cfg: DictConfig, seed: int):
    """Build a ``DecisionTreeRegressor`` from config."""
    from sklearn.tree import DecisionTreeRegressor  # local import keeps deps optional

    return DecisionTreeRegressor(
        max_depth=int(cfg.max_depth),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=seed,
    )


def _build_xgboost(cfg: DictConfig, seed: int):
    """Build an ``XGBRegressor`` from config."""
    import xgboost as xgb  # local import keeps deps optional

    return xgb.XGBRegressor(
        max_depth=int(cfg.max_depth),
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=int(cfg.min_samples_leaf),
        verbosity=0,
        random_state=seed,
        seed=seed,
    )


_BACKENDS = {
    "decision_tree": _build_sklearn_tree,
    "xgboost": _build_xgboost,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TreeBaseline:
    """Tree-based regression baseline for pronunciation assessment.

    Args:
        cfg:  Hydra DictConfig from ``configs/model/tree.yaml`` merged on
              top of ``configs/base.yaml``.  Must contain:
              ``model.model_type``, ``model.max_depth``,
              ``model.min_samples_leaf``, ``seed``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        model_cfg = cfg.model
        model_type: str = str(model_cfg.model_type).lower()
        if model_type not in _BACKENDS:
            raise ValueError(
                f"Unknown tree model_type '{model_type}'. "
                f"Choose from: {list(_BACKENDS)}."
            )
        seed: int = int(cfg.seed)
        self._model = _BACKENDS[model_type](model_cfg, seed)
        self._model_type: str = model_type
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "TreeBaseline":
        """Fit the tree model on the training partition.

        Args:
            X_train: shape (N_sentences, n_features) — pooled features.
            y_train: shape (N_sentences,)             — target scores.

        Returns:
            self
        """
        logger.info(
            "Fitting TreeBaseline (%s) on %d samples, %d features.",
            self._model_type,
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
            raise RuntimeError("TreeBaseline must be fitted before calling predict.")
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist the fitted model to disk using joblib.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "model_type": self._model_type,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(payload, path)
        logger.info("TreeBaseline (%s) saved to %s", self._model_type, path)

    @classmethod
    def load(cls, path: Path | str, cfg: DictConfig) -> "TreeBaseline":
        """Restore a previously saved TreeBaseline.

        Args:
            path: Path to the joblib file created by ``save``.
            cfg:  Config (used to reconstruct the instance shell).

        Returns:
            Restored ``TreeBaseline`` instance.
        """
        payload = joblib.load(path)
        instance = cls(cfg=cfg)
        instance._model = payload["model"]
        instance._model_type = payload["model_type"]
        instance._is_fitted = payload["is_fitted"]
        logger.info("TreeBaseline loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def params_dict(self) -> dict:
        """Return a flat dict of hyperparameters for ``mlflow.log_params``."""
        base = {
            "baseline_model": self._model_type,
            "tree_max_depth": int(self.cfg.model.max_depth),
        }
        if self._model_type == "decision_tree":
            base["tree_min_samples_leaf"] = int(self.cfg.model.min_samples_leaf)
        return base
