"""
Weighted multi-task MSE loss for hierarchical pronunciation assessment.

Loss weights (phoneme / word / sentence) are read from the Hydra config
and must be logged to MLflow at run initialization so every experiment
remains fully reproducible. Default weights defined in configs/base.yaml:

    loss_weights:
        phoneme: 1.0
        word:    2.0
        sentence: 5.0

The weights apply at the *level* (phoneme / word / sentence), not per
individual metric, so all word-level metrics share ``loss_weights.word``.

Design note — gradient imbalance:
    The phoneme level produces far more samples per forward pass than
    word or sentence level.  Without upweighting, phoneme gradients
    dominate training and word/sentence heads under-train.  The weights
    above compensate for this imbalance; increase `sentence` further if
    the sentence-level PCC is unsatisfactory.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


def _level_of(metric_key: str) -> str:
    """Return the hierarchical level for a canonical metric key.

    Examples:
        ``phoneme_accuracy`` → ``"phoneme"``
        ``word_stress``      → ``"word"``
        ``sentence_fluency`` → ``"sentence"``
    """
    return metric_key.split("_")[0]


class MultiTaskLoss(nn.Module):
    """Weighted sum of MSE losses across all active metrics.

    The active metrics and their levels are determined by
    ``cfg.score_mode`` and ``cfg.metrics.<score_mode>``.

    Each metric's loss is weighted by the level weight defined in
    ``cfg.loss_weights``:

    * ``phoneme_*`` metrics → ``loss_weights.phoneme``
    * ``word_*``    metrics → ``loss_weights.word``
    * ``sentence_*``metrics → ``loss_weights.sentence``

    Args:
        cfg: Hydra DictConfig containing ``loss_weights.phoneme``,
             ``loss_weights.word``, ``loss_weights.sentence``,
             ``score_mode``, and ``metrics.<score_mode>``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        lw = cfg.loss_weights
        self._level_weights: dict[str, float] = {
            "phoneme": float(lw.phoneme),
            "word":    float(lw.word),
            "sentence": float(lw.sentence),
        }
        self._mse = nn.MSELoss(reduction="mean")

        # The ordered list of active metric keys mirrors the model's heads
        from omegaconf import OmegaConf  # local import to keep top-level clean

        self.active_metrics: list[str] = list(
            OmegaConf.to_container(
                cfg.metrics[cfg.score_mode], resolve=True
            ).keys()  # type: ignore[union-attr]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total weighted loss and individual component losses.

        Args:
            predictions: dict mapping canonical metric key → predicted tensor.
                         Must contain an entry for every key in
                         ``self.active_metrics``. Visualisation keys
                         (``word_attn_weights``, ``sentence_attn_weights``)
                         are silently ignored.
            targets:     dict mapping canonical metric key → ground-truth
                         tensor.  Same keys as ``predictions``.

        Returns:
            total_loss:  Scalar tensor used for ``loss.backward()``.
            components:  Dict with float values for MLflow logging.
                         Contains one ``"loss_<metric>"`` entry per active
                         metric plus ``"loss_total"``.
        """
        total: torch.Tensor | None = None
        components: dict[str, float] = {}

        for metric in self.active_metrics:
            if metric not in predictions or metric not in targets:
                raise KeyError(
                    f"Metric '{metric}' missing from predictions or targets. "
                    f"Available prediction keys: {list(predictions.keys())}."
                )
            pred = predictions[metric]
            tgt = targets[metric]
            level = _level_of(metric)
            weight = self._level_weights[level]

            metric_loss = self._mse(pred, tgt)
            components[f"loss_{metric}"] = metric_loss.item()

            weighted = weight * metric_loss
            total = weighted if total is None else total + weighted

        if total is None:
            raise RuntimeError("No active metrics — loss cannot be computed.")

        components["loss_total"] = total.item()
        return total, components

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def weights_dict(self) -> dict[str, float]:
        """Return loss weights for MLflow ``log_params``.

        Returns the level-based weights AND the active metric list so the
        full loss configuration is captured in a single params call.
        """
        out: dict[str, float] = {
            f"loss_weight_{level}": w
            for level, w in self._level_weights.items()
        }
        # Log which metrics are active (as a string, MLflow stores as str)
        out["active_metrics"] = ",".join(self.active_metrics)  # type: ignore[assignment]
        return out
