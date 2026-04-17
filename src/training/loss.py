"""
Weighted multi-task MSE loss for hierarchical pronunciation assessment.

Loss weights (phoneme / word / sentence) are read from the Hydra config
and must be logged to MLflow at run initialization so every experiment
remains fully reproducible. Default weights defined in configs/base.yaml:

    loss_weights:
        phoneme: 1.0
        word:    2.0
        sentence: 5.0

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


class MultiTaskLoss(nn.Module):
    """Weighted sum of MSE losses at phoneme, word, and sentence levels.

    Args:
        cfg: Hydra DictConfig containing ``loss_weights.phoneme``,
             ``loss_weights.word``, and ``loss_weights.sentence``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        lw = cfg.loss_weights
        self.w_phoneme: float = float(lw.phoneme)
        self.w_word: float = float(lw.word)
        self.w_sentence: float = float(lw.sentence)
        self._mse = nn.MSELoss(reduction="mean")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_phoneme: torch.Tensor,
        pred_word: torch.Tensor,
        pred_sentence: torch.Tensor,
        target_phoneme: torch.Tensor,
        target_word: torch.Tensor,
        target_sentence: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total weighted loss and individual component losses.

        Args:
            pred_phoneme:   (N_phonemes,)  — predicted phoneme scores.
            pred_word:      (N_words,)     — predicted word scores.
            pred_sentence:  (N_sentences,) — predicted sentence scores.
            target_phoneme:   same shape as pred_phoneme.
            target_word:      same shape as pred_word.
            target_sentence:  same shape as pred_sentence.

        Returns:
            total_loss:  scalar tensor used for ``loss.backward()``.
            components:  dict with float values for MLflow logging:
                         ``{"loss_phoneme": …, "loss_word": …,
                            "loss_sentence": …, "loss_total": …}``.
        """
        loss_p = self._mse(pred_phoneme, target_phoneme)
        loss_w = self._mse(pred_word, target_word)
        loss_s = self._mse(pred_sentence, target_sentence)

        total = (
            self.w_phoneme * loss_p
            + self.w_word * loss_w
            + self.w_sentence * loss_s
        )

        components: dict[str, float] = {
            "loss_phoneme": loss_p.item(),
            "loss_word": loss_w.item(),
            "loss_sentence": loss_s.item(),
            "loss_total": total.item(),
        }

        return total, components

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def weights_dict(self) -> dict[str, float]:
        """Return loss weights for MLflow ``log_params``."""
        return {
            "loss_weight_phoneme": self.w_phoneme,
            "loss_weight_word": self.w_word,
            "loss_weight_sentence": self.w_sentence,
        }
