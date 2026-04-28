"""
Primary Model — Hierarchical Multi-Task Bi-directional GRU.

Architecture (per ``APA_Antigravity.md`` and implementation plan):

  Input  →  Bi-GRU Encoder  →  Dropout
                │
                ├── Phoneme heads  (MLP on raw hidden states per phoneme)
                │   e.g. phoneme_accuracy → sigmoid × 2.0
                │
                ├─→ Word-level Attention Pooling
                │   └── Word heads (MLP per metric)
                │       e.g. word_accuracy → sigmoid × 10.0
                │           word_stress   → sigmoid × 10.0  (all_metrics only)
                │
                └─→ Sentence-level Attention Pooling
                    └── Sentence heads (MLP per metric)
                        e.g. sentence_accuracy      → sigmoid × 10.0
                             sentence_completeness  → sigmoid × 1.0   (all_metrics)
                             sentence_fluency       → sigmoid × 10.0  (all_metrics)
                             sentence_prosodic      → sigmoid × 10.0  (all_metrics)

Score ranges come from ``configs/base.yaml`` under ``metrics.<score_mode>``
so adding a new axis only requires a YAML edit and a re-run.

Metric-level routing — each canonical key is prefixed with its level:
  * ``phoneme_*``  →  per-phoneme hidden state (no pooling)
  * ``word_*``     →  word-level attention pooling
  * ``sentence_*`` →  sentence-level attention pooling

Attention pooling:
  Two independent learned attention heads — one for word-level aggregation
  (pools within word boundaries) and one for sentence-level aggregation
  (pools over the whole sequence).  Separate heads prevent the two
  objectives from interfering.

Regularisation:
  * Dropout applied after the Bi-GRU encoder.
  * L2 weight decay is passed to the optimiser by the trainer, NOT here.

Global random seeds are set via the trainer before model instantiation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — metric routing
# ---------------------------------------------------------------------------

def _level_of(metric_key: str) -> str:
    """Return the hierarchical level for a canonical metric key.

    Convention: canonical keys are prefixed with their level followed by '_'.
    Examples:
        ``phoneme_accuracy`` → ``"phoneme"``
        ``word_stress``      → ``"word"``
        ``sentence_fluency`` → ``"sentence"``
    """
    return metric_key.split("_")[0]


# ---------------------------------------------------------------------------
# Attention pooling head
# ---------------------------------------------------------------------------


class AttentionPooling(nn.Module):
    """Learned scalar-attention pooler over a variable-length sequence.

    Given hidden states ``H`` of shape ``(B, T, D)``, computes a
    context vector as a weighted sum:

        e_t = tanh(H_t W_e + b_e)
        a_t = softmax(e_t w_a)
        c   = Σ a_t H_t

    This is a lightweight, single-head additive attention mechanism.

    Args:
        hidden_size: Dimensionality of the input vectors.  For a
                     Bi-GRU with ``hidden_size`` per direction, the
                     concatenated output is ``2 * hidden_size``.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: ``(B, T, D)`` hidden states to pool.
            mask:   ``(B, T)`` boolean mask; True → valid position.
                    If ``None``, all positions are considered valid.

        Returns:
            context:  ``(B, D)`` pooled representation.
            weights:  ``(B, T)`` attention weights (for visualisation).
        """
        # (B, T, D) → (B, T, D)
        energy = torch.tanh(self.projection(hidden))
        # (B, T, 1) → (B, T)
        scores = self.context_vector(energy).squeeze(-1)

        if mask is not None:
            # Fill padding positions with -inf so softmax → 0
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, T)
        # Weighted sum: (B, T, 1) * (B, T, D) → (B, D)
        context = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        return context, weights


# ---------------------------------------------------------------------------
# Prediction head (shared MLP structure)
# ---------------------------------------------------------------------------


class PredictionHead(nn.Module):
    """Two-layer MLP prediction head with sigmoid × max_score output.

    Args:
        input_size: Dimensionality of the incoming representation.
        hidden_size: Width of the hidden layer.
        max_score:   Upper bound for the sigmoid output scaling.
        dropout:     Dropout probability applied before the final layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        max_score: float,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_score = max_score
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  ``(N, input_size)``

        Returns:
            scores: ``(N,)`` in range ``[0, max_score]``.
        """
        raw = self.net(x).squeeze(-1)  # (N,)
        return torch.sigmoid(raw) * self.max_score


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class HierarchicalBiGRU(nn.Module):
    """Hierarchical Multi-Task Bi-directional GRU for pronunciation assessment.

    The model receives a *padded* sequence of eGeMAPS feature vectors per
    utterance and predicts scores at three levels simultaneously.

    Which metrics are predicted is controlled by ``cfg.score_mode``:

    * ``"major_scores"`` (default) — one head per level (phoneme accuracy,
      word accuracy, sentence accuracy).
    * ``"all_metrics"`` — adds word stress, sentence completeness,
      sentence fluency, and sentence prosodic heads.

    The full metric-to-max-score mapping lives in
    ``cfg.metrics.<score_mode>`` (``configs/base.yaml``); adding a new
    metric requires only a YAML change.

    Args:
        cfg: Hydra DictConfig.  Expected keys (from ``configs/model/bigru.yaml``
             merged on top of ``configs/base.yaml``):

            - ``model.hidden_size``      (int,   default 128)
            - ``model.num_layers``       (int,   default 2)
            - ``model.dropout``          (float, default 0.2)
            - ``model.l2_weight_decay``  — used by trainer, not here.
            - ``model.learning_rate``    — used by trainer, not here.
            - ``score_mode``             (str, ``"major_scores"`` or
                                          ``"all_metrics"``)
            - ``metrics.<score_mode>``   mapping of metric key → max score.

        input_size:  Number of acoustic features per phoneme (88 for eGeMAPS).
    """

    def __init__(self, cfg: DictConfig, input_size: int = 88) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size

        hidden_size: int = int(cfg.model.hidden_size)
        num_layers: int = int(cfg.model.num_layers)
        dropout: float = float(cfg.model.dropout)
        self.score_mode: str = str(cfg.score_mode)

        # Resolve the active metric → max_score mapping from config
        self.metric_max_scores: dict[str, float] = {
            k: float(v)
            for k, v in OmegaConf.to_container(  # type: ignore[call-overload]
                cfg.metrics[self.score_mode], resolve=True
            ).items()
        }

        # The Bi-GRU doubles the output dimensionality
        bigru_out_size: int = 2 * hidden_size

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_dropout = nn.Dropout(p=dropout)

        # ------------------------------------------------------------------
        # Attention poolers (shared across all word / sentence heads)
        # ------------------------------------------------------------------
        self.word_attention = AttentionPooling(hidden_size=bigru_out_size)
        self.sentence_attention = AttentionPooling(hidden_size=bigru_out_size)

        # ------------------------------------------------------------------
        # Prediction heads — one ModuleDict per level, keyed by metric name
        # ------------------------------------------------------------------
        phoneme_heads: dict[str, nn.Module] = {}
        word_heads: dict[str, nn.Module] = {}
        sentence_heads: dict[str, nn.Module] = {}

        for metric, max_score in self.metric_max_scores.items():
            level = _level_of(metric)
            head = PredictionHead(
                input_size=bigru_out_size,
                hidden_size=hidden_size,
                max_score=max_score,
                dropout=dropout,
            )
            if level == "phoneme":
                phoneme_heads[metric] = head
            elif level == "word":
                word_heads[metric] = head
            elif level == "sentence":
                sentence_heads[metric] = head
            else:
                raise ValueError(
                    f"Unknown level '{level}' for metric '{metric}'. "
                    "Canonical metric keys must start with 'phoneme_', "
                    "'word_', or 'sentence_'."
                )

        self.phoneme_heads = nn.ModuleDict(phoneme_heads)
        self.word_heads = nn.ModuleDict(word_heads)
        self.sentence_heads = nn.ModuleDict(sentence_heads)

        self._init_weights()

        logger.info(
            "HierarchicalBiGRU initialised — score_mode=%s, hidden=%d, "
            "layers=%d, dropout=%.2f, metrics=%s, params=%s",
            self.score_mode,
            hidden_size,
            num_layers,
            dropout,
            list(self.metric_max_scores.keys()),
            f"{sum(p.numel() for p in self.parameters()):,}",
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier/orthogonal initialisation for GRU weights; zeros for biases."""
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        all_heads = (
            list(self.phoneme_heads.values())
            + list(self.word_heads.values())
            + list(self.sentence_heads.values())
        )
        for head in all_heads:
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        phoneme_features: torch.Tensor,
        phoneme_mask: torch.Tensor,
        word_boundaries: list[list[tuple[int, int]]],
    ) -> dict[str, torch.Tensor]:
        """Run the hierarchical forward pass.

        Args:
            phoneme_features:
                ``(B, T_max, 88)`` padded eGeMAPS feature sequences.
            phoneme_mask:
                ``(B, T_max)`` boolean — True for valid phoneme positions.
            word_boundaries:
                For each sentence in the batch, a list of ``(start, end)``
                index tuples (exclusive end) indicating which phoneme indices
                belong to each word.  Length of the outer list equals B.

        Returns:
            A flat dict keyed by canonical metric name.  Values are tensors:

            * phoneme metrics → ``(N_valid_phonemes,)``
            * word metrics    → ``(N_words_in_batch,)``
            * sentence metrics→ ``(B,)``

            Additional keys for visualisation:
            * ``"word_attn_weights"``     — ``list[Tensor]``
            * ``"sentence_attn_weights"`` — ``(B, T_max)``
        """
        # ---- 1. Encode ---------------------------------------------------
        lengths = phoneme_mask.sum(dim=1).cpu()  # (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            phoneme_features, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (B, T_max, 2*H)
        hidden = self.encoder_dropout(hidden)

        outputs: dict[str, torch.Tensor] = {}

        # ---- 2. Phoneme-level predictions --------------------------------
        if self.phoneme_heads:
            valid_hidden = hidden[phoneme_mask]  # (N_valid, 2*H)
            for metric, head in self.phoneme_heads.items():
                outputs[metric] = head(valid_hidden)  # (N_valid,)

        # ---- 3. Word-level predictions -----------------------------------
        if self.word_heads:
            # Collect per-word context vectors across the whole batch
            word_contexts: list[torch.Tensor] = []
            all_word_attn: list[torch.Tensor] = []
            for b_idx, word_bounds in enumerate(word_boundaries):
                for (w_start, w_end) in word_bounds:
                    word_hidden = hidden[b_idx, w_start:w_end, :]  # (W_len, 2*H)
                    word_hidden = word_hidden.unsqueeze(0)           # (1, W_len, 2*H)
                    ctx, attn = self.word_attention(word_hidden)     # (1, 2*H)
                    word_contexts.append(ctx)
                    all_word_attn.append(attn.squeeze(0))

            batched_word_ctx = torch.cat(word_contexts, dim=0)  # (N_words, 2*H)
            for metric, head in self.word_heads.items():
                outputs[metric] = head(batched_word_ctx)         # (N_words,)
            outputs["word_attn_weights"] = all_word_attn  # type: ignore[assignment]

        # ---- 4. Sentence-level predictions -------------------------------
        if self.sentence_heads:
            sent_ctx, sent_attn = self.sentence_attention(
                hidden, mask=phoneme_mask
            )  # (B, 2*H), (B, T_max)
            for metric, head in self.sentence_heads.items():
                outputs[metric] = head(sent_ctx)             # (B,)
            outputs["sentence_attn_weights"] = sent_attn

        return outputs

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def params_dict(self) -> dict:
        """Return a flat dict of hyperparameters for ``mlflow.log_params``."""
        base = {
            "model_name": "hierarchical_bigru",
            "model_input_size": self.input_size,
            "model_hidden_size": int(self.cfg.model.hidden_size),
            "model_num_layers": int(self.cfg.model.num_layers),
            "model_dropout": float(self.cfg.model.dropout),
            "model_l2_weight_decay": float(self.cfg.model.l2_weight_decay),
            "model_learning_rate": float(self.cfg.model.learning_rate),
            "model_batch_size": int(self.cfg.model.batch_size),
            "model_epochs": int(self.cfg.model.epochs),
            "score_mode": self.score_mode,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        # Log each metric's max score for full reproducibility
        for metric, max_score in self.metric_max_scores.items():
            base[f"max_score_{metric}"] = max_score

        # Scheduler hyperparameters
        sched_cfg = self.cfg.model.get("scheduler", None)
        if sched_cfg is not None:
            base["scheduler_factor"] = float(sched_cfg.factor)
            base["scheduler_patience"] = int(sched_cfg.patience)
            base["scheduler_min_lr"] = float(sched_cfg.min_lr)

        # Early stopping hyperparameters
        es_cfg = self.cfg.model.get("early_stopping", None)
        if es_cfg is not None:
            base["early_stopping_patience"] = int(es_cfg.patience)
            base["early_stopping_min_delta"] = float(es_cfg.min_delta)

        return base
