import hydra
import torch
import glob
import os
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from omegaconf import DictConfig

from src.training.trainer import _build_dataloaders
from src.models.bigru import HierarchicalBiGRU
from src.evaluation.score_comparison import compare_score_predictions, summarise_by_level, format_report

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="configs", config_name="experiment/default")
def main(cfg: DictConfig):
    log.info("Starting score comparison evaluation...")
    
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Get val loader (we use val loader to evaluate if no dedicated test loader)
    batch_size = int(cfg.model.batch_size)
    _, val_loader = _build_dataloaders(cfg, batch_size)
    
    # Infer feature dim
    sample_batch = next(iter(val_loader))
    n_lld = sample_batch["phoneme_features"].shape[-1]
    
    # 3. Instantiate model
    model = HierarchicalBiGRU(cfg, input_size=n_lld).to(device)
    
    # 4. Find the latest model checkpoint
    models_dir = Path("models")
    if not models_dir.exists():
        log.warning("Models directory not found, using untrained model")
    else:
        # Find latest .pth file
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            log.info(f"Loading model weights from {latest_model}")
            model.load_state_dict(torch.load(latest_model, map_location=device))
        else:
            log.warning("No .pth files found in models/, using untrained model")
            
    # 5. Run inference
    model.eval()
    all_targets = defaultdict(list)
    all_preds = defaultdict(list)
    
    log.info("Running inference...")
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                features = batch["phoneme_features"].to(device)
                mask = batch["phoneme_mask"].to(device)
                word_bounds = batch["word_boundaries"]
                targets = {k: v.to(device) for k, v in batch["targets"].items()}
            else:
                features, mask, word_bounds, targets = batch
                features = features.to(device)
                mask = mask.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
            preds = model(features, mask, word_bounds)
            
            for k in targets.keys():
                all_targets[k].extend(targets[k].cpu().numpy())
                all_preds[k].extend(preds[k].cpu().numpy())
                
    # Convert lists to numpy arrays
    final_targets = {k: np.array(v) for k, v in all_targets.items()}
    final_preds = {k: np.array(v) for k, v in all_preds.items()}
    
    # 6. Generate report
    log.info("Computing metrics...")
    df = compare_score_predictions(final_targets, final_preds, str(cfg.score_mode), cfg)
    
    print("\n--- Cross-Metric Prediction Accuracy Report ---")
    print(format_report(df))
    print("\n--- Summary By Level ---")
    print(summarise_by_level(df).to_string(index=False))

if __name__ == "__main__":
    main()
