import logging

log = logging.getLogger(__name__)

import time
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from dataloaders import OCT_SKIN, frames_from_directories, load_manifest_frames
from functions import CombinedLoss, train_one_epoch, validate_one_epoch
from model_utils import build_unet
from splits import assert_no_patient_overlap, describe_split, split_frames


def load_frames(cfg: DictConfig, orig_cwd: Path):
    """Training frames, from the manifest when available."""
    data_cfg = cfg.data
    if data_cfg.source == "manifest":
        manifest_dir = orig_cwd / data_cfg.manifest_dir
        frames = load_manifest_frames(manifest_dir)
        log.info("Loaded %d annotated frames from manifest %s", len(frames), manifest_dir)
        return frames

    frames = frames_from_directories(
        orig_cwd / data_cfg.train_images, orig_cwd / data_cfg.train_masks
    )
    log.warning(
        "Using directory-based frames (%d found). These carry no patient identity, so the "
        "train/val split cannot be patient-grouped.",
        len(frames),
    )
    return frames


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    log.info(f"Configuration:\n{cfg}")
    hp = cfg.hyperparameters
    target_size = tuple(cfg.data.target_size)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {DEVICE}")

    orig_cwd = Path(hydra.utils.get_original_cwd())
    # Each run writes into its own timestamped Hydra directory. Writing to a
    # shared models/ path would silently destroy the previous run's checkpoint
    # -- Hydra does not change the working directory under version_base 1.2, so
    # Path.cwd() here is the repo root, not a per-run directory.
    OUTPUT_DIR = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Saving outputs to: {OUTPUT_DIR}")

    # --- DATA ---
    frames = load_frames(cfg, orig_cwd)
    if len(frames) == 0:
        log.error("No annotated frames found. Nothing to train on.")
        return

    train_indices, val_indices = split_frames(frames, hp.val_split_ratio, hp.seed)
    assert_no_patient_overlap(frames, train_indices, val_indices)
    log.info("Split: %s", describe_split(frames, train_indices, val_indices))

    if not val_indices:
        log.warning("Validation set is empty; best-checkpoint selection is disabled.")

    train_dataset = OCT_SKIN(frames, target_size=target_size, augment=True)
    val_dataset = OCT_SKIN(frames, target_size=target_size, augment=False)

    train_dataloader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=hp.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=True,
    )

    # --- MODEL ---
    POS_WEIGHT = torch.tensor([float(hp.pos_weight)]).to(DEVICE)
    model = build_unet(encoder_weights="imagenet").to(DEVICE)

    CRITERION = CombinedLoss(alpha=hp.dice_alpha, pos_weight=POS_WEIGHT)
    OPTIMIZER = optim.AdamW(model.parameters(), lr=hp.learning_rate)
    SCHEDULER = CosineAnnealingLR(optimizer=OPTIMIZER, T_max=hp.num_epochs, eta_min=1e-7)

    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH = OUTPUT_DIR / "models" / "best_unet_model.pth"
    HISTORY_PATH = OUTPUT_DIR / "models" / "training_history.pt"

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "epoch": [], "test_dice": None}
    best_val_dice = 0.0

    log.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, hp.num_epochs + 1):
        train_loss = train_one_epoch(train_dataloader, model, CRITERION, OPTIMIZER, DEVICE)

        if val_indices:
            val_loss, val_dice = validate_one_epoch(val_dataloader, model, CRITERION, DEVICE)
        else:
            val_loss, val_dice = float("nan"), 0.0

        log.info(
            f"Epoch {epoch}/{hp.num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val DICE: {val_dice:.4f}"
        )
        SCHEDULER.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["epoch"].append(epoch)

        if val_dice > best_val_dice:
            log.info(f"   --> New best model! DICE {best_val_dice:.4f} -> {val_dice:.4f}. Saving.")
            best_val_dice = val_dice
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            torch.save(history, HISTORY_PATH)

    if not BEST_MODEL_PATH.exists():
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        torch.save(history, HISTORY_PATH)

    # --- TEST EVALUATION ---
    test_frames = frames_from_directories(
        orig_cwd / cfg.data.test_images, orig_cwd / cfg.data.test_masks
    )
    if len(test_frames) > 0:
        log.info(f"Found {len(test_frames)} test frames. Loading best model for evaluation...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval()

        test_dataset = OCT_SKIN(test_frames, target_size=target_size, augment=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
        )
        _, test_dice = validate_one_epoch(test_dataloader, model, CRITERION, DEVICE)
        log.info(f"Test Set Evaluation -> DICE: {test_dice:.4f}")

        history["test_dice"] = test_dice
        torch.save(history, HISTORY_PATH)
    else:
        log.warning(
            "No test frames found at %s. Skipping test evaluation.",
            orig_cwd / cfg.data.test_images,
        )

    log.info(f"Training finished. Model saved to {BEST_MODEL_PATH}")
    log.info(
        f"Training Complete in {time.time() - start_time:.2f} seconds. Best DICE: {best_val_dice:.4f}"
    )

    promote_path = orig_cwd / "models" / "best_unet_model.pth"
    if cfg.data.get("promote_checkpoint", False):
        if promote_path.exists() and not cfg.data.get("overwrite_promoted_checkpoint", False):
            log.warning(
                "Not promoting: %s already exists. Re-run with "
                "data.overwrite_promoted_checkpoint=true to replace it.",
                promote_path,
            )
        else:
            promote_path.parent.mkdir(parents=True, exist_ok=True)
            promote_path.write_bytes(BEST_MODEL_PATH.read_bytes())
            log.info("Promoted checkpoint to %s", promote_path)
    else:
        log.info(
            "Checkpoint left in the run directory. To make it the project model, copy it to "
            "%s or re-run with data.promote_checkpoint=true",
            promote_path,
        )


if __name__ == "__main__":
    main()
