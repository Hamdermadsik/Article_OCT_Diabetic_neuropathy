import logging
log = logging.getLogger(__name__)

import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp

from dataloaders import OCT_SKIN
from functions import train_one_epoch, validate_one_epoch, CombinedLoss

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Log configuration (Hydra allows overriding these via CLI, e.g., hyperparameters.learning_rate=0.01)
    log.info(f"Configuration:\n{cfg}")
    
    # Extract params from config
    hp = cfg.hyperparameters
    
    # --- 1. DEVICE ALLOCATION ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {DEVICE}")

    # --- 2. PATHS ---
    # Hydra changes the working directory to the output directory of the run.
    # We use hydra.utils.get_original_cwd() to find the data.
    orig_cwd = Path(hydra.utils.get_original_cwd())
    DATA_DIR = orig_cwd / 'data' / 'train'
    images_path = DATA_DIR / 'images'
    masks_path = DATA_DIR / 'mask'
    
    OUTPUT_DIR = Path.cwd() # Current working dir is the hydra output dir
    log.info(f"Saving outputs to: {OUTPUT_DIR}")

    # --- 3. DATA LOADERS ---
    TARGET_SIZE = (512, 512)
    
    # Create two separate dataset instances to control augmentation independently
    train_base_dataset = OCT_SKIN(
        images_dir=images_path, 
        masks_dir=masks_path,
        target_size=TARGET_SIZE,
        augment=True
    )

    val_base_dataset = OCT_SKIN(
        images_dir=images_path, 
        masks_dir=masks_path,
        target_size=TARGET_SIZE,
        augment=False
    )

    # Calculate Split Sizes
    total_samples = len(train_base_dataset)
    if total_samples == 0:
        log.error(f"No valid image/mask pairs found in {images_path}! Check file extensions (.jpeg) and naming.")
        return

    val_samples = int(hp.val_split_ratio * total_samples)
    train_samples = total_samples - val_samples
    log.info(f"Data mapping: total={total_samples} | Train={train_samples} | Validation={val_samples}")

    # Create fixed internal indices and shuffle once
    torch.manual_seed(hp.seed)
    shuffled_indices = torch.randperm(total_samples).tolist()
    
    train_indices = shuffled_indices[val_samples:]
    val_indices = shuffled_indices[:val_samples]

    train_subset = Subset(train_base_dataset, train_indices)
    val_subset = Subset(val_base_dataset, val_indices)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=hp.num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=True
    )

    # --- 4. MODEL SETUP ---
    POS_WEIGHT = torch.tensor([9.0]).to(DEVICE) 
    
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    ).to(DEVICE)

    CRITERION = CombinedLoss(alpha=hp.dice_alpha, pos_weight=POS_WEIGHT)
    OPTIMIZER = optim.AdamW(model.parameters(), lr=hp.learning_rate)

    SCHEDULER = CosineAnnealingLR(
        optimizer=OPTIMIZER,
        T_max=hp.num_epochs,
        eta_min=1e-7
    )

    # --- 5. TRAINING LOOP ---
    # Ensure models directory exists
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)

    BEST_MODEL_PATH = OUTPUT_DIR / "models" / "best_unet_model.pth"
    HISTORY_PATH = OUTPUT_DIR / "models" / "training_history.pt"

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'epoch': [], 'test_dice': None}
    best_val_dice = 0.0

    log.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, hp.num_epochs + 1):
        # 1. Training Phase
        train_loss = train_one_epoch(train_dataloader, model, CRITERION, OPTIMIZER, DEVICE)
        
        # 2. Validation Phase
        val_loss, val_dice = validate_one_epoch(val_dataloader, model, CRITERION, DEVICE)

        # 3. Epoch Summary
        log.info(f"Epoch {epoch}/{hp.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val DICE: {val_dice:.4f}")

        SCHEDULER.step()

        # 4. Save History (keep only val_dice for plotting if no test set)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['epoch'].append(epoch)

        # 5. Checkpoint
        if val_dice > best_val_dice:
            log.info(f"   --> New best model! DICE improved {best_val_dice:.4f} -> {val_dice:.4f}. Saving.")
            best_val_dice = val_dice
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            # We defer saving history until end or here, usually we want to save history too
            torch.save(history, HISTORY_PATH)

    # --- 6. TEST EVALUATION (After training completes) ---
    TEST_IMAGES_DIR = orig_cwd / 'data' / 'test' / 'images'
    TEST_MASKS_DIR = orig_cwd / 'data' / 'test' / 'mask'
    
    # Check if directory exists and has files
    test_files = list(TEST_IMAGES_DIR.glob("*.jpeg")) if TEST_IMAGES_DIR.exists() else []
    
    if TEST_IMAGES_DIR.exists() and TEST_MASKS_DIR.exists() and len(test_files) > 0:
        log.info(f"Found {len(test_files)} images at {TEST_IMAGES_DIR}. Loading best model for evaluation...")
        
        # Load Best Model
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval()
        
        # Create Test Loader (No Augmentation, No Noise)
        test_dataset = OCT_SKIN(
            images_dir=TEST_IMAGES_DIR, 
            masks_dir=TEST_MASKS_DIR,
            target_size=TARGET_SIZE,
            augment=False,
            n_samples=10000, # Load all test images
            noise_sigma=0 # Test data is clean? Or should we add noise? Usually clean.
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers
        )
        
        # Run Evaluation
        _, test_dice = validate_one_epoch(test_dataloader, model, CRITERION, DEVICE)
        log.info(f"Test Set Evaluation -> DICE: {test_dice:.4f}")
        
        # Update and Save History
        history['test_dice'] = test_dice
        torch.save(history, HISTORY_PATH)
    else:
        log.warning(f"Test data not found at {TEST_IMAGES_DIR}. Skipping test evaluation.")

    log.info(f"Training finished. Model saved to {BEST_MODEL_PATH}")
    total_duration = time.time() - start_time
    log.info(f"Training Complete in {total_duration:.2f} seconds. Best DICE: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()
