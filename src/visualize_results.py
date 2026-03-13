import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from dataloaders import OCT_SKIN
from functions import dice_coefficient

def visualize_predictions(model_path, data_dir, device='cuda', num_samples=5):
    # --- 1. SET UP PATHS ---
    images_dir = Path(data_dir) / 'train' / 'images' # Check train if test is empty or vice versa
    masks_dir = Path(data_dir) / 'train' / 'mask'
    
    # Check if test exist, override if it does
    if (Path(data_dir) / 'test' / 'images').exists():
        images_dir = Path(data_dir) / 'test' / 'images'
        masks_dir = Path(data_dir) / 'test' / 'mask'
    
    print(f"Loading images from: {images_dir}")
    print(f"Loading masks from: {masks_dir}")

    # --- 2. LOAD MODEL ---
    print(f"Loading model from {model_path}...")
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None, # Loading local weights
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- 3. PREPARE DATASET ---
    dataset = OCT_SKIN(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
        augment=False
    )
    
    if len(dataset) == 0:
        print("No test images found!")
        return

    # --- 4. RUN PREDICTIONS & PLOT ---
    print(f"Calculating metrics and visualizing {min(num_samples, len(dataset))} samples...")
    
    # Calculate DICE for ALL items in dataset
    all_dices = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, mask = dataset[i]
            image_in = image.unsqueeze(0).to(device)
            output = model(image_in)
            dice = dice_coefficient(output, mask.unsqueeze(0).to(device))
            all_dices.append(dice)
            
    mean_dice = np.mean(all_dices)
    std_dice = np.std(all_dices)
    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Total Samples: {len(dataset)}")
    print(f"Mean DICE: {mean_dice:.4f}")
    print(f"Std DICE:  {std_dice:.4f}")
    print(f"---------------------------\n")

    # Pick samples for visualization
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_path = dataset.data_pairs[idx]['image'].name
            image_in = image.unsqueeze(0).to(device)
            
            # Forward pass
            output = model(image_in)
            dice = dice_coefficient(output, mask.unsqueeze(0).to(device))
            
            # Post-processing
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # Convert tensors back to viewable images
            # Need to undo the normalization for visualization: img = img * 0.5 + 0.5
            display_img = (image.squeeze().cpu().numpy() * 0.5 + 0.5) * 255.0
            display_img = display_img.astype(np.uint8)
            
            display_mask = mask.squeeze().cpu().numpy()
            
            # Plotting
            # Handle cases where num_samples=1 (axes is not 2D)
            curr_axes = axes[i] if num_samples > 1 else axes
            
            # Subplot 1: Input Image
            curr_axes[0].imshow(display_img, cmap='gray')
            curr_axes[0].set_title(f"Sample {idx}\n{image_path}")
            curr_axes[0].axis('off')
            
            # Subplot 2: Overlay Ground Truth
            curr_axes[1].imshow(display_img, cmap='gray')
            curr_axes[1].imshow(display_mask, cmap='jet', alpha=0.4) # Overlay with transparency
            curr_axes[1].set_title(f"Ground Truth Overlay")
            curr_axes[1].axis('off')
            
            # Subplot 3: Overlay Prediction
            curr_axes[2].imshow(display_img, cmap='gray')
            curr_axes[2].imshow(pred_binary, cmap='jet', alpha=0.4) # Overlay with transparency
            curr_axes[2].set_title(f"Prediction Overlay\nDice: {dice:.4f}")
            curr_axes[2].axis('off')

    plt.tight_layout()
    output_png = 'test_predictions.png'
    plt.savefig(output_png)
    print(f"Saved visualization to {output_png}")

if __name__ == "__main__":
    # Settings
    PROJECT_ROOT = "/zhome/29/b/146867/Article_OCT/Article_OCT_Diabetic_neuropathy"
    MODEL_PATH = f"{PROJECT_ROOT}/models/best_unet_model.pth"
    DATA_PATH = f"{PROJECT_ROOT}/data"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visualize_predictions(MODEL_PATH, DATA_PATH, device=device)
