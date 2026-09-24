import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import cv2
from pathlib import Path

from image_utils import letterbox_resize, letterbox_unresize
from model_utils import (
    DEFAULT_TARGET_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    default_model_path,
    load_pytorch_model,
    project_root,
)

def import_dicom_volume(path):
    """Loads a DICOM volume into a numpy array."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    dicom_image = reader.Execute()
    return sitk.GetArrayFromImage(dicom_image)

def predict_mask(model, image, device, target_size=DEFAULT_TARGET_SIZE):
    """Runs inference with proper normalization mapping to [-1, 1]."""
    
    # Scale to [0, 1]
    max_val = 65535.0 if image.dtype == np.uint16 else 255.0
    img_norm = image.astype(np.float32) / max_val
    
    # Extract structural channel if RGB
    if len(img_norm.shape) == 3:
        img_for_model = img_norm[:, :, 0]
    else:
        img_for_model = img_norm

    # Resize and scale to [-1, 1] for model
    img_resized, letterbox = letterbox_resize(
        img_for_model, target_size, interpolation=cv2.INTER_LINEAR
    )
    img_resized = (img_resized - NORMALIZE_MEAN) / NORMALIZE_STD
    
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.sigmoid(pred_logits)
        pred_mask = (pred_mask > 0.5).float()
    
    # Resize output mask back to native size
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    resized_mask = letterbox_unresize(pred_mask_np, letterbox, interpolation=cv2.INTER_NEAREST)
    
    return img_for_model, resized_mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Project paths
    PROJECT_ROOT = project_root()
    MODEL_PATH = default_model_path()
    DICOM_PATH = PROJECT_ROOT / "data" / "dicom" / "JJ_Right Hallux_L1030_S2678__24_05_2021.dcm" 
    
    print("\nLoading Model...")
    model = load_pytorch_model(MODEL_PATH, device)
    
    print(f"Loading DICOM volume from {DICOM_PATH.name}...")
    dicom_data = import_dicom_volume(DICOM_PATH)
    
    num_slices_total = dicom_data.shape[0]
    # Pick 5 equidistant slices ranging from start to finish
    test_slices = np.linspace(0, num_slices_total - 1, 5, dtype=int)
    
    # Setup our layout: 5 rows (slices) x 3 columns (Original, Mask, Overlay)
    fig, axes = plt.subplots(len(test_slices), 3, figsize=(18, 4 * len(test_slices)))
    
    print("Running inference and generating visual slices...")
    for i, slice_idx in enumerate(test_slices):
        print(f" -> Processing Slice {slice_idx}/{num_slices_total}")
        
        slice_data = dicom_data[slice_idx]
        img_structural, pred_mask = predict_mask(model, slice_data, device)
        
        # Build Red-Overlay Image
        overlay = np.stack([img_structural]*3, axis=-1)
        overlay[pred_mask > 0] = [1, 0, 0] # Solid red where mask == 1
        
        # Original Image
        axes[i, 0].imshow(img_structural, cmap='gray')
        axes[i, 0].set_title(f"Slice {slice_idx}: Structural OCT")
        axes[i, 0].axis('off')
        
        # Standalone Mask
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title(f"Slice {slice_idx}: Raw Prediction")
        axes[i, 1].axis('off')
        
        # Mask overlaid on original
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Slice {slice_idx}: Mask Overlay")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    output_image = PROJECT_ROOT / "dicom_segmentation_verification.png"
    plt.savefig(output_image)
    plt.close()
    
    print(f"\nSuccess! Verification plots strictly saved to: {output_image.name}")