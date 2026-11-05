import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
from skimage.transform import resize
import cv2
from skimage import measure, morphology
import math
from pathlib import Path

from image_utils import resize_for_model, restore_from_model, structural_image
from model_utils import (
    DEFAULT_TARGET_SIZE,
    LEGACY_PREPROCESSING,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    default_model_path,
    load_pytorch_model,
    project_root,
)

# --- DICOM Import ---

def import_dicom(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    dicom_image = reader.Execute()
    
    image = sitk.GetArrayFromImage(dicom_image)
    spacing = dicom_image.GetSpacing()
    metadata = {key: dicom_image.GetMetaData(key) for key in dicom_image.GetMetaDataKeys()}
    
    return image, spacing, metadata 

# --- Processing Functions ---

def test_model_on_sample_image(model, image, device, target_size=None, plot=False):
    
    max_val = 65535.0 if image.dtype == np.uint16 else 255.0
    img_norm = image.astype(np.float32) / max_val
    
    img_for_model = structural_image(img_norm)

    # Prepare input exactly as this model's training data was prepared.
    mode = getattr(model, "preprocessing", LEGACY_PREPROCESSING)
    target_size = target_size or getattr(model, "target_size", DEFAULT_TARGET_SIZE)
    img_resized, resize_params = resize_for_model(
        img_for_model, target_size, mode, interpolation=cv2.INTER_LINEAR
    )
    img_resized = (img_resized - NORMALIZE_MEAN) / NORMALIZE_STD
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.sigmoid(pred_logits)
        pred_mask = (pred_mask > 0.5).float()
    
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    resized_mask = restore_from_model(pred_mask_np, resize_params, interpolation=cv2.INTER_NEAREST)

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.imshow(img_for_model, cmap='gray'); plt.title('Structural OCT'); plt.axis('off')
        overlay = np.stack([img_for_model]*3, axis=-1)
        overlay[resized_mask > 0] = [1, 0, 0]
        plt.subplot(1, 2, 2); plt.imshow(overlay); plt.title('Mask Overlay'); plt.axis('off')
        plt.savefig("segmentation_test.png")
        plt.close()

    return resized_mask

def remove_epidermis_from_image(image, p_mask, plot=False):
    upper_line = np.argmax(p_mask, axis=0)
    lower_line = p_mask.shape[0] - np.argmax(np.flipud(p_mask), axis=0) - 1
    
    mask_exists = np.any(p_mask, axis=0)
    upper_line[~mask_exists] = 0
    lower_line[~mask_exists] = 0

    cols = np.arange(p_mask.shape[1])
    upper_coords = np.column_stack((upper_line, cols))
    lower_coords = np.column_stack((lower_line, cols))

    img_modified = image.copy()
    for col in range(image.shape[1]):
        if mask_exists[col]:
            img_modified[:int(lower_line[col]), col] = 0 

    if plot:
        plt.figure(figsize=(12, 5))
        plt.imshow(image, cmap='gray')
        plt.plot(cols, upper_line, 'r', label='Surface')
        plt.plot(cols, lower_line, 'b', label='DEJ')
        plt.legend()
        plt.savefig("epidermis_removal_lines.png")
        plt.close()

    return img_modified, upper_coords, lower_coords

def shift_line_coordinates(line_coords, shift_amount, img_height):
    new_coords = line_coords.copy()
    new_coords[:, 0] = np.clip(new_coords[:, 0] + shift_amount, 0, img_height - 1)
    return new_coords

def create_mask_between_lines(img_height, img_width, upper_coords, lower_coords):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for col in range(img_width):
        y1, y2 = int(upper_coords[col, 0]), int(lower_coords[col, 0])
        if y1 < y2: mask[y1:y2+1, col] = 1
    return mask

def calculate_tortuosity(coords):
    coords = np.asarray(coords)
    if len(coords) < 2: return 0
    diffs = np.diff(coords, axis=0)
    L = np.sum(np.linalg.norm(diffs, axis=1))
    D = np.linalg.norm(coords[-1] - coords[0])
    return L / D if D != 0 else 1.0

def identify_and_plot_dots(
    image,
    min_size=50,
    intensity_threshold=50,
    plot=True,
    circularity_threshold=0.7,
    out_path="detected_capillary_loops.png",
    title=None,
):
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    binary_image = (image_norm > intensity_threshold).astype(np.uint8)
    # Equivalent to the deprecated min_size=min_size under scikit-image 0.26
    # (verified), despite the deprecation note suggesting an off-by-one.
    cleaned_image = morphology.remove_small_objects(
        binary_image.astype(bool), max_size=min_size
    )
    labeled_image = measure.label(cleaned_image)
    regions = measure.regionprops(labeled_image)

    circular_objects = np.zeros_like(labeled_image)
    dot_coords = []
    for region in regions:
        if region.perimeter == 0: continue
        circularity = 4 * math.pi * region.area / (region.perimeter ** 2)
        if circularity >= circularity_threshold:
            dot_coords.append(region.centroid)
            circular_objects[labeled_image == region.label] = 1

    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.imshow(circular_objects > 0, cmap='hot', alpha=0.5)
        plt.title(title or f'Detected Loops: {len(dot_coords)}')
        plt.savefig(out_path)
        plt.close()

    return dot_coords, circular_objects > 0


# Depth window below the predicted DEJ, in pixels, that holds the papillary loops.
SLAB_TOP_PX = 0
SLAB_BOTTOM_PX = 15
EN_FACE_SIZE = (1355, 1355)
DEFAULT_DICOM = Path("data") / "dicom" / "RS_Right Hallux_L1076_S3070__26_05_2021.dcm"


def extract_flow(volume_slice):
    """Flow signal: in the RGB renderings, flow is the red tint over grey."""
    if volume_slice.ndim == 3:
        red = volume_slice[:, :, 0].astype(float)
        green = volume_slice[:, :, 1].astype(float)
        return np.clip(red - green, 0, None)
    return volume_slice.astype(float)


def papillary_dermis_mip(volume, model, device, slab=(SLAB_TOP_PX, SLAB_BOTTOM_PX)):
    """En-face maximum-intensity projection of flow in the papillary dermis slab.

    Returns the MIP (slices x width), the DEJ depth per slice and column (-1
    where the model found no skin layer), and the DEJ tortuosity per slice.
    """
    n_slices, height, width = volume.shape[:3]
    mip = np.zeros((n_slices, width), dtype=np.float32)
    dej = np.full((n_slices, width), -1, dtype=int)
    tortuosities = []

    for i in range(n_slices):
        p_mask = test_model_on_sample_image(model, volume[i], device)
        _, _, lower_line = remove_epidermis_from_image(volume[i], p_mask)
        found = p_mask.astype(bool).any(axis=0)

        slab_top = shift_line_coordinates(lower_line, slab[0], height)
        slab_bottom = shift_line_coordinates(lower_line, slab[1], height)
        slab_mask = create_mask_between_lines(height, width, slab_top, slab_bottom)
        # A column with no detected skin layer has no DEJ. Without this, its
        # slab would be taken from the top rows of the image, above the skin.
        slab_mask[:, ~found] = 0

        mip[i] = np.where(slab_mask == 1, extract_flow(volume[i]), 0).max(axis=0)
        dej[i, found] = lower_line[found, 0]
        tortuosities.append(calculate_tortuosity(lower_line[found]))

    return mip, dej, tortuosities


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Papillary dermis en-face MIP and capillary loop count.")
    parser.add_argument(
        "dicoms", nargs="*", type=Path, help=f"RGB OCTA DICOM volumes (default: {DEFAULT_DICOM})"
    )
    parser.add_argument("--out-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    dicoms = args.dicoms or [project_root() / DEFAULT_DICOM]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_pytorch_model(default_model_path(), device)

    for path in dicoms:
        volume, _, _ = import_dicom(path)
        print(f"{path.name}: processing {volume.shape[0]} slices...")
        mip, dej, tortuosities = papillary_dermis_mip(volume, model, device)

        stem = path.stem
        np.save(args.out_dir / f"{stem}_mip.npy", mip)
        np.save(args.out_dir / f"{stem}_dej.npy", dej)

        mip_final = resize(mip, EN_FACE_SIZE, anti_aliasing=True, order=1)
        plt.figure(figsize=(10, 10))
        plt.imshow(mip_final, cmap='gray')
        plt.title(f"Papillary dermis en-face MIP\n{stem}")
        plt.savefig(args.out_dir / f"{stem}_papillary_dermis_mip.png")
        plt.close()

        dots, _ = identify_and_plot_dots(
            mip_final,
            min_size=30,
            intensity_threshold=75,
            circularity_threshold=0.45,
            out_path=args.out_dir / f"{stem}_detected_capillary_loops.png",
        )
        no_dej = (dej < 0).mean()
        print(
            f"  loops={len(dots)}  mean DEJ tortuosity={np.mean(tortuosities):.4f}  "
            f"columns without a detected DEJ={no_dej:.1%}"
        )


if __name__ == "__main__":
    main()
