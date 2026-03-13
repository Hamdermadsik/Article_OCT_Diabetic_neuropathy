import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import SimpleITK as sitk
from skimage.transform import resize
import cv2
from skimage import measure, morphology
import math
from pathlib import Path

# --- PyTorch Model Loading ---

def load_pytorch_model(model_path, device):
    """
    Loads the smp.Unet model trained with train.py.
    """
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

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

def test_model_on_sample_image(model, image, device, target_size=(512, 512), plot=False):
    orig_shape = image.shape
    
    max_val = 65535.0 if image.dtype == np.uint16 else 255.0
    img_norm = image.astype(np.float32) / max_val
    
    # If the image has 3 channels, it's an RGB export where:
    # Channel 0/1/2 are grayscale structural base, and Red is the OCTA overlay.
    if len(img_norm.shape) == 3:
        # Use first channel as the structural base for the segmentation model
        img_for_model = img_norm[:, :, 0]
    else:
        img_for_model = img_norm

    img_resized = cv2.resize(img_for_model, target_size, interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.sigmoid(pred_logits)
        pred_mask = (pred_mask > 0.5).float()
    
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    resized_mask = cv2.resize(pred_mask_np, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.imshow(img_for_model, cmap='gray'); plt.title('Structural OCT'); plt.axis('off')
        overlay = np.stack([img_for_model]*3, axis=-1)
        overlay[resized_mask > 0] = [1, 0, 0]
        plt.subplot(1, 2, 2); plt.imshow(overlay); plt.title('Mask Overlay'); plt.axis('off')
        plt.show()

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
        plt.legend(); plt.show()

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

def identify_and_plot_dots(image, min_size=50, intensity_threshold=50, plot=True, circularity_threshold=0.7):
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    binary_image = (image_norm > intensity_threshold).astype(np.uint8)
    cleaned_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=min_size)
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
        plt.title(f'Detected Loops: {len(dot_coords)}')
        plt.show()

    return dot_coords, circular_objects > 0

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    MODEL_PATH = "/home/madsl/article/Article_OCT_Diabetic_neuropathy/models/best_unet_model.pth"
    
    # Path to separate OCT and OCTA files as per the notebook logic
    # _S.dcm = Structural (OCT), _D.dcm = Flow (OCTA)
    OCT_PATH = "/home/madsl/article/Article_OCT_Diabetic_neuropathy/data/dicom/JJ_Right Hallux_L1030_S2678__24_05_2021.dcm"
    OCTA_PATH = "/home/madsl/article/Article_OCT_Diabetic_neuropathy/data/dicom/JJ_Right Hallux_L1030_S2678__24_05_2021.dcm" # Usually different file, e.g. ..._D.dcm
    
    model = load_pytorch_model(MODEL_PATH, device)

    # 1. Import separate datasets
    OCT_data, spacing, _ = import_dicom(OCT_PATH)
    OCTA_data, _, _ = import_dicom(OCTA_PATH)
    x_sp, y_sp, z_sp = spacing
    
    # Initialize the 3D volume for the papillary dermis slab
    # Using the shape of the OCTA data
    papillary_dermis_3d = np.zeros_like(OCTA_data, dtype=np.float32)
    tortuosities = []
    
    print(f"Processing {OCT_data.shape[0]} slices...")
    for i in range(OCT_data.shape[0]):
        # B-scan from structural OCT for segmenting the epidermis
        oct_slice = OCT_data[i]
        # B-scan from OCTA for extracting flow
        octa_slice = OCTA_data[i]
        
        # Segment epidermis on OCT slice
        p_mask = test_model_on_sample_image(model, oct_slice, device)
        # Find DEJ (lower line)
        _, _, lower_line = remove_epidermis_from_image(oct_slice, p_mask)
        
        # Define slab from -5 to +20 pixels relative to DOJ
        new_lower = shift_line_coordinates(lower_line, 20, oct_slice.shape[0])
        new_upper = shift_line_coordinates(lower_line, -5, oct_slice.shape[0])
        p_dermi_mask = create_mask_between_lines(oct_slice.shape[0], oct_slice.shape[1], new_upper, new_lower)
        
        # Extract Flow from OCTA data
        # If the file is RGB, the 'red stuff' is the OCTA signal.
        if len(octa_slice.shape) == 3:
            # Extract red signal by subtracting Green from Red
            r = octa_slice[:, :, 0].astype(float)
            g = octa_slice[:, :, 1].astype(float)
            flow_signal = np.clip(r - g, 0, None)
        else:
            flow_signal = octa_slice

        papillary_dermis_3d[i] = np.where(p_dermi_mask == 1, flow_signal, 0)
        tortuosities.append(calculate_tortuosity(lower_line))

    # EN-FACE MIP: Projected along Axis 1 (Depth)
    mip = np.max(papillary_dermis_3d, axis=1) 
    mip_final = resize(mip, (1355, 1355), anti_aliasing=True, order=1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mip_final, cmap='gray')
    plt.title("Papillary Dermis EN-FACE MIP (Aligned with Notebook)")
    plt.show()
    print(f"Mean DEJ Tortuosity: {np.mean(tortuosities):.4f}")
    
    dots, _ = identify_and_plot_dots(mip_final, min_size=30, intensity_threshold=75, circularity_threshold=0.45)
    print(f"Capillary Loops Found: {len(dots)}")
