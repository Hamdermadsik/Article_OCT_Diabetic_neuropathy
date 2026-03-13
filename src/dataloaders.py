import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 # Used for image reading and resizing
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF # Use TF for programmatic transforms
import random


# --- Configuration (Adjust if needed) ---
TARGET_SIZE = (512, 512) 

# --- PyTorch Dataset Class ---

class OCT_SKIN(Dataset):
    """
    A custom PyTorch Dataset for OCT skin segmentation.
    Handles resizing, normalization, and data type conversion, and augmentation.
    """
    # TARGET_SIZE is now a class variable
    TARGET_SIZE = (512, 512) 

    def __init__(self, images_dir, masks_dir, target_size=TARGET_SIZE, augment=False):
        """
        Args:
            images_dir (Path): Path to the folder containing original OCT images.
            masks_dir (Path): Path to the folder containing binary segmentation masks.
            target_size (tuple): Desired (Height, Width) for resizing.
            augment (bool): Whether to apply geometric data augmentation.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        self.augment = augment # New augmentation flag
        
        # 1. Map all available images and match them to masks
        image_files = sorted(list(self.images_dir.glob('*.png'))) 
        self.data_pairs = []
        
        for img_path in image_files:
            # Match image "100.png" with mask "mask_100.png"
            mask_name = f"mask_{img_path.name}"
            mask_path = self.masks_dir / mask_name
            
            if mask_path.exists():
                self.data_pairs.append({'image': img_path, 'mask': mask_path})
            # Warning print removed for cleaner output
            
        print(f"Dataset initialized with {len(self.data_pairs)} valid pairs. Augment={self.augment}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.data_pairs[idx]
        image_path = pair['image']
        mask_path = pair['mask']

        # --- 1. Load Data (as NumPy arrays first) ---
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise RuntimeError(f"Failed to load image or mask at index {idx}")

        # --- 2. Initial Resizing and Pre-processing (NumPy to Tensor) ---
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0 # Scale 0-255 to 0-1
        mask = (mask > 0).astype(np.float32)     # Convert 0/255 to 0/1 
        
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        
        # --- 3. Geometric Augmentation (Paired Transforms) ---
        if self.augment:
            # Horizontal Flip (Paired) - KEPT
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)
                
            # Vertical Flip (Paired) - REMOVED
            # This was removed as it's not anatomically realistic for retina scans.

            # Random Small-Angle Rotation (Paired) - KEPT
            if random.random() > 0.5:
                # Using a slightly wider but still realistic angle range.
                angle = random.uniform(-15, 15) 
                image_tensor = TF.rotate(image_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)
                
            # Random 90-degree Rotation - REMOVED
            # This was removed as it's not a realistic orientation for scans.

        # --- 4. Photometric Augmentation (Image Only) ---
        if self.augment:
            # Apply these transforms independently to get more combinations.
            
            # Adjust Brightness/Contrast - KEPT & IMPROVED
            if random.random() > 0.5:
                # Use torchvision's built-in ColorJitter for a convenient way to do this.
                # For grayscale, it adjusts brightness and contrast.
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
                image_tensor = jitter(image_tensor)
            
            # Add Gaussian Noise - NEW
            if random.random() > 0.5:
                # Add noise with a random standard deviation.
                std_dev = random.uniform(0.01, 0.08)
                noise = torch.randn_like(image_tensor) * std_dev
                image_tensor = image_tensor + noise
                # We must clamp the image to ensure pixel values stay in the [0, 1] range.
                image_tensor = torch.clamp(image_tensor, 0., 1.)

            
        # Final Step: Normalization (Applied after all other augmentations)
        # This standardizes the image tensor's range to [-1, 1], which helps training.
        image_tensor = TF.normalize(image_tensor, mean=[0.5], std=[0.5])


        # Final Check on Types before returning
        return image_tensor.float(), mask_tensor.float()

