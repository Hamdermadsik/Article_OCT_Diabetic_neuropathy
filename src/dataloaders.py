import numpy as np
from pathlib import Path
from OCTVol import OCTVol  # can be installed via: pip install oct-vol
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 # Used for image reading and resizing
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF # Use TF for programmatic transforms
import random

class OCTDataManager:
    """
    
    Manages loading and saving of OCT volume data in various formats.

    """

    def __init__(self, pixel_spacing_microns=3.87): # Initialize with default pixel spacing
        self.pixel_spacing = pixel_spacing_microns  

    def load_vol_file(self, path):
        """Loads a .vol file """
        vol = OCTVol(str(path)) # Load OCT volume
        raw_data = vol.b_scans # Get raw B-scan data as numpy array

        if raw_data.shape[2] < raw_data.shape[0]:  # Check dimensions to ensure (slices, height, width)
            raw_data = np.transpose(raw_data, (2, 0, 1)) 
        return raw_data

    def load_numpy_file(self, path):
        """Loads a .npy file and fixes dimensions."""

        data = np.load(path)
        if data.shape[2] < data.shape[0]: # Ensure shape is (slices, height, width)
             data = np.transpose(data, (2, 0, 1))
        return data

    def save_numpy_file(self, data, path):
        np.save(path, data)
        print(f"Saved volume with shape {data.shape} to {path}")
    

# --- Configuration (Adjust if needed) ---
TARGET_SIZE = (512, 512) 

# --- PyTorch Dataset Class ---

class RetinaSegDataset(Dataset):
    """
    A custom PyTorch Dataset for OCT Retina Segmentation (U-Net training).
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
        # Assuming .jpeg is the correct extension based on your successful run
        image_files = sorted(list(self.images_dir.glob('*.jpeg'))) 
        self.data_pairs = []
        
        for img_path in image_files:
            mask_name = f"{img_path.stem}_mask.png"
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


class RetinaSegDataset_size_n(Dataset):
    """
    A custom PyTorch Dataset for OCT Retina Segmentation (U-Net training).
    Handles resizing, normalization, and data type conversion, and augmentation.
    Takes number N for number of samples to load.
    """
    # TARGET_SIZE is now a class variable
    TARGET_SIZE = (512, 512) 

    def __init__(self, images_dir, masks_dir, target_size=TARGET_SIZE, augment=False, n_samples=10, noise_sigma=0):
        """
        Args:
            images_dir (Path): Path to the folder containing original OCT images.
            masks_dir (Path): Path to the folder containing binary segmentation masks.
            target_size (tuple): Desired (Height, Width) for resizing.
            augment (bool): Whether to apply geometric data augmentation.
            n_samples (int): Number of samples to load from the dataset.
            noise_sigma (float): Standard deviation of Gaussian noise to add (0-255 scale).
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        self.augment = augment # New augmentation flag
        self.n_samples = n_samples
        self.noise_sigma = noise_sigma
        
        # 1. Map all available images and match them to masks
        # Assuming .jpeg is the correct extension based on your successful run
        image_files = sorted(list(self.images_dir.glob('*.jpeg'))) 
        self.data_pairs = []
        
        for img_path in image_files[:self.n_samples]:
            mask_name = f"{img_path.stem}_mask.png"
            mask_path = self.masks_dir / mask_name
            
            if mask_path.exists():
                self.data_pairs.append({'image': img_path, 'mask': mask_path})
            # Warning print removed for cleaner output
            
        print(f"Dataset initialized with {len(self.data_pairs)} valid pairs. Augment={self.augment}, Noise sigma={self.noise_sigma}")

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

        # --- 1.b Apply Fixed Gaussian Noise (if configured) ---
        # This simulates noisy sensor data for the EXPERIMENT
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

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

