# --- DICE Score Implementation ---
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculates the Sørensen–Dice Coefficient (often just 'Dice Score').
    
    Args:
        pred (torch.Tensor): Model output logits (e.g., after Sigmoid or just the raw logits).
        target (torch.Tensor): Ground truth mask (0 or 1).
        smooth (float): Smoothing factor to prevent division by zero.
        
    Returns:
        float: The calculated Dice Score.
    """
    # 1. Convert logits to predictions (0 or 1)
    # Since we use BCEWithLogitsLoss, pred is raw logits. Apply sigmoid and threshold.
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > 0.5).float()
    
    # 2. Flatten tensors (keep batch dimension intact)
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # 3. Calculate Intersection and Union
    intersection = (pred_flat * target_flat).sum()
    sum_of_areas = pred_flat.sum() + target_flat.sum()
    
    # 4. Calculate Dice Score
    dice = (2. * intersection + smooth) / (sum_of_areas + smooth)
    
    # Handle the case where both target and prediction are entirely zero
    if sum_of_areas == 0:
        return 1.0 # Perfect score if nothing is there and nothing is predicted

    return dice.item() # Return as a standard float

def train_one_epoch(dataloader, model, criterion, optimizer, device):
    # ... (same as before, no changes needed inside this function) ...
    # (function implementation from the previous answer)
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device) 
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate_one_epoch(dataloader, model, criterion, device):
    """Handles the validation logic for a single epoch and calculates metrics."""
    
    # Set the model to evaluation mode (turns off dropout, etc.)
    model.eval()
    total_loss = 0
    total_dice = 0
    
    # Ensure no gradients are calculated during validation/testing
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Loss and Metric Calculation
            loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)

            total_loss += loss.item()
            total_dice += dice

            loop.set_postfix(v_loss=loss.item(), dice=dice)

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply Sigmoid to the raw logits (pred)
        pred = torch.sigmoid(pred) 
        
        # Flatten tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # Calculate Intersection and Union
        intersection = (pred * target).sum()
        sum_of_areas = pred.sum() + target.sum()
        
        # Calculate Dice Score (for the foreground class, 1)
        dice_score = (2. * intersection + self.smooth) / (sum_of_areas + self.smooth)
        
        # Dice Loss is 1 - Dice Score
        dice_loss = 1.0 - dice_score
        
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, pos_weight=None):
        super(CombinedLoss, self).__init__()
        # Use pos_weight in BCE
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
        self.dice_loss = DiceLoss()
        self.alpha = alpha 

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice
    

# helper to add Gaussian noise
def add_gaussian_noise(img, sigma):
    """Adds Gaussian noise to an image."""
    # Ensure image is float for calculation
    img_float = img.astype(np.float32)
    noise = np.random.normal(0, sigma, img_float.shape)
    noisy_img = img_float + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)