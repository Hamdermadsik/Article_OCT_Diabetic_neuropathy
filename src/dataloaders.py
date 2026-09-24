"""Dataset for OCT skin segmentation, driven by the manifest.

Image/mask pairing comes from an explicit manifest rather than being inferred
from filenames at load time. A frame that is missing its mask is therefore a
visible, queryable state instead of an image that silently disappears from the
training set.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

from image_utils import letterbox_resize
from model_utils import DEFAULT_TARGET_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

log = logging.getLogger(__name__)

TARGET_SIZE = DEFAULT_TARGET_SIZE

# Rotation mixes the depth and lateral axes, which carry different physical
# scales, so it is deliberately small. Vertical flips are excluded entirely:
# inverting depth would put dermis above epidermis, which no real scan shows.
MAX_ROTATION_DEGREES = 7.0


def frames_from_directories(images_dir: Path, masks_dir: Path) -> pd.DataFrame:
    """Build a frames table from the legacy ``N.png`` / ``mask_N.png`` layout.

    Unmatched images are reported rather than dropped in silence.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    rows, unmatched = [], []
    for image_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / f"mask_{image_path.name}"
        if mask_path.exists():
            rows.append(
                {
                    "frame_id": image_path.stem,
                    "png_path": str(image_path),
                    "mask_path": str(mask_path),
                    "patient_id": image_path.stem,
                }
            )
        else:
            unmatched.append(image_path.name)

    if unmatched:
        log.warning(
            "%d image(s) in %s have no matching mask and are excluded: %s",
            len(unmatched),
            images_dir,
            ", ".join(unmatched[:10]) + ("..." if len(unmatched) > 10 else ""),
        )
    return pd.DataFrame(rows, columns=["frame_id", "png_path", "mask_path", "patient_id"])


def load_manifest_frames(
    manifest_dir: Path, annotation_status: str | None = "verified"
) -> pd.DataFrame:
    """Load frames from a built manifest, keeping only annotated ones."""
    frames_path = Path(manifest_dir) / "frames.csv"
    if not frames_path.exists():
        raise FileNotFoundError(
            f"No frames.csv at {frames_path}. Run src/manifest/build_manifest.py first."
        )
    frames = pd.read_csv(frames_path)
    if annotation_status is not None and "annotation_status" in frames.columns:
        frames = frames[frames["annotation_status"] == annotation_status]
    return frames[frames["mask_path"].notna()].reset_index(drop=True)


class OCT_SKIN(Dataset):
    """Paired OCT image/mask frames, letterboxed to a fixed model input size."""

    TARGET_SIZE = DEFAULT_TARGET_SIZE

    def __init__(
        self,
        frames: pd.DataFrame,
        target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
        augment: bool = False,
    ):
        """
        Args:
            frames: table with ``png_path`` and ``mask_path`` columns.
            target_size: model input size as (height, width).
            augment: apply geometric and photometric augmentation.
        """
        missing_columns = {"png_path", "mask_path"} - set(frames.columns)
        if missing_columns:
            raise ValueError(f"frames table is missing column(s): {sorted(missing_columns)}")

        usable = frames[frames["mask_path"].notna() & frames["png_path"].notna()]
        if len(usable) < len(frames):
            log.warning(
                "%d frame(s) lack an image or mask path and are excluded",
                len(frames) - len(usable),
            )

        self.frames = usable.reset_index(drop=True)
        self.target_size = tuple(target_size)
        self.augment = augment
        log.info("Dataset initialized with %d pairs (augment=%s)", len(self.frames), augment)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def patient_ids(self) -> list:
        """Grouping key for leakage-free splitting, when the manifest has one."""
        if "patient_id" in self.frames.columns:
            return self.frames["patient_id"].tolist()
        return self.frames.index.tolist()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.frames.iloc[idx]
        image = cv2.imread(str(row["png_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(
                f"Failed to load frame {row.get('frame_id', idx)}: "
                f"image={row['png_path']} mask={row['mask_path']}"
            )
        if image.shape != mask.shape:
            raise RuntimeError(
                f"Image/mask geometry mismatch for frame {row.get('frame_id', idx)}: "
                f"{image.shape} vs {mask.shape}"
            )

        # Identical letterbox parameters for both, so the pair stays aligned.
        image, _ = letterbox_resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask, _ = letterbox_resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:
            image_tensor, mask_tensor = self._augment_geometry(image_tensor, mask_tensor)
            image_tensor = self._augment_photometry(image_tensor)

        image_tensor = TF.normalize(image_tensor, mean=[NORMALIZE_MEAN], std=[NORMALIZE_STD])
        return image_tensor.float(), mask_tensor.float()

    def _augment_geometry(self, image_tensor, mask_tensor):
        if random.random() > 0.5:
            image_tensor = TF.hflip(image_tensor)
            mask_tensor = TF.hflip(mask_tensor)

        if random.random() > 0.5:
            angle = random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
            image_tensor = TF.rotate(
                image_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR
            )
            mask_tensor = TF.rotate(
                mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST
            )
        return image_tensor, mask_tensor

    def _augment_photometry(self, image_tensor):
        if random.random() > 0.5:
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
            image_tensor = jitter(image_tensor)

        if random.random() > 0.5:
            noise = torch.randn_like(image_tensor) * random.uniform(0.01, 0.08)
            image_tensor = torch.clamp(image_tensor + noise, 0.0, 1.0)
        return image_tensor
