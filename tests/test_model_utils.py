"""A model must be fed input prepared the way its training data was.

The original checkpoint scores Dice 0.88 on its test frames with the square
resize it was trained on, and 0.77 with letterboxing. Checkpoints therefore
record their preprocessing, and loading must recover it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from image_utils import LETTERBOX, SQUASH
from model_utils import build_unet, checkpoint_payload, load_pytorch_model

REPO = Path(__file__).resolve().parents[1]


def test_a_bare_state_dict_is_treated_as_a_squash_model(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save(build_unet().state_dict(), path)

    model = load_pytorch_model(path, "cpu")

    assert model.preprocessing == SQUASH


def test_a_new_checkpoint_carries_its_preprocessing(tmp_path):
    path = tmp_path / "new.pth"
    torch.save(checkpoint_payload(build_unet(), LETTERBOX, (256, 384)), path)

    model = load_pytorch_model(path, "cpu")

    assert model.preprocessing == LETTERBOX
    assert model.target_size == (256, 384)


@pytest.mark.skipif(
    not (REPO / "models" / "best_unet_model.pth").exists(), reason="no trained model on disk"
)
def test_the_projects_current_model_is_recognised_as_squash_trained():
    model = load_pytorch_model(REPO / "models" / "best_unet_model.pth", "cpu")

    assert model.preprocessing == SQUASH
