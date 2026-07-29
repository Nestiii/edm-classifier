"""Tests for the Short-chunk CNN + ResNet architecture."""

from __future__ import annotations

import pytest
import torch

from edm_classifier.config import NUM_CLASSES
from edm_classifier.models.short_chunk_cnn import ResBlock2d, build_model


def test_model_forward_output_shape():
    model = build_model().eval()
    x = torch.randn(3, 1, 128, 86)  # 3 segments, mel 128 x ~2s frames
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (3, NUM_CLASSES)


def test_model_handles_single_segment_in_eval():
    model = build_model().eval()
    x = torch.randn(1, 1, 128, 86)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, NUM_CLASSES)


def test_model_rejects_wrong_input_rank():
    model = build_model().eval()
    with pytest.raises(ValueError):
        model(torch.randn(3, 128, 86))


def test_resblock_projection_changes_channels():
    block = ResBlock2d(1, 16, stride=2).eval()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out = block(x)
    assert out.shape[1] == 16
    # stride 2 halves spatial dims.
    assert out.shape[2] == 16 and out.shape[3] == 16


def test_resblock_identity_path_when_same_shape():
    block = ResBlock2d(8, 8, stride=1).eval()
    assert block.needs_projection is False
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape
