"""Tests for the `gradient_routing` knob of the PyTorch mixture-of-transformers (arXiv:2611.05249).

`full` (the default) propagates the action expert's loss into the VLM (PaliGemma) trunk;
`stop_vlm` insulates the trunk, yielding exactly zero gradients for every VLM parameter while
the action expert keeps training.
"""

import pytest
import torch

import openpi.models.gemma as _gemma


def _torch_grad_masses(gradient_routing: str) -> tuple[float, float]:
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

    config = _gemma.get_config("dummy")
    model = PaliGemmaWithExpertModel(
        config, config, use_adarms=[False, False], precision="float32", gradient_routing=gradient_routing
    )
    prefix = torch.randn(2, 5, config.width)
    suffix = torch.randn(2, 3, config.width)
    attention_mask = torch.zeros(2, 1, 8, 8)
    position_ids = torch.arange(8)[None, :].repeat(2, 1)
    outputs, _ = model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
    )
    outputs[1].square().mean().backward()

    def mass(module):
        return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)

    return mass(model.paligemma), mass(model.gemma_expert)


def test_stop_vlm_insulates_vlm_pytorch():
    # Requires the patched transformers from `models_pytorch/transformers_replace`.
    pytest.importorskip("transformers.models.siglip.check", reason="transformers_replace is not installed")
    vlm, expert = _torch_grad_masses("full")
    assert vlm > 0.0
    assert expert > 0.0
    vlm, expert = _torch_grad_masses("stop_vlm")
    assert vlm == 0.0
    assert expert > 0.0
