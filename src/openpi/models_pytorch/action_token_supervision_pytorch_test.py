"""Tests for the PyTorch action-token supervision port (arXiv:2611.05249)."""

import numpy as np
import torch

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
import openpi.models_pytorch.action_token_supervision_pytorch as _action_tokens


def _expected_token_ids(actions: np.ndarray, n_bins: int) -> np.ndarray:
    boundaries = np.linspace(-1.0, 1.0, n_bins + 1)[:-1].astype(np.float32).astype(np.float64)
    bins = np.clip(np.digitize(actions.astype(np.float64), bins=boundaries) - 1, 0, n_bins - 1)
    return PALIGEMMA_VOCAB_SIZE - 1 - _action_tokens.SPECIAL_TOKEN_MARGIN - bins.reshape(actions.shape[0], -1)


def test_tokenize_actions_torch_matches_numpy_digitize():
    actions = np.random.RandomState(0).uniform(-1.5, 1.5, (3, 4, 5)).astype(np.float32)
    token_ids = _action_tokens.tokenize_actions_torch(torch.from_numpy(actions)).numpy()
    np.testing.assert_array_equal(token_ids, _expected_token_ids(actions, _action_tokens.N_BINS))
