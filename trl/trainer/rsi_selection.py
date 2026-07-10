# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Relative Surprisal Index (RSI) token selection for RLVR.

Implements the Relative Surprisal Index and RSI Selection (RSI-S) from
*Which Tokens Matter? Adaptive Token Selection for RLVR with the Relative Surprisal
Index* (https://huggingface.co/papers/2606.31575). RSI couples the sampled token's
surprisal with the predictive entropy at each position, collapsing two competing
token-selection signals into one scalar:

    RSI = surprisal(selected token) / entropy(distribution) = -log p_selected / H

A token with RSI ~ 0 is near-deterministic (redundant low-surprisal); a token with
RSI ~ 1 is "typical"; a token with RSI >> 1 lies in the high-surprisal tail (unstable).
RSI-S keeps tokens within a stable interval [low, high], filtering both ends at once.
This reconciles the "prioritize high-entropy tokens" and "drop low-probability tokens"
heuristics, which each only see one side of the same coupled signal.
"""

import torch


def relative_surprisal_index(
    per_token_logps: torch.Tensor,
    entropies: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-token Relative Surprisal Index (RSI): surprisal of the selected token divided by the predictive entropy.

    Args:
        per_token_logps (`torch.Tensor`):
            Log-probabilities of the selected tokens, shape `(batch_size, seq_len)`.
        entropies (`torch.Tensor`):
            Predictive entropy of the distribution at each position, shape `(batch_size, seq_len)`.
        mask (`torch.Tensor`):
            Binary mask where `1` marks valid completion tokens and `0` marks padding, same shape as `entropies`.
        eps (`float`, *optional*, defaults to `1e-8`):
            Numerical-stability floor for the entropy denominator.

    Returns:
        `torch.Tensor`:
            RSI per token, shape `(batch_size, seq_len)`. Padding positions are set to zero.

    Examples:

    ```python
    >>> rsi = relative_surprisal_index(per_token_logps, entropies, completion_mask)
    ```
    """
    # Surprisal of the actually-sampled token: s = -log p_selected.
    surprisal = -per_token_logps
    # RSI = surprisal / entropy. Where the distribution is deterministic (H ~ 0 and p ~ 1), the token is redundant by
    # definition, so its RSI is set to 0 (surprisal is also ~0 there, but the ratio is numerically unstable).
    rsi = torch.where(
        entropies > eps,
        surprisal / entropies.clamp(min=eps),
        torch.zeros_like(surprisal),
    )
    return rsi * mask.float()


def rsi_selection_mask(
    per_token_logps: torch.Tensor,
    entropies: torch.Tensor,
    mask: torch.Tensor,
    low: float | None = None,
    high: float | None = None,
) -> torch.Tensor:
    """
    RSI Selection (RSI-S) mask: keep tokens whose RSI falls within `[low, high]`.

    Filtering both ends at once reconciles the "prioritize high-entropy tokens" and "drop low-probability tokens"
    heuristics: tokens with RSI < `low` are redundant low-surprisal tokens, tokens with RSI > `high` are unstable
    high-surprisal tail tokens.

    Args:
        per_token_logps (`torch.Tensor`):
            Log-probabilities of the selected tokens, shape `(batch_size, seq_len)`.
        entropies (`torch.Tensor`):
            Predictive entropy of the distribution at each position, shape `(batch_size, seq_len)`.
        mask (`torch.Tensor`):
            Binary mask where `1` marks valid completion tokens and `0` marks padding, same shape as `entropies`.
        low (`float` or `None`, *optional*):
            Lower RSI bound. `None` leaves the lower end unbounded.
        high (`float` or `None`, *optional*):
            Upper RSI bound. `None` leaves the upper end unbounded.

    Returns:
        `torch.Tensor`:
            Boolean mask of shape `(batch_size, seq_len)`, where `True` indicates kept tokens.

    Examples:

    ```python
    >>> keep = rsi_selection_mask(per_token_logps, entropies, completion_mask, low=0.5, high=2.0)
    ```
    """
    rsi = relative_surprisal_index(per_token_logps, entropies, mask)
    keep = torch.ones_like(rsi, dtype=torch.bool)
    if low is not None:
        keep = keep & (rsi >= low)
    if high is not None:
        keep = keep & (rsi <= high)
    return keep & mask.bool()
