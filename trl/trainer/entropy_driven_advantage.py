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

import torch


def compute_entropy_driven_advantage(
    advantages: torch.Tensor,
    per_token_entropy: torch.Tensor,
    completion_mask: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """
    Entropy-Driven Advantage (EDA) reweighting from EDGE-GRPO
    (https://huggingface.co/papers/2507.21848).

    Each response's advantage is divided by its group-normalized policy entropy so that
    confident-correct responses (low entropy, positive advantage) receive a larger advantage
    and confident-wrong responses (low entropy, negative advantage) receive a harsher penalty.
    This increases advantage diversity within a group and mitigates the advantage-collapse
    problem of GRPO under sparse, group-identical rewards.

    Per-response entropy `P_i` is the mean per-token policy entropy over the completion
    (Eq. 3). It is group-normalized as `P̂_i = P_i / mean({P_1, ..., P_G})` over the
    `num_generations` completions sharing a prompt (Eq. 6), and the advantage is scaled as
    `Â_i = A_i / P̂_i` (Eq. 7). Because `P̂_i` averages to 1 within each group, the overall
    advantage scale is preserved.

    Args:
        advantages (`torch.Tensor`):
            Per-response advantages of shape `(B,)`, where `B` is the number of completions.
            `B` must be divisible by `num_generations` and laid out as contiguous groups
            `[prompt_0_gen_0, ..., prompt_0_gen_{G-1}, prompt_1_gen_0, ...]`.
        per_token_entropy (`torch.Tensor`):
            Per-token policy entropy of shape `(B, T)`, aligned with `completion_mask`.
        completion_mask (`torch.Tensor`):
            Completion mask of shape `(B, T)` (1 for real completion tokens, 0 otherwise).
        num_generations (`int`):
            Group size `G`; the number of completions sampled per prompt.

    Returns:
        `torch.Tensor`: Entropy-driven advantages of shape `(B,)`.

    Examples:

    ```python
    >>> # Two groups of 2 generations; the first completion of each group is more confident
    >>> # (lower entropy). Its advantage is amplified (+3 -> +6, -3 -> -6), the uncertain
    >>> # one is dampened (+3 -> +2, -3 -> -2).
    >>> advantages = torch.tensor([3.0, 3.0, -3.0, -3.0])
    >>> entropy = torch.tensor([[1.0], [3.0], [1.0], [3.0]])
    >>> mask = torch.ones(4, 1)
    >>> compute_entropy_driven_advantage(advantages, entropy, mask, num_generations=2).tolist()
    [6.0, 2.0, -6.0, -2.0]
    ```
    """
    # Per-response policy entropy P_i = mean per-token entropy over the completion (Eq. 3)
    response_entropy = (per_token_entropy * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    # Group-normalize: P̂_i = P_i / mean(P) over the group of num_generations (Eq. 6)
    grouped_entropy = response_entropy.view(-1, num_generations)
    scaled_entropy = grouped_entropy / grouped_entropy.mean(-1, keepdim=True).clamp(min=1e-6)
    # Entropy-driven advantage: Â_i = A_i / P̂_i (Eq. 7); floor P̂_i to avoid dividing by ~0
    return advantages / scaled_entropy.reshape(-1).clamp(min=1e-6)
