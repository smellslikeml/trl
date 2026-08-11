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

from dataclasses import dataclass

from ...trainer.grpo_config import GRPOConfig


@dataclass
class PerturbedParameterGRPOConfig(GRPOConfig):
    """
    Configuration class for [`PerturbedParameterGRPOTrainer`].

    Adds parameter-space exploration to GRPO: at generation time the policy weights are perturbed with
    Gaussian noise to sample a diverse *behavior* policy, rollouts are generated under it, and the update
    is importance-corrected back to the base policy. This is a complementary exploration lever to
    action-space (temperature) scaling and targets the zero-advantage-group instability of GRPO.

    Adapted from "Parameter Exploration for RLVR via Variational Learning" (Perturbed Parameter Policy
    Optimization / 3PO), https://arxiv.org/abs/2608.09805. The variational posterior of the original
    method is replaced by a parameter-free Gaussian perturbation of the trainable weights.

    Args:
        perturbation_sigma (`float`, *optional*, defaults to `0.0`):
            Relative standard deviation of the parameter-space noise added to each trainable weight at
            generation time. The perturbation applied to a parameter `w` is `sigma * clamp(w.std(),
            perturbation_min_std) * eps` with `eps ~ N(0, I)`. `0.0` disables perturbation and reduces
            the trainer to standard GRPO.

        perturbation_min_std (`float`, *optional*, defaults to `1e-4`):
            Floor on the per-parameter standard deviation used to scale the noise, so near-constant
            tensors are not over-perturbed.
    """

    perturbation_sigma: float = 0.0
    perturbation_min_std: float = 1e-4

    def __post_init__(self):
        super().__post_init__()

        if self.perturbation_sigma < 0.0:
            raise ValueError(f"perturbation_sigma must be non-negative, got {self.perturbation_sigma}")

        if self.perturbation_min_std <= 0.0:
            raise ValueError(f"perturbation_min_std must be positive, got {self.perturbation_min_std}")
