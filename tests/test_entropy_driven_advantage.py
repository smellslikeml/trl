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

from trl import GRPOConfig
from trl.trainer.entropy_driven_advantage import compute_entropy_driven_advantage

from .testing_utils import TrlTestCase


class TestEntropyDrivenAdvantageConfig(TrlTestCase):
    """The `entropy_driven_advantage` option must be wired through GRPOConfig (the call-site config)."""

    def test_default_is_disabled(self):
        config = GRPOConfig(output_dir=self.tmp_dir, report_to="none", bf16=False)
        assert config.entropy_driven_advantage is False

    def test_can_be_enabled(self):
        config = GRPOConfig(output_dir=self.tmp_dir, report_to="none", bf16=False, entropy_driven_advantage=True)
        assert config.entropy_driven_advantage is True


class TestEntropyDrivenAdvantage(TrlTestCase):
    """Math of the Entropy-Driven Advantage reweighting (EDGE-GRPO, Eqs. 3/6/7)."""

    def test_confident_correct_amplified_and_confident_wrong_penalized(self):
        # Two groups of 2 generations. Within each group the first completion is more confident
        # (lower entropy) than the second.
        advantages = torch.tensor([3.0, 3.0, -3.0, -3.0])
        entropy = torch.tensor([[1.0], [3.0], [1.0], [3.0]])
        mask = torch.ones(4, 1)
        scaled = compute_entropy_driven_advantage(advantages, entropy, mask, num_generations=2)
        # Group-normalized entropy is [0.5, 1.5] per group, so the confident responses are divided by 0.5
        # (amplified) and the uncertain ones by 1.5 (dampened).
        torch.testing.assert_close(scaled, torch.tensor([6.0, 2.0, -6.0, -2.0]))
        # Sign is preserved: positive advantages stay positive, negative stay negative.
        assert (scaled.sign() == advantages.sign()).all()

    def test_advantage_collapse_group_unchanged(self):
        # When every advantage in a group is already zero (advantage collapse), scaling by entropy leaves it at
        # zero — the Entropy-Driven Advantage does not invent signal for a fully collapsed group.
        advantages = torch.zeros(4)
        entropy = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
        mask = torch.ones(4, 1)
        scaled = compute_entropy_driven_advantage(advantages, entropy, mask, num_generations=2)
        torch.testing.assert_close(scaled, torch.zeros(4))

    def test_uniform_entropy_is_noop(self):
        # If policy entropy is identical across a group, the normalization is 1 everywhere and the
        # advantage is returned unchanged.
        advantages = torch.tensor([2.0, -1.0, 0.5, 3.0])
        entropy = torch.full((4, 1), 0.7)
        mask = torch.ones(4, 1)
        scaled = compute_entropy_driven_advantage(advantages, entropy, mask, num_generations=2)
        torch.testing.assert_close(scaled, advantages)

    def test_groups_normalized_independently(self):
        # The two groups have very different entropy baselines (group 1 is 10x group 0), yet per-group
        # normalization yields the identical rescaling for both — proof the mean is taken within each group.
        # Group 0: entropies [1, 3] (mean 2); group 1: entropies [10, 30] (mean 20). Each gives P̂ = [0.5, 1.5].
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
        entropy = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
        mask = torch.ones(4, 1)
        scaled = compute_entropy_driven_advantage(advantages, entropy, mask, num_generations=2)
        torch.testing.assert_close(scaled, torch.tensor([2.0, 1.0 / 1.5, 2.0, 1.0 / 1.5]))

    def test_completion_mask_excludes_padding(self):
        # Padding tokens (mask 0) must not contribute to the per-response entropy.
        advantages = torch.tensor([1.0, 1.0])
        entropy = torch.tensor([[0.2, 0.4, 0.0], [0.6, 0.6, 0.6]])  # last token of row 0 is padding
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        scaled = compute_entropy_driven_advantage(advantages, entropy, mask, num_generations=2)
        # P_0 = (0.2 + 0.4) / 2 = 0.3, P_1 = 0.6, group mean = 0.45 -> P̂ = [0.3/0.45, 0.6/0.45].
        torch.testing.assert_close(scaled, torch.tensor([1.5, 0.75]))
