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

import pytest
import torch
from datasets import Dataset

from trl.experimental.off_context_grpo import OffContextGRPOConfig, OffContextGRPOTrainer
from trl.trainer.grpo_trainer import GRPOTrainer

from ..testing_utils import TrlTestCase


def completion_length_reward(completions, **kwargs):
    """Simple reward based on completion length."""
    return [min(1.0, len(completion) / 10.0) for completion in completions]


def math_guidance_func(prompt: str) -> str | None:
    """Simple guidance function for math-like prompts."""
    if "solve" in prompt.lower() or "calculate" in prompt.lower():
        return "Hint: Start by identifying the key variables. "
    return None


class TestOffContextGRPOTrainer(TrlTestCase):
    def test_trainer_initialization(self):
        """Test that OffContextGRPOTrainer can be initialized."""
        dataset = Dataset.from_dict({"prompt": ["Solve: 2 + 2", "Calculate: 5 * 3"]})

        config = OffContextGRPOConfig(
            output_dir=self.tmp_dir,
            off_context_grpo_ratio=0.5,
            guidance_func=math_guidance_func,
            report_to="none",
        )

        trainer = OffContextGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_length_reward,
            args=config,
            train_dataset=dataset,
        )

        assert isinstance(trainer, GRPOTrainer)
        assert trainer.off_context_grpo_ratio == 0.5
        assert trainer.guidance_func is not None

    def test_guidance_application(self):
        """Test that guidance is applied to prompts correctly."""
        dataset = Dataset.from_dict({"prompt": ["Solve: 2 + 2", "Calculate: 5 * 3"]})

        config = OffContextGRPOConfig(
            output_dir=self.tmp_dir,
            off_context_grpo_ratio=0.5,
            guidance_func=math_guidance_func,
            report_to="none",
            per_device_train_batch_size=2,
            max_completion_length=8,
        )

        trainer = OffContextGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_length_reward,
            args=config,
            train_dataset=dataset,
        )

        # Test guidance application
        prompts = ["Solve: 2 + 2", "Calculate: 5 * 3"]
        guided_prompts = trainer._apply_guidance_to_prompts(prompts, [0])

        # First prompt should have guidance prepended
        assert guided_prompts[0].startswith("Hint:")
        # Second prompt should be unchanged
        assert guided_prompts[1] == prompts[1]

    def test_guidance_with_conversational_format(self):
        """Test that guidance is applied to conversational prompts correctly."""
        dataset = Dataset.from_dict({"prompt": [
            [{"role": "user", "content": "Solve: 2 + 2"}],
            [{"role": "user", "content": "Calculate: 5 * 3"}]
        ]})

        config = OffContextGRPOConfig(
            output_dir=self.tmp_dir,
            off_context_grpo_ratio=0.5,
            guidance_func=math_guidance_func,
            report_to="none",
            per_device_train_batch_size=2,
            max_completion_length=8,
        )

        trainer = OffContextGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_length_reward,
            args=config,
            train_dataset=dataset,
        )

        # Test guidance application to conversational format
        prompts = [
            [{"role": "user", "content": "Solve: 2 + 2"}],
            [{"role": "user", "content": "Calculate: 5 * 3"}]
        ]
        guided_prompts = trainer._apply_guidance_to_prompts(prompts, [0])

        # First prompt should have guidance prepended to content
        assert guided_prompts[0][0]["content"].startswith("Hint:")
        # Second prompt should be unchanged
        assert guided_prompts[1] == prompts[1]

    def test_config_validation(self):
        """Test that config validation works correctly."""
        dataset = Dataset.from_dict({"prompt": ["Solve: 2 + 2"]})

        # Test invalid off_context_grpo_ratio
        with pytest.raises(ValueError, match="off_context_grpo_ratio must be between"):
            config = OffContextGRPOConfig(
                output_dir=self.tmp_dir,
                off_context_grpo_ratio=1.5,
                guidance_func=math_guidance_func,
            )
            OffContextGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=completion_length_reward,
                args=config,
                train_dataset=dataset,
            )

        # Test missing guidance_func when ratio > 0
        with pytest.raises(ValueError, match="guidance_func must be provided"):
            config = OffContextGRPOConfig(
                output_dir=self.tmp_dir,
                off_context_grpo_ratio=0.3,
                guidance_func=None,
            )
            OffContextGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=completion_length_reward,
                args=config,
                train_dataset=dataset,
            )

    def test_train(self):
        """Test that training runs without errors."""
        dataset = Dataset.from_dict({"prompt": ["Solve: 2 + 2", "Calculate: 5 * 3"]})

        config = OffContextGRPOConfig(
            output_dir=self.tmp_dir,
            off_context_grpo_ratio=0.5,
            guidance_func=math_guidance_func,
            report_to="none",
            per_device_train_batch_size=2,
            max_completion_length=8,
            learning_rate=0.1,
        )

        trainer = OffContextGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_length_reward,
            args=config,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
