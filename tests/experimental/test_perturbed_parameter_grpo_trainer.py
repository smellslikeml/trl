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

from trl.experimental.perturbed_parameter_grpo import (
    PerturbedParameterGRPOConfig,
    PerturbedParameterGRPOTrainer,
)
from trl.trainer.grpo_trainer import GRPOTrainer

from ..testing_utils import TrlTestCase


def completion_parity_reward(completions, **kwargs):
    """Completion-dependent binary reward, so samples within a prompt vary (nonzero advantages)."""
    return [float(len(completion) % 2 == 0) for completion in completions]


class TestPerturbedParameterGRPOTrainer(TrlTestCase):
    def test_is_grpo_subclass(self):
        # The trainer is wired into the existing GRPO machinery by inheritance, overriding generation and loss.
        assert issubclass(PerturbedParameterGRPOTrainer, GRPOTrainer)

    def test_disabled_falls_back_to_grpo(self):
        # With sigma=0 the generation hook delegates to GRPO and returns no behavior log-probs.
        dataset = Dataset.from_dict({"prompt": ["The capital of France is", "Two plus two equals"]})
        trainer = PerturbedParameterGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=PerturbedParameterGRPOConfig(
                output_dir=self.tmp_dir, report_to="none", perturbation_sigma=0.0, bf16=False
            ),
            train_dataset=dataset,
        )
        prompt_ids = [trainer.processing_class("The capital of France is")["input_ids"]]
        _, logprobs = trainer._generate_single_turn(prompt_ids, None, {})
        assert logprobs is None

    def test_perturbation_restores_parameters(self):
        # The sampled behavior policy must be reverted exactly so the optimizer still updates the base policy.
        dataset = Dataset.from_dict({"prompt": ["The capital of France is", "Two plus two equals"]})
        trainer = PerturbedParameterGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=PerturbedParameterGRPOConfig(
                output_dir=self.tmp_dir, report_to="none", perturbation_sigma=1e-2, bf16=False
            ),
            train_dataset=dataset,
        )
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

        prompt_ids = [trainer.processing_class("The capital of France is")["input_ids"]]
        completion_ids, behavior_logps = trainer._generate_single_turn(prompt_ids, None, {})

        # Behavior log-probs are returned (one list per completion) and consumed by the loss as the
        # importance-sampling reference.
        assert behavior_logps is not None
        assert len(behavior_logps) == len(completion_ids)
        for completion, logps in zip(completion_ids, behavior_logps, strict=True):
            assert len(logps) == len(completion)
            assert all(torch.isfinite(torch.tensor(x)) for x in logps)

        # Weights are restored to the base policy bit-for-bit.
        for n, p in trainer.model.named_parameters():
            assert torch.equal(before[n], p), f"Parameter {n} was not restored after perturbation"

    def test_train(self):
        dataset = Dataset.from_dict(
            {"prompt": ["The capital of France is", "Two plus two equals", "Water is made of", "The sky is"]}
        )
        training_args = PerturbedParameterGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # higher lr because gradients are tiny and the default lr can stall updates
            per_device_train_batch_size=2,
            max_completion_length=8,
            num_generations=2,
            perturbation_sigma=1e-2,
            bf16=False,
            report_to="none",
        )
        trainer = PerturbedParameterGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_invalid_sigma_rejected(self):
        with pytest.raises(ValueError, match="perturbation_sigma"):
            PerturbedParameterGRPOConfig(output_dir=self.tmp_dir, perturbation_sigma=-1.0, bf16=False)
