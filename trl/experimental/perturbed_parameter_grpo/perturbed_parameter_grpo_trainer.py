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

import textwrap

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc
from ...trainer.utils import pad
from .perturbed_parameter_grpo_config import PerturbedParameterGRPOConfig


class PerturbedParameterGRPOTrainer(GRPOTrainer):
    """
    Trainer for parameter-space exploration in GRPO.

    Instead of exploring only in action space (e.g. via temperature scaling, which cannot reorder tokens),
    this trainer samples a perturbed *behavior* policy `theta_tilde = theta + sigma * std(theta) * eps` at
    generation time, generates rollouts under `theta_tilde`, and then importance-corrects the update back
    to the base policy `theta`. Sampling more or less diverse policies is thus a complementary lever over
    exploration, and producing rollouts from a shifted policy reduces the all-equal-reward (zero-advantage)
    groups that stall GRPO training.

    The correction reuses GRPO's existing importance-sampling channel: the behavior policy's per-token
    log-probs are returned from generation as `sampling_per_token_logps`, then threaded into the loss as
    `old_per_token_logps`. The base GRPO ratio is therefore `pi_theta / pi_theta_tilde`, with the standard
    PPO clipping handling the off-policyness.

    Adapted from "Parameter Exploration for RLVR via Variational Learning" (3PO),
    https://arxiv.org/abs/2608.09805. The variational posterior is replaced by a parameter-free Gaussian
    perturbation, and the multi-parameter-sample grouping of the original family is reduced to a single
    sampled policy per generation step. Perturbation currently runs on the regular (transformers) text
    generation path; vLLM, continuous batching, and VLM rollouts fall back to standard GRPO.

    Example:

    ```python
    >>> from datasets import load_dataset
    >>> from trl.experimental.perturbed_parameter_grpo import PerturbedParameterGRPOConfig, PerturbedParameterGRPOTrainer

    >>> dataset = load_dataset("trl-lib/tldr", split="train")


    >>> def reward_func(completions, **kwargs):
    ...     return [float(len(c) % 2 == 0) for c in completions]


    >>> config = PerturbedParameterGRPOConfig(
    ...     perturbation_sigma=1e-2,  # 0.0 recovers standard GRPO
    ... )

    >>> trainer = PerturbedParameterGRPOTrainer(
    ...     model="Qwen/Qwen2.5-Math-7B",
    ...     reward_funcs=reward_func,
    ...     args=config,
    ...     train_dataset=dataset,
    ... )

    >>> trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards (same as GRPO).
        args ([`PerturbedParameterGRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training.
        eval_dataset: Same requirements as train_dataset.
        processing_class: Processing class (tokenizer/processor) for the model.
        reward_processing_classes: Processing classes for reward models.
        callbacks: Training callbacks.
        optimizers: Optimizer and scheduler tuple.
        peft_config: PEFT configuration if using parameter-efficient fine-tuning.
    """

    _tag_names = ["trl", "perturbed-parameter-grpo"]
    _name = "PerturbedParameterGRPO"
    _paper = {
        "title": "Parameter Exploration for RLVR via Variational Learning",
        "id": "2608.09805",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @misc{parameter_exploration_rlvr_2026,
                title        = {{Parameter Exploration for RLVR via Variational Learning}},
                author       = {{3PO Authors}},
                year         = 2026,
                url          = {https://arxiv.org/abs/2608.09805},
                archivePrefix= {arXiv},
                eprint       = {2608.09805},
                primaryClass = {cs.LG}
            }"""
        ),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: PerturbedParameterGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = PerturbedParameterGRPOConfig(f"{model_name}-PerturbedParameterGRPO")

        self.perturbation_sigma = args.perturbation_sigma
        self.perturbation_min_std = args.perturbation_min_std

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        # Fall back to standard GRPO generation when exploration is disabled, or on paths whose weight
        # syncing is not handled here (vLLM, continuous batching, and VLM multimodal rollouts).
        if (
            self.perturbation_sigma <= 0.0
            or self.use_vllm
            or self.use_transformers_continuous_batching
            or self._is_vlm
        ):
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

        # Sample a perturbed behavior policy theta_tilde, keeping the originals to restore exactly so the
        # optimizer step still updates the base policy theta.
        saved_parameters = self._perturb_parameters()
        try:
            completion_ids, _ = super()._generate_single_turn(prompt_ids, images, multimodal_fields)
            # Capture log pi(theta_tilde) for the sampled tokens; this becomes the importance-sampling
            # reference (old_per_token_logps) in the loss, so the GRPO ratio is pi(theta)/pi(theta_tilde).
            behavior_logps = self._compute_behavior_logps(prompt_ids, completion_ids)
        finally:
            self._restore_parameters(saved_parameters)
        return completion_ids, behavior_logps

    def _perturb_parameters(self) -> dict[str, torch.Tensor]:
        saved_parameters = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                saved_parameters[name] = param.detach().clone()
                scale = torch.clamp(param.std(), min=self.perturbation_min_std)
                param.add_(self.perturbation_sigma * scale * torch.randn_like(param))
        return saved_parameters

    def _restore_parameters(self, saved_parameters: dict[str, torch.Tensor]) -> None:
        for name, param in self.model.named_parameters():
            if name in saved_parameters:
                param.data.copy_(saved_parameters[name])

    def _compute_behavior_logps(self, prompt_ids, completion_ids) -> list[list[float]]:
        device = self.accelerator.device
        prompt_completion_ids = [
            torch.tensor(p + c, dtype=torch.long, device=device)
            for p, c in zip(prompt_ids, completion_ids, strict=True)
        ]
        attention_mask = [torch.ones_like(ids) for ids in prompt_completion_ids]
        padded_ids = pad(prompt_completion_ids, padding_value=self._tokenizer.pad_token_id, padding_side="left")
        padded_mask = pad(attention_mask, padding_value=0, padding_side="left")
        completion_lengths = [len(c) for c in completion_ids]
        logits_to_keep = max(completion_lengths) if completion_lengths else 0
        if logits_to_keep == 0:
            return [[] for _ in completion_ids]

        with torch.no_grad():
            logps, _, _ = self._get_per_token_logps_and_entropies(
                self.model, padded_ids.to(device), padded_mask.to(device), logits_to_keep
            )

        # Completion tokens are right-aligned within the kept slice (we left-pad the prompt), so each
        # sequence's behavior log-probs are the last len(completion) entries, returned in token order.
        behavior_logps = []
        for n, row in zip(completion_lengths, logps, strict=True):
            behavior_logps.append(row[-n:].tolist() if n > 0 else [])
        return behavior_logps

    def _compute_loss(self, model, inputs):
        # Use the perturbed behavior policy as the importance-sampling reference, overriding the base
        # policy log-probs that GRPO would otherwise recompute under theta. With perturbation disabled
        # (or on a delegated path with no behavior log-probs), this is a no-op and GRPO is unchanged.
        behavior_logps = inputs.get("sampling_per_token_logps")
        if self.perturbation_sigma > 0.0 and behavior_logps is not None:
            inputs["old_per_token_logps"] = behavior_logps
        return super()._compute_loss(model, inputs)
