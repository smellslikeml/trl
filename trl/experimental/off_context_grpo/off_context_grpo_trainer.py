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

from typing import Any

import torch

from ...data_utils import is_conversational
from ...trainer.grpo_trainer import GRPOTrainer
from .off_context_grpo_config import OffContextGRPOConfig


class OffContextGRPOTrainer(GRPOTrainer):
    """
    Trainer for Off-Context GRPO (OC-GRPO).

    OC-GRPO addresses the learning cliff problem in reinforcement learning with verifiable rewards. When a model
    cannot generate correct solutions on hard problems, it receives zero learning signal. OC-GRPO overcomes this by
    using privileged information (e.g., solution prefixes) during training to guide generations toward correct
    solutions, while applying importance correction to steer the update back toward the original unguided objective.

    This trainer extends [`GRPOTrainer`] with the following modifications:

    1. Adaptive hint triggering: After initial generation, identifies zero-reward groups where `n_correct <=
       `hint_trigger_threshold` (default 0) and regenerates those groups with privileged guidance. This focuses
       assistance on learning cliffs where the model consistently fails.
    2. Two-phase trigger mode: When `hint_trigger_mode="two_phase"`, hints are only triggered when the first half
       of trajectories are all incorrect AND the full set meets the threshold condition.
    3. Importance sampling correction is applied to guided completions to account for the distribution shift between
       guided and unguided generation.

    Example:

    ```python
    >>> from trl.experimental import OffContextGRPOTrainer
    >>> from trl.rewards import accuracy_reward
    >>> from datasets import load_dataset

    >>> def math_guidance(prompt: str) -> str | None:
    ...     # Provide hints for math problems
    ...     if "solve" in prompt.lower():
    ...         return "First, identify the key variables. "
    ...     return None

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = OffContextGRPOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     reward_funcs=accuracy_reward,
    ...     train_dataset=dataset,
    ...     args=OffContextGRPOConfig(
    ...         hint_trigger_threshold=0,  # Only hint when all completions are incorrect
    ...         hint_trigger_mode="simple",  # Use "two_phase" for stricter triggering
    ...         guidance_func=math_guidance,
    ...     ),
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Passed to [`GRPOTrainer`].
        reward_funcs (`RewardFunc` or `list[RewardFunc]`):
            Reward functions for computing rewards. Passed to [`GRPOTrainer`].
        args ([`OffContextGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. Passed to [`GRPOTrainer`].
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. Passed to [`GRPOTrainer`].
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. Passed to [`GRPOTrainer`].
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes for reward models. Passed to [`GRPOTrainer`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Passed to [`GRPOTrainer`].
        optimizers (`tuple[torch.optim.Optimizer | None, torch.lr_scheduler.LambdaLR | None]`, *optional*):
            Optimizer and scheduler. Passed to [`GRPOTrainer`].
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration. Passed to [`GRPOTrainer`].
        tools (list of `Callable`, *optional*):
            Tool functions. Passed to [`GRPOTrainer`].
        rollout_func (`RolloutFunc`, *optional*):
            Rollout function. Passed to [`GRPOTrainer`].
        environment_factory (`EnvironmentFactory`, *optional*):
            Environment factory. Passed to [`GRPOTrainer`].
    """

    _tag_names = ["trl", "grpo", "experimental", "off_context_grpo"]
    _name = "OffContextGRPO"
    _paper = {
        "title": "Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information",
        "id": "2607.19313v1",
    }

    def __init__(self, args: OffContextGRPOConfig | None = None, **kwargs):
        if args is None:
            args = OffContextGRPOConfig()
        super().__init__(args=args, **kwargs)

        self.hint_trigger_threshold = args.hint_trigger_threshold
        self.hint_trigger_mode = args.hint_trigger_mode
        self.guidance_func = args.guidance_func

    def _apply_guidance_to_prompts(self, prompts: list[str | list[dict]], guided_indices: list[int]) -> list[str | list[dict]]:
        """
        Apply privileged guidance to selected prompts.

        Args:
            prompts: Original prompts (can be strings or conversational format).
            guided_indices: Indices of prompts that should receive guidance.

        Returns:
            Prompts with guidance applied to selected indices.
        """
        guided_prompts = []
        for i, prompt in enumerate(prompts):
            if i in guided_indices:
                guidance = self.guidance_func(prompt)
                if guidance is not None:
                    # Apply guidance based on prompt format
                    if is_conversational({"prompt": prompt}):
                        # Conversational format: append to last user message
                        guided_prompt = [msg.copy() for msg in prompt]  # Deep copy
                        last_user_idx = len(guided_prompt) - 1
                        while last_user_idx >= 0 and guided_prompt[last_user_idx].get("role") != "user":
                            last_user_idx -= 1
                        if last_user_idx >= 0:
                            content = guided_prompt[last_user_idx]["content"]
                            if isinstance(content, str):
                                guided_prompt[last_user_idx]["content"] = guidance + content
                            elif isinstance(content, list):
                                # Prepend guidance text to existing content
                                guided_prompt[last_user_idx]["content"] = [
                                    {"type": "text", "text": guidance}
                                ] + content
                        guided_prompts.append(guided_prompt)
                    else:
                        # String format: prepend guidance
                        guided_prompts.append(guidance + prompt)
                else:
                    # No guidance available, use original prompt
                    guided_prompts.append(prompt)
            else:
                guided_prompts.append(prompt)
        return guided_prompts

    def _regenerate_zero_reward_groups(
        self,
        generation_batch: dict[str, torch.Tensor | Any],
        zero_reward_groups: list[dict],
    ) -> dict[str, torch.Tensor | Any]:
        """
        Regenerate completions for zero-reward groups with guided prompts.

        Args:
            generation_batch: Original generation batch with prompts.
            zero_reward_groups: List of zero-reward group info dicts containing:
                - group_idx: Index of the prompt group
                - sample_indices: List of sample indices in this group

        Returns:
            Updated generation_batch with regenerated completions for zero-reward groups.
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in generation_batch]

        # Get group indices for prompting (one prompt per group)
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        num_groups = len(prompts) // num_generations

        # Identify which groups receive guidance
        guided_group_indices = [g["group_idx"] for g in zero_reward_groups]
        guided_prompt_indices = [i * num_generations for i in guided_group_indices]  # First index in each group

        # Apply guidance to selected prompts
        guided_prompts = self._apply_guidance_to_prompts(prompts, guided_prompt_indices)

        # Build new inputs for regeneration (only zero-reward groups)
        regenerate_inputs = []
        for group_info in zero_reward_groups:
            group_idx = group_info["group_idx"]
            sample_indices = group_info["sample_indices"]
            for idx in sample_indices:
                new_input = generation_batch.copy()
                new_input["prompt"] = guided_prompts[group_idx * num_generations]
                regenerate_inputs.append(new_input)

        if not regenerate_inputs:
            return generation_batch

        # Generate with guidance
        regenerated = super()._generate_and_score_completions(regenerate_inputs)

        # Update original batch with regenerated completions
        updated_batch = generation_batch.copy()
        for i, group_info in enumerate(zero_reward_groups):
            sample_indices = group_info["sample_indices"]
            regenerated_idx_offset = i * num_generations
            for j, orig_idx in enumerate(sample_indices):
                regen_idx = regenerated_idx_offset + j
                # Update all relevant fields
                for key in regenerated:
                    if key not in updated_batch:
                        continue
                    if isinstance(regenerated[key], torch.Tensor):
                        updated_batch[key][orig_idx] = regenerated[key][regen_idx]
                    elif isinstance(regenerated[key], list):
                        updated_batch[key][orig_idx] = regenerated[key][regen_idx]

        # Store which indices were guided for importance correction
        is_guided = torch.zeros(len(prompts), dtype=torch.bool, device=device)
        for group_info in zero_reward_groups:
            for idx in group_info["sample_indices"]:
                is_guided[idx] = True
        updated_batch["is_guided"] = is_guided

        return updated_batch

    def _identify_zero_reward_groups(
        self,
        generation_batch: dict[str, torch.Tensor | Any],
    ) -> list[dict]:
        """
        Identify zero-reward groups based on hint_trigger_threshold.

        Groups where n_correct <= hint_trigger_threshold are selected for hint regeneration.

        Args:
            generation_batch: Batch with computed rewards.

        Returns:
            List of zero-reward group info dicts.
        """
        device = self.accelerator.device
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        num_groups = len(generation_batch["prompt"]) // num_generations

        # Get reward sums for each completion
        if "reward" in generation_batch:
            rewards = generation_batch["reward"]
            # Sum across reward functions if multiple
            if rewards.dim() > 1:
                reward_sums = rewards.sum(dim=-1)
            else:
                reward_sums = rewards
        else:
            # No rewards computed yet (shouldn't happen in normal flow)
            return []

        zero_reward_groups = []
        for group_idx in range(num_groups):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            group_rewards = reward_sums[start_idx:end_idx]

            n_correct = int((group_rewards > 0).sum())
            n_total = num_generations

            should_hint = False
            if self.hint_trigger_mode == "two_phase":
                half = n_total // 2
                first_half_correct = int((group_rewards[:half] > 0).sum())
                if first_half_correct == 0 and n_correct <= self.hint_trigger_threshold:
                    should_hint = True
            else:  # simple mode
                if n_correct <= self.hint_trigger_threshold:
                    should_hint = True

            if should_hint:
                zero_reward_groups.append({
                    "group_idx": group_idx,
                    "sample_indices": list(range(start_idx, end_idx)),
                    "n_correct": n_correct,
                })

        return zero_reward_groups

    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        """
        Override to add adaptive hint regeneration.

        Flow:
        1. Generate initial completions without hints
        2. Compute rewards
        3. Identify zero-reward groups (n_correct <= hint_trigger_threshold)
        4. Regenerate those groups with hints
        5. Return updated batch
        """
        mode = "train" if self.model.training else "eval"

        # Skip adaptive hints if guidance_func not provided or hints disabled
        if self.guidance_func is None or self.hint_trigger_threshold < 0:
            return super()._prepare_inputs(generation_batch)

        # During training, we need to handle the generation batch differently for adaptive hints
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # First pass: generate without hints, compute rewards
                generation_batch = super()._generate_and_score_completions(generation_batch)

                # Identify zero-reward groups
                zero_reward_groups = self._identify_zero_reward_groups(generation_batch)

                # Log metrics
                num_generations = self.num_generations
                num_groups = len(generation_batch["prompt"]) // num_generations
                n_hinted = len(zero_reward_groups)
                self._metrics["train"]["hint_trigger_threshold"] = self.hint_trigger_threshold
                self._metrics["train"]["n_hint_candidates"] = n_hinted
                self._metrics["train"]["pct_hinted"] = n_hinted / max(num_groups, 1)

                # Second pass: regenerate zero-reward groups with hints
                if zero_reward_groups:
                    generation_batch = self._regenerate_zero_reward_groups(
                        generation_batch, zero_reward_groups
                    )

                from ...trainer.grpo_trainer import shuffle_sequence_dict, split_tensor_dict, split_pixel_values_by_grid, unsplit_pixel_values_by_grid

                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, generate first, then identify and regenerate
            generation_batch = super()._generate_and_score_completions(generation_batch)
            zero_reward_groups = self._identify_zero_reward_groups(generation_batch)
            if zero_reward_groups:
                generation_batch = self._regenerate_zero_reward_groups(
                    generation_batch, zero_reward_groups
                )
            inputs = generation_batch

        return inputs
