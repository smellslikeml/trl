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

from collections.abc import Callable
from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


GuidanceFunc = Callable[[str], str | None]


@dataclass
class OffContextGRPOConfig(GRPOConfig):
    r"""
    Configuration class for [`OffContextGRPOTrainer`].

    This is an extension of [`GRPOConfig`] with parameters specific to Off-Context GRPO (OC-GRPO).

    OC-GRPO addresses the learning cliff problem in reinforcement learning with verifiable rewards (RLVR). When a
    model cannot generate any correct solutions on difficult problems, it receives zero learning signal. OC-GRPO
    overcomes this by using privileged information (e.g., solution prefixes or hints) during training to guide
    generations toward correct solutions, while applying importance correction to steer the update back toward
    the original unguided objective.

    This is an experimental implementation of the methods described in the paper [Off-Context GRPO: Learning to
    Reason on Hard Problems using Privileged Information](https://huggingface.co/papers/2607.19313v1).

    Parameters:
        off_context_grpo_ratio (`float`, *optional*, defaults to `0.0`):
            **[DEPRECATED - use hint_trigger_threshold instead]** Fraction of rollouts (0.0 to 1.0) to generate
            with privileged guidance. This parameter is kept for backward compatibility but adaptive hint
            triggering via `hint_trigger_threshold` is preferred.
        guidance_func (`GuidanceFunc`, *optional*):
            A callable that provides privileged information for a given prompt. The function receives a single prompt
            (string or list of dict) and should return either a guidance string (e.g., a solution prefix, hint, or
            structured hint) or `None` if no guidance is available for that prompt. The guidance is prepended to the
            original prompt during generation for guided rollouts only. For example:

            ```python
            def math_guidance_func(prompt: str) -> str | None:
                # Extract problem and return a helpful hint
                if "solve" in prompt.lower():
                    return "Hint: Start by identifying the key variables."
                return None
            ```
        hint_trigger_threshold (`int`, *optional*, defaults to `0`):
            Adaptive hint triggering threshold. Hints are generated only for groups where the number of correct
            completions (`n_correct`) is less than or equal to this threshold. A value of 0 means hints are only
            used when the model produces zero correct solutions for a problem (all-zero reward groups). This focuses
            guidance on learning cliffs where the model consistently fails, rather than on problems the model can
            already solve.
        hint_trigger_mode (`str`, *optional*, defaults to `"simple"`):
            Mode for adaptive hint triggering. Options:
            - `"simple"`: Hints are triggered when `n_correct <= hint_trigger_threshold` across all trajectories.
            - `"two_phase"`: Two-phase trigger — hints are triggered only when the first half of trajectories are
              all incorrect (`first_half_correct == 0`) AND the full set meets `n_correct <= hint_trigger_threshold`.
              This stricter mode ensures hints are used only when the model consistently fails, reducing false
              positives on stochastic successes.
    """

    off_context_grpo_ratio: float = field(
        default=0.0,
        metadata={
            "help": "[DEPRECATED - use hint_trigger_threshold] Fraction of rollouts to generate with privileged guidance."
        },
    )
    guidance_func: GuidanceFunc | None = field(
        default=None,
        metadata={
            "help": "A callable that provides privileged information for a given prompt. Receives a prompt "
                    "(string or list of dict) and returns guidance string or None. Guidance is prepended to prompts."
        },
    )
    hint_trigger_threshold: int = field(
        default=0,
        metadata={
            "help": "Adaptive hint triggering threshold. Hints are generated only for groups where n_correct <= "
                    "this threshold. Default 0 means only zero-reward groups (all incorrect) receive hints."
        },
    )
    hint_trigger_mode: str = field(
        default="simple",
        metadata={
            "help": "Mode for hint triggering: 'simple' (n_correct <= threshold) or 'two_phase' (first_half "
                    "all-zero AND n_correct <= threshold). Default is 'simple'."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.off_context_grpo_ratio <= 1.0:
            raise ValueError(
                f"off_context_grpo_ratio must be between 0.0 and 1.0, got {self.off_context_grpo_ratio}"
            )
        if self.off_context_grpo_ratio > 0.0 and self.guidance_func is None:
            raise ValueError(
                "guidance_func must be provided when off_context_grpo_ratio > 0.0"
            )
        if self.hint_trigger_mode not in ("simple", "two_phase"):
            raise ValueError(
                f"hint_trigger_mode must be 'simple' or 'two_phase', got {self.hint_trigger_mode}"
            )
        if self.hint_trigger_threshold < 0:
            raise ValueError(
                f"hint_trigger_threshold must be >= 0, got {self.hint_trigger_threshold}"
            )
