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
        off_context_grpo_ratio (`float`, *optional*, defaults to `0.3`):
            Fraction of rollouts (0.0 to 1.0) to generate with privileged guidance. For example, with
            `off_context_grpo_ratio=0.3` and `num_generations=8`, approximately 2-3 generations per prompt will
            use guidance and the rest will be unguided. This balances the benefit of guided rollouts with the
            computational efficiency of standard GRPO.
        guidance_func (`GuidanceFunc`, *optional*):
            A callable that provides privileged information for a given prompt. The function receives a single prompt
            (string) and should return either a guidance string (e.g., a solution prefix, hint, or structured hint)
            or `None` if no guidance is available for that prompt. The guidance is appended to the original prompt
            during generation for guided rollouts only. For example:

            ```python
            def math_guidance_func(prompt: str) -> str | None:
                # Extract problem and return a helpful hint
                if "solve" in prompt.lower():
                    return "Hint: Start by identifying the key variables."
                return None
            ```
    """

    off_context_grpo_ratio: float = field(
        default=0.3,
        metadata={
            "help": "Fraction of rollouts to generate with privileged guidance. Must be between 0.0 and 1.0. "
                    "Guided rollouts help overcome learning cliffs by steering generations toward correct solutions."
        },
    )
    guidance_func: GuidanceFunc | None = field(
        default=None,
        metadata={
            "help": "A callable that provides privileged information for a given prompt. Receives a prompt string "
                    "and returns guidance string or None. Guidance is only used for guided rollouts."
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
