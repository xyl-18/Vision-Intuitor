# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any

import torch


def inject_outcome_scores_at_eos(response_mask: torch.Tensor, outcome_scores: torch.Tensor) -> torch.Tensor:
    """Convert per-response scalar scores into token-level scores at EOS positions."""
    outcome_scores = outcome_scores.to(response_mask.device)
    eos_idx = response_mask.long().sum(dim=-1).sub(1).clamp_min(0)
    token_scores = torch.zeros(response_mask.shape, dtype=outcome_scores.dtype, device=response_mask.device)
    token_scores.scatter_(-1, eos_idx.unsqueeze(-1), outcome_scores.unsqueeze(-1))
    return token_scores


@dataclass
class IntuitorAuxRewardMixer:
    """Compose auxiliary outcome signals for Intuitor advantage fusion.

    The mixer builds a single auxiliary outcome score per sample from configurable
    components (format, external reward and length penalty), then projects it to token-level EOS
    positions for GRPO-style group normalization.
    """

    format_weight: float = 0.1
    external_weight: float = 0.0
    length_weight: float = 0.0
    external_reward_key: str = "accuracy"

    @classmethod
    def from_algorithm_config(cls, algorithm_config: Any) -> "IntuitorAuxRewardMixer":
        format_weight = float(getattr(algorithm_config, "intuitor_aux_format_weight", 0.1))
        external_weight = float(getattr(algorithm_config, "intuitor_aux_external_weight", 0.0))
        length_weight = float(getattr(algorithm_config, "intuitor_aux_length_weight", 0.0))
        external_reward_key = str(getattr(algorithm_config, "intuitor_aux_external_reward_key", "accuracy"))

        if format_weight < 0.0 or external_weight < 0.0 or length_weight < 0.0:
            raise ValueError(
                "`intuitor_aux_format_weight`, `intuitor_aux_external_weight`, "
                "and `intuitor_aux_length_weight` must be >= 0."
            )
        if format_weight + external_weight + length_weight > 1.0:
            raise ValueError(
                "`intuitor_aux_format_weight + intuitor_aux_external_weight + intuitor_aux_length_weight` "
                "must be <= 1."
            )

        return cls(
            format_weight=format_weight,
            external_weight=external_weight,
            length_weight=length_weight,
            external_reward_key=external_reward_key,
        )

    @property
    def intuitor_weight(self) -> float:
        return max(0.0, 1.0 - self.format_weight - self.external_weight - self.length_weight)

    @property
    def enabled(self) -> bool:
        return self.format_weight > 0.0 or self.external_weight > 0.0 or self.length_weight > 0.0

    def _summarize_tensor(self, values: torch.Tensor, prefix: str) -> dict[str, float]:
        values = torch.nan_to_num(values, nan=0.0)
        return {
            f"{prefix}_mean": values.mean().item(),
            f"{prefix}_max": values.max().item(),
            f"{prefix}_min": values.min().item(),
        }

    def _metric_to_tensor(
        self,
        reward_metrics: dict[str, list[float]],
        key: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = reward_metrics.get(key, None)
        if values is None:
            return torch.zeros(batch_size, dtype=torch.float32, device=device)
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        return torch.nan_to_num(tensor, nan=0.0)

    def build_component_token_level_scores(
        self,
        reward_tensor: torch.Tensor,
        reward_metrics: dict[str, list[float]],
        response_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Build format/external component token-level scores and diagnostics for logging."""
        batch_size = reward_tensor.shape[0]
        device = reward_tensor.device

        component_scores: dict[str, torch.Tensor] = {}
        metrics = {
            "algorithm/intuitor_weight": self.intuitor_weight,
            "algorithm/format_weight": self.format_weight,
            "algorithm/external_weight": self.external_weight,
            "algorithm/length_weight": self.length_weight,
        }

        if self.format_weight > 0.0:
            format_scores = self._metric_to_tensor(reward_metrics, "format", batch_size, device)
            component_scores["format"] = inject_outcome_scores_at_eos(response_mask, format_scores)
            metrics.update(self._summarize_tensor(format_scores, "reward_component/format"))

        if self.external_weight > 0.0:
            if self.external_reward_key == "overall":
                metric_overall = reward_metrics.get("overall", None)
                if metric_overall is not None:
                    external_scores = torch.tensor(metric_overall, dtype=torch.float32, device=device)
                else:
                    external_scores = reward_tensor.sum(dim=-1)
            else:
                external_scores = self._metric_to_tensor(
                    reward_metrics, self.external_reward_key, batch_size, device
                )
            external_scores = torch.nan_to_num(external_scores, nan=0.0)
            component_scores["external"] = inject_outcome_scores_at_eos(response_mask, external_scores)
            metrics.update(self._summarize_tensor(external_scores, "reward_component/external"))

        if self.length_weight > 0.0:
            if "length" in reward_metrics:
                length_scores = self._metric_to_tensor(reward_metrics, "length", batch_size, device)
            else:
                response_lengths = response_mask.sum(dim=-1).float()
                max_response_length = float(response_mask.shape[-1])
                length_scores = -response_lengths / max_response_length
            component_scores["length"] = inject_outcome_scores_at_eos(response_mask, length_scores)
            metrics.update(self._summarize_tensor(length_scores, "reward_component/length"))

        return component_scores, metrics
