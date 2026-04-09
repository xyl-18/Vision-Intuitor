from collections import defaultdict
from typing import Tuple

import torch

from verl.protocol import DataProto
from verl.workers.reward.function import AutoRewardManager


class SelfRewardManager(AutoRewardManager):
    """Reward manager that forwards description_answers to the reward function."""

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        has_desc = "description_answers" in data.non_tensor_batch
        has_second_mask = "second_hop__response_mask" in data.batch

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "max_response_length": int(response_ids.shape[-1]),
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            if has_desc:
                description_answer = data.non_tensor_batch["description_answers"][i]
                reward_input["description_answer"] = description_answer
                if has_second_mask:
                    second_mask = data.batch["second_hop__response_mask"][i]
                    reward_input["description_response_length"] = int(second_mask.sum().item())
                    reward_input["description_max_response_length"] = int(second_mask.shape[-1])
                else:
                    reward_input["description_response_length"] = len(
                        self.tokenizer.encode(str(description_answer), add_special_tokens=False)
                    )
                    reward_input["description_max_response_length"] = int(response_ids.shape[-1])

            score = self.reward_fn(reward_input)
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        has_desc = "description_answers" in data.non_tensor_batch
        has_second_mask = "second_hop__response_mask" in data.batch

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "max_response_length": int(response_ids.shape[-1]),
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            if has_desc:
                description_answer = data.non_tensor_batch["description_answers"][i]
                reward_input["description_answer"] = description_answer
                if has_second_mask:
                    second_mask = data.batch["second_hop__response_mask"][i]
                    reward_input["description_response_length"] = int(second_mask.sum().item())
                    reward_input["description_max_response_length"] = int(second_mask.shape[-1])
                else:
                    reward_input["description_response_length"] = len(
                        self.tokenizer.encode(str(description_answer), add_special_tokens=False)
                    )
                    reward_input["description_max_response_length"] = int(response_ids.shape[-1])

            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
