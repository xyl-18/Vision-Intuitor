"""Vision-Intuitor-SR1 trainer.

SR1 extension points:
1. Build and keep second-hop generation in each training/validation batch.
2. Compute second-hop internal intuitor signals (self-certainty/entropy) via
   actor recomputation, aligned with first-hop compute_log_probs.
3. Combine multi-component advantages with configurable first-hop and second-hop
   weights while keeping the base PPO update flow intact.
"""

import re
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage
from verl.trainer.intuitor_mixer import inject_outcome_scores_at_eos
from verl.trainer.metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
)
from verl.utils import torch_functional as VF
from verl.utils.logger import Tracker
from verl.utils.py_functional import convert_dict_to_str, timer, unflatten_dict

SECOND_HOP_PREFIX = "second_hop__"
SECOND_HOP_MAX_PROMPT_LENGTH = 2048

SELF_REWARD_VERIFY_PROMPT = (
    "Text description: {Description}\n"
    "Question: {Question}\n"
    "You are provided a text description of a problem and a question. "
    "Determine the answer to the question based on the text description. "
    "First provide an internal step-by-step reasoning within <think> </think> "
    "tags, then provide a single word or phrase answer in \\boxed{{}}."
)


def extract_description(text: str) -> str:
    """Extract content between <description> tags, or return full text if absent."""
    match = re.search(r"<description>([\s\S]*?)</description>", text, re.DOTALL)
    if not match:
        return text
    return match.group(1).strip()


class IntuitorSR1Trainer(RayPPOTrainer):
    """RayPPOTrainer + SR1 second-hop data flow with dual-hop intuitor advantages."""

    def _decode_response_texts(self, response_ids: torch.Tensor) -> list[str]:
        if response_ids.dim() == 3:
            response_ids = response_ids.view(-1, response_ids.size(-1))
        return self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _build_second_hop_prompts(self, first_hop_texts: list[str], questions: list[str], n: int) -> list[str]:
        if n > 1 and len(first_hop_texts) == len(questions) * n:
            questions_expanded = [q for q in questions for _ in range(n)]
        else:
            questions_expanded = list(questions)

        prompts = []
        for response_text, question in zip(first_hop_texts, questions_expanded):
            description = extract_description(response_text)
            prompt_text = SELF_REWARD_VERIFY_PROMPT.format(
                Description=description,
                Question=question.replace("<image>", "").strip(),
            )
            prompts.append(prompt_text)

        return prompts

    def _generate_second_hop(
        self,
        gen_batch_output: DataProto,
        questions: list[str],
        n: int,
    ) -> tuple[list[str], DataProto]:
        """Generate text-only answers from first-hop descriptions."""
        with torch.no_grad():
            first_hop_texts = self._decode_response_texts(gen_batch_output.batch["responses"])
            second_hop_prompts = self._build_second_hop_prompts(first_hop_texts, questions, n)

            max_prompt_len = min(SECOND_HOP_MAX_PROMPT_LENGTH, self.config.data.max_prompt_length)
            chat_prompts = []
            raw_ids_list = []
            for prompt in second_hop_prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                chat_prompts.append(chat_prompt)
                raw_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)
                if len(raw_ids) > max_prompt_len:
                    raw_ids = raw_ids[:max_prompt_len]
                raw_ids_list.append(raw_ids)

            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer(
                chat_prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = original_padding_side

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            if (
                self.processor is not None
                and hasattr(self.processor, "image_processor")
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from verl.models.transformers.qwen3_vl import get_rope_index
                else:
                    from verl.models.transformers.qwen2_vl import get_rope_index

                batch_position_ids = []
                seq_len = input_ids.shape[1]
                for i in range(input_ids.shape[0]):
                    sample_input_ids = input_ids[i]
                    sample_attention_mask = attention_mask[i]
                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=sample_input_ids,
                        image_grid_thw=None,
                        video_grid_thw=None,
                        second_per_grid_ts=None,
                        attention_mask=sample_attention_mask,
                    )  # (3, seq_len)
                    text_position_ids = torch.clip(sample_attention_mask.long().cumsum(-1) - 1, min=0).unsqueeze(0)
                    text_position_ids.masked_fill_(sample_attention_mask.unsqueeze(0) == 0, 0)
                    sample_position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
                    if sample_position_ids.shape[-1] != seq_len:
                        sample_position_ids = sample_position_ids[..., -seq_len:]
                    batch_position_ids.append(sample_position_ids)

                position_ids = torch.stack(batch_position_ids, dim=0)  # (bs, 4, seq_len)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=1) - 1, min=0)
            raw_prompt_ids = np.array(raw_ids_list, dtype=object)

            second_gen_batch = DataProto.from_single_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "raw_prompt_ids": raw_prompt_ids,
                },
                meta_info={
                    "n": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "min_pixels": self.config.data.min_pixels,
                    "max_pixels": self.config.data.max_pixels,
                    "video_fps": self.config.data.video_fps,
                },
            )

            second_gen_batch, pad_size = pad_dataproto_to_divisor(
                second_gen_batch,
                self.actor_rollout_ref_wg.world_size,
            )
            second_out = self.actor_rollout_ref_wg.generate_sequences(second_gen_batch)
            second_out = unpad_dataproto(second_out, pad_size=pad_size)

            return self._decode_response_texts(second_out.batch["responses"]), second_out

    def _attach_second_hop_batch(self, batch: DataProto, second_hop_batch: DataProto) -> None:
        """Attach second-hop tensors to batch with a safe key prefix."""
        for key, value in second_hop_batch.batch.items():
            batch.batch[f"{SECOND_HOP_PREFIX}{key}"] = value

    def _extract_second_hop_batch(self, batch: DataProto) -> Optional[DataProto]:
        """Extract prefixed second-hop tensors into a standalone DataProto."""
        tensors = {}
        for key in batch.batch.keys():
            if key.startswith(SECOND_HOP_PREFIX):
                tensors[key[len(SECOND_HOP_PREFIX) :] ] = batch.batch[key]

        if not tensors:
            return None

        return DataProto.from_dict(tensors=tensors)

    def _log_entropy_metric(self, metrics: dict[str, Any], entropys: torch.Tensor, mask: torch.Tensor, name: str) -> None:
        response_entropy = VF.masked_mean(entropys.detach(), mask=mask.to(entropys.dtype), dim=-1)
        response_entropy = torch.nan_to_num(response_entropy, nan=0.0)
        metrics[f"{name}/mean"] = response_entropy.mean().item()
        metrics[f"{name}/max"] = response_entropy.max().item()
        metrics[f"{name}/min"] = response_entropy.min().item()

    def _scalar_advantage_to_token_advantage(
        self,
        scalar_scores: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
    ) -> torch.Tensor:
        token_level_scores = inject_outcome_scores_at_eos(response_mask, scalar_scores)
        component_advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_scores,
            response_mask=response_mask,
            index=index,
        )
        return component_advantages

    def _scalar_advantage_to_full_mask_advantage(
        self,
        scalar_scores: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        id2score: dict[str, list[torch.Tensor]] = defaultdict(list)
        bsz = scalar_scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scalar_scores[i])

        id2mean, id2std = {}, {}
        for idx, vals in id2score.items():
            t = torch.stack(vals)
            id2mean[idx] = t.mean()
            id2std[idx] = t.std()

        normed = scalar_scores.clone()
        for i in range(bsz):
            normed[i] = (scalar_scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

        return normed.unsqueeze(-1) * response_mask

    def _metric_scores(self, reward_metrics: dict[str, list[float]], key: str, batch_size: int, device: torch.device) -> torch.Tensor:
        values = reward_metrics.get(key, None)
        if values is None:
            return torch.zeros(batch_size, dtype=torch.float32, device=device)
        return torch.nan_to_num(torch.tensor(values, dtype=torch.float32, device=device), nan=0.0)

    def _compute_sr1_intuitor_advantages(
        self,
        batch: DataProto,
        reward_metrics: dict[str, list[float]],
        metrics: dict[str, Any],
    ) -> None:
        """Compute weighted SR1 intuitor advantages with first-hop and second-hop components."""
        # first-hop components
        response_mask = batch.batch["response_mask"]
        mask1 = response_mask.to(torch.float32)
        index = batch.non_tensor_batch["uid"]
        device = mask1.device
        batch_size = mask1.shape[0]
        # internal
        internal1_token = batch.batch["intuitor_internal_token_level_scores"].detach()
        internal1 = VF.masked_mean(internal1_token, mask=mask1, dim=-1)
        internal1 = torch.nan_to_num(internal1, nan=0.0)
        # external
        format1 = self._metric_scores(reward_metrics, "format", batch_size, device)
        external1 = self._metric_scores(reward_metrics, self.intuitor_aux_mixer.external_reward_key, batch_size, device,)
        length1 = self._metric_scores(reward_metrics, "length", batch_size, device)
            
        # second-hop components
        mask2 = batch.batch.get("second_hop__response_mask", None)
        # internal
        internal2_token = batch.batch.get("second_hop__intuitor_internal_token_level_scores", None)
        if mask2 is not None and internal2_token is not None:
            internal2 = VF.masked_mean(internal2_token.detach(), mask=mask2, dim=-1)
            internal2 = torch.nan_to_num(internal2, nan=0.0)
        else:
            internal2 = torch.zeros(batch_size, dtype=torch.float32, device=device)
        # external
        format2 = self._metric_scores(reward_metrics, "description_format", batch_size, device)
        external2 = self._metric_scores(reward_metrics, self.config.algorithm.intuitor_sr1_second_external_reward_key, batch_size, device)
        second_length_scores = self._metric_scores(reward_metrics, "description_length", batch_size, device)
        
        # weights
        w1_format = self.intuitor_aux_mixer.format_weight
        w1_external = self.intuitor_aux_mixer.external_weight
        w1_length = self.intuitor_aux_mixer.length_weight
        w1_internal = self.intuitor_aux_mixer.intuitor_weight

        w2_internal = float(getattr(self.config.algorithm, "intuitor_sr1_second_internal_weight", 0.0))
        w2_format = float(getattr(self.config.algorithm, "intuitor_sr1_second_format_weight", 0.0))
        w2_external = float(getattr(self.config.algorithm, "intuitor_sr1_second_external_weight", 0.0))
        w2_length = float(getattr(self.config.algorithm, "intuitor_sr1_second_length_weight", 0.0))

        # combine advantages with weights
        component_specs: list[tuple[torch.Tensor, float, str, bool]] = [
            (internal1, w1_internal, "intuitor_internal_first", True),
            (format1, w1_format, "format_first", False),
            (external1, w1_external, "external_first", False),
            (length1, w1_length, "length_first", False),
            (internal2, w2_internal, "intuitor_internal_second", True),
            (format2, w2_format, "format_second", False),
            (external2, w2_external, "external_second", False),
            (second_length_scores, w2_length, "length_second", False),
        ]

        # compute combined advantage and log component metrics
        combined_adv = torch.zeros_like(mask1)
        active = []
        for scalar_scores, weight, name, use_eos_injection in component_specs:
            if weight <= 0.0:
                continue
            if use_eos_injection:
                adv = self._scalar_advantage_to_token_advantage(scalar_scores, response_mask, index)
            else:
                adv = self._scalar_advantage_to_full_mask_advantage(scalar_scores, mask1, index)
            combined_adv = combined_adv + weight * adv
            active.append(name)
            metrics[f"reward_component/{name}_mean"] = scalar_scores.mean().item()
            metrics[f"reward_component/{name}_adv_mean"] = VF.masked_mean(adv, mask=mask1, dim=-1).mean().item()
            metrics[f"reward_component/{name}_projection_is_eos"] = 1.0 if use_eos_injection else 0.0

        batch.batch["advantages"] = combined_adv
        batch.batch["returns"] = combined_adv

        metrics["algorithm/intuitor_weight"] = w1_internal
        metrics["algorithm/format_weight"] = w1_format
        metrics["algorithm/external_weight"] = w1_external
        metrics["algorithm/length_weight"] = w1_length
        metrics["algorithm/sr1_second_internal_weight"] = w2_internal
        metrics["algorithm/sr1_second_format_weight"] = w2_format
        metrics["algorithm/sr1_second_external_weight"] = w2_external
        metrics["algorithm/sr1_second_length_weight"] = w2_length
        metrics["algorithm/sr1_active_components"] = ",".join(active)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)

        print("Start validation (with second-hop)...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )

            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch,
                self.actor_rollout_ref_wg.world_size,
            )
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            questions = test_batch.non_tensor_batch.get("question")
            if questions is None:
                description_answers = [""] * len(test_output_gen_batch)
            else:
                description_answers, _ = self._generate_second_hop(
                    test_output_gen_batch,
                    questions.tolist(),
                    repeat_times,
                )

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch.non_tensor_batch["description_answers"] = np.array(description_answers, dtype=object)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch["prompts"]]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch["responses"]
            ]
            scores = reward_tensor.sum(-1).cpu().tolist()

            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self._maybe_save_responses(sample_inputs, sample_outputs, sample_labels, sample_scores)

        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        n = self.config.worker.rollout.n

        print("Start generating batch (with second-hop)...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                dtype=object,
            )

            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            questions = new_batch.non_tensor_batch.get("question")
            if questions is None:
                description_answers = [""] * len(gen_batch_output)
                second_hop_batch = None
            else:
                description_answers, second_hop_batch = self._generate_second_hop(gen_batch_output, questions.tolist(), n)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=n, interleave=True)
            new_batch.non_tensor_batch["description_answers"] = np.array(description_answers, dtype=object)
            new_batch = new_batch.union(gen_batch_output)
            if second_hop_batch is not None:
                self._attach_second_hop_batch(new_batch, second_hop_batch)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for key, value in reward_metrics.items():
                    all_metrics[key].extend(value)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low
                    and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. "
                        "Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                return batch[: rollout_batch_size * n]

    def fit(self):
        """Training loop with SR1 dual-hop intuitor advantage computation."""
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            reward_metrics: dict[str, list[float]] = {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                    self._log_entropy_metric(
                        metrics,
                        entropys=batch.batch["entropys"],
                        mask=batch.batch["response_mask"],
                        name="actor/response_entropy",
                    )

                    second_hop_batch = self._extract_second_hop_batch(batch)
                    if second_hop_batch is not None:
                        second_log_probs = self.actor_rollout_ref_wg.compute_log_probs(second_hop_batch)
                        batch.batch[f"{SECOND_HOP_PREFIX}old_log_probs"] = second_log_probs.batch["old_log_probs"]
                        batch.batch[f"{SECOND_HOP_PREFIX}entropys"] = second_log_probs.batch["entropys"]
                        if "intuitor_internal_token_level_scores" in second_log_probs.batch:
                            batch.batch[f"{SECOND_HOP_PREFIX}intuitor_internal_token_level_scores"] = (
                                second_log_probs.batch["intuitor_internal_token_level_scores"]
                            )
                        if "response_mask" in second_hop_batch.batch and "entropys" in second_log_probs.batch:
                            self._log_entropy_metric(
                                metrics,
                                entropys=second_log_probs.batch["entropys"],
                                mask=second_hop_batch.batch["response_mask"],
                                name="actor/response_entropy_second",
                            )

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})

                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.INTUITOR:
                        self._compute_sr1_intuitor_advantages(
                            batch=batch,
                            reward_metrics=reward_metrics,
                            metrics=metrics,
                        )
                    else:
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.non_tensor_batch))

                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.non_tensor_batch))

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
