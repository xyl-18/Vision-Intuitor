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
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`, `intuitor`"""
    intuitor_reward_method: str = "self_certainty"
    """internal intuitor signal, support `self_certainty`, `entropy`, `rlsf`"""
    intuitor_aux_format_weight: float = 0.0
    """format reward weight in final intuitor advantage combination"""
    intuitor_aux_external_weight: float = 0.0
    """external reward weight in final intuitor advantage combination"""
    intuitor_aux_length_weight: float = 0.0
    """length penalty weight in final intuitor advantage combination"""
    intuitor_aux_external_reward_key: str = "accuracy"
    """reward metric key used as external signal, e.g. `overall`, `accuracy`, `description_accuracy`"""
    intuitor_window_enabled: bool = False
    """enable sliding-window pooling for internal intuitor reward"""
    intuitor_window_size: int = 16
    """sliding window size over response tokens"""
    intuitor_window_stride: int = 8
    """sliding window stride over response tokens"""
    intuitor_window_strategy: str = "min"
    """window selection strategy, support `min` and `bottom_p_mean`"""
    intuitor_window_bottom_p: float = 0.25
    """bottom-p ratio used when intuitor_window_strategy is `bottom_p_mean`"""
    intuitor_window_top_p: Optional[float] = None
    """deprecated alias for intuitor_window_bottom_p; kept for backward compatibility"""
    intuitor_global_weight: float = 1.0
    """global (full-response mean) internal intuitor reward weight"""
    intuitor_window_weight: float = 0.0
    """window-pooled internal intuitor reward weight"""
    intuitor_internal_weight_normalize: bool = True
    """normalize global/window internal weights to sum to 1 when both are active"""
    intuitor_sr1_second_internal_weight: float = 0.0
    """second-hop internal intuitor reward weight"""
    intuitor_sr1_second_format_weight: float = 0.0
    """second-hop format reward weight"""
    intuitor_sr1_second_external_weight: float = 0.0
    """second-hop external reward weight"""
    intuitor_sr1_second_length_weight: float = 0.0
    """second-hop length penalty weight"""
    intuitor_sr1_second_external_reward_key: str = "description_accuracy"
    """second-hop external reward key, e.g. `description_accuracy`, `overall`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""

    def post_init(self):
        valid_methods = {"self_certainty", "entropy", "rlsf"}
        if self.intuitor_reward_method not in valid_methods:
            raise ValueError(
                f"Unknown intuitor_reward_method: {self.intuitor_reward_method}. "
                f"Expected one of {sorted(valid_methods)}."
            )

        for key, value in {
            "intuitor_aux_format_weight": self.intuitor_aux_format_weight,
            "intuitor_aux_external_weight": self.intuitor_aux_external_weight,
            "intuitor_aux_length_weight": self.intuitor_aux_length_weight,
            "intuitor_global_weight": self.intuitor_global_weight,
            "intuitor_window_weight": self.intuitor_window_weight,
            "intuitor_sr1_second_internal_weight": self.intuitor_sr1_second_internal_weight,
            "intuitor_sr1_second_format_weight": self.intuitor_sr1_second_format_weight,
            "intuitor_sr1_second_external_weight": self.intuitor_sr1_second_external_weight,
            "intuitor_sr1_second_length_weight": self.intuitor_sr1_second_length_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{key} must be non-negative, got {value}")

        if self.intuitor_window_weight > 0.0 and not self.intuitor_window_enabled:
            raise ValueError("intuitor_window_weight > 0 requires intuitor_window_enabled=true")
        if self.intuitor_global_weight <= 0.0 and self.intuitor_window_weight <= 0.0:
            raise ValueError("At least one of intuitor_global_weight or intuitor_window_weight must be > 0")

        if self.intuitor_window_size <= 0:
            raise ValueError(f"intuitor_window_size must be > 0, got {self.intuitor_window_size}")
        if self.intuitor_window_stride <= 0:
            raise ValueError(f"intuitor_window_stride must be > 0, got {self.intuitor_window_stride}")
        if self.intuitor_window_strategy not in {"min", "bottom_p_mean"}:
            raise ValueError(
                f"Unknown intuitor_window_strategy: {self.intuitor_window_strategy}. "
                "Expected one of ['min', 'bottom_p_mean']."
            )
        if self.intuitor_window_top_p is not None:
            self.intuitor_window_bottom_p = self.intuitor_window_top_p
        if not (0.0 < self.intuitor_window_bottom_p <= 1.0):
            raise ValueError(f"intuitor_window_bottom_p must be in (0, 1], got {self.intuitor_window_bottom_p}")


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    response_path: Optional[str] = None
    """path to save all validation responses as JSONL (e.g. ./evaluation/responses/model_name/dataset.jsonl)"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.adv_estimator = self.algorithm.adv_estimator
        self.worker.actor.intuitor_reward_method = self.algorithm.intuitor_reward_method

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
