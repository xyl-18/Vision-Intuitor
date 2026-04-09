#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=${1:-/root/autodl-tmp/code/Vision-SR1-main/LLaMA-Factory-Cold-Start/saves/sft_models/Vision-SR1-Cold-Start/checkpoint-1500-for-rl}
AUX_FORMAT_WEIGHT=${AUX_FORMAT_WEIGHT:-0.1}
AUX_EXTERNAL_WEIGHT=${AUX_EXTERNAL_WEIGHT:-0.6}
AUX_LENGTH_WEIGHT=${AUX_LENGTH_WEIGHT:-0.1}
AUX_EXTERNAL_KEY=${AUX_EXTERNAL_KEY:-accuracy}
INTUITOR_REWARD_METHOD=${INTUITOR_REWARD_METHOD:-self_certainty} # supported methods: self_certainty, entropy
INTUITOR_WINDOW_ENABLED=${INTUITOR_WINDOW_ENABLED:-false}
WINDOW_SIZE=${WINDOW_SIZE:-10}
WINDOW_STRIDE=${WINDOW_STRIDE:-5}
SELECT_WINDOW_STRATEGY=${SELECT_WINDOW_STRATEGY:-bottom_p_mean} # supported: min, bottom_p_mean
WINDOW_BOTTOM_P=${WINDOW_BOTTOM_P:-0.1}

python3 -m vision_intuitor_lora.main \
    config=vision_intuitor_lora/config.yaml \
    data.train_files=/root/autodl-tmp/code/Vision-SR1-main/data/Vision-SR1-10K \
    data.val_files=zli12321/mmstar@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.n=8 \
    algorithm.intuitor_reward_method=${INTUITOR_REWARD_METHOD} \
    algorithm.intuitor_aux_format_weight=${AUX_FORMAT_WEIGHT} \
    algorithm.intuitor_aux_external_weight=${AUX_EXTERNAL_WEIGHT} \
    algorithm.intuitor_aux_length_weight=${AUX_LENGTH_WEIGHT} \
    algorithm.intuitor_aux_external_reward_key=${AUX_EXTERNAL_KEY} \
    algorithm.intuitor_window_enabled=${INTUITOR_WINDOW_ENABLED} \
    algorithm.intuitor_window_size=${WINDOW_SIZE} \
    algorithm.intuitor_window_stride=${WINDOW_STRIDE} \
    algorithm.intuitor_window_strategy=${SELECT_WINDOW_STRATEGY} \
    algorithm.intuitor_window_bottom_p=${WINDOW_BOTTOM_P} \
    trainer.total_epochs=1 \
    trainer.experiment_name=v3.3_qwen2_5_vl_3b_visionIntuitor_lora \
    trainer.save_checkpoint_path=./saves/v3.3_3b_intuitor_lora \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=true \
    trainer.val_only=false
