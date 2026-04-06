#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=${1:-/root/autodl-tmp/code/Vision-SR1-main/LLaMA-Factory-Cold-Start/saves/sft_models/Vision-SR1-Cold-Start/checkpoint-1500-for-rl}
AUX_FORMAT_WEIGHT=${AUX_FORMAT_WEIGHT:-0.1}
AUX_EXTERNAL_WEIGHT=${AUX_EXTERNAL_WEIGHT:-0.0}
AUX_LENGTH_WEIGHT=${AUX_LENGTH_WEIGHT:-0.3}
AUX_EXTERNAL_KEY=${AUX_EXTERNAL_KEY:-accuracy}
INTUITOR_REWARD_METHOD=${INTUITOR_REWARD_METHOD:-self_certainty} # supported methods: self_certainty, entropy

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
    trainer.total_epochs=1 \
    trainer.experiment_name=v3.2_qwen2_5_vl_3b_visionIntuitor_lora \
    trainer.save_checkpoint_path=./saves/v3.2_3b_intuitor_lora \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=true \
    trainer.val_only=false
