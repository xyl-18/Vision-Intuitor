nohup bash -c "
echo '===== checkpoint eval 1 =====';
bash evaluation/full_rl/eval_vision_r1.sh;
" > eval.log 2>&1 &

# nohup bash -c "
# echo '===== checkpoint eval 1 =====';
# bash evaluation/lora_rl/eval_vision_sr1_lora.sh /root/autodl-tmp/code/Vision-SR1-main/saves/3b_grpo_self_reward_lora/global_step_60;
# echo '===== checkpoint eval 2 =====';
# bash evaluation/lora_rl/eval_vision_r1_lora.sh /root/autodl-tmp/code/Vision-SR1-main/saves/3b_grpo_accuracy_lora/global_step_45;
# " > eval.log 2>&1 &