### 此文档用于记录科研目标与进展
### 目标是将内部反馈机制从LLM中迁移至VLM，查看是否有用，同时进一步改进，故分为两节

## 1.复现，迁移至LLM
预实践的内部反馈机制有entropy、length、intuitor、TTRL
计划在三个vlm_grpo策略上实践intuitor，分别是vision_r1_lora、vision_sr1_lora，和FAST-GRPO
目前已经是实现
vision_r1_lora->vision_intuitor_lora（已实现1）和vision_sr1_lora->vision_intuitor_sr1_lora （已实现2）,包括entropy、length、intuitor
下一步计划在上述两个框架上打通TTRL策略（未实现任务1），然后再在FAST-GRAPO中迁移上述四个内部监督机制（未实现任务2）

## 2.改进Intuitor等方法
改进方向，分为两种，
第一种，pure-Inutior策略的改进，比如将entropy、intuitor变成滑动窗口，取bottom_p or max window的置信度值作为奖励，同样迁移至三种vlm_grpo策略中（未实现任务3）
第二种，mixer-Inutior策略改进，综合之前的最佳内部反馈机制，和外部监督奖励形成混合奖励，看是否有提升（未实现任务4）
