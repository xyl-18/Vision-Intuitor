# Vision-Intuitor

**内部反馈驱动的视觉语言模型后训练研究**  
Internal-Feedback Reinforcement Learning for Vision-Language Models

[项目代码](https://github.com/xyl-18/Vision-Intuitor) · [上游 Vision-SR1](https://github.com/zli12321/Vision-SR1) · [训练入口](vision_intuitor_lora/main.py) · [两阶段训练入口](vision_intuitor_sr1_lora/main.py)

## 1. 项目简介

Vision-Intuitor 是谢育林本科毕业设计《基于内部反馈监督的多模态大模型推理能力增强研究》的代码实现，在 **Vision-SR1 / verl / EasyR1** 基础上扩展内部奖励、局部奖励聚合和多奖励优势融合。

本项目关注三个研究问题：

1. **内部反馈能否用于 VLM 后训练？** 将自确定性（RLIF）、熵（EM-RL）、概率间隔（RLSF）与长度惩罚（LP）迁移至单阶段 Vision-R1 和两阶段 Vision-SR1 流程。
2. **如何避免全局奖励掩盖局部薄弱片段？** 提出滑动窗口 Min / Bottom-p 聚合，以局部统计构造序列级奖励，分析其对性能与训练退化的影响。
3. **内部奖励如何辅助答案监督？** 对内部、格式、答案、长度等分量分别计算组内标准化优势，再加权融合，比较单一奖励与混合奖励。

实验基于 **Qwen2.5-VL-3B、9K 冷启动 SFT 数据和 10K RL 题目**，由项目作者完成基线复现、方法实现、训练和评测。在 8 项基准约 11.5K 道题的样本加权平均准确率上，答案奖励 + 自确定性达到 **38.5%**（复现 Vision-R1：36.5%）；自确定性 + 长度惩罚达到 **37.9%**（仅长度惩罚：36.9%）。详细结果与适用边界见第 4 节。

“不使用答案奖励”仅指相应 **RL 奖励配置**；冷启动 SFT、验证选点和测试评估仍使用监督信息，不代表全流程无标签。

## 2. 相比原项目的改动

### 继承与新增的边界

Vision-SR1 的视觉描述与文本推理解耦思想、原始 R1/SR1 基线及冷启动数据来自上游；分布式执行、FSDP、vLLM rollout 和基础 PPO/GRPO 能力来自原有训练栈。本项目的贡献集中在 **内部反馈迁移、局部奖励建模、混合优势计算、双阶段内部信号接入及对照实验**，不将这些底层框架归为原创。

| 本项目实现 | 核心逻辑 | 代码入口 |
| --- | --- | --- |
| 多种内部信号接入 | 在 Actor 概率计算链路提取自确定性、熵或概率间隔，传递至训练器 | [fsdp_workers.py](verl/workers/fsdp_workers.py) |
| 局部窗口奖励 | 在有效响应范围内滑窗求均值，取最小窗口或最低 p 比例窗口的均值，再构造序列奖励 | [compute_intuitor_outcome_advantage](verl/trainer/core_algos.py) |
| 可配置混合奖励 | 新增辅助奖励混合器，管理格式、答案和长度分量，并按响应 mask 写入末尾有效 token | [intuitor_mixer.py](verl/trainer/intuitor_mixer.py) |
| 分量独立优势归一化 | 各分量分别执行 GRPO 组内标准化，再按权重融合；支持全局/窗口分支组合 | [core_algos.py](verl/trainer/core_algos.py) |
| 双阶段内部反馈 | 在 SR1 的视觉描述 → 纯文本推理流程中保留第二阶段采样，重算 Actor 内部信号并融合双阶段优势 | [IntuitorSR1Trainer](vision_intuitor_sr1_lora/ray_trainer.py)、[reward_manager.py](vision_intuitor_sr1_lora/reward_manager.py) |
| 训练诊断 | 记录奖励分量、组内标准差及训练指标，结合熵、响应长度和验证表现分析奖励捷径与塌陷 | [ray_trainer.py](verl/trainer/ray_trainer.py) |

窗口方法仍是**序列级奖励重构**，不是逐 token 的因果信用分配。二阶段反馈沿用 SR1 的感知—推理解耦结构，并扩展其内部信号计算与优势融合。

仓库还包含 `internal_reward_adaptive` 实验开关，用答案正确性校准高置信输出；这是代码中的探索分支，下方论文结果表不将其计为已验证增益。

## 3. 训练与评测

### 环境与数据

以下命令在 **Linux + NVIDIA GPU** 环境下从仓库根目录执行。现有 [setup.sh](setup.sh) 包含 Torch 2.9.1 / CUDA 12.8、vLLM 0.11.0、FlashAttention 2.8.3 和 Transformers 4.57.0 的安装流程，也包含系统包安装及 W&B 登录；请按机器环境检查后执行。这里不提供跨 CUDA 版本的兼容性保证。

```bash
git clone https://github.com/xyl-18/Vision-Intuitor.git
cd Vision-Intuitor
conda create -n vision-intuitor python=3.12 -y
conda activate vision-intuitor
bash setup.sh
```

- SFT 数据来自上游 [Vision-SR1-Cold-9K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-Cold-9K)。
- 论文 RL 实验使用从 [Vision-SR1-47K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-47K) 抽取的 **10K 子集**。默认 YAML 指向 47K 数据，直接使用默认数据不等价于论文设置；复现历史结果还需要相同子集。
- 数据字段为 `problem`、`answer`、`images`；验证集默认使用 `zli12321/mmstar@test`。
- 现有 `train.sh` 保留了实验机器的绝对路径及历史参数。下方直接调用 Python 入口，显式覆盖路径与关键参数，避免误用脚本默认配置。

### 冷启动 SFT

SFT 使用上游集成的 LLaMA-Factory，建议与 RL 环境分开安装。运行前检查 YAML 中数据注册、输出目录及模型设置。

```bash
python download-sft-data.py
conda create -n vision-intuitor-sft python=3.11 -y
conda activate vision-intuitor-sft
cd LLaMA-Factory-Cold-Start
pip install -e ".[torch,metrics]" --no-build-isolation
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/Vision-SR1-Cold-Start.yaml
cd ..
conda activate vision-intuitor
```

该 [SFT 配置](LLaMA-Factory-Cold-Start/examples/train_full/Vision-SR1-Cold-Start.yaml) 使用 Qwen2.5-VL-3B-Instruct，并冻结视觉塔与多模态投影层。RL 的 `MODEL_PATH` 应指向可被 Transformers 加载的冷启动模型目录，而不是未合并的分片目录。

### 单阶段内部奖励训练

下面是**配置入口示例，不是论文逐项最优检查点的精确复现配方**。论文采用验证集选点/早停；当前仓库的默认脚本参数也不完全对应历史实验。

```bash
export MODEL_PATH=/path/to/qwen2.5-vl-3b-sft
export TRAIN_DATA=/path/to/Vision-SR1-10K
export N_GPUS=4

python -m vision_intuitor_lora.main \
  config=vision_intuitor_lora/config.yaml \
  data.train_files="$TRAIN_DATA" \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.actor.model.lora.rank=64 \
  worker.actor.optim.lr=1e-5 \
  worker.rollout.n=8 \
  algorithm.intuitor_reward_method=self_certainty \
  algorithm.intuitor_weight=1.0 \
  algorithm.intuitor_aux_format_weight=0.1 \
  algorithm.intuitor_aux_external_weight=0.0 \
  algorithm.intuitor_aux_length_weight=0.0 \
  algorithm.internal_reward_adaptive=false \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_epochs=1 \
  trainer.experiment_name=vision_intuitor_rlif \
  trainer.save_checkpoint_path=./saves/vision_intuitor_rlif
```

替换或追加以下参数可以选择不同实验分支；权重是配置示例，需要独立验证，不代表表中实验的最优超参数。

| 变体 | 参数变化 |
| --- | --- |
| EM-RL | `algorithm.intuitor_reward_method=entropy` |
| RLSF | `algorithm.intuitor_reward_method=rlsf` |
| 仅长度惩罚 LP | `algorithm.intuitor_weight=0.0 algorithm.intuitor_aux_length_weight=1.0` |
| RLIF + LP | `algorithm.intuitor_aux_length_weight=0.1` |
| 答案奖励 + RLIF | `algorithm.intuitor_aux_external_weight=1.0` |
| Window-RLIF | `algorithm.intuitor_window_enabled=true algorithm.intuitor_global_weight=0.0 algorithm.intuitor_window_weight=1.0 algorithm.intuitor_window_size=32 algorithm.intuitor_window_stride=4 algorithm.intuitor_window_strategy=bottom_p_mean algorithm.intuitor_window_bottom_p=0.1` |

窗口必须同时启用 `intuitor_window_enabled` 并给予非零 `intuitor_window_weight` 才会参与计算；`min` 和 `bottom_p_mean` 分别对应最小窗口与最低 p 比例窗口平均。

### 两阶段内部反馈训练

```bash
python -m vision_intuitor_sr1_lora.main \
  config=vision_intuitor_sr1_lora/config.yaml \
  data.train_files="$TRAIN_DATA" \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.actor.model.lora.rank=64 \
  worker.rollout.n=8 \
  algorithm.intuitor_reward_method=self_certainty \
  algorithm.intuitor_weight=1.0 \
  algorithm.intuitor_aux_format_weight=0.1 \
  algorithm.intuitor_aux_external_weight=0.0 \
  algorithm.intuitor_aux_length_weight=0.0 \
  algorithm.intuitor_sr1_second_internal_weight=1.0 \
  algorithm.intuitor_sr1_second_format_weight=0.1 \
  algorithm.intuitor_sr1_second_external_weight=0.0 \
  algorithm.intuitor_sr1_second_length_weight=0.0 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_epochs=1 \
  trainer.experiment_name=vision_intuitor_sr1 \
  trainer.save_checkpoint_path=./saves/vision_intuitor_sr1
```

这里显式关闭两个阶段的答案奖励。混合监督实验可分别设置 `intuitor_aux_external_weight` 和 `intuitor_sr1_second_external_weight`；第二阶段答案奖励键由 `intuitor_sr1_second_external_reward_key` 配置，默认是 `description_accuracy`。两阶段 shell 示例主要列出自确定性/熵分支，其他组合请核对训练器实现及实际验证情况。

### 检查点评测

```bash
# CHECKPOINT_PATH 指向 global_step_* 目录；BASE_MODEL 必须匹配训练时的 SFT 底座。
export CHECKPOINT_PATH=./saves/vision_intuitor_rlif/global_step_100
export BASE_MODEL="$MODEL_PATH"
bash evaluation/lora_rl/eval_vision_intuitor_lora.sh "$CHECKPOINT_PATH" "$BASE_MODEL"

# 两阶段模型：将路径换成实际保存的 SR1 检查点。
bash evaluation/lora_rl/eval_vision_intuitor_sr1_lora.sh \
  ./saves/vision_intuitor_sr1/global_step_100 "$BASE_MODEL"
```

`global_step_100` 是路径示例，并非论文指定检查点。评测脚本当前固定 `trainer.n_gpus_per_node=2`；请在运行前根据硬件修改。两个入口使用各自匹配的输出模板，依次生成响应、提取答案并调用本地 LLM Judge。

```bash
# 已有生成结果时，可以单独重跑评判和汇总。
python evaluation/llm_judge.py \
  --response_dir /path/to/responses \
  --judge_model Qwen/Qwen2.5-14B-Instruct --tp_size 2
python evaluation/print_accuracy.py --judgment_dir /path/to/judgments
```

评测采用 greedy decoding，并区分规则准确率与 LLM-Judge 准确率。修改 Judge、提示模板或样本过滤策略会改变结果，不能与历史表格直接混用。保留每项基准的样本数和逐题输出，使用 `sum(correct) / sum(samples)` 汇总，不对表中的已四舍五入百分比直接求平均。

## 4. 实验结果

以下为本科毕业论文中 **Vision-R1 单阶段分支**的结果，不混入两阶段分支。全部模型以 Qwen2.5-VL-3B 为底座，基线由作者自行训练和评测。数值单位为准确率 %；AVG 为约 11.5K 题的样本加权平均。

| 方法 | MMMU-Pro | MMMU | MMVet | RealWorldQA | VisNum | MathVerse | MathVision | HallusionBench | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | 22.7 | 34.2 | 55.0 | 43.4 | 23.3 | 33.0 | 14.7 | 60.4 | 29.2 |
| SFT | 30.0 | 43.6 | 50.9 | 60.7 | 39.6 | 32.9 | 17.0 | 62.8 | 34.6 |
| Vision-R1（答案奖励） | 30.8 | 46.3 | 52.3 | 61.6 | 42.7 | 34.6 | 18.9 | 64.6 | 36.5 |
| RLIF | 32.7 | 45.9 | 51.8 | 58.6 | 40.8 | 33.7 | 18.5 | 66.2 | 36.0 |
| LP | 31.2 | 46.7 | 53.7 | 63.5 | 42.7 | 35.0 | 19.5 | 63.1 | 36.9 |
| Window-RLIF | 31.1 | 45.7 | 53.7 | 61.2 | 42.4 | 34.5 | 19.4 | 67.6 | 36.7 |
| RLIF + LP | 31.8 | 46.8 | 52.3 | 63.5 | 43.5 | 36.3 | 20.3 | 67.0 | 37.9 |
| Acc + LP | 32.5 | 49.8 | 53.2 | 61.7 | 45.8 | 37.9 | 21.1 | 68.6 | **39.2** |
| Acc + RLIF | 33.0 | 46.7 | 57.3 | 60.5 | 42.2 | 37.0 | 22.1 | 68.6 | 38.5 |
| Acc + LP + RLIF | 34.2 | 46.9 | 52.3 | 62.9 | 42.7 | 38.2 | 20.4 | 67.5 | 38.6 |
| Acc + LP + Window-RLIF | 32.4 | 48.4 | 51.4 | 63.3 | 44.5 | 37.5 | 21.3 | 66.8 | 38.7 |

RLIF = 自确定性；LP = 长度惩罚；Acc = 答案正确性奖励。最高 AVG 由 Acc + LP 取得，不能将该增益归因于自确定性。

### 从实验中得到的发现

- **内部反馈可作为辅助信号。** Acc + RLIF 相比 Vision-R1 提升 2.0 个百分点；RLIF + LP 相比 LP 提升 1.0 个百分点，8 项基准中 6 项提升、1 项持平、1 项下降，收益具有任务差异。
- **局部窗口有改善，但未解决全部退化。** Window-RLIF 相比 RLIF 的 AVG 提升 0.7 个百分点；训练曲线中观察到窗口策略缓解 EM-RL 塌陷，对 RLSF 未见同样改善，Window-RLIF 后期仍可能退化。
- **奖励项并非越多越好。** Acc + LP + RLIF 为 38.6%，低于 Acc + LP 的 39.2%；这支持进一步研究奖励冲突与动态融合，而非直接声称多信号总能互补。

### 结果边界与复现状态

论文采用验证集选点/早停，不同方法没有保证相同训练步数或生成预算。当前结果未报告多随机种子重复运行、置信区间和固定生成预算对照，因此不据此声称统计显著或跨模型稳定增益。窗口大小、步长与选择策略也尚缺系统消融。

代码入口与结果表已公开；精确的 10K 子集清单、每行结果对应的完整配置/检查点/随机种子，以及逐题评测归档仍需补齐，才能做到历史结果的严格复现。上述命令是依据现有入口整理的运行示例，本次文档更新未重新执行 GPU 训练或评测。

## 致谢与引用

本项目基于 [Vision-SR1](https://github.com/zli12321/Vision-SR1) 开发，感谢其作者开源感知—推理解耦训练方法、代码与数据。训练基础设施来自 [verl](https://github.com/volcengine/verl) 和 [EasyR1](https://github.com/hiyouga/EasyR1)，SFT 使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。相关内部反馈研究及上游作者的贡献应分别引用。

上游发布的 [Vision-SR1 模型](https://huggingface.co/LMMs-Lab-Turtle/SelfRewarded-R1-7B)、[冷启动模型](https://huggingface.co/LMMs-Lab-Turtle/Qwen-2.5VL-7B-Cold-Start) 和原仓库中的示意图/训练曲线属于上游工作，不作为本项目新增实验的展示。

本项目研究来源：谢育林，《基于内部反馈监督的多模态大模型推理能力增强研究》，本科毕业设计，2026。仓库未提供独立论文链接，不使用上游论文链接代指本项目。

以下保留原 README 的引用条目：



使用上游方法时请引用：

```bibtex
@misc{li2025selfrewardingvisionlanguagemodelreasoning,
      title={Self-Rewarding Vision-Language Model via Reasoning Decomposition}, 
      author={Zongxia Li and Wenhao Yu and Chengsong Huang and Rui Liu and Zhenwen Liang and Fuxiao Liu and Jingxi Che and Dian Yu and Jordan Boyd-Graber and Haitao Mi and Dong Yu},
      year={2025},
      eprint={2508.19652},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.19652}, 
}

@article{huang2508self,
  title={Self-evolving reasoning llm from zero data, 2025},
  author={Huang, Chengsong and Yu, Wenhao and Wang, Xiaoyang and Zhang, Hongming and Li, Zongxia and Li, Ruosen and Huang, Jiaxin and Mi, Haitao},
  journal={URL https://arxiv. org/abs/2508.05004}
}

@article{he2025visplay,
  title={Visplay: Self-evolving vision-language models from images},
  author={He, Yicheng and Huang, Chengsong and Li, Zongxia and Huang, Jiaxin and Yang, Yonghui},
  journal={arXiv preprint arXiv:2511.15661},
  year={2025}
}
```

We recommend to also cite the sourcecode work.

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```
