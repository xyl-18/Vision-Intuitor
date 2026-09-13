# Vision-Intuitor

**Internal-Feedback Reinforcement Learning for Vision-Language Models**

[方法与实现](#相比-vision-sr1-的改动) · [训练与评测](#训练与评测) · [实验结果](#实验结果) · [引用](#致谢与引用)

Vision-Intuitor 是基于 [Vision-SR1](https://github.com/zli12321/Vision-SR1) 的 VLM 强化学习实现，使用模型内部信号替代或补充答案奖励。我们将自确定性、熵、概率间隔与长度奖励接入单阶段和双阶段训练，并实现局部窗口奖励及分量独立的优势融合。

全序列平均容易掩盖推理中的低置信片段。为此，我们对有效响应滑窗计算奖励，聚合最小窗口或最低 p 比例窗口，再与全局、答案和长度信号组合。在 Qwen2.5-VL-3B 上，自确定性与答案奖励组合比复现的 Vision-R1 提升 **2.0 个百分点**；窗口聚合比全局自确定性提升 **0.7 个百分点**。

## 相比 Vision-SR1 的改动

基于上游的感知—推理解耦流程，我们扩展了奖励计算、优势估计和双阶段训练器：

| 模块 | 实现 | 代码 |
| --- | --- | --- |
| 内部反馈 | 接入自确定性（RLIF）、熵（EM-RL）、概率间隔（RLSF），以长度惩罚（LP）作为对照与辅助奖励 | [fsdp_workers.py](verl/workers/fsdp_workers.py) |
| 局部奖励 | 对有效响应滑窗，取最小窗口或最低 p 比例窗口的均值，构造序列级奖励 | [core_algos.py](verl/trainer/core_algos.py) |
| 混合优势 | 内部、格式、答案、长度分量分别执行 GRPO 组内标准化，再加权融合 | [intuitor_mixer.py](verl/trainer/intuitor_mixer.py)、[core_algos.py](verl/trainer/core_algos.py) |
| 双阶段反馈 | 扩展第二阶段采样、Actor 内部信号重算及双阶段优势融合 | [IntuitorSR1Trainer](vision_intuitor_sr1_lora/ray_trainer.py) |
| 训练诊断 | 记录分量奖励、组内标准差，结合熵、响应长度与验证表现分析训练塌陷 | [ray_trainer.py](verl/trainer/ray_trainer.py) |

## 训练与评测

### 安装

使用 Linux + NVIDIA GPU。当前 [安装脚本](setup.sh) 配置为 PyTorch 2.9.1 / CUDA 12.8、vLLM 0.11.0、FlashAttention 2.8.3、Transformers 4.57.0，包含系统依赖安装与 W&B 登录。

```bash
git clone https://github.com/xyl-18/Vision-Intuitor.git
cd Vision-Intuitor
conda create -n vision-intuitor python=3.12 -y
conda activate vision-intuitor
bash setup.sh
```

### 数据

实验使用 [Vision-SR1-Cold-9K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-Cold-9K) 进行 SFT，从 [Vision-SR1-47K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-47K) 抽取 10K 题进行 RL。数据字段为 `problem`、`answer`、`images`，验证集为 `zli12321/mmstar@test`。

<details>
<summary>冷启动 SFT（LLaMA-Factory）</summary>

使用独立环境：

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

运行前检查 [SFT 配置](LLaMA-Factory-Cold-Start/examples/train_full/Vision-SR1-Cold-Start.yaml) 的数据注册和输出目录。

</details>

`MODEL_PATH` 指向 Transformers 格式的 SFT 模型，`TRAIN_DATA` 指向本地 10K 数据。以下命令直接调用 Python 入口，覆盖原 shell 脚本中的本机路径。

### 单阶段训练

默认示例使用自确定性与格式奖励，关闭答案奖励。复现条件见[实验设置](#实验设置)。

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

在上述命令中替换或追加参数：

| 变体 | 参数变化 |
| --- | --- |
| EM-RL | `algorithm.intuitor_reward_method=entropy` |
| RLSF | `algorithm.intuitor_reward_method=rlsf` |
| 仅长度惩罚 LP | `algorithm.intuitor_weight=0.0 algorithm.intuitor_aux_length_weight=1.0` |
| RLIF + LP | `algorithm.intuitor_aux_length_weight=0.1` |
| 答案奖励 + RLIF | `algorithm.intuitor_aux_external_weight=1.0` |

窗口奖励配置（追加到单阶段命令末尾）：

```bash
algorithm.intuitor_window_enabled=true \
algorithm.intuitor_global_weight=0.0 \
algorithm.intuitor_window_weight=1.0 \
algorithm.intuitor_window_size=32 \
algorithm.intuitor_window_stride=4 \
algorithm.intuitor_window_strategy=bottom_p_mean \
algorithm.intuitor_window_bottom_p=0.1
```

`min` 取最小窗口；`bottom_p_mean` 取最低 p 比例窗口的均值。窗口分支需同时启用开关并设置非零权重。

### 两阶段训练

<details>
<summary>视觉描述 → 文本推理：训练命令</summary>

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

两个阶段的答案奖励均设为 0。混合监督可分别调整 `intuitor_aux_external_weight` 和 `intuitor_sr1_second_external_weight`；第二阶段答案奖励键默认为 `description_accuracy`。

</details>

### 评测

检查点需与训练时的 SFT 底座匹配。`global_step_100` 为示例路径；评测脚本默认使用 2 张 GPU，运行前按硬件调整。

```bash
export CHECKPOINT_PATH=./saves/vision_intuitor_rlif/global_step_100
export BASE_MODEL="$MODEL_PATH"
bash evaluation/lora_rl/eval_vision_intuitor_lora.sh "$CHECKPOINT_PATH" "$BASE_MODEL"

# 两阶段模型：将路径换成实际保存的 SR1 检查点。
bash evaluation/lora_rl/eval_vision_intuitor_sr1_lora.sh \
  ./saves/vision_intuitor_sr1/global_step_100 "$BASE_MODEL"
```

流程为 greedy decoding → 答案提取 → 本地 LLM Judge → 分基准汇总。已有响应文件时可单独评判：

```bash
python evaluation/llm_judge.py \
  --response_dir /path/to/responses \
  --judge_model Qwen/Qwen2.5-14B-Instruct --tp_size 2
python evaluation/print_accuracy.py --judgment_dir /path/to/judgments
```

## 实验结果

**Qwen2.5-VL-3B · 单阶段训练 · 8 项基准 · 约 11.5K 题**

所有基线均在本项目中复现，使用相同评测流程。数值为准确率（%），AVG 按样本数加权；评测采用本地 LLM Judge，不等同于各基准的官方计分。

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

RLIF：自确定性；LP：长度惩罚；Acc：答案正确性奖励。关闭 Acc 仅表示 RL 不使用答案奖励，SFT 与验证评测仍使用监督信息。

### 结果分析

RLIF + LP 相比 LP 在 6 项基准上提升、1 项持平、1 项下降。Acc + LP 的平均分最高；继续叠加 RLIF 后由 39.2% 降至 38.6%，说明这些奖励的组合并非始终有效。

窗口聚合改善了 RLIF，并在训练曲线中缓解 EM-RL 退化，对 RLSF 未观察到同样效果。Window-RLIF 后期仍可能塌陷。

### 实验设置

- 使用 9K SFT 数据和 10K RL 子集，按验证集选点/早停；各方法训练步数与生成预算不完全相同。
- RLIF、LP 等配置关闭 RL 答案奖励；SFT、验证选点和评测仍使用监督信息。
- 表中为单次实验结果，未报告多种子置信区间。完整配置、10K 子集清单与检查点尚未逐项归档；训练命令展示配置用法，不保证精确复现表中数值。
- `internal_reward_adaptive` 为探索分支，表中结果未使用该分支。

## 致谢与引用

感谢 [Vision-SR1](https://github.com/zli12321/Vision-SR1) 提供感知—推理解耦方法、基线代码与数据。分布式训练基于 [verl](https://github.com/volcengine/verl) 和 [EasyR1](https://github.com/hiyouga/EasyR1)，冷启动训练使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。使用相关方法时，请引用原始工作。

<details>
<summary>BibTeX</summary>

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

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

</details>
