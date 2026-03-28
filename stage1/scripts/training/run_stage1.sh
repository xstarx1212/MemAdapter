#!/bin/bash
# Stage I 完整流程脚本

set -e  # 遇到错误立即退出
# 切换到项目根目录
cd "$(dirname "$0")/../.." || exit 1

echo "=========================================="
echo "Stage I: MemAdapter 训练流程"
echo "=========================================="
echo ""

# 配置
MAX_TRAIN_SAMPLES=1000  # 调整为你需要的样本数
MAX_DEV_SAMPLES=500
STUDENT_MODEL="Qwen/Qwen2.5-1.5B"
DISTILL_MODE="kl"  # "ce" 或 "kl"

# 步骤 1: 安装依赖
echo "步骤 1: 安装依赖"
pip install -r requirements.txt
echo ""

# 步骤 2: 下载 HotpotQA 数据集
echo "步骤 2: 下载和预处理 HotpotQA"
python -m src.data.build_hotpot \
  --out data/hotpotqa_raw \
  --max_train ${MAX_TRAIN_SAMPLES} \
  --max_dev ${MAX_DEV_SAMPLES}
echo ""

# 步骤 3: 生成教师监督信号 (训练集, 两阶段)
echo "步骤 3: 生成教师监督信号 (训练集) - 两阶段生成"
python -m src.data.teacher_generate_vllm_safe \
  --dataset data/hotpotqa_raw/train.jsonl \
  --out data/teacher_supervision/hotpot_train_graph.jsonl \
  --save_teacher_logprobs \
  --logprobs_dir data/teacher_supervision/logprobs \
  --step1_prompt prompts/step1_full_graph_strict.txt \
  --step2_prompt prompts/step2_evidence_subgraph_strict.txt \
  --max_docs 4 \
  --max_tokens_step1 4096 \
  --max_tokens_step2 1024 \
  --cache_full_graphs data/teacher_supervision/full_graphs_cache.jsonl \
  --max_samples ${MAX_TRAIN_SAMPLES}
echo ""

# 步骤 4: 生成教师监督信号 (验证集, 复用缓存的完整图)
echo "步骤 4: 生成教师监督信号 (验证集) - 复用完整图缓存"
python -m src.data.teacher_generate_vllm_safe \
  --dataset data/hotpotqa_raw/dev.jsonl \
  --out data/teacher_supervision/hotpot_dev_graph.jsonl \
  --save_teacher_logprobs \
  --logprobs_dir data/teacher_supervision/logprobs \
  --step1_prompt prompts/step1_full_graph_strict.txt \
  --step2_prompt prompts/step2_evidence_subgraph_strict.txt \
  --max_docs 4 \
  --max_tokens_step1 4096 \
  --max_tokens_step2 1024 \
  --cache_full_graphs data/teacher_supervision/full_graphs_cache.jsonl \
  --max_samples ${MAX_DEV_SAMPLES}
echo ""

# 步骤 5: 健全性检查
echo "步骤 5: 健全性检查"
python -m src.eval.sanity_check \
  --data data/teacher_supervision/hotpot_train_graph.jsonl \
  --mode check \
  --verbose
echo ""

echo "步骤 5.1: 查看样本示例"
python -m src.eval.sanity_check \
  --data data/teacher_supervision/hotpot_train_graph.jsonl \
  --mode inspect \
  --num_inspect 3
echo ""

# 步骤 6: 训练学生模型
echo "步骤 6: 训练学生模型 (Mode: ${DISTILL_MODE})"
python -m src.train.train_stage1 \
  --train data/teacher_supervision/hotpot_train_graph.jsonl \
  --eval data/teacher_supervision/hotpot_dev_graph.jsonl \
  --student_model ${STUDENT_MODEL} \
  --distill_mode ${DISTILL_MODE} \
  --lr 2e-5 \
  --batch_size 8 \
  --grad_accum 4 \
  --num_epochs 2 \
  --max_input_len 4096 \
  --max_output_len 512 \
  --bf16 true \
  --out_dir outputs/stage1_qwen1p5b_${DISTILL_MODE} \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 500
echo ""

echo "=========================================="
echo "Stage I 训练完成！"
echo "模型保存在: outputs/stage1_qwen1p5b_${DISTILL_MODE}/final/"
echo "=========================================="
