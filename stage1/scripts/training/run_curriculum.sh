#!/bin/bash
# Stage I 课程学习完整流程

set -e
# 切换到项目根目录
cd "$(dirname "$0")/../.." || exit 1

echo "=========================================="
echo "Stage I: 课程学习训练流程"
echo "=========================================="
echo ""
echo "核心策略:"
echo "1. Teacher 先标注简单+短的数据，确保监督稳定"
echo "2. Student 先用高置信度样本训练，不追求覆盖"
echo "3. 如果不收敛，不加难度，只加同分布数据"
echo ""

# ========================================
# Step 1: 安装依赖
# ========================================
echo "Step 1: 安装依赖"
pip install -r requirements.txt pyyaml
echo ""

# ========================================
# Step 2: 下载 HotpotQA
# ========================================
echo "Step 2: 下载 HotpotQA"
python -m src.data.build_hotpot \
  --out data/hotpotqa_raw \
  --max_train 50000 \
  --max_dev 5000
echo ""

# ========================================
# Step 3: Teacher 分阶段标注（先短后长）
# ========================================
echo "Step 3: Teacher 分阶段标注"
echo ""

# Phase 0: Sanity check (300 samples, max 1200 tokens)
echo "Phase 0: Sanity check - 短文档快速验证"
python -m src.data.teacher_generate_vllm_safe \
  --dataset data/hotpotqa_raw/train.jsonl \
  --out data/teacher_supervision/phase_0_sanity.jsonl \
  --save_teacher_logprobs \
  --logprobs_dir data/teacher_supervision/logprobs \
  --step1_prompt prompts/step1_full_graph_strict.txt \
  --step2_prompt prompts/step2_evidence_subgraph_strict.txt \
  --max_docs 3 \
  --max_tokens_step1 2048 \
  --max_tokens_step2 512 \
  --cache_full_graphs data/teacher_supervision/full_graphs_cache.jsonl \
  --max_samples 300
echo ""

# 健全性检查
echo "健全性检查 Phase 0..."
python -m src.eval.sanity_check \
  --data data/teacher_supervision/phase_0_sanity.jsonl \
  --mode check
echo ""

# Phase 1: Core (10000 samples, max 2400 tokens)
echo "Phase 1: Core - 核心高质量样本"
python -m src.data.teacher_generate_vllm_safe \
  --dataset data/hotpotqa_raw/train.jsonl \
  --out data/teacher_supervision/phase_1_core.jsonl \
  --save_teacher_logprobs \
  --logprobs_dir data/teacher_supervision/logprobs \
  --step1_prompt prompts/step1_full_graph_strict.txt \
  --step2_prompt prompts/step2_evidence_subgraph_strict.txt \
  --max_docs 4 \
  --max_tokens_step1 3072 \
  --max_tokens_step2 768 \
  --cache_full_graphs data/teacher_supervision/full_graphs_cache.jsonl \
  --max_samples 10000
echo ""

# Phase 2: Expand (30000 samples, max 3600 tokens)
echo "Phase 2: Expand - 扩展覆盖"
python -m src.data.teacher_generate_vllm_safe \
  --dataset data/hotpotqa_raw/train.jsonl \
  --out data/teacher_supervision/phase_2_expand.jsonl \
  --save_teacher_logprobs \
  --logprobs_dir data/teacher_supervision/logprobs \
  --step1_prompt prompts/step1_full_graph_strict.txt \
  --step2_prompt prompts/step2_evidence_subgraph_strict.txt \
  --max_docs 4 \
  --max_tokens_step1 4096 \
  --max_tokens_step2 1024 \
  --cache_full_graphs data/teacher_supervision/full_graphs_cache.jsonl \
  --max_samples 30000
echo ""

# ========================================
# Step 4: 课程学习训练
# ========================================
echo "Step 4: 课程学习训练"
echo "按置信度从高到低逐步训练..."
echo ""

python -m src.train.curriculum_trainer \
  --config configs/student_training.yaml

echo ""
echo "=========================================="
echo "课程学习训练完成！"
echo ""
echo "可以用以下命令查看各阶段结果:"
echo "  - Phase 0: outputs/stage1_curriculum/phase_0_overfit/final/"
echo "  - Phase 1: outputs/stage1_curriculum/phase_1_stable/final/"
echo "  - Phase 2: outputs/stage1_curriculum/phase_2_expand/final/"
echo "=========================================="
