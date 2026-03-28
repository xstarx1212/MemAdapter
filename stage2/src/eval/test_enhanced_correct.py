#!/usr/bin/env python3
"""
MemAdapter 增强测试脚本 - 正确版本

正确的测试流程:
1. Baseline (StreamingLLM/MemoryLLM) 处理 context+question → memory + baseline_answer
2. Alignment Projection: memory → aligned_memory [1536] (已对齐到 anchor space)
3. Stage1: aligned_memory → get_prefix_embeddings → prefix → generate evidence
   ⚠️ 不需要 full_graph！aligned_memory 已经编码了相关信息
4. Reasoner: evidence + question → enhanced_answer
5. 比较 baseline_answer vs enhanced_answer

数据来源选项:
- 已有: 加载 baseline 的 .jsonl (baseline_answer) + .npy (memory)
- 实时: 运行 baseline 模型提取 memory + baseline_answer
"""

import argparse
import json
import os
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import tempfile
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] PEFT not available.")

# Add stage1 and stage2 src to path
stage1_src = "/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Memadapter/stage1/src"
stage2_src = "/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Memadapter/stage2/src"
if stage1_src not in sys.path:
    sys.path.insert(0, stage1_src)
if stage2_src not in sys.path:
    sys.path.insert(0, stage2_src)

from model.student import StudentRetriever
from models.projection import MemoryProjection


# ============================================================================
# Configuration
# ============================================================================

MODEL_SIZE_CONFIG = {
    "1.5B": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "hidden_dim": 1536,
    },
    "3B": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "hidden_dim": 2048,
    },
    "7B": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
    },
}

BASE = "/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Baselines"
BASELINE_RESULTS_PATHS = {
    "streaming": {
        "1.5B": f"{BASE}/streaming-llm/results/qwen_Qwen2.5-1.5B",
        "3B": f"{BASE}/streaming-llm/results/qwen_Qwen2.5-3B",
        "7B": f"{BASE}/streaming-llm/results/qwen_Qwen2.5-7B",
    },
    "memoryllm": {
        "1.5B": f"{BASE}/MemoryLLM/results/qwen_Qwen2.5-1.5B",
        "3B": f"{BASE}/MemoryLLM/results/qwen_Qwen2.5-3B",
        "7B": f"{BASE}/MemoryLLM/results/qwen_7B",
    },
    "amem": {
        "1.5B": f"{BASE}/A-mem/results/qwen_Qwen2.5-1.5B",
        "3B": f"{BASE}/A-mem/results/qwen_Qwen2.5-3B",
        "7B": f"{BASE}/A-mem/results/qwen_Qwen2.5-7B",
    },
    "mem0": {
        "1.5B": f"{BASE}/mem0/results/qwen_1.5B",
        "3B": f"{BASE}/mem0/results/qwen_3B",
        "7B": f"{BASE}/mem0/results/qwen_7B",
    },
    "lmlm": {
        "1.5B": f"{BASE}/LMLM/results/qwen_1.5B",
        "3B": f"{BASE}/LMLM/results/qwen_3B",
        "7B": f"{BASE}/LMLM/results/qwen_7B",
    },
    "care": {
        "1.5B": f"{BASE}/CARE/results/qwen_1.5B",
        "3B": f"{BASE}/CARE/results/qwen_3B",
        "7B": f"{BASE}/CARE/results/qwen_7B",
    },
}


@dataclass
class BaselineSample:
    """Baseline 结果样本"""
    id: str
    context: str
    question: str
    gold_answer: str
    baseline_answer: str
    memory: np.ndarray  # [hidden_dim]
    dataset_source: str


# ============================================================================
# Data Loading
# ============================================================================

def load_baseline_results(
    paradigm: str,
    model_size: str,
) -> Tuple[List[BaselineSample], bool]:
    """
    加载已有的 baseline 结果
    
    Returns:
        samples: List of BaselineSample
        has_memory: 是否有 memory npy 文件
    """
    base_path = Path(BASELINE_RESULTS_PATHS[paradigm][model_size])
    
    # Find results file
    jsonl_files = list(base_path.glob("*_results.jsonl"))
    jsonl_files = [f for f in jsonl_files if 'memory' not in f.name]
    
    if not jsonl_files:
        print(f"  Warning: No results file found in {base_path}")
        return [], False
    
    jsonl_path = jsonl_files[0]
    print(f"  Loading results from: {jsonl_path}")
    
    # Load jsonl
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Load memory npy if available
    npy_files = list(base_path.glob("*_memory_implicit.npy"))
    memories = None
    has_memory = False
    
    if npy_files:
        npy_path = npy_files[0]
        print(f"  Loading memory from: {npy_path}")
        memories = np.load(npy_path)
        has_memory = True
        print(f"  Memory shape: {memories.shape}")
        
        if memories.shape[0] != len(results):
            print(f"  Warning: Memory count ({memories.shape[0]}) != Results count ({len(results)})")
            has_memory = False
    
    # Build samples
    samples = []
    for i, r in enumerate(results):
        memory = memories[i] if has_memory else None
        sample = BaselineSample(
            id=r.get('id', f'sample_{i}'),
            context=r.get('context', ''),
            question=r.get('question', ''),
            gold_answer=r.get('gold_answer', r.get('answer', '')),
            baseline_answer=r.get('generated_answer', ''),
            memory=memory,
            dataset_source=r.get('dataset_source', 'unknown'),
        )
        samples.append(sample)
    
    print(f"  Loaded {len(samples)} samples, has_memory={has_memory}")
    return samples, has_memory


# ============================================================================
# Model Loading
# ============================================================================

def load_alignment_projection(
    checkpoint_dir: str,
    paradigm: str,
    model_size: str,
    device: str = "cuda",
) -> MemoryProjection:
    """加载 Stage2 alignment projection"""
    config = MODEL_SIZE_CONFIG[model_size]
    input_dim = config["hidden_dim"]
    anchor_dim = 1536
    
    paradigm_key = f"{paradigm}_{model_size.replace('.', '_')}"
    print(f"  Loading alignment: {paradigm_key} ({input_dim} -> {anchor_dim})")
    
    projection = MemoryProjection(input_dim=input_dim, anchor_dim=anchor_dim)
    
    # Find checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    subdirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()], reverse=True)
    
    loaded = False
    for subdir in subdirs:
        for ckpt_name in ['best_model.pt', 'final_model.pt']:
            ckpt_path = subdir / ckpt_name
            if ckpt_path.exists():
                print(f"  Loading from {ckpt_path}")
                state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                
                if 'projections' in state:
                    for key in state['projections'].keys():
                        if paradigm in key:
                            projection.load_state_dict(state['projections'][key])
                            loaded = True
                            print(f"  Loaded: {key}")
                            break
                
                if loaded:
                    break
        if loaded:
            break
    
    if not loaded:
        print(f"  WARNING: Could not load weights!")
    
    projection = projection.to(device).to(torch.bfloat16)
    projection.eval()
    return projection


def load_stage1_retriever(checkpoint_path: str, device: str = "cuda") -> StudentRetriever:
    """加载 Stage1 retriever"""
    checkpoint_path = Path(checkpoint_path)
    base_model = "Qwen/Qwen2.5-1.5B"
    
    print(f"  Loading Stage1 retriever...")
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter
    adapter_path = checkpoint_path / "adapter_model.safetensors"
    adapter_config_path = checkpoint_path / "adapter_config.json"
    
    model = base_model_obj
    if (adapter_path.exists() or adapter_config_path.exists()) and PEFT_AVAILABLE:
        print("  Loading LoRA adapter...")
        try:
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                unsupported_fields = [
                    'alora_invocation_tokens', 'arrow_config', 'corda_config',
                    'eva_config', 'layer_replication', 'layers_pattern',
                    'layers_to_transform', 'megatron_config', 'megatron_core',
                    'qalora_group_size', 'target_parameters', 'trainable_token_indices'
                ]
                cleaned_config = {k: v for k, v in adapter_config.items() if k not in unsupported_fields}
                
                temp_checkpoint = tempfile.mkdtemp(prefix='peft_load_')
                with open(os.path.join(temp_checkpoint, 'adapter_config.json'), 'w') as f:
                    json.dump(cleaned_config, f, indent=2)
                if adapter_path.exists():
                    shutil.copy(adapter_path, os.path.join(temp_checkpoint, 'adapter_model.safetensors'))
                
                model = PeftModel.from_pretrained(base_model_obj, temp_checkpoint).to(device)
                shutil.rmtree(temp_checkpoint)
            else:
                model = PeftModel.from_pretrained(base_model_obj, str(checkpoint_path)).to(device)
            print("  LoRA adapter loaded")
        except Exception as e:
            print(f"  Warning: Failed to load LoRA: {e}")
    
    # Build StudentRetriever
    student_model = StudentRetriever(
        model_name_or_path=base_model,
        device=device,
        frozen_anchor_align=False
    )
    student_model.model = model
    student_model.tokenizer = tokenizer
    
    model_dtype = next(model.parameters()).dtype
    student_model.model_dtype = model_dtype
    student_model.anchor_align = student_model.anchor_align.to(device).to(dtype=model_dtype)
    student_model.prefix_projection = student_model.prefix_projection.to(device).to(dtype=model_dtype)
    
    # Load anchor components
    anchor_components_path = checkpoint_path / "anchor_components.pt"
    if anchor_components_path.exists():
        print("  Loading anchor components...")
        components = torch.load(anchor_components_path, map_location=device, weights_only=False)
        student_model.anchor_align.load_state_dict(components['anchor_align'])
        student_model.prefix_projection.load_state_dict(components['prefix_projection'])
        student_model.anchor_align = student_model.anchor_align.to(device).to(dtype=model_dtype)
        student_model.prefix_projection = student_model.prefix_projection.to(device).to(dtype=model_dtype)
    
    student_model.eval()
    print("  Stage1 retriever ready")
    return student_model


def load_reasoner(model_size: str, device: str = "cuda"):
    """加载 Reasoner (与 Baseline 相同的 LLM)"""
    config = MODEL_SIZE_CONFIG[model_size]
    model_name = config["model_name"]
    
    print(f"  Loading Reasoner: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    print(f"  Reasoner loaded")
    
    return model, tokenizer


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def enhanced_inference(
    sample: BaselineSample,
    alignment_projection: Optional[MemoryProjection],
    stage1_retriever: StudentRetriever,
    reasoner_model,
    reasoner_tokenizer,
    device: str = "cuda",
    use_alignment: bool = True,
    use_explicit_context: bool = False,
) -> Dict:
    """
    单样本增强推理
    
    流程:
    - use_explicit_context=True: 显式 context 文本 → Stage1.encode_context → anchor_align → prefix → evidence（统一用我们的编码器）
    - use_alignment=True:  memory → alignment → aligned_memory [1536] → Stage1 → evidence
    - use_alignment=False: memory [1536] 直接 → Stage1 generate_from_aligned_memory（仅当 dim=1536 时有效）
    """
    if use_explicit_context:
        # 统一成显式 context：用我们的编码器编码 sample.context → anchor_align → prefix
        if not (sample.context or "").strip():
            evidence = "[ERROR: empty context]"
        else:
            try:
                g = stage1_retriever.encode_context(sample.context)
                if g.dim() == 1:
                    g = g.unsqueeze(0)
                g = g.to(device)
                if hasattr(stage1_retriever, "model_dtype"):
                    g = g.to(dtype=stage1_retriever.model_dtype)
                h_a = stage1_retriever.anchor_align(g)
                prefix_input = h_a  # [1, 1536]
                evidence = stage1_retriever.generate_from_aligned_memory(
                    aligned_memory=prefix_input.squeeze(0),
                    query=sample.question,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False
                )
            except Exception as e:
                evidence = f"[ERROR: {str(e)}]"
    else:
        memory_tensor = torch.from_numpy(sample.memory).unsqueeze(0).to(device).to(torch.bfloat16)
        if use_alignment and alignment_projection is not None:
            prefix_input = alignment_projection(memory_tensor)  # [1, 1536]
        else:
            if memory_tensor.shape[1] != 1536:
                raise ValueError(
                    f"no_align 仅支持 1536 维 memory，当前为 {memory_tensor.shape[1]}。"
                    "请使用 use_alignment=True 或 1.5B 的 baseline memory。"
                )
            prefix_input = memory_tensor  # [1, 1536]
        try:
            evidence = stage1_retriever.generate_from_aligned_memory(
                aligned_memory=prefix_input.squeeze(0),  # [1536]
                query=sample.question,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=False
            )
        except Exception as e:
            evidence = f"[ERROR: {str(e)}]"
    
    # Step 3: Reasoner generates enhanced answer
    prompt = f"""Based on the following evidence, answer the question with a short phrase.

Evidence:
{evidence}

Question: {sample.question}

Answer:"""
    
    inputs = reasoner_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    outputs = reasoner_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=reasoner_tokenizer.pad_token_id,
        eos_token_id=reasoner_tokenizer.eos_token_id,
    )
    
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    enhanced_answer = reasoner_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Evaluate
    gold = sample.gold_answer.strip().lower()
    baseline_ans = sample.baseline_answer.strip().lower()
    enhanced_ans = enhanced_answer.strip().lower()
    
    return {
        "id": sample.id,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "baseline_answer": sample.baseline_answer,
        "enhanced_answer": enhanced_answer,
        "evidence": evidence[:500] if len(evidence) > 500 else evidence,
        "dataset_source": sample.dataset_source,
        "baseline_match": gold in baseline_ans or baseline_ans in gold,
        "enhanced_match": gold in enhanced_ans or enhanced_ans in gold,
        "use_alignment": use_alignment and not use_explicit_context,
        "use_explicit_context": use_explicit_context,
    }


def _tokenize(text: str):
    import re
    return re.findall(r'\w+', text.lower())


def _compute_f1(gold_tokens: list, pred_tokens: list) -> float:
    from collections import Counter
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(gold_tokens)
    return 2 * p * r / (p + r)


def compute_metrics(results: List[Dict]) -> Dict:
    """计算指标：EM、Contains、F1、ROUGE-1/2/L"""
    from collections import defaultdict
    try:
        from rouge_score import rouge_scorer
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        has_rouge = True
    except ImportError:
        has_rouge = False

    total = len(results)
    baseline_em = sum(1 for r in results if r.get('baseline_match', False))
    enhanced_em = sum(1 for r in results if r.get('enhanced_match', False))
    baseline_contains = sum(
        1 for r in results
        if (r.get('gold_answer', '').strip().lower() in r.get('baseline_answer', '').strip().lower()
            or r.get('baseline_answer', '').strip().lower() in r.get('gold_answer', '').strip().lower())
    )
    enhanced_contains = sum(
        1 for r in results
        if (r.get('gold_answer', '').strip().lower() in r.get('enhanced_answer', '').strip().lower()
            or r.get('enhanced_answer', '').strip().lower() in r.get('gold_answer', '').strip().lower())
    )

    f1_baseline, f1_enhanced = [], []
    r1_baseline, r1_enhanced = [], []
    r2_baseline, r2_enhanced = [], []
    rL_baseline, rL_enhanced = [], []
    for r in results:
        gold = (r.get('gold_answer') or '').strip()
        base = (r.get('baseline_answer') or '').strip()
        enh = (r.get('enhanced_answer') or '').strip()
        gt, bt, et = _tokenize(gold), _tokenize(base), _tokenize(enh)
        f1_baseline.append(_compute_f1(gt, bt))
        f1_enhanced.append(_compute_f1(gt, et))
        if has_rouge:
            try:
                sb = rouge.score(gold, base)
                se = rouge.score(gold, enh)
                r1_baseline.append(sb['rouge1'].fmeasure)
                r1_enhanced.append(se['rouge1'].fmeasure)
                r2_baseline.append(sb['rouge2'].fmeasure)
                r2_enhanced.append(se['rouge2'].fmeasure)
                rL_baseline.append(sb['rougeL'].fmeasure)
                rL_enhanced.append(se['rougeL'].fmeasure)
            except Exception:
                pass
    n = total if total > 0 else 1
    nr = len(r1_baseline) if r1_baseline else 1
    metrics = {
        "total": total,
        "baseline_exact_match": baseline_em,
        "baseline_em_rate": baseline_em / n,
        "enhanced_exact_match": enhanced_em,
        "enhanced_em_rate": enhanced_em / n,
        "improvement": (enhanced_em - baseline_em) / n,
        "baseline_contains": baseline_contains,
        "baseline_contains_rate": baseline_contains / n,
        "enhanced_contains": enhanced_contains,
        "enhanced_contains_rate": enhanced_contains / n,
        "baseline_f1": round(sum(f1_baseline) / n * 100, 2) if f1_baseline else 0,
        "enhanced_f1": round(sum(f1_enhanced) / n * 100, 2) if f1_enhanced else 0,
        "baseline_rouge1": round(sum(r1_baseline) / nr * 100, 2) if r1_baseline else 0,
        "enhanced_rouge1": round(sum(r1_enhanced) / nr * 100, 2) if r1_enhanced else 0,
        "baseline_rouge2": round(sum(r2_baseline) / nr * 100, 2) if r2_baseline else 0,
        "enhanced_rouge2": round(sum(r2_enhanced) / nr * 100, 2) if r2_enhanced else 0,
        "baseline_rougeL": round(sum(rL_baseline) / nr * 100, 2) if rL_baseline else 0,
        "enhanced_rougeL": round(sum(rL_enhanced) / nr * 100, 2) if rL_enhanced else 0,
    }

    # Per-dataset
    dataset_results = defaultdict(list)
    for r in results:
        dataset_results[r.get('dataset_source', 'unknown')].append(r)
    metrics['per_dataset'] = {}
    for dataset, ds_results in dataset_results.items():
        ds_total = len(ds_results)
        ds_baseline_em = sum(1 for r in ds_results if r.get('baseline_match', False))
        ds_enhanced_em = sum(1 for r in ds_results if r.get('enhanced_match', False))
        metrics['per_dataset'][dataset] = {
            'total': ds_total,
            'baseline_em': ds_baseline_em,
            'baseline_em_rate': ds_baseline_em / ds_total if ds_total > 0 else 0,
            'enhanced_em': ds_enhanced_em,
            'enhanced_em_rate': ds_enhanced_em / ds_total if ds_total > 0 else 0,
        }
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MemAdapter Enhanced Testing - Correct Version')
    
    # Model config
    parser.add_argument('--paradigm', type=str, required=True,
                        choices=['streaming', 'memoryllm', 'amem', 'mem0', 'lmlm', 'care'],
                        help='Paradigm (baseline) to test')
    parser.add_argument('--model_size', type=str, required=True,
                        choices=['1.5B', '3B', '7B'],
                        help='Model size')
    
    # Checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--stage2_checkpoint', type=str, required=True)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results/enhanced')
    parser.add_argument('--max_samples', type=int, default=None)
    
    # Alignment 对比：不经过 alignment 直接喂 memory 进 retriever（仅 1.5B 时 dim=1536 有效）
    parser.add_argument('--no_align', action='store_true',
                        help='不经过 alignment，直接把 memory 喂给 Stage1 retriever（仅 1.5B）')
    parser.add_argument('--compare_both', action='store_true',
                        help='同时跑 with_align 和 no_align，输出两边指标对比')
    # 统一成显式 context：用我们的编码器编码 jsonl 里的 context，不用 baseline 的隐式 memory
    parser.add_argument('--use_explicit_context', action='store_true',
                        help='用显式 context 文本 + 我们的 Stage1 编码器（encode_context→anchor_align→evidence），不读 .npy')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MemAdapter Enhanced Testing - Correct Version")
    print("=" * 80)
    print(f"Paradigm: {args.paradigm}")
    print(f"Model size: {args.model_size}")
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [1] Load baseline results
    print("\n[1] Loading baseline results...")
    samples, has_memory = load_baseline_results(args.paradigm, args.model_size)
    
    if not samples:
        print("ERROR: No baseline results found!")
        return
    
    if not has_memory and not args.use_explicit_context:
        print("ERROR: No memory data found! Use --use_explicit_context to run with explicit context only, or run baseline first.")
        return
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Using {len(samples)} samples")
    
    use_align = not args.no_align and not args.use_explicit_context
    if args.compare_both:
        use_align = True  # 会跑两遍，先 with_align 再 no_align
    
    # [2] Load alignment projection（no_align / use_explicit_context 且非 compare_both 时不加载）
    alignment_projection = None
    if (use_align or args.compare_both) and not args.use_explicit_context:
        print(f"\n[2] Loading alignment projection...")
        alignment_projection = load_alignment_projection(
            checkpoint_dir=args.stage2_checkpoint,
            paradigm=args.paradigm,
            model_size=args.model_size,
            device=args.device
        )
    else:
        print(f"\n[2] Skipping alignment (--no_align). 仅 1.5B 时 memory 维度须为 1536。")
        if args.model_size != "1.5B":
            print("  WARNING: --no_align 仅对 1.5B 有效（dim=1536），当前 model_size 非 1.5B，可能报错。")
    
    # [3] Load Stage1 retriever
    print(f"\n[3] Loading Stage1 retriever...")
    stage1_retriever = load_stage1_retriever(
        checkpoint_path=args.stage1_checkpoint,
        device=args.device
    )
    
    # [4] Load Reasoner
    print(f"\n[4] Loading Reasoner ({args.model_size})...")
    reasoner_model, reasoner_tokenizer = load_reasoner(
        model_size=args.model_size,
        device=args.device
    )
    
    def run_inference(use_alignment: bool, desc: str, use_explicit_context: bool = False):
        res_list = []
        for i, sample in enumerate(tqdm(samples, desc=desc)):
            try:
                r = enhanced_inference(
                    sample=sample,
                    alignment_projection=alignment_projection,
                    stage1_retriever=stage1_retriever,
                    reasoner_model=reasoner_model,
                    reasoner_tokenizer=reasoner_tokenizer,
                    device=args.device,
                    use_alignment=use_alignment,
                    use_explicit_context=use_explicit_context,
                )
                res_list.append(r)
            except Exception as e:
                print(f"\nSample {i} failed: {e}")
                res_list.append({
                    "id": getattr(sample, "id", f"sample_{i}"),
                    "question": getattr(sample, "question", ""),
                    "gold_answer": getattr(sample, "gold_answer", ""),
                    "baseline_answer": getattr(sample, "baseline_answer", ""),
                    "enhanced_answer": "",
                    "evidence": "",
                    "dataset_source": getattr(sample, "dataset_source", "unknown"),
                    "baseline_match": False,
                    "enhanced_match": False,
                    "error": str(e),
                    "use_alignment": use_alignment,
                    "use_explicit_context": use_explicit_context,
                })
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        return res_list
    
    # [5] Run enhanced inference
    if args.compare_both:
        print(f"\n[5] Comparing with_align vs no_align on {len(samples)} samples...")
        results_with = run_inference(use_alignment=True, desc="with_align")
        results_no = run_inference(use_alignment=False, desc="no_align")
        metrics_with = compute_metrics(results_with)
        metrics_no = compute_metrics(results_no)
        print(f"\n{'='*60}")
        print("Comparison: with_align vs no_align")
        print(f"{'='*60}")
        print(f"{'Metric':<16} {'with_align':<14} {'no_align':<14} {'Δ (no - with)':>14}")
        print("-" * 62)
        print(f"{'EM (acc)':<16} {metrics_with['enhanced_em_rate']:>10.2%}   {metrics_no['enhanced_em_rate']:>10.2%}   {metrics_no['enhanced_em_rate'] - metrics_with['enhanced_em_rate']:>+12.2%}")
        print(f"{'Contains':<16} {metrics_with.get('enhanced_contains_rate', 0):>10.2%}   {metrics_no.get('enhanced_contains_rate', 0):>10.2%}   {metrics_no.get('enhanced_contains_rate', 0) - metrics_with.get('enhanced_contains_rate', 0):>+12.2%}")
        print(f"{'F1 (%)':<16} {metrics_with.get('enhanced_f1', 0):>10.2f}   {metrics_no.get('enhanced_f1', 0):>10.2f}   {metrics_no.get('enhanced_f1', 0) - metrics_with.get('enhanced_f1', 0):>+12.2f}")
        print(f"{'ROUGE-1 (%)':<16} {metrics_with.get('enhanced_rouge1', 0):>10.2f}   {metrics_no.get('enhanced_rouge1', 0):>10.2f}   {metrics_no.get('enhanced_rouge1', 0) - metrics_with.get('enhanced_rouge1', 0):>+12.2f}")
        print(f"{'ROUGE-2 (%)':<16} {metrics_with.get('enhanced_rouge2', 0):>10.2f}   {metrics_no.get('enhanced_rouge2', 0):>10.2f}   {metrics_no.get('enhanced_rouge2', 0) - metrics_with.get('enhanced_rouge2', 0):>+12.2f}")
        print(f"{'ROUGE-L (%)':<16} {metrics_with.get('enhanced_rougeL', 0):>10.2f}   {metrics_no.get('enhanced_rougeL', 0):>10.2f}   {metrics_no.get('enhanced_rougeL', 0) - metrics_with.get('enhanced_rougeL', 0):>+12.2f}")
        print("-" * 62)
        print(f"{'baseline EM':<16} {metrics_with['baseline_em_rate']:>10.2%}   (same)")
        print(f"{'improvement':<16} {metrics_with['improvement']:>10.2%}   {metrics_no['improvement']:>10.2%}   {metrics_no['improvement'] - metrics_with['improvement']:>+12.2%}")
        exp_name = f"enhanced_{args.paradigm}_{args.model_size}"
        for tag, res, met in [("with_align", results_with, metrics_with), ("no_align", results_no, metrics_no)]:
            out_r = output_dir / f"{exp_name}_{tag}_results.jsonl"
            out_m = output_dir / f"{exp_name}_{tag}_metrics.json"
            with open(out_r, 'w', encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            with open(out_m, 'w', encoding='utf-8') as f:
                json.dump(met, f, indent=2, ensure_ascii=False)
            print(f"  {tag}: {out_r}, {out_m}")
        print("=" * 80)
        return
    
    mode_desc = "explicit_context" if args.use_explicit_context else f"use_alignment={use_align}"
    print(f"\n[5] Running enhanced inference on {len(samples)} samples ({mode_desc})...")
    results = run_inference(use_alignment=use_align, desc=f"{args.paradigm}_{args.model_size}", use_explicit_context=args.use_explicit_context)
    
    # [6] Compute metrics
    print(f"\n[6] Computing metrics...")
    metrics = compute_metrics(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary - {args.paradigm} + {args.model_size} ({mode_desc})")
    print(f"{'='*60}")
    print(f"Total: {metrics['total']}")
    print(f"Baseline EM: {metrics['baseline_exact_match']} ({metrics['baseline_em_rate']:.2%})")
    print(f"Enhanced EM: {metrics['enhanced_exact_match']} ({metrics['enhanced_em_rate']:.2%})")
    print(f"Improvement: {metrics['improvement']:+.2%}")
    
    print("\nPer-dataset:")
    for dataset, ds_m in metrics['per_dataset'].items():
        print(f"  {dataset}: Baseline={ds_m['baseline_em_rate']:.2%}, Enhanced={ds_m['enhanced_em_rate']:.2%}")
    
    # Save results
    if args.use_explicit_context:
        align_suffix = "_explicit_context"
    else:
        align_suffix = "" if use_align else "_no_align"
    exp_name = f"enhanced_{args.paradigm}_{args.model_size}{align_suffix}"
    
    results_file = output_dir / f"{exp_name}_results.jsonl"
    with open(results_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    metrics_file = output_dir / f"{exp_name}_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
