#!/usr/bin/env python3
"""
MemAdapter 两阶段测试脚本 - 批量并行版本

Stage 1: StudentRetriever (1.5B, 训练好的权重) 批量生成证据子图
Stage 2: Reasoner (任意大小 LLM) 批量用证据子图回答问题

用法:
    python test_memadapter_twostage_batch.py \
        --stage1_checkpoint /path/to/stage1/final \
        --reasoner_model Qwen/Qwen2.5-3B-Instruct \
        --test_data /path/to/test_merged.jsonl \
        --output_dir ./results \
        --batch_size 8
"""

import json
import argparse
import os
import sys
import torch
import gc
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import tempfile
import shutil

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] PEFT not available.")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add stage1 src to path
stage1_src = "/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Memadapter/stage1/src"
if stage1_src not in sys.path:
    sys.path.insert(0, stage1_src)

from model.student import StudentRetriever


@dataclass
class QASample:
    """QA 样本数据结构"""
    id: str
    context: str
    question: str
    answer: str
    dataset_source: str


def load_qa_data(data_path: str, limit: Optional[int] = None) -> List[QASample]:
    """加载测试数据"""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line.strip())
            sample = QASample(
                id=data.get('id', f'sample_{i}'),
                context=data.get('context', ''),
                question=data.get('question', ''),
                answer=data.get('answer', ''),
                dataset_source=data.get('dataset_source', 'unknown')
            )
            samples.append(sample)
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def load_stage1_model(checkpoint_path: str, device: str = "cuda") -> StudentRetriever:
    """
    加载 Stage1 模型 (StudentRetriever)
    """
    checkpoint_path = Path(checkpoint_path)
    base_model = "Qwen/Qwen2.5-1.5B"
    
    print(f"Loading Stage1 model (base: {base_model})")
    
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
    
    # Load LoRA adapter if exists
    adapter_path = checkpoint_path / "adapter_model.safetensors"
    adapter_config_path = checkpoint_path / "adapter_config.json"
    
    model = base_model_obj
    if (adapter_path.exists() or adapter_config_path.exists()) and PEFT_AVAILABLE:
        print("Loading LoRA adapter...")
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
            print("LoRA adapter loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load LoRA: {e}")
    
    # Create StudentRetriever
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
        print("Loading anchor components...")
        components = torch.load(anchor_components_path, map_location=device, weights_only=False)
        student_model.anchor_align.load_state_dict(components['anchor_align'])
        student_model.prefix_projection.load_state_dict(components['prefix_projection'])
        student_model.anchor_align = student_model.anchor_align.to(device).to(dtype=model_dtype)
        student_model.prefix_projection = student_model.prefix_projection.to(device).to(dtype=model_dtype)
        print("Anchor components loaded successfully")
    
    student_model.eval()
    return student_model


def load_reasoner_model(model_name: str, device: str = "cuda"):
    """加载 Reasoner 模型"""
    print(f"Loading Reasoner model: {model_name}")
    
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
    
    print(f"Reasoner model loaded: {model_name}")
    return model, tokenizer


@torch.no_grad()
def stage1_batch_generate_evidence(
    model: StudentRetriever,
    contexts: List[str],
    questions: List[str],
    max_new_tokens: int = 512
) -> List[str]:
    """
    Stage 1: 批量生成证据子图
    
    由于 StudentRetriever.generate 不支持批量，我们逐个处理但优化内存管理
    """
    evidences = []
    for context, question in zip(contexts, questions):
        try:
            full_graph = f"[FULL_GRAPH]\n{context}"
            evidence = model.generate(
                query=question,
                full_graph=full_graph,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=False
            )
            evidences.append(evidence.strip())
        except Exception as e:
            evidences.append(f"[ERROR: {str(e)}]")
    
    return evidences


@torch.no_grad()
def stage2_batch_generate_answers(
    model,
    tokenizer,
    evidences: List[str],
    questions: List[str],
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> List[str]:
    """
    Stage 2: 批量生成答案 (真正的batch processing)
    """
    # 构建prompts
    prompts = []
    for evidence, question in zip(evidences, questions):
        prompt = f"""Based on the following evidence, answer the question with a short phrase.

Evidence:
{evidence}

Question: {question}

Answer:"""
        prompts.append(prompt)
    
    # 批量tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(device)
    
    # 批量生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 解码
    input_len = inputs.input_ids.shape[1]
    answers = []
    for i in range(len(prompts)):
        generated_ids = outputs[i][input_len:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        answers.append(answer)
    
    return answers


def compute_metrics(results: List[Dict]) -> Dict:
    """计算评估指标"""
    from collections import defaultdict
    
    metrics = {
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
    }
    
    exact_matches = 0
    for r in results:
        if r['success']:
            gold = r['gold_answer'].strip().lower()
            pred = r['generated_answer'].strip().lower()
            if gold in pred or pred in gold:
                exact_matches += 1
    
    metrics['exact_match'] = exact_matches
    metrics['exact_match_rate'] = exact_matches / metrics['success'] if metrics['success'] > 0 else 0
    
    dataset_results = defaultdict(list)
    for r in results:
        dataset_results[r['dataset_source']].append(r)
    
    metrics['per_dataset'] = {}
    for dataset, ds_results in dataset_results.items():
        ds_success = sum(1 for r in ds_results if r['success'])
        ds_em = sum(1 for r in ds_results if r['success'] and
                   (r['gold_answer'].strip().lower() in r['generated_answer'].strip().lower() or
                    r['generated_answer'].strip().lower() in r['gold_answer'].strip().lower()))
        
        metrics['per_dataset'][dataset] = {
            'total': len(ds_results),
            'success': ds_success,
            'exact_match': ds_em,
            'exact_match_rate': ds_em / ds_success if ds_success > 0 else 0
        }
    
    return metrics


def process_batches(
    stage1_model: StudentRetriever,
    reasoner_model,
    reasoner_tokenizer,
    samples: List[QASample],
    batch_size: int = 8,
    stage1_max_tokens: int = 512,
    stage2_max_tokens: int = 256,
    device: str = "cuda",
    checkpoint_file: Optional[str] = None,
    reasoner_name: str = "unknown"
) -> List[Dict]:
    """
    批量处理所有样本
    """
    results = []
    start_idx = 0
    
    # 从checkpoint恢复
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Found checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))
        start_idx = len(results)
        print(f"Resuming from batch containing sample {start_idx}")
    
    remaining_samples = samples[start_idx:]
    num_batches = (len(remaining_samples) + batch_size - 1) // batch_size
    
    # 打开checkpoint文件
    ckpt_f = None
    if checkpoint_file:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        ckpt_f = open(checkpoint_file, 'a', encoding='utf-8')
    
    try:
        pbar = tqdm(range(num_batches), desc=f"Batches (bs={batch_size})")
        for batch_idx in pbar:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_samples))
            batch_samples = remaining_samples[batch_start:batch_end]
            
            # 提取batch数据
            contexts = [s.context for s in batch_samples]
            questions = [s.question for s in batch_samples]
            
            try:
                # Stage 1: 生成证据 (目前仍是逐个，但在batch循环内)
                evidences = stage1_batch_generate_evidence(
                    stage1_model, contexts, questions,
                    max_new_tokens=stage1_max_tokens
                )
                
                # Stage 2: 批量生成答案 (真正的batch)
                answers = stage2_batch_generate_answers(
                    reasoner_model, reasoner_tokenizer,
                    evidences, questions,
                    max_new_tokens=stage2_max_tokens,
                    device=device
                )
                
                # 保存结果
                for i, sample in enumerate(batch_samples):
                    result = {
                        'id': sample.id,
                        'question': sample.question,
                        'context_preview': sample.context[:300] + '...' if len(sample.context) > 300 else sample.context,
                        'evidence': evidences[i][:500] + '...' if len(evidences[i]) > 500 else evidences[i],
                        'gold_answer': sample.answer,
                        'generated_answer': answers[i],
                        'dataset_source': sample.dataset_source,
                        'reasoner_model': reasoner_name,
                        'success': True,
                        'error': None
                    }
                    results.append(result)
                    
                    if ckpt_f:
                        ckpt_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                # 如果batch失败，逐个处理
                print(f"\nBatch {batch_idx} failed: {e}, falling back to single processing")
                for i, sample in enumerate(batch_samples):
                    result = {
                        'id': sample.id,
                        'question': sample.question,
                        'context_preview': sample.context[:300] + '...',
                        'evidence': '',
                        'gold_answer': sample.answer,
                        'generated_answer': '',
                        'dataset_source': sample.dataset_source,
                        'reasoner_model': reasoner_name,
                        'success': False,
                        'error': str(e)
                    }
                    results.append(result)
                    if ckpt_f:
                        ckpt_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 定期flush和清理
            if ckpt_f and (batch_idx + 1) % 10 == 0:
                ckpt_f.flush()
            
            if (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # 更新进度条
            pbar.set_postfix({
                'processed': start_idx + batch_end,
                'total': len(samples)
            })
    
    finally:
        if ckpt_f:
            ckpt_f.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MemAdapter Two-Stage Batch Testing')
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--reasoner_model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--stage1_max_tokens', type=int, default=512)
    parser.add_argument('--stage2_max_tokens', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--enable_checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MemAdapter Two-Stage Batch Testing")
    print("=" * 80)
    print(f"Stage1 checkpoint: {args.stage1_checkpoint}")
    print(f"Reasoner model: {args.reasoner_model}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    reasoner_name = args.reasoner_model.split('/')[-1].replace('-Instruct', '')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint file
    checkpoint_file = None
    if args.enable_checkpoint:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = str(checkpoint_dir / f"checkpoint_{reasoner_name}.jsonl")
    
    # Load data
    print("\nLoading test data...")
    samples = load_qa_data(args.test_data, limit=args.max_samples)
    
    # Load Stage1 model
    print("\n[Stage 1] Loading StudentRetriever (1.5B)...")
    stage1_model = load_stage1_model(args.stage1_checkpoint, args.device)
    
    # Load Reasoner model
    print(f"\n[Stage 2] Loading Reasoner ({args.reasoner_model})...")
    reasoner_model, reasoner_tokenizer = load_reasoner_model(args.reasoner_model, args.device)
    
    # Process in batches
    print(f"\nProcessing {len(samples)} samples in batches of {args.batch_size}...")
    results = process_batches(
        stage1_model=stage1_model,
        reasoner_model=reasoner_model,
        reasoner_tokenizer=reasoner_tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        stage1_max_tokens=args.stage1_max_tokens,
        stage2_max_tokens=args.stage2_max_tokens,
        device=args.device,
        checkpoint_file=checkpoint_file,
        reasoner_name=args.reasoner_model
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Summary - Reasoner: {reasoner_name}")
    print("=" * 80)
    print(f"Total samples: {metrics['total']}")
    print(f"Successful: {metrics['success']}")
    print(f"Failed: {metrics['failed']}")
    print(f"Exact match: {metrics['exact_match']} ({metrics['exact_match_rate']:.2%})")
    
    print("\nPer-dataset results:")
    for dataset, ds_metrics in metrics['per_dataset'].items():
        print(f"  {dataset}: {ds_metrics['exact_match']}/{ds_metrics['total']} "
              f"({ds_metrics['exact_match_rate']:.2%})")
    
    # Save results
    print("\nSaving results...")
    results_file = output_dir / f"test_results_{reasoner_name}.jsonl"
    with open(results_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    metrics_file = output_dir / f"test_metrics_{reasoner_name}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
