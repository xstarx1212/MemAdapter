#!/usr/bin/env python3
"""
Stage1 模型 Memory 提取脚本

在 stage2_alignment_merged.jsonl 上提取 Stage1 模型的 memory 信息（显式、隐式），
与 StreamingLLM/MemoryLLM 的 memory 格式保持一致。

输出格式:
- stage2_alignment_stage1_results.jsonl: 生成的 evidence subgraph
- stage2_alignment_stage1_results_memory_explicit.jsonl: 显式 memory (生成的 subgraph)
- stage2_alignment_stage1_results_memory_implicit.npy: 隐式 memory (aligned embedding)
- stage2_alignment_stage1_results_memory_meta.json: 元信息
"""

import json
import argparse
import os
import sys
import torch
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
    print("[WARNING] PEFT not available. LoRA loading may fail.")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add stage1 src to path
stage1_src = "/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Memadapter/stage1/src"
if stage1_src not in sys.path:
    sys.path.insert(0, stage1_src)

from model.student import StudentRetriever


@dataclass
class AlignmentSample:
    """Alignment 样本数据结构"""
    id: str
    context: str
    question: str
    answer: str
    dataset_source: str


def load_alignment_data(data_path: str, limit: Optional[int] = None) -> List[AlignmentSample]:
    """加载 stage2_alignment_merged.jsonl 格式的数据"""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line.strip())
            sample = AlignmentSample(
                id=data.get('id', f'sample_{i}'),
                context=data.get('context', ''),
                question=data.get('question', ''),
                answer=data.get('answer', ''),
                dataset_source=data.get('dataset_source', 'unknown')
            )
            samples.append(sample)
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def format_input(question: str, context: str) -> str:
    """格式化输入，与训练时一致"""
    return f"""[QUERY]
{question}

[CONTEXT]
{context}

[GENERATE_EVIDENCE]
"""


def load_stage1_model(
    checkpoint_path: str,
    base_model: str,
    device: str = "cuda"
) -> Tuple:
    """
    加载 Stage1 训练好的模型
    
    返回:
        - model: PeftModel (LoRA 微调后的模型，用于生成)
        - tokenizer: tokenizer
        - student_model: StudentRetriever (用于提取隐式 memory)
    """
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading base model: {base_model}")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter for generation
    model = base_model_obj
    if PEFT_AVAILABLE:
        print("Loading LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(base_model_obj, str(checkpoint_path))
            print("LoRA adapter loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter: {e}")
            print("Using base model only")
    
    model.eval()
    
    # Create StudentRetriever for implicit memory extraction
    student_model = StudentRetriever(
        model_name_or_path=base_model,
        device=device,
        frozen_anchor_align=False
    )
    
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    student_model.model_dtype = model_dtype
    student_model.anchor_align = student_model.anchor_align.to(device).to(dtype=model_dtype)
    student_model.prefix_projection = student_model.prefix_projection.to(device).to(dtype=model_dtype)
    
    # Load anchor components for implicit memory
    anchor_components_path = checkpoint_path / "anchor_components.pt"
    if anchor_components_path.exists():
        print("Loading anchor components...")
        components = torch.load(anchor_components_path, map_location=device, weights_only=False)
        
        ckpt_d_model = components.get('d_model', 1536)
        current_d_model = student_model.d_model
        
        if ckpt_d_model == current_d_model:
            student_model.anchor_align.load_state_dict(components['anchor_align'])
            student_model.prefix_projection.load_state_dict(components['prefix_projection'])
            student_model.anchor_align = student_model.anchor_align.to(device).to(dtype=model_dtype)
            student_model.prefix_projection = student_model.prefix_projection.to(device).to(dtype=model_dtype)
            if 'prefix_length' in components:
                student_model.prefix_length = components['prefix_length']
            print(f"Anchor components loaded successfully (d_model={ckpt_d_model})")
        else:
            print(f"Warning: Dimension mismatch! Checkpoint d_model={ckpt_d_model}, model d_model={current_d_model}")
    else:
        print("Warning: anchor_components.pt not found, using default initialization")
    
    student_model.eval()
    return model, tokenizer, student_model


def generate_evidence(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> str:
    """
    使用格式化输入生成 evidence (与 quick_test_stage1.py 一致)
    """
    input_text = format_input(question, context)
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=4096
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 只取生成的部分
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result


def extract_implicit_memory(
    student_model: StudentRetriever,
    context: str
) -> np.ndarray:
    """
    提取隐式 memory (h_a)
    """
    with torch.no_grad():
        g = student_model.encode_full_graph(context)
        if len(g.shape) == 1:
            g = g.unsqueeze(0)
        h_a = student_model.anchor_align(g)
        implicit_memory = h_a.squeeze(0).cpu().float().numpy()
    return implicit_memory


def extract_implicit_memory_batch(
    student_model: StudentRetriever,
    contexts: List[str]
) -> np.ndarray:
    """
    批量提取隐式 memory
    """
    with torch.no_grad():
        batch_g = []
        for context in contexts:
            g = student_model.encode_full_graph(context)
            if len(g.shape) == 1:
                g = g.unsqueeze(0)
            batch_g.append(g)
        
        g_stacked = torch.cat(batch_g, dim=0)  # [batch, d_model]
        h_a = student_model.anchor_align(g_stacked)  # [batch, d_model]
        
        return h_a.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description='Extract memory from Stage1 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage1 checkpoint (e.g., outputs/stage1_full_40k/stage1_enhanced_xxx/best)')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-1.5B',
                       help='Base model name or path')
    parser.add_argument('--data_file', type=str, 
                       default='/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Memadapter/stage2/data/merged/stage2_alignment_merged.jsonl',
                       help='Path to alignment data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum new tokens to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for implicit memory extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Stage1 Memory Extraction for Stage2 Alignment")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Data file: {args.data_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model, tokenizer, student_model = load_stage1_model(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        device=args.device
    )
    
    # Get hidden dim for meta info
    hidden_dim = student_model.d_model
    print(f"Model hidden dimension: {hidden_dim}")
    
    # Load data
    print("\nLoading alignment data...")
    samples = load_alignment_data(args.data_file, limit=args.max_samples)
    
    # Process samples
    print(f"\nExtracting memory ({len(samples)} samples)...")
    
    results = []
    explicit_memories = []
    all_implicit_memories = []
    
    # 分批处理
    batch_size = args.batch_size
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
        
        # 1. 批量提取隐式 memory (使用 StudentRetriever)
        contexts = [s.context for s in batch_samples]
        batch_implicit = extract_implicit_memory_batch(student_model, contexts)
        
        # 2. 逐个生成 evidence (使用格式化输入 + LoRA model)
        for i, sample in enumerate(batch_samples):
            try:
                generated_text = generate_evidence(
                    model=model,
                    tokenizer=tokenizer,
                    question=sample.question,
                    context=sample.context,
                    max_new_tokens=args.max_new_tokens
                )
                success = True
                error = None
            except Exception as e:
                print(f"Warning: Generation failed for {sample.id}: {e}")
                generated_text = ""
                success = False
                error = str(e)
            
            # Result entry
            results.append({
                'id': sample.id,
                'question': sample.question,
                'context_preview': sample.context[:500] + '...' if len(sample.context) > 500 else sample.context,
                'gold_answer': sample.answer,
                'generated': generated_text,
                'dataset_source': sample.dataset_source,
                'success': success,
                'error': error
            })
            
            # Explicit memory entry
            explicit_memories.append({
                'id': sample.id,
                'memory_type': 'explicit',
                'memory_items': [
                    {'text': generated_text, 'metadata': {'type': 'evidence_subgraph'}}
                ],
                'num_memories': 1 if generated_text else 0
            })
            
            # Implicit memory
            all_implicit_memories.append(batch_implicit[i])
        
        # GPU cache cleanup after each batch
        torch.cuda.empty_cache()
    
    # Save results
    print("\nSaving results...")
    
    # 1. Main results
    results_path = os.path.join(args.output_dir, 'stage2_alignment_stage1_results.jsonl')
    with open(results_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Results saved to: {results_path}")
    
    # 2. Explicit memory
    explicit_path = os.path.join(args.output_dir, 'stage2_alignment_stage1_results_memory_explicit.jsonl')
    with open(explicit_path, 'w', encoding='utf-8') as f:
        for e in explicit_memories:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f"Explicit memory saved to: {explicit_path}")
    
    # 3. Implicit memory
    implicit_arr = np.stack(all_implicit_memories, axis=0)  # [N, hidden_dim]
    implicit_path = os.path.join(args.output_dir, 'stage2_alignment_stage1_results_memory_implicit.npy')
    np.save(implicit_path, implicit_arr)
    print(f"Implicit memory saved to: {implicit_path}")
    print(f"  Shape: {implicit_arr.shape}")
    
    # 4. Meta info
    meta = {
        'paradigm': 'stage1',
        'model': args.base_model,
        'checkpoint': args.checkpoint,
        'num_samples': len(samples),
        'hidden_dim': hidden_dim,
        'implicit_shape': list(implicit_arr.shape),
        'data_source': args.data_file
    }
    meta_path = os.path.join(args.output_dir, 'stage2_alignment_stage1_results_memory_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Meta info saved to: {meta_path}")
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples: {len(samples)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(samples) - success_count}")
    print(f"\nOutput files:")
    print(f"  - {results_path}")
    print(f"  - {explicit_path}")
    print(f"  - {implicit_path}")
    print(f"  - {meta_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
