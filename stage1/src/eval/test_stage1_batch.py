"""
Stage1 + Reasoner 批量测试

支持批量生成以充分利用 H200 GPU
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def normalize_answer(s: str) -> str:
    """标准化答案"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        import string
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def format_stage1_input(question: str, context: str) -> str:
    return f"""[QUERY]
{question}

[CONTEXT]
{context}

[GENERATE_EVIDENCE]
"""


def format_reasoner_input(question: str, context: str, evidence: str) -> str:
    return f"""Answer the question with ONLY the answer entity/value. Do NOT include explanations or full sentences.

Context:
{context[:3000]}

Evidence:
{evidence}

Question: {question}

Answer (only the answer, nothing else):"""


def load_stage1_model(model_path: str, base_model: str = "Qwen/Qwen2.5-1.5B"):
    print(f"[Stage1] 加载模型: {base_model} + {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 批量生成需要左填充
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model_obj, model_path)
    model.eval()
    return model, tokenizer


def load_reasoner_model(model_name: str):
    print(f"[Reasoner] 加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 批量生成需要左填充
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def batch_generate_evidence(model, tokenizer, batch_data: List[Dict], max_new_tokens: int = 256, batch_size: int = 8) -> List[Tuple[str, bool]]:
    """批量生成 evidence"""
    results = []
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        input_texts = [format_stage1_input(item['question'], item['context']) for item in batch]
        
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            generated = output[inputs['input_ids'].shape[1]:]
            evidence = tokenizer.decode(generated, skip_special_tokens=True)
            
            # 清理 evidence
            is_valid = True
            if len(evidence.strip()) < 20:
                is_valid = False
            for marker in ["[QUERY]", "[CONTEXT]", "[END_OF_TEXT]"]:
                if marker in evidence:
                    evidence = evidence.split(marker)[0].strip()
            if len(evidence.strip()) < 20:
                is_valid = False
            
            results.append((evidence.strip(), is_valid))
    
    return results


def batch_generate_answers(model, tokenizer, prompts: List[str], max_new_tokens: int = 64, batch_size: int = 8) -> List[str]:
    """批量生成答案"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            generated = output[inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            answer = answer.split('\n')[0].strip()
            if '.' in answer:
                answer = answer.split('.')[0].strip()
            results.append(answer)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_model', type=str, required=True)
    parser.add_argument('--stage1_base', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--reasoner_model', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--memory_output', type=str, default=None, help='保存完整 memory/evidence 的文件路径')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--evidence_max_tokens', type=int, default=256)
    parser.add_argument('--answer_max_tokens', type=int, default=64)
    parser.add_argument('--evidence_batch_size', type=int, default=8)
    parser.add_argument('--answer_batch_size', type=int, default=16)
    args = parser.parse_args()
    
    # 自动生成 memory 输出路径
    if args.memory_output is None:
        args.memory_output = args.output.replace('.jsonl', '_memory.jsonl')
    
    print("=" * 70)
    print("Stage1 + Reasoner 批量测试")
    print("=" * 70)
    print(f"Stage1 Model: {args.stage1_model}")
    print(f"Reasoner Model: {args.reasoner_model}")
    print(f"Evidence Batch Size: {args.evidence_batch_size}")
    print(f"Answer Batch Size: {args.answer_batch_size}")
    print("=" * 70)
    
    # 加载模型
    stage1_model, stage1_tokenizer = load_stage1_model(args.stage1_model, args.stage1_base)
    reasoner_model, reasoner_tokenizer = load_reasoner_model(args.reasoner_model)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_data = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    # 分块处理
    chunk_size = 500  # 每次处理 500 个样本
    results = []
    total_em, total_f1, valid_count = 0.0, 0.0, 0
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # 打开两个输出文件：结果文件 和 memory文件
    memory_file = open(args.memory_output, 'w', encoding='utf-8')
    print(f"Memory 将保存到: {args.memory_output}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk_start in tqdm(range(0, len(test_data), chunk_size), desc="Processing chunks"):
            chunk = test_data[chunk_start:chunk_start + chunk_size]
            
            # Step 1: 批量生成 evidence
            evidence_results = batch_generate_evidence(
                stage1_model, stage1_tokenizer, chunk,
                max_new_tokens=args.evidence_max_tokens,
                batch_size=args.evidence_batch_size
            )
            
            # Step 2: 准备 Reasoner 输入
            reasoner_prompts = []
            for i, item in enumerate(chunk):
                evidence, _ = evidence_results[i]
                prompt = format_reasoner_input(item['question'], item['context'], evidence)
                reasoner_prompts.append(prompt)
            
            # Step 3: 批量生成答案
            answers = batch_generate_answers(
                reasoner_model, reasoner_tokenizer, reasoner_prompts,
                max_new_tokens=args.answer_max_tokens,
                batch_size=args.answer_batch_size
            )
            
            # Step 4: 计算指标并保存
            for i, item in enumerate(chunk):
                evidence, is_valid = evidence_results[i]
                predicted_answer = answers[i]
                gold_answer = item['answer']
                
                em = compute_em(predicted_answer, gold_answer)
                f1 = compute_f1(predicted_answer, gold_answer)
                
                total_em += em
                total_f1 += f1
                if is_valid:
                    valid_count += 1
                
                result = {
                    'id': item['id'],
                    'question': item['question'],
                    'gold_answer': gold_answer,
                    'evidence': evidence[:500],
                    'evidence_valid': is_valid,
                    'predicted_answer': predicted_answer,
                    'em': em,
                    'f1': f1,
                    'dataset_source': item.get('dataset_source', '')
                }
                
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                results.append(result)
                
                # 保存完整的 memory/evidence 信息
                memory_record = {
                    'id': item['id'],
                    'question': item['question'],
                    'context': item['context'],
                    'evidence': evidence,  # 完整的 evidence，不截断
                    'evidence_valid': is_valid,
                    'gold_answer': gold_answer,
                    'dataset_source': item.get('dataset_source', '')
                }
                memory_file.write(json.dumps(memory_record, ensure_ascii=False) + '\n')
            
            f.flush()
            memory_file.flush()
            
            # 打印进度
            n = len(results)
            print(f"\n[Progress {n}/{len(test_data)}] EM: {total_em/n:.4f}, F1: {total_f1/n:.4f}, Valid Evidence: {valid_count/n:.2%}")
    
    memory_file.close()
    
    # 最终结果
    n = len(results)
    print("\n" + "=" * 70)
    print("最终结果")
    print("=" * 70)
    print(f"样本数: {n}")
    print(f"EM: {total_em/n:.4f}")
    print(f"F1: {total_f1/n:.4f}")
    print(f"Valid Evidence: {valid_count/n:.2%}")
    
    # 保存汇总
    summary = {
        'stage1_model': args.stage1_model,
        'reasoner_model': args.reasoner_model,
        'num_samples': n,
        'em': total_em/n,
        'f1': total_f1/n,
        'valid_evidence_ratio': valid_count/n
    }
    summary_path = args.output.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == "__main__":
    main()
