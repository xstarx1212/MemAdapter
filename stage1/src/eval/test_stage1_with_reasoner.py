"""
Stage1 + Reasoner 完整测试

流程:
1. Stage1 模型生成 evidence
2. Evidence 拼接到 context
3. Reasoner (1.5B/3B/7B) 生成答案
4. 评估答案质量 (EM, F1)
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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
    """Exact Match"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1"""
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
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def format_stage1_input(question: str, context: str) -> str:
    """Stage1 输入格式"""
    return f"""[QUERY]
{question}

[CONTEXT]
{context}

[GENERATE_EVIDENCE]
"""


def format_reasoner_input(question: str, context: str, evidence: str) -> str:
    """Reasoner 输入格式 (带 evidence)"""
    return f"""Based on the following context and extracted evidence, answer the question concisely.

Context:
{context[:3000]}

Extracted Evidence:
{evidence}

Question: {question}

Answer:"""


def format_reasoner_input_no_evidence(question: str, context: str) -> str:
    """Reasoner 输入格式 (不带 evidence，作为对照)"""
    return f"""Based on the following context, answer the question concisely.

Context:
{context[:4000]}

Question: {question}

Answer:"""


def load_stage1_model(model_path: str, base_model: str = "Qwen/Qwen2.5-1.5B"):
    """加载 Stage1 模型"""
    print(f"[Stage1] 加载 base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"[Stage1] 加载 LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(base_model_obj, model_path)
    model.eval()
    
    return model, tokenizer


def load_reasoner_model(model_name: str, device: str = "cuda"):
    """加载 Reasoner 模型"""
    print(f"[Reasoner] 加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    return model, tokenizer


def generate_evidence(model, tokenizer, question: str, context: str, max_new_tokens: int = 256) -> Tuple[str, bool]:
    """
    生成 evidence
    返回: (evidence, is_valid)
    """
    input_text = format_stage1_input(question, context)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    evidence = tokenizer.decode(generated, skip_special_tokens=True)
    
    # 检查 evidence 是否有效
    # 1. 不应该太短
    # 2. 不应该包含 [QUERY] 等循环标记
    is_valid = True
    if len(evidence.strip()) < 20:
        is_valid = False
    if "[QUERY]" in evidence or "[CONTEXT]" in evidence:
        # 截断到第一个 [QUERY] 或 [CONTEXT]
        for marker in ["[QUERY]", "[CONTEXT]", "[END_OF_TEXT]"]:
            if marker in evidence:
                evidence = evidence.split(marker)[0].strip()
        is_valid = len(evidence.strip()) >= 20
    
    return evidence.strip(), is_valid


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """生成答案"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    
    # 截取第一行或第一句作为答案
    answer = answer.split('\n')[0].strip()
    if '.' in answer:
        answer = answer.split('.')[0].strip()
    
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_model', type=str, required=True, help='Stage1 模型路径')
    parser.add_argument('--stage1_base', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--reasoner_model', type=str, required=True, help='Reasoner 模型名称')
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--evidence_max_tokens', type=int, default=256)
    parser.add_argument('--answer_max_tokens', type=int, default=64)
    parser.add_argument('--compare_no_evidence', action='store_true', help='同时测试不带 evidence 的版本')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Stage1 + Reasoner 完整测试")
    print("=" * 70)
    print(f"Stage1 Model: {args.stage1_model}")
    print(f"Reasoner Model: {args.reasoner_model}")
    print(f"Test Data: {args.test_data}")
    print(f"Output: {args.output}")
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
    
    # 统计
    total_em = 0.0
    total_f1 = 0.0
    total_em_no_ev = 0.0
    total_f1_no_ev = 0.0
    valid_evidence_count = 0
    invalid_evidence_count = 0
    
    results = []
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(test_data, desc="Testing")):
            question = item['question']
            context = item['context']
            gold_answer = item['answer']
            
            # Step 1: 生成 evidence
            evidence, is_valid = generate_evidence(
                stage1_model, stage1_tokenizer,
                question, context,
                max_new_tokens=args.evidence_max_tokens
            )
            
            if is_valid:
                valid_evidence_count += 1
            else:
                invalid_evidence_count += 1
            
            # Step 2: 用 Reasoner 生成答案 (带 evidence)
            reasoner_prompt = format_reasoner_input(question, context, evidence)
            predicted_answer = generate_answer(
                reasoner_model, reasoner_tokenizer,
                reasoner_prompt,
                max_new_tokens=args.answer_max_tokens
            )
            
            # 计算指标
            em = compute_em(predicted_answer, gold_answer)
            f1 = compute_f1(predicted_answer, gold_answer)
            total_em += em
            total_f1 += f1
            
            result = {
                'id': item['id'],
                'question': question,
                'gold_answer': gold_answer,
                'evidence': evidence[:500],
                'evidence_valid': is_valid,
                'predicted_answer': predicted_answer,
                'em': em,
                'f1': f1,
                'dataset_source': item.get('dataset_source', '')
            }
            
            # 可选：同时测试不带 evidence 的版本
            if args.compare_no_evidence:
                prompt_no_ev = format_reasoner_input_no_evidence(question, context)
                answer_no_ev = generate_answer(
                    reasoner_model, reasoner_tokenizer,
                    prompt_no_ev,
                    max_new_tokens=args.answer_max_tokens
                )
                em_no_ev = compute_em(answer_no_ev, gold_answer)
                f1_no_ev = compute_f1(answer_no_ev, gold_answer)
                total_em_no_ev += em_no_ev
                total_f1_no_ev += f1_no_ev
                
                result['answer_no_evidence'] = answer_no_ev
                result['em_no_evidence'] = em_no_ev
                result['f1_no_evidence'] = f1_no_ev
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            results.append(result)
            
            # 每 50 个样本打印进度
            if (idx + 1) % 50 == 0:
                avg_em = total_em / (idx + 1)
                avg_f1 = total_f1 / (idx + 1)
                valid_ratio = valid_evidence_count / (idx + 1)
                print(f"\n[Progress {idx+1}/{len(test_data)}]")
                print(f"  EM: {avg_em:.4f}, F1: {avg_f1:.4f}")
                print(f"  Valid Evidence: {valid_ratio:.2%} ({valid_evidence_count}/{idx+1})")
                if args.compare_no_evidence:
                    avg_em_no = total_em_no_ev / (idx + 1)
                    avg_f1_no = total_f1_no_ev / (idx + 1)
                    print(f"  [No Evidence] EM: {avg_em_no:.4f}, F1: {avg_f1_no:.4f}")
    
    # 最终结果
    n = len(results)
    avg_em = total_em / n
    avg_f1 = total_f1 / n
    valid_ratio = valid_evidence_count / n
    
    print("\n" + "=" * 70)
    print("最终结果")
    print("=" * 70)
    print(f"样本数: {n}")
    print(f"Reasoner: {args.reasoner_model}")
    print(f"\n[With Evidence]")
    print(f"  EM: {avg_em:.4f}")
    print(f"  F1: {avg_f1:.4f}")
    print(f"  Valid Evidence: {valid_ratio:.2%} ({valid_evidence_count}/{n})")
    
    if args.compare_no_evidence:
        avg_em_no = total_em_no_ev / n
        avg_f1_no = total_f1_no_ev / n
        print(f"\n[Without Evidence (Baseline)]")
        print(f"  EM: {avg_em_no:.4f}")
        print(f"  F1: {avg_f1_no:.4f}")
        print(f"\n[Improvement]")
        print(f"  EM: +{(avg_em - avg_em_no):.4f}")
        print(f"  F1: +{(avg_f1 - avg_f1_no):.4f}")
    
    # 保存汇总
    summary = {
        'reasoner_model': args.reasoner_model,
        'stage1_model': args.stage1_model,
        'num_samples': n,
        'with_evidence': {
            'em': avg_em,
            'f1': avg_f1,
            'valid_evidence_ratio': valid_ratio
        }
    }
    if args.compare_no_evidence:
        summary['without_evidence'] = {
            'em': avg_em_no,
            'f1': avg_f1_no
        }
    
    summary_path = args.output.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == "__main__":
    main()
