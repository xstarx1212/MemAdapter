"""
使用教师模型 (Qwen2.5-72B) 生成图监督信号。

两阶段生成策略:
Step 1: 从文档生成完整记忆图 G* (不考虑 query)
Step 2: 从 G* 和 query 生成证据子图 r*

支持两种模式:
- Mode A: 保存 logits 用于 KL 蒸馏
- Mode B: 只保存文本用于交叉熵
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt_template(prompt_path: str) -> str:
    """加载 prompt 模板"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_step1_prompt(template: str, documents: List[str]) -> str:
    """格式化 Step 1 prompt (生成完整图)"""
    # 合并文档
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
    
    # 替换占位符（支持新旧两种格式）
    prompt = template.replace("{DOCUMENTS}", docs_text)
    prompt = prompt.replace("{DOCS}", docs_text)  # 向后兼容
    
    return prompt


def format_step2_prompt(template: str, query: str, full_graph: str) -> str:
    """格式化 Step 2 prompt (生成证据子图)"""
    # 替换占位符（支持新旧两种格式）
    prompt = template.replace("{QUESTION}", query)
    prompt = prompt.replace("{QUERY}", query)  # 向后兼容
    prompt = prompt.replace("{FULL_GRAPH}", full_graph)
    
    return prompt


def parse_step1_output(output_text: str) -> str:
    """
    解析 Step 1 输出，提取完整图
    """
    full_start = output_text.find("[FULL_GRAPH]")
    
    if full_start == -1:
        raise ValueError("Step 1 输出格式错误: 缺少 [FULL_GRAPH] 标记")
    
    # 提取完整图 (从 [FULL_GRAPH] 到结尾)
    full_graph = output_text[full_start:].strip()
    
    return full_graph


def parse_step2_output(output_text: str) -> Tuple[str, float]:
    """
    解析 Step 2 输出，提取证据子图和置信度
    
    Returns:
        (subgraph, confidence)
    """
    evidence_start = output_text.find("[EVIDENCE_SUBGRAPH]")
    
    if evidence_start == -1:
        raise ValueError("Step 2 输出格式错误: 缺少 [EVIDENCE_SUBGRAPH] 标记")
    
    # 查找 [CONFIDENCE] 标记
    confidence_start = output_text.find("[CONFIDENCE]")
    
    if confidence_start == -1:
        # 如果没有置信度，返回默认值
        subgraph = output_text[evidence_start:].strip()
        return subgraph, 0.5
    
    # 提取证据子图 (从 [EVIDENCE_SUBGRAPH] 到 [CONFIDENCE])
    subgraph = output_text[evidence_start:confidence_start].strip()
    
    # 提取置信度
    confidence_text = output_text[confidence_start:].strip()
    
    # 解析置信度数字
    import re
    match = re.search(r'([0-9]*\.?[0-9]+)', confidence_text)
    
    if match:
        confidence = float(match.group(1))
        # 确保在 [0, 1] 范围内
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.5  # 默认值
    
    return subgraph, confidence


def generate_with_teacher(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    save_logits: bool = False,
    topk_logits: int = 50
) -> Dict:
    """
    使用教师模型生成图结构
    
    返回:
        {
            "output_text": str,
            "logits": Optional[List[torch.Tensor]]  # 如果 save_logits=True
        }
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        if save_logits:
            # Mode A: 保存 logits
            # 使用 generate 但需要 return_dict_in_generate=True 和 output_scores=True
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # outputs.scores 是每个生成步骤的 logits 元组
            # 每个元素形状: (batch_size, vocab_size)
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 保存 top-k logits 以节省空间
            topk_logits_list = []
            for step_logits in outputs.scores:
                # step_logits: (1, vocab_size)
                step_logits = step_logits[0]  # (vocab_size,)
                topk_vals, topk_indices = torch.topk(step_logits, k=topk_logits)
                topk_logits_list.append({
                    "indices": topk_indices.cpu(),
                    "values": topk_vals.cpu()
                })
            
            return {
                "output_text": output_text,
                "logits": topk_logits_list,
                "generated_ids": generated_ids.cpu()
            }
        else:
            # Mode B: 只生成文本
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return {
                "output_text": output_text,
                "logits": None
            }


def main():
    parser = argparse.ArgumentParser(description="生成教师监督信号 (两阶段)")
    parser.add_argument('--dataset', type=str, required=True, help='输入数据集 (JSONL)')
    parser.add_argument('--out', type=str, required=True, help='输出文件路径')
    parser.add_argument('--teacher_model', type=str, default='Qwen/Qwen2.5-72B-Instruct')
    parser.add_argument('--step1_prompt', type=str, default='prompts/step1_full_graph.txt', 
                       help='Step 1 prompt (生成完整图)')
    parser.add_argument('--step2_prompt', type=str, default='prompts/step2_evidence_subgraph.txt',
                       help='Step 2 prompt (生成证据子图)')
    parser.add_argument('--max_docs', type=int, default=4, help='每个样本最多使用的文档数')
    parser.add_argument('--max_tokens_step1', type=int, default=4096, help='Step 1 最大生成 token 数')
    parser.add_argument('--max_tokens_step2', type=int, default=1024, help='Step 2 最大生成 token 数')
    parser.add_argument('--save_teacher_logits', type=bool, default=False, help='是否保存教师 logits (Mode A)')
    parser.add_argument('--topk_logits', type=int, default=50, help='保存 top-k logits')
    parser.add_argument('--logits_dir', type=str, default='data/teacher_supervision/logits')
    parser.add_argument('--max_samples', type=int, default=None, help='最多处理的样本数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cache_full_graphs', type=str, default=None, 
                       help='缓存完整图的文件路径 (避免重复生成)')
    
    args = parser.parse_args()
    
    print("=== 教师模型生成配置 (两阶段) ===")
    print(f"教师模型: {args.teacher_model}")
    print(f"输入数据: {args.dataset}")
    print(f"输出文件: {args.out}")
    print(f"保存 logits: {args.save_teacher_logits}")
    print(f"设备: {args.device}")
    print(f"完整图缓存: {args.cache_full_graphs}")
    print()
    
    # 加载 prompt 模板
    step1_template = load_prompt_template(args.step1_prompt)
    step2_template = load_prompt_template(args.step2_prompt)
    
    # 加载教师模型
    print("正在加载教师模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("教师模型加载完成！")
    print()
    
    # 准备输出目录
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.save_teacher_logits:
        logits_dir = Path(args.logits_dir)
        logits_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据集
    print(f"正在读取数据集: {args.dataset}")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
    print(f"数据集大小: {len(dataset)}")
    print()
    
    # 加载或生成完整图缓存
    full_graphs_cache = {}
    docs_to_process = {}  # {docs_hash: [example_ids]}
    
    if args.cache_full_graphs and Path(args.cache_full_graphs).exists():
        print(f"正在加载完整图缓存: {args.cache_full_graphs}")
        with open(args.cache_full_graphs, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                full_graphs_cache[item['docs_hash']] = item['gold_full_graph']
        print(f"已加载 {len(full_graphs_cache)} 个缓存的完整图")
    
    # 第一遍: 收集需要生成的文档集合
    print("\n第一阶段: 分析数据集...")
    for example in tqdm(dataset, desc="分析样本"):
        documents = example['documents'][:args.max_docs]
        # 使用文档内容的哈希作为键
        docs_hash = hash(tuple(documents))
        
        if docs_hash not in full_graphs_cache:
            if docs_hash not in docs_to_process:
                docs_to_process[docs_hash] = {
                    'documents': documents,
                    'example_ids': []
                }
            docs_to_process[docs_hash]['example_ids'].append(example['id'])
    
    print(f"需要生成 {len(docs_to_process)} 个唯一的完整图")
    
    # Step 1: 生成完整图 (针对唯一的文档集合)
    if docs_to_process:
        print("\nStep 1: 生成完整记忆图...")
        for docs_hash, doc_info in tqdm(docs_to_process.items(), desc="生成完整图"):
            try:
                documents = doc_info['documents']
                
                # 构建 Step 1 prompt
                prompt = format_step1_prompt(step1_template, documents)
                
                # 生成完整图
                teacher_output = generate_with_teacher(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens_step1,
                    save_logits=False  # Step 1 不需要保存 logits
                )
                
                # 解析输出
                full_graph = parse_step1_output(teacher_output['output_text'])
                
                # 缓存
                full_graphs_cache[docs_hash] = full_graph
                
            except Exception as e:
                print(f"\n警告: 生成完整图时出错 (hash={docs_hash}): {e}")
                continue
        
        # 保存完整图缓存
        if args.cache_full_graphs:
            cache_path = Path(args.cache_full_graphs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                for docs_hash, full_graph in full_graphs_cache.items():
                    f.write(json.dumps({
                        'docs_hash': docs_hash,
                        'gold_full_graph': full_graph
                    }, ensure_ascii=False) + '\n')
            print(f"\n已保存完整图缓存: {cache_path}")
    
    # Step 2: 为每个 query 生成证据子图
    print("\nStep 2: 生成证据子图...")
    results = []
    for i, example in enumerate(tqdm(dataset, desc="生成证据子图")):
        try:
            # 限制文档数量
            documents = example['documents'][:args.max_docs]
            query = example['query']
            docs_hash = hash(tuple(documents))
            
            # 获取完整图
            if docs_hash not in full_graphs_cache:
                print(f"\n警告: 样本 {example['id']} 的完整图未找到，跳过")
                continue
            
            full_graph = full_graphs_cache[docs_hash]
            
            # 构建 Step 2 prompt
            prompt = format_step2_prompt(step2_template, query, full_graph)
            
            # 生成证据子图
            teacher_output = generate_with_teacher(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens_step2,
                save_logits=args.save_teacher_logits,
                topk_logits=args.topk_logits
            )
            
            # 解析输出（包含置信度）
            subgraph, confidence = parse_step2_output(teacher_output['output_text'])
            
            # 计算文档总长度（用于后续排序）
            doc_token_length = sum(len(doc.split()) for doc in documents)
            
            # 构建结果
            result = {
                "id": example['id'],
                "documents": documents,
                "query": query,
                "gold_full_graph": full_graph,
                "gold_subgraph": subgraph,
                "confidence": confidence,
                "doc_token_length": doc_token_length
            }
            
            # 保存 logits (如果需要)
            if args.save_teacher_logits and teacher_output['logits'] is not None:
                logits_path = logits_dir / f"{example['id']}.pt"
                torch.save({
                    "logits": teacher_output['logits'],
                    "generated_ids": teacher_output['generated_ids']
                }, logits_path)
                result['teacher_logits_path'] = str(logits_path)
            
            results.append(result)
            
        except Exception as e:
            print(f"\n警告: 处理样本 {example['id']} 时出错: {e}")
            continue
    
    # 保存结果
    print(f"\n正在保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"完成！成功生成 {len(results)} 个样本的监督信号。")


if __name__ == "__main__":
    main()
