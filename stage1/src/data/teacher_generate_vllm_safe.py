"""
使用 vLLM 服务器进行教师生成（安全版本：实时写入 + 断点续传）

安全特性：
1. 实时写入：每个样本处理完立即写入，避免数据丢失
2. 断点续传：启动时检查已有结果，自动跳过已处理的样本
3. 追加模式：支持增量处理，不会覆盖已有结果
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Any
import asyncio
import aiohttp
from tqdm import tqdm
import torch


def load_prompt_template(prompt_path: str) -> str:
    """加载 prompt 模板"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_step1_prompt(template: str, documents: List[str]) -> str:
    """格式化 Step 1 prompt (生成完整图)"""
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
    prompt = template.replace("{DOCUMENTS}", docs_text)
    prompt = prompt.replace("{DOCS}", docs_text)
    return prompt


def format_step2_prompt(template: str, query: str, full_graph: str) -> str:
    """格式化 Step 2 prompt (生成证据子图)"""
    prompt = template.replace("{QUESTION}", query)
    prompt = prompt.replace("{QUERY}", query)
    prompt = prompt.replace("{FULL_GRAPH}", full_graph)
    return prompt


def parse_step1_output(output_text: str) -> str:
    """解析 Step 1 输出，提取完整图"""
    full_start = output_text.find("[FULL_GRAPH]")
    if full_start == -1:
        raise ValueError("Step 1 输出格式错误: 缺少 [FULL_GRAPH] 标记")
    return output_text[full_start:].strip()


def parse_step2_output(output_text: str) -> Tuple[str, float]:
    """解析 Step 2 输出，提取证据子图和置信度"""
    evidence_start = output_text.find("[EVIDENCE_SUBGRAPH]")
    if evidence_start == -1:
        raise ValueError("Step 2 输出格式错误: 缺少 [EVIDENCE_SUBGRAPH] 标记")
    
    confidence_start = output_text.find("[CONFIDENCE]")
    if confidence_start == -1:
        subgraph = output_text[evidence_start:].strip()
        return subgraph, 0.5
    
    subgraph = output_text[evidence_start:confidence_start].strip()
    confidence_text = output_text[confidence_start:].strip()
    
    import re
    match = re.search(r'([0-9]*\.?[0-9]+)', confidence_text)
    if match:
        confidence = float(match.group(1))
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.5
    
    return subgraph, confidence


def load_existing_results(output_file: str) -> Tuple[Dict[str, dict], Set[str]]:
    """
    加载已有结果，返回结果字典和已处理的样本ID集合
    
    Returns:
        (results_dict, processed_ids): 结果字典和已处理ID集合
    """
    if not os.path.exists(output_file):
        return {}, set()
    
    results_dict = {}
    processed_ids = set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    sample_id = result.get('id')
                    if sample_id:
                        processed_ids.add(sample_id)
                        results_dict[sample_id] = result
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"警告: 读取已有结果文件失败: {e}")
        return {}, set()
    
    return results_dict, processed_ids


def write_result_safe(output_file: str, result: dict, mode: str = 'a'):
    """
    安全写入单个结果（追加模式）
    
    Args:
        output_file: 输出文件路径
        result: 结果字典
        mode: 写入模式 ('a' 追加, 'w' 覆盖)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用追加模式，确保不会丢失数据
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()  # 立即刷新到磁盘
        os.fsync(f.fileno())  # 强制同步到磁盘


class VLLMClient:
    """vLLM 客户端，支持多服务器负载均衡"""
    
    def __init__(self, api_urls: List[str]):
        self.api_urls = api_urls
        self.current_server = 0
        self.session = None
        self._loop = None
    
    def _get_next_server(self) -> str:
        """轮询获取下一个服务器"""
        url = self.api_urls[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.api_urls)
        return url
    
    async def _ensure_session(self):
        """确保 session 已创建"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        异步生成
        
        Returns:
            If return_logprobs=False: str (生成的文本)
            If return_logprobs=True: Dict with 'text' and 'logprobs'
        """
        await self._ensure_session()
        
        url = self._get_next_server()
        api_url = f"{url}/v1/completions"
        
        payload = {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": None
        }
        
        # 如果需要 logprobs，添加参数
        if return_logprobs:
            payload["logprobs"] = 1  # 返回 top-1 logprob（可以增加到更大的值）
        
        try:
            await self._ensure_session()
            async with self.session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as response:
                if response.status == 200:
                    result = await response.json()
                    choice = result['choices'][0]
                    
                    if return_logprobs:
                        return {
                            "text": choice['text'],
                            "logprobs": choice.get('logprobs', None),
                            "tokens": choice.get('logprobs', {}).get('tokens', []) if choice.get('logprobs') else []
                        }
                    else:
                        return choice['text']
                else:
                    error_text = await response.text()
                    raise Exception(f"vLLM API 错误 (状态码 {response.status}): {error_text}")
        except asyncio.TimeoutError:
            raise Exception(f"vLLM API 超时: {url}")
        except Exception as e:
            print(f"警告: 服务器 {url} 失败: {e}，尝试下一个服务器...", flush=True)
            url = self._get_next_server()
            api_url = f"{url}/v1/completions"
            # 确保 session 仍然有效
            await self._ensure_session()
            async with self.session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as response:
                if response.status == 200:
                    result = await response.json()
                    choice = result['choices'][0]
                    
                    if return_logprobs:
                        return {
                            "text": choice['text'],
                            "logprobs": choice.get('logprobs', None),
                            "tokens": choice.get('logprobs', {}).get('tokens', []) if choice.get('logprobs') else []
                        }
                    else:
                        return choice['text']
                else:
                    error_text = await response.text()
                    raise Exception(f"所有服务器都失败，最后错误: {error_text}")
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()


async def generate_batch_async(
    client: VLLMClient,
    prompts: List[str],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_concurrent: int = 10,
    return_logprobs: bool = False
) -> List[Union[str, Dict[str, Any]]]:
    """
    批量异步生成
    
    Returns:
        If return_logprobs=False: List[str] (生成的文本列表)
        If return_logprobs=True: List[Dict] (包含 'text' 和 'logprobs' 的字典列表)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_one(prompt: str):
        async with semaphore:
            return await client.generate_async(
                prompt, max_tokens, temperature, top_p, return_logprobs=return_logprobs
            )
    
    tasks = [generate_one(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    outputs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"警告: 样本 {i} 生成失败: {result}", flush=True)
            if return_logprobs:
                outputs.append({"text": "", "logprobs": None, "tokens": []})
            else:
                outputs.append("")
        else:
            outputs.append(result)
    
    return outputs


async def main_async():
    """异步主函数"""
    parser = argparse.ArgumentParser(description="使用 vLLM 服务器进行教师生成（安全版本）")
    parser.add_argument('--dataset', type=str, required=True, help='输入数据集 (JSONL)')
    parser.add_argument('--out', type=str, required=True, help='输出文件路径')
    parser.add_argument('--vllm_urls', type=str, nargs='+', 
                       default=['http://localhost:8000', 'http://localhost:8001'],
                       help='vLLM 服务器 URL 列表')
    parser.add_argument('--step1_prompt', type=str, 
                       default='prompts/step1_full_graph_strict.txt',
                       help='Step 1 prompt 模板')
    parser.add_argument('--step2_prompt', type=str,
                       default='prompts/step2_evidence_subgraph_strict.txt',
                       help='Step 2 prompt 模板')
    parser.add_argument('--max_docs', type=int, default=4, help='每个样本最多使用的文档数')
    parser.add_argument('--max_tokens_step1', type=int, default=4096, help='Step 1 最大生成 token 数')
    parser.add_argument('--max_tokens_step2', type=int, default=1024, help='Step 2 最大生成 token 数')
    parser.add_argument('--max_samples', type=int, default=None, help='最多处理的样本数')
    parser.add_argument('--cache_full_graphs', type=str, default=None, help='完整图缓存文件路径')
    parser.add_argument('--batch_size', type=int, default=20, help='批量大小（并发数）')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p 采样')
    parser.add_argument('--force', action='store_true', help='强制重新处理所有样本（忽略已有结果）')
    parser.add_argument('--write_interval', type=int, default=1, help='每处理N个样本写入一次（默认1，即实时写入）')
    parser.add_argument('--start_index', type=int, default=None, help='起始样本索引（用于多GPU分配）')
    parser.add_argument('--end_index', type=int, default=None, help='结束样本索引（用于多GPU分配，不包含）')
    parser.add_argument('--max_doc_tokens', type=int, default=None, help='最大文档token长度（用于按长度过滤）')
    parser.add_argument('--min_doc_tokens', type=int, default=None, help='最小文档token长度（用于按长度过滤）')
    parser.add_argument('--save_teacher_logprobs', action='store_true', help='保存 teacher logprobs 用于 KL 蒸馏（vLLM 支持）')
    parser.add_argument('--logprobs_dir', type=str, default='data/teacher_supervision/logprobs', help='logprobs 保存目录')
    
    args = parser.parse_args()
    
    print("="*70, flush=True)
    print("使用 vLLM 服务器进行教师生成（安全版本：实时写入 + 断点续传）", flush=True)
    print("="*70, flush=True)
    print(f"vLLM 服务器: {args.vllm_urls}", flush=True)
    print(f"输入数据集: {args.dataset}", flush=True)
    print(f"输出文件: {args.out}", flush=True)
    print(f"批量大小: {args.batch_size}", flush=True)
    print(f"写入间隔: {args.write_interval} (实时写入)", flush=True)
    print(flush=True)
    
    # 检查已有结果（断点续传）
    existing_results, processed_ids = load_existing_results(args.out)
    
    if existing_results:
        if args.force:
            print(f"🔄 强制模式：忽略已有 {len(existing_results)} 个结果，重新处理所有样本")
            existing_results = {}
            processed_ids = set()
            # 如果强制模式，删除旧文件
            if os.path.exists(args.out):
                os.remove(args.out)
        else:
            print(f"📋 发现已有结果: {len(existing_results)} 个样本")
            print(f"⏭️  将跳过已处理的样本，继续处理剩余样本")
    else:
        print("🚀 开始新任务，处理所有样本")
    
    print()
    
    # 加载 prompt 模板
    step1_template = load_prompt_template(args.step1_prompt)
    step2_template = load_prompt_template(args.step2_prompt)
    
    # 加载数据集
    print("加载数据集...", flush=True)
    dataset = []
    line_count = 0
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  已加载 {line_count} 个样本...", flush=True)
    
    original_count = len(dataset)
    print(f"原始数据集: {original_count} 个样本", flush=True)
    
    # 计算所有样本的文档长度（用于排序和过滤）
    print("计算文档长度...", flush=True)
    for i, sample in enumerate(dataset):
        doc_length = sum(len(doc.split()) for doc in sample.get('documents', []))
        sample['doc_token_length'] = doc_length
        if (i + 1) % 10000 == 0:
            print(f"  已计算 {i + 1}/{original_count} 个样本的文档长度...", flush=True)
    
    # 正确的逻辑：先按文档长度分类，再在每个类别中选择样本
    # 这样可以确保从所有数据中选择，而不是先限制数量再分类
    print("按文档长度分类（先分类，再选择样本）...", flush=True)
    
    # 第一步：按文档长度过滤（分类）
    # 支持多个长度范围：如果同时设置了min_doc_tokens和max_doc_tokens，且min_doc_tokens <= max_doc_tokens，
    # 则处理该范围内的样本；否则，如果只设置了其中一个，则处理单边范围
    if args.min_doc_tokens is not None or args.max_doc_tokens is not None:
        filtered_dataset = []
        for sample in dataset:
            doc_length = sample.get('doc_token_length', 0)
            
            # 检查长度范围
            # 如果同时设置了min和max，且min <= max，则处理该范围内的样本
            # 如果min > max，则处理两个范围：<= max 或 >= min（用于处理不连续的范围）
            if args.min_doc_tokens is not None and args.max_doc_tokens is not None:
                if args.min_doc_tokens <= args.max_doc_tokens:
                    # 正常范围：min <= length <= max
                    if doc_length < args.min_doc_tokens or doc_length > args.max_doc_tokens:
                        continue
                else:
                    # 异常情况：min > max，处理两个范围：<= max 或 >= min
                    if doc_length > args.max_doc_tokens and doc_length < args.min_doc_tokens:
                        continue
            elif args.min_doc_tokens is not None:
                if doc_length < args.min_doc_tokens:
                    continue
            elif args.max_doc_tokens is not None:
                if doc_length > args.max_doc_tokens:
                    continue
            
            filtered_dataset.append(sample)
        
        dataset = filtered_dataset
        if args.min_doc_tokens is not None and args.max_doc_tokens is not None and args.min_doc_tokens > args.max_doc_tokens:
            print(f"按文档长度过滤: {len(dataset)} 个样本 (长度范围: <= {args.max_doc_tokens} 或 >= {args.min_doc_tokens} tokens)", flush=True)
        else:
            print(f"按文档长度过滤: {len(dataset)} 个样本 (长度范围: {args.min_doc_tokens or 0} - {args.max_doc_tokens or '∞'} tokens)", flush=True)
    
    # 第二步：在过滤后的类别中，按文档长度排序（短文档优先，课程学习策略）
    print("在类别内按文档长度排序（短文档优先，符合课程学习策略）...", flush=True)
    dataset = sorted(dataset, key=lambda x: x.get('doc_token_length', 0))
    if dataset:
        print(f"排序完成，最短: {dataset[0].get('doc_token_length', 0)} tokens, 最长: {dataset[-1].get('doc_token_length', 0)} tokens", flush=True)
    else:
        print("排序完成，但数据集为空", flush=True)
    
    # 第三步：应用最大样本数限制（课程学习：只标注前N个高质量样本）
    # 注意：如果同时设置了索引范围，max_samples应该至少等于END_INDEX，以确保能取到索引范围内的所有样本
    if args.start_index is not None or args.end_index is not None:
        # 如果设置了索引范围，确保max_samples至少等于END_INDEX
        end_idx = args.end_index if args.end_index is not None else len(dataset)
        if args.max_samples and args.max_samples < end_idx:
            print(f"警告: max_samples ({args.max_samples}) < end_index ({end_idx})，将max_samples调整为 {end_idx} 以确保能取到索引范围内的所有样本", flush=True)
            args.max_samples = end_idx
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"限制最大样本数: {args.max_samples} (课程学习策略：从该类别中选择前N个样本用于初始训练)", flush=True)
    
    # 应用样本范围过滤（用于多GPU分配，如果指定了索引范围）
    # 注意：这个在max_samples之后，用于进一步细分
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(dataset)
        dataset = dataset[start_idx:end_idx]
        print(f"应用样本索引范围: [{start_idx}:{end_idx}] (从 {len(dataset) + (end_idx - start_idx) - len(dataset)} 个样本中)")
    
    print(f"最终处理: {len(dataset)} 个样本")
    if dataset:
        print(f"  文档长度范围: {min(s['doc_token_length'] for s in dataset)} - {max(s['doc_token_length'] for s in dataset)} tokens")
        print(f"  平均长度: {sum(s['doc_token_length'] for s in dataset) / len(dataset):.0f} tokens")
    
    # 保存选中的样本ID列表（用于追踪和验证）
    selected_sample_ids = [ex.get('id') for ex in dataset]
    selected_ids_file = args.out.replace('.jsonl', '_selected_ids.jsonl')
    with open(selected_ids_file, 'w', encoding='utf-8') as f:
        for sample_id in selected_sample_ids:
            f.write(json.dumps({'id': sample_id}, ensure_ascii=False) + '\n')
    print(f"📋 已保存选中的样本ID列表: {selected_ids_file} ({len(selected_sample_ids)} 个样本)")
    print()
    
    # 过滤已处理的样本
    if not args.force and processed_ids:
        original_count = len(dataset)
        dataset = [ex for ex in dataset if ex.get('id') not in processed_ids]
        skipped_count = original_count - len(dataset)
        print(f"⏭️  跳过 {skipped_count} 个已处理的样本")
        print(f"📝 剩余 {len(dataset)} 个样本需要处理")
    
    if not dataset:
        print("✅ 所有样本都已处理完成！")
        return
    
    print()
    
    # 初始化 vLLM 客户端
    client = VLLMClient(args.vllm_urls)
    
    # 创建 logprobs 保存目录（如果需要）
    logprobs_dir = None
    tokenizer = None
    if args.save_teacher_logprobs:
        logprobs_dir = Path(args.logprobs_dir)
        logprobs_dir.mkdir(parents=True, exist_ok=True)
        print(f"将保存 teacher logprobs 到: {logprobs_dir}")
        
        # 加载 tokenizer（用于 tokenization，确保与 teacher 模型一致）
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-32B-Instruct",
            trust_remote_code=True
        )
        print(f"已加载 tokenizer (vocab_size={tokenizer.vocab_size})")
        print()
    
    # 完整图缓存（支持断点续传）
    full_graphs_cache = {}
    cache_file_exists = args.cache_full_graphs and os.path.exists(args.cache_full_graphs)
    if cache_file_exists:
        print(f"加载完整图缓存: {args.cache_full_graphs}")
        with open(args.cache_full_graphs, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cache_item = json.loads(line)
                    full_graphs_cache[cache_item['docs_hash']] = cache_item['gold_full_graph']
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效的缓存行: {e}")
        print(f"已加载 {len(full_graphs_cache)} 个缓存的完整图")
        print()
    
    # 准备缓存文件的写入模式（追加模式，支持断点续传）
    cache_write_mode = 'a' if cache_file_exists else 'w'
    
    # Step 1: 生成完整图
    print("="*70)
    print("Step 1: 生成完整记忆图")
    print("="*70)
    print()
    
    unique_docs = {}
    for example in dataset:
        documents = example['documents'][:args.max_docs]
        docs_hash = hash(tuple(sorted(documents)))
        if docs_hash not in unique_docs:
            unique_docs[docs_hash] = documents
    
    print(f"唯一文档集合数: {len(unique_docs)}")
    print()
    
    step1_prompts = []
    step1_hashes = []
    
    for docs_hash, documents in unique_docs.items():
        if docs_hash in full_graphs_cache:
            continue
        
        prompt = format_step1_prompt(step1_template, documents)
        step1_prompts.append(prompt)
        step1_hashes.append(docs_hash)
    
    if step1_prompts:
        print(f"需要生成 {len(step1_prompts)} 个完整图...")
        
        start_time = time.time()
        step1_outputs_raw = await generate_batch_async(
            client,
            step1_prompts,
            max_tokens=args.max_tokens_step1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_concurrent=args.batch_size,
            return_logprobs=False  # Step 1 不需要 logprobs
        )
        
        # 提取文本（兼容旧接口）
        step1_outputs = []
        for output in step1_outputs_raw:
            if isinstance(output, dict):
                step1_outputs.append(output.get('text', ''))
            else:
                step1_outputs.append(output)
        step1_time = time.time() - start_time
        
        print(f"Step 1 完成: {step1_time:.2f}s ({len(step1_prompts)/step1_time:.2f} 样本/秒)")
        print()
        
        # 解析并缓存（实时写入，支持断点续传）
        # 注意: asyncio.gather 保持顺序，所以 step1_outputs[i] 对应 step1_hashes[i]
        assert len(step1_outputs) == len(step1_hashes), "Step 1 输出数量与hash数量不匹配"
        
        # 实时写入：每生成一个完整图立即追加写入，避免任务中断导致数据丢失
        newly_generated_count = 0
        if args.cache_full_graphs:
            cache_file_handle = open(args.cache_full_graphs, cache_write_mode, encoding='utf-8')
        
        try:
            for i, output in enumerate(step1_outputs):
                if output:
                    try:
                        full_graph = parse_step1_output(output)
                        docs_hash = step1_hashes[i]
                        
                        # 只保存新生成的完整图（避免重复写入）
                        if docs_hash not in full_graphs_cache:
                            full_graphs_cache[docs_hash] = full_graph
                            newly_generated_count += 1
                            
                            # 立即写入缓存文件（实时写入，支持断点续传）
                            if args.cache_full_graphs:
                                cache_file_handle.write(json.dumps({
                                    'docs_hash': docs_hash,
                                    'gold_full_graph': full_graph
                                }, ensure_ascii=False) + '\n')
                                cache_file_handle.flush()  # 强制刷新到磁盘
                    except Exception as e:
                        print(f"警告: 解析完整图失败 (hash={step1_hashes[i]}): {e}", flush=True)
        finally:
            if args.cache_full_graphs:
                cache_file_handle.close()
        
        if args.cache_full_graphs:
            print(f"✅ 已实时保存 {newly_generated_count} 个新生成的完整图到缓存: {args.cache_full_graphs}")
            print(f"   缓存文件总记录数: {len(full_graphs_cache)}")
            print()
    else:
        print("所有完整图都已缓存，跳过 Step 1")
        print()
    
    # Step 2: 生成证据子图（实时写入）
    print("="*70)
    print("Step 2: 生成证据子图（实时写入）")
    print("="*70)
    print()
    
    # 确定写入模式
    write_mode = 'a' if os.path.exists(args.out) and not args.force else 'w'
    
    # 准备需要处理的样本
    step2_tasks = []
    for i, example in enumerate(dataset):
        # 如果已有结果且不是强制模式，跳过
        if not args.force and example.get('id') in processed_ids:
            continue
        
        documents = example['documents'][:args.max_docs]
        docs_hash = hash(tuple(sorted(documents)))
        
        if docs_hash not in full_graphs_cache:
            print(f"警告: 样本 {example['id']} 的完整图未找到，跳过")
            continue
        
        full_graph = full_graphs_cache[docs_hash]
        query = example['query']
        
        prompt = format_step2_prompt(step2_template, query, full_graph)
        step2_tasks.append({
            'index': i,
            'example': example,
            'prompt': prompt,
            'docs_hash': docs_hash
        })
    
    print(f"需要生成 {len(step2_tasks)} 个证据子图...")
    print(f"写入模式: {'追加' if write_mode == 'a' else '新建'}")
    print()
    
    # 批量处理，但实时写入
    total_processed = 0
    total_successful = 0
    start_time = time.time()
    
    # 分批处理
    batch_size = args.batch_size
    for batch_start in range(0, len(step2_tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(step2_tasks))
        batch_tasks = step2_tasks[batch_start:batch_end]
        
        # 提取 prompts
        batch_prompts = [task['prompt'] for task in batch_tasks]
        
        # 批量生成（支持 logprobs）
        batch_outputs_raw = await generate_batch_async(
            client,
            batch_prompts,
            max_tokens=args.max_tokens_step2,
            temperature=args.temperature,
            top_p=args.top_p,
            max_concurrent=batch_size,
            return_logprobs=args.save_teacher_logprobs
        )
        
        # 处理并实时写入每个结果
        # 注意: asyncio.gather 保持顺序，所以 batch_outputs_raw[i] 对应 batch_tasks[i]
        assert len(batch_outputs_raw) == len(batch_tasks), f"Step 2 批次输出数量不匹配: {len(batch_outputs_raw)} != {len(batch_tasks)}"
        for task, output_raw in zip(batch_tasks, batch_outputs_raw):
            # 提取文本和 logprobs
            if isinstance(output_raw, dict):
                output = output_raw.get('text', '')
                logprobs_data = output_raw.get('logprobs', None)
            else:
                output = output_raw
                logprobs_data = None
            example = task['example']
            docs_hash = task['docs_hash']
            
            # 验证: 确保使用正确的完整图
            if docs_hash not in full_graphs_cache:
                print(f"错误: 样本 {example['id']} 的完整图未找到 (hash={docs_hash})")
                result = {
                    'id': example['id'],
                    'documents': example['documents'],
                    'query': example['query'],
                    'success': False,
                    'error': f'完整图未找到 (hash={docs_hash})'
                }
                write_result_safe(args.out, result, mode=write_mode)
                write_mode = 'a'
                total_processed += 1
                continue
            
            if not output:
                result = {
                    'id': example['id'],
                    'documents': example['documents'],
                    'query': example['query'],
                    'success': False,
                    'error': '生成失败'
                }
            else:
                try:
                    subgraph, confidence = parse_step2_output(output)
                    full_graph = full_graphs_cache[docs_hash]
                    
                    # Strict subset validation (paper Appendix B.1):
                    # The evidence subgraph must be consistent with the full graph
                    # in terms of node IDs and (src, dst, relation).
                    validation_passed = True
                    validation_errors = []
                    try:
                        from src.data.serialize_graph import GraphParser
                        parser = GraphParser()
                        full_parsed = parser.parse_graph(full_graph)
                        sub_parsed = parser.parse_graph(subgraph)
                        validation_passed, validation_errors = parser.validate_graph(full_parsed, sub_parsed)
                    except Exception as e:
                        validation_passed = False
                        validation_errors = [f"验证解析错误: {str(e)}"]

                    # Reject strict validation failures
                    if validation_passed:
                        result = {
                            'id': example['id'],
                            'documents': example['documents'],
                            'query': example['query'],
                            'gold_full_graph': full_graph,
                            'gold_subgraph': subgraph,
                            'confidence': confidence,
                            'doc_token_length': sum(len(d.split()) for d in example['documents']),
                            'success': True,
                            'validation_passed': validation_passed,
                            'validation_errors': validation_errors if validation_errors else None
                        }
                        
                        # 保存 logprobs（如果启用）- 完整版本
                        if args.save_teacher_logprobs and logprobs_data is not None and tokenizer is not None:
                            logprobs_path = logprobs_dir / f"{example['id']}.pt"
                            
                            # Tokenize 生成的文本
                            generated_token_ids = tokenizer.encode(
                                output,
                                add_special_tokens=False
                            )
                            
                            # Tokenize 完整 prompt（用于对齐）
                            full_prompt = format_step2_prompt(step2_template, query, full_graph)
                            prompt_token_ids = tokenizer.encode(
                                full_prompt,
                                add_special_tokens=True
                            )
                            
                            # 完整序列 token IDs
                            full_sequence_token_ids = prompt_token_ids + generated_token_ids
                            
                            # 保存完整信息
                            torch.save({
                                # Logprobs 数据（来自 vLLM）
                                'logprobs': logprobs_data,
                                'tokens': output_raw.get('tokens', []) if isinstance(output_raw, dict) else [],
                                
                                # Token IDs（用于对齐）
                                'generated_token_ids': generated_token_ids,
                                'prompt_token_ids': prompt_token_ids,
                                'full_sequence_token_ids': full_sequence_token_ids,
                                
                                # 文本（用于验证）
                                'generated_text': output,
                                'prompt_text': full_prompt,
                                
                                # 序列信息
                                'prompt_length': len(prompt_token_ids),
                                'generated_length': len(generated_token_ids),
                                'sequence_length': len(full_sequence_token_ids),
                                
                                # 模型和参数
                                'model_name': 'Qwen/Qwen2.5-32B-Instruct',
                                'vocab_size': tokenizer.vocab_size,
                                'temperature': args.temperature,
                                'top_p': args.top_p,
                                'max_tokens': args.max_tokens_step2,
                                'step': 2,  # Step 2 生成
                                
                                # 元数据
                                'sample_id': example['id'],
                                'timestamp': time.time()
                            }, logprobs_path)
                            result['teacher_logprobs_path'] = str(logprobs_path)
                        
                        total_successful += 1
                    else:
                        # Strict mode: reject invalid subgraph
                        result = {
                        'id': example['id'],
                        'documents': example['documents'],
                        'query': example['query'],
                        'gold_full_graph': full_graph,
                        'gold_subgraph': subgraph,
                        # 严格验证失败的样本：将 confidence 置为 0，避免后续课程学习选择器抽样
                        'confidence': 0.0,
                        'doc_token_length': sum(len(d.split()) for d in example['documents']),
                        'success': False,
                        'validation_passed': False,
                        'validation_errors': validation_errors,
                        'error': 'Subset validation failed (graph is not a strict subgraph).'
                        }
                        
                        # 保存 logprobs（如果启用，即使验证有问题也保存）- 完整版本
                        if args.save_teacher_logprobs and logprobs_data is not None and tokenizer is not None:
                            logprobs_path = logprobs_dir / f"{example['id']}.pt"
                            
                            # Tokenize 生成的文本
                            generated_token_ids = tokenizer.encode(
                                output,
                                add_special_tokens=False
                            )
                            
                            # Tokenize 完整 prompt（用于对齐）
                            full_prompt = format_step2_prompt(step2_template, query, full_graph)
                            prompt_token_ids = tokenizer.encode(
                                full_prompt,
                                add_special_tokens=True
                            )
                            
                            # 完整序列 token IDs
                            full_sequence_token_ids = prompt_token_ids + generated_token_ids
                            
                            # 保存完整信息
                            torch.save({
                                # Logprobs 数据（来自 vLLM）
                                'logprobs': logprobs_data,
                                'tokens': output_raw.get('tokens', []) if isinstance(output_raw, dict) else [],
                                
                                # Token IDs（用于对齐）
                                'generated_token_ids': generated_token_ids,
                                'prompt_token_ids': prompt_token_ids,
                                'full_sequence_token_ids': full_sequence_token_ids,
                                
                                # 文本（用于验证）
                                'generated_text': output,
                                'prompt_text': full_prompt,
                                
                                # 序列信息
                                'prompt_length': len(prompt_token_ids),
                                'generated_length': len(generated_token_ids),
                                'sequence_length': len(full_sequence_token_ids),
                                
                                # 模型和参数
                                'model_name': 'Qwen/Qwen2.5-32B-Instruct',
                                'vocab_size': tokenizer.vocab_size,
                                'temperature': args.temperature,
                                'top_p': args.top_p,
                                'max_tokens': args.max_tokens_step2,
                                'step': 2,  # Step 2 生成
                                
                                # 元数据
                                'sample_id': example['id'],
                                'timestamp': time.time()
                            }, logprobs_path)
                            result['teacher_logprobs_path'] = str(logprobs_path)
                        
                        total_successful += 1
                except Exception as e:
                    result = {
                        'id': example['id'],
                        'documents': example['documents'],
                        'query': example['query'],
                        'success': False,
                        'error': str(e)
                    }
            
            # 实时写入（每个样本处理完立即写入）
            write_result_safe(args.out, result, mode=write_mode)
            write_mode = 'a'  # 后续都使用追加模式
            
            total_processed += 1
            
            # 进度显示
            if total_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"进度: {total_processed}/{len(step2_tasks)} ({total_processed/len(step2_tasks)*100:.1f}%) | "
                      f"成功: {total_successful} | 速度: {rate:.2f} 样本/秒")
    
    total_time = time.time() - start_time
    print()
    print(f"Step 2 完成: {total_time:.2f}s ({len(step2_tasks)/total_time:.2f} 样本/秒)")
    print()
    
    # 最终统计
    all_results, _ = load_existing_results(args.out)
    total_count = len(all_results)
    successful_count = sum(1 for r in all_results.values() if r.get('success', False))
    
    print("="*70)
    print("最终统计")
    print("="*70)
    print(f"总样本数: {total_count}")
    print(f"成功: {successful_count} ({successful_count/total_count*100:.1f}%)")
    print(f"失败: {total_count - successful_count}")
    print(f"结果文件: {args.out}")
    print("="*70)
    
    # 关闭客户端
    await client.close()


def main():
    """同步入口函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
