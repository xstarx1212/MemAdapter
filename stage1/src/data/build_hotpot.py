"""
Download and preprocess HotpotQA dataset.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path) -> None:
    """下载文件"""
    print(f"正在下载: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"下载完成: {output_path}")


def parse_hotpot_example(example: Dict) -> Dict:
    """
    解析 HotpotQA 样本
    
    输入格式:
    {
        "_id": "...",
        "question": "...",
        "answer": "...",
        "supporting_facts": [[title, sent_id], ...],
        "context": [[title, [sent1, sent2, ...]], ...]
    }
    """
    # 提取文档
    documents = []
    for title, sentences in example['context']:
        # 合并该文档的所有句子
        doc_text = f"Title: {title}\n" + " ".join(sentences)
        documents.append(doc_text)
    
    # 构建输出
    output = {
        "id": example['_id'],
        "documents": documents,
        "query": example['question'],
        "answer": example.get('answer', ''),
        "supporting_facts": example.get('supporting_facts', []),
        "level": example.get('level', 'unknown'),
        "type": example.get('type', 'unknown')
    }
    
    return output


def process_hotpot_file(input_path: Path, output_path: Path, max_samples: int = None) -> None:
    """处理 HotpotQA JSON 文件"""
    print(f"正在处理: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = []
    for i, example in enumerate(tqdm(data, desc="处理样本")):
        if max_samples and i >= max_samples:
            break
        
        try:
            parsed = parse_hotpot_example(example)
            processed.append(parsed)
        except Exception as e:
            print(f"警告: 处理样本 {example.get('_id', i)} 时出错: {e}")
            continue
    
    # 保存为 JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存 {len(processed)} 个样本到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="下载并预处理 HotpotQA 数据集")
    parser.add_argument('--out', type=str, required=True, help='输出目录')
    parser.add_argument('--max_train', type=int, default=None, help='最大训练样本数')
    parser.add_argument('--max_dev', type=int, default=None, help='最大验证样本数')
    parser.add_argument('--skip_download', action='store_true', help='跳过下载，直接处理')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # HotpotQA 下载链接
    urls = {
        'train': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json',
        'dev_distractor': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
    }
    
    # 下载原始文件
    if not args.skip_download:
        for split, url in urls.items():
            output_file = out_dir / f"{split}_raw.json"
            if not output_file.exists():
                download_file(url, output_file)
            else:
                print(f"文件已存在，跳过下载: {output_file}")
    
    # 处理文件
    print("\n开始处理数据...")
    
    # 训练集
    train_raw = out_dir / "train_raw.json"
    if train_raw.exists():
        process_hotpot_file(
            train_raw, 
            out_dir / "train.jsonl",
            max_samples=args.max_train
        )
    
    # 验证集
    dev_raw = out_dir / "dev_distractor_raw.json"
    if dev_raw.exists():
        process_hotpot_file(
            dev_raw,
            out_dir / "dev.jsonl",
            max_samples=args.max_dev
        )
    
    print("\n完成！")
    print(f"输出目录: {out_dir}")
    print(f"  - train.jsonl")
    print(f"  - dev.jsonl")


if __name__ == "__main__":
    main()
