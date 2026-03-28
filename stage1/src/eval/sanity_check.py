"""
健全性检查: 验证训练数据的质量
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.serialize_graph import validate_teacher_output, GraphParser


def check_format(example: dict) -> Tuple[bool, List[str]]:
    """检查单个样本的格式"""
    errors = []
    
    # 检查必需字段
    required_fields = ['id', 'documents', 'query', 'gold_full_graph', 'gold_subgraph']
    for field in required_fields:
        if field not in example:
            errors.append(f"缺少必需字段: {field}")
    
    if errors:
        return False, errors
    
    # 检查图的有效性
    is_valid, graph_errors = validate_teacher_output(
        example['gold_full_graph'],
        example['gold_subgraph']
    )
    
    if not is_valid:
        errors.extend(graph_errors)
    
    return len(errors) == 0, errors


def check_dataset(data_path: str, max_samples: int = None, verbose: bool = False):
    """检查整个数据集"""
    print(f"正在检查数据集: {data_path}")
    print()
    
    # 读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    
    if max_samples:
        examples = examples[:max_samples]
    
    print(f"样本总数: {len(examples)}")
    print()
    
    # 统计
    valid_count = 0
    invalid_count = 0
    error_types = {}
    
    # 检查每个样本
    for i, example in enumerate(tqdm(examples, desc="检查样本")):
        is_valid, errors = check_format(example)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            
            # 统计错误类型
            for error in errors:
                error_types[error] = error_types.get(error, 0) + 1
            
            # 如果 verbose，打印错误
            if verbose:
                print(f"\n样本 {example.get('id', i)} 无效:")
                for error in errors:
                    print(f"  - {error}")
    
    # 输出统计
    print("\n" + "="*50)
    print("检查结果:")
    print("="*50)
    print(f"有效样本: {valid_count} ({valid_count/len(examples)*100:.1f}%)")
    print(f"无效样本: {invalid_count} ({invalid_count/len(examples)*100:.1f}%)")
    
    if error_types:
        print("\n常见错误类型:")
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for error, count in sorted_errors[:10]:
            print(f"  - {error}: {count} 次")
    
    return valid_count, invalid_count


def inspect_samples(data_path: str, num_samples: int = 5):
    """随机检查几个样本"""
    import random
    
    print(f"正在随机抽取 {num_samples} 个样本进行检查...")
    print()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    
    samples = random.sample(examples, min(num_samples, len(examples)))
    
    parser = GraphParser()
    
    for i, example in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"样本 {i}: {example['id']}")
        print(f"{'='*60}")
        
        print(f"\n查询: {example['query']}")
        
        print(f"\n文档数: {len(example['documents'])}")
        for j, doc in enumerate(example['documents'][:2], 1):
            print(f"\n文档 {j} (前200字符):")
            print(doc[:200] + "...")
        
        print(f"\n完整图:")
        print(example['gold_full_graph'][:500] + "...")
        
        print(f"\n证据子图:")
        print(example['gold_subgraph'])
        
        # 解析并显示统计
        full = parser.parse_graph(example['gold_full_graph'])
        sub = parser.parse_graph(example['gold_subgraph'])
        
        print(f"\n图统计:")
        print(f"  完整图: {len(full['nodes'])} 节点, {len(full['edges'])} 边")
        print(f"  证据子图: {len(sub['nodes'])} 节点, {len(sub['edges'])} 边")
        
        # 验证
        is_valid, errors = check_format(example)
        print(f"\n有效性: {'✓ 有效' if is_valid else '✗ 无效'}")
        if errors:
            print("错误:")
            for error in errors:
                print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description="健全性检查工具")
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--mode', type=str, default='check', 
                       choices=['check', 'inspect'], 
                       help='检查模式')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='最多检查的样本数')
    parser.add_argument('--num_inspect', type=int, default=5, 
                       help='inspect 模式下查看的样本数')
    parser.add_argument('--verbose', action='store_true', 
                       help='显示详细错误信息')
    
    args = parser.parse_args()
    
    if args.mode == 'check':
        check_dataset(args.data, max_samples=args.max_samples, verbose=args.verbose)
    elif args.mode == 'inspect':
        inspect_samples(args.data, num_samples=args.num_inspect)


if __name__ == "__main__":
    main()
