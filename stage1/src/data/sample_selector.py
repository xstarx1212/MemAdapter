"""
样本选择器：基于 teacher 置信度筛选训练样本

核心原则：
- 只使用 teacher 的 confidence score
- 不使用任何规则打分
- 按置信度排序，支持阈值过滤
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_samples(input_paths: List[str]) -> List[Dict]:
    """加载所有样本"""
    samples = []
    
    for path in input_paths:
        if not Path(path).exists():
            print(f"警告: 文件不存在，跳过: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)
    
    return samples


def select_samples(
    samples: List[Dict],
    min_confidence: float = 0.0,
    max_samples: int = None,
    sort_by_confidence: bool = True
) -> List[Dict]:
    """
    选择样本
    
    Args:
        samples: 所有样本
        min_confidence: 最小置信度阈值
        max_samples: 最大样本数
        sort_by_confidence: 是否按置信度排序
    
    Returns:
        筛选后的样本列表
    """
    # 1. 按置信度排序（高到低）
    if sort_by_confidence:
        samples = sorted(
            samples, 
            key=lambda x: x.get('confidence', 0.0), 
            reverse=True
        )
    
    # 2. 过滤低置信度样本
    filtered = [
        s for s in samples
        if s.get('confidence', 0.0) >= min_confidence
    ]
    
    # 3. 限制数量
    if max_samples is not None and max_samples > 0:
        filtered = filtered[:max_samples]
    
    return filtered


def analyze_samples(samples: List[Dict]) -> Dict:
    """分析样本统计信息"""
    if not samples:
        return {}
    
    confidences = [s.get('confidence', 0.0) for s in samples]
    doc_lengths = [s.get('doc_token_length', 0) for s in samples]
    
    stats = {
        'total_samples': len(samples),
        'confidence': {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'median': sorted(confidences)[len(confidences) // 2]
        },
        'doc_length': {
            'mean': sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
            'min': min(doc_lengths) if doc_lengths else 0,
            'max': max(doc_lengths) if doc_lengths else 0
        },
        'confidence_distribution': {
            '0.9+': sum(1 for c in confidences if c >= 0.9),
            '0.8-0.9': sum(1 for c in confidences if 0.8 <= c < 0.9),
            '0.7-0.8': sum(1 for c in confidences if 0.7 <= c < 0.8),
            '0.6-0.7': sum(1 for c in confidences if 0.6 <= c < 0.7),
            '<0.6': sum(1 for c in confidences if c < 0.6)
        }
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="选择训练样本 (只基于 teacher 置信度)"
    )
    
    # 输入输出
    parser.add_argument(
        '--input', 
        type=str, 
        nargs='+', 
        required=True,
        help='输入文件路径（可以多个）'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='输出文件路径'
    )
    
    # 选择标准
    parser.add_argument(
        '--min_confidence', 
        type=float, 
        default=0.75,
        help='最小置信度阈值 (默认 0.75)'
    )
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=None,
        help='最大样本数 (默认无限制)'
    )
    
    # 分析
    parser.add_argument(
        '--analyze', 
        action='store_true',
        help='显示样本统计信息'
    )
    
    args = parser.parse_args()
    
    print("=== 样本选择器 ===")
    print(f"输入文件: {args.input}")
    print(f"最小置信度: {args.min_confidence}")
    print(f"最大样本数: {args.max_samples or '无限制'}")
    print()
    
    # 加载样本
    print("正在加载样本...")
    all_samples = load_samples(args.input)
    print(f"加载了 {len(all_samples)} 个样本")
    print()
    
    # 分析原始样本
    if args.analyze:
        print("=== 原始样本统计 ===")
        orig_stats = analyze_samples(all_samples)
        print(f"样本总数: {orig_stats['total_samples']}")
        print(f"置信度: 均值={orig_stats['confidence']['mean']:.3f}, "
              f"范围=[{orig_stats['confidence']['min']:.3f}, {orig_stats['confidence']['max']:.3f}]")
        print("\n置信度分布:")
        for range_name, count in orig_stats['confidence_distribution'].items():
            percentage = count / orig_stats['total_samples'] * 100
            print(f"  {range_name}: {count} ({percentage:.1f}%)")
        print()
    
    # 选择样本
    print("正在选择样本...")
    selected = select_samples(
        all_samples,
        min_confidence=args.min_confidence,
        max_samples=args.max_samples,
        sort_by_confidence=True
    )
    
    print(f"选择了 {len(selected)} 个样本")
    print()
    
    # 分析选择后的样本
    if args.analyze and selected:
        print("=== 选择后样本统计 ===")
        sel_stats = analyze_samples(selected)
        print(f"样本总数: {sel_stats['total_samples']}")
        print(f"置信度: 均值={sel_stats['confidence']['mean']:.3f}, "
              f"范围=[{sel_stats['confidence']['min']:.3f}, {sel_stats['confidence']['max']:.3f}]")
        print("\n置信度分布:")
        for range_name, count in sel_stats['confidence_distribution'].items():
            percentage = count / sel_stats['total_samples'] * 100
            print(f"  {range_name}: {count} ({percentage:.1f}%)")
        print()
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"已保存到: {output_path}")
    print("\n完成！")


if __name__ == "__main__":
    main()
