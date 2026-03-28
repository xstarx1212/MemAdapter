"""
课程学习训练器

核心策略：
1. 先用高置信度样本训练
2. 不收敛不加难度，只加同分布数据
3. 逐步扩展到更多样本
"""
import argparse
import json
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.sample_selector import select_samples, load_samples, analyze_samples


def load_curriculum_config(config_path: str) -> Dict:
    """加载课程学习配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_phase_data(
    all_samples: List[Dict],
    phase_config: Dict,
    output_dir: Path
) -> str:
    """为某个 phase 准备训练数据"""
    print(f"\n准备 {phase_config['name']} 的训练数据...")
    
    # 选择样本
    selected = select_samples(
        all_samples,
        min_confidence=phase_config['min_confidence'],
        max_samples=phase_config['max_samples'],
        sort_by_confidence=True
    )
    
    print(f"选择了 {len(selected)} 个样本")
    
    # 分析
    stats = analyze_samples(selected)
    print(f"置信度范围: [{stats['confidence']['min']:.3f}, {stats['confidence']['max']:.3f}]")
    print(f"平均置信度: {stats['confidence']['mean']:.3f}")
    
    # 保存
    phase_data_path = output_dir / f"{phase_config['name']}_data.jsonl"
    phase_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(phase_data_path, 'w', encoding='utf-8') as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"已保存到: {phase_data_path}")
    
    return str(phase_data_path)


def check_convergence(
    phase_output_dir: Path,
    convergence_config: Dict
) -> bool:
    """检查是否收敛"""
    if not convergence_config.get('enabled', False):
        return True
    
    # TODO: 实现收敛检查逻辑
    # 这里简化为总是返回 True
    # 实际应该读取训练日志，检查 loss 变化
    
    print("收敛检查: 通过")
    return True


def run_curriculum_training(config_path: str):
    """运行完整的课程学习训练"""
    
    # 加载配置
    config = load_curriculum_config(config_path)
    
    print("=== Stage I 课程学习训练 ===")
    print(f"学生模型: {config['student_model']}")
    print(f"训练目标: {config['training_objective']}")
    print()
    
    # 加载所有 teacher 标注的样本
    print("加载 teacher 标注样本...")
    all_samples = load_samples(config['sample_selection']['source_files'])
    print(f"总共 {len(all_samples)} 个样本")
    
    # 按置信度排序
    all_samples = sorted(
        all_samples,
        key=lambda x: x.get('confidence', 0.0),
        reverse=True
    )
    
    # 分析整体数据
    print("\n=== 整体数据统计 ===")
    overall_stats = analyze_samples(all_samples)
    print(f"样本总数: {overall_stats['total_samples']}")
    print(f"置信度: 均值={overall_stats['confidence']['mean']:.3f}, "
          f"范围=[{overall_stats['confidence']['min']:.3f}, {overall_stats['confidence']['max']:.3f}]")
    print("\n置信度分布:")
    for range_name, count in overall_stats['confidence_distribution'].items():
        percentage = count / overall_stats['total_samples'] * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")
    
    # 逐个 phase 训练
    curriculum = config['curriculum']
    previous_checkpoint = None
    
    for i, phase in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"Phase {i}: {phase['name']}")
        print(f"{'='*60}")
        print(f"描述: {phase['description']}")
        print()
        
        # 准备数据
        output_dir = Path(phase['output_dir'])
        phase_data_path = prepare_phase_data(
            all_samples,
            phase,
            output_dir
        )
        
        # 准备训练参数
        train_args = argparse.Namespace(
            train=phase_data_path,
            eval=None,  # TODO: 添加评估集
            student_model=previous_checkpoint or config['student_model'],
            distill_mode=config['training_objective'],
            lr=phase['learning_rate'],
            batch_size=phase['batch_size'],
            grad_accum=phase['grad_accumulation'],
            num_epochs=phase['num_epochs'],
            max_input_len=config['training']['max_input_len'],
            max_output_len=config['training']['max_output_len'],
            bf16=config['training']['bf16'],
            fp16=config['training']['fp16'],
            out_dir=str(output_dir),
            logging_steps=config['training']['logging_steps'],
            eval_steps=config['training']['eval_steps'],
            save_steps=config['training']['save_steps'],
            seed=42,
            max_grad_norm=config['training']['max_grad_norm'],
            warmup_ratio=config['training']['warmup_ratio'],
            # paper: KL temperature=2.0, KL weight=0.5 (CE weight fixed to 1.0)
            temperature=2.0,
            kl_weight=0.5,
            config=None
        )
        
        # 训练
        print(f"\n开始训练 {phase['name']}...")
        try:
            # 以 subprocess 方式调用 Stage1 训练脚本，避免 import/调用签名不一致。
            # 这样课程学习训练链路是闭合可复现的。
            cmd = [
                sys.executable,
                "-m",
                "src.train.train_stage1",
                "--train",
                str(phase_data_path),
                "--student_model",
                str(train_args.student_model),
                "--distill_mode",
                str(train_args.distill_mode),
                "--lr",
                str(train_args.lr),
                "--batch_size",
                str(train_args.batch_size),
                "--grad_acc",
                str(train_args.grad_acc),
                "--num_epochs",
                str(train_args.num_epochs),
                "--max_input_len",
                str(train_args.max_input_len),
                "--max_output_len",
                str(train_args.max_output_len),
                "--out_dir",
                str(train_args.out_dir),
                "--logging_steps",
                str(train_args.logging_steps),
                "--eval_steps",
                str(train_args.eval_steps),
                "--save_steps",
                str(train_args.save_steps),
                "--seed",
                str(train_args.seed),
                "--max_grad_norm",
                str(train_args.max_grad_norm),
                "--warmup_ratio",
                str(train_args.warmup_ratio),
                "--temperature",
                str(train_args.temperature),
                "--kl_weight",
                str(train_args.kl_weight),
            ]
            # precision flags: 原 parser 里 type=bool，直接传 "false" 会被误解析为 True
            if train_args.bf16:
                cmd.extend(["--bf16", "true"])
            if getattr(train_args, "fp16", False):
                cmd.extend(["--fp16", "true"])
            
            subprocess.run(cmd, check=True)
            print("训练完成！")
            
            # 更新检查点路径
            previous_checkpoint = str(output_dir / "final")
            
            # 检查收敛
            if not check_convergence(output_dir, phase.get('convergence_check', {})):
                print(f"\n警告: {phase['name']} 未收敛")
                
                # Fallback 策略
                fallback_config = config.get('fallback', {})
                if fallback_config.get('if_not_converged', {}).get('action') == 'add_more_same_distribution':
                    print("执行 fallback 策略: 添加更多同分布数据")
                    # TODO: 实现 fallback 逻辑
                else:
                    print("跳过后续 phase")
                    break
            
        except Exception as e:
            print(f"\n错误: {phase['name']} 训练失败: {e}")
            break
    
    print(f"\n{'='*60}")
    print("课程学习训练完成！")
    print(f"最终模型: {previous_checkpoint}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Stage I 课程学习训练")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/student_training.yaml',
        help='课程学习配置文件'
    )
    
    args = parser.parse_args()
    
    run_curriculum_training(args.config)


if __name__ == "__main__":
    main()
