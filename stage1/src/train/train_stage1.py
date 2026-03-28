"""
Stage I 训练脚本
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed
)

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.train.collator import Stage1DataCollator

# LoRA支持
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] PEFT not available. LoRA training will be disabled.")


def compute_kl_loss_from_logprobs(
    student_logits: torch.Tensor,
    teacher_logprobs_list,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    基于 teacher token-level logprobs 的 KL 近似蒸馏（与 enhanced 版本一致的近似方法）。
    """
    device = student_logits.device
    dtype = student_logits.dtype
    batch_size, seq_len, vocab_size = student_logits.shape
    # Student 的 logits 覆盖了 [PREFIX] + [QUERY] + [TARGET]。
    # labels 只覆盖 [TARGET] 部分，因此需要把 target 位置映射回 logits 的绝对位置。
    target_len = labels.shape[1]
    prompt_len = seq_len - target_len

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    valid_count = 0

    for b in range(batch_size):
        if teacher_logprobs_list[b] is None:
            continue

        teacher_token_logprobs = teacher_logprobs_list[b].get("token_logprobs", [])
        teacher_token_ids = teacher_logprobs_list[b].get("generated_token_ids", [])

        if not teacher_token_ids or not teacher_token_logprobs:
            continue

        valid_positions = (labels[b] != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue

        num_teacher_tokens = min(len(teacher_token_ids), len(teacher_token_logprobs))
        num_to_process = min(len(valid_positions), num_teacher_tokens)
        if num_to_process == 0:
            continue

        for i in range(num_to_process):
            pos = valid_positions[i].item()
            logit_pos = prompt_len + pos
            if logit_pos >= seq_len:
                continue

            teacher_token_id = teacher_token_ids[i]
            teacher_log_prob = teacher_token_logprobs[i]

            if teacher_token_id is None:
                continue
            # 兼容 teacher_token_id 可能是 int 或 0-d tensor
            if isinstance(teacher_token_id, torch.Tensor):
                teacher_token_id = teacher_token_id.item()
            if teacher_token_id >= vocab_size or teacher_token_id < 0:
                continue
            if isinstance(teacher_log_prob, torch.Tensor):
                teacher_log_prob = teacher_log_prob.item()

            student_log_probs = F.log_softmax(student_logits[b, logit_pos] / temperature, dim=-1)
            student_log_prob_at_teacher_token = student_log_probs[teacher_token_id]

            teacher_prob = torch.exp(
                torch.tensor(teacher_log_prob, device=device, dtype=dtype)
            ).clamp(min=1e-8, max=1.0)

            # teacher_prob * (-log p_S(token|...))
            total_loss = total_loss + teacher_prob * (-student_log_prob_at_teacher_token)
            valid_count += 1

    if valid_count > 0:
        return (total_loss / valid_count) * (temperature ** 2)
    return torch.tensor(0.0, device=device, dtype=dtype)


class Stage1Dataset(Dataset):
    """Stage I 数据集"""
    
    def __init__(self, data_path: str, max_samples: int = None):
        self.examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.examples.append(json.loads(line))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def split_dataset(data_path: str, train_ratio: float = 0.9, seed: int = 42, max_samples: int = None):
    """
    划分数据集为训练集和测试集
    
    Args:
        data_path: 数据文件路径
        train_ratio: 训练集比例（默认0.9，即90%训练，10%测试）
        seed: 随机种子
        max_samples: 最大样本数（如果设置，只使用前N个样本）
    
    Returns:
        train_examples, test_examples: 训练集和测试集的样本列表
    """
    examples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))
    
    # 设置随机种子
    import random
    random.seed(seed)
    random.shuffle(examples)
    
    # 划分
    train_size = int(len(examples) * train_ratio)
    train_examples = examples[:train_size]
    test_examples = examples[train_size:]
    
    print(f"[数据划分] 总样本数: {len(examples)}")
    print(f"[数据划分] 训练集: {len(train_examples)} ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"[数据划分] 测试集: {len(test_examples)} ({len(test_examples)/len(examples)*100:.1f}%)")
    
    return train_examples, test_examples


def train_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    epoch: int,
    args
):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    total_kl_loss = 0.0
    total_ce_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(progress_bar):
        # 前向传播 - 使用新架构
        # 新架构需要 full_graph_texts 和 query_texts
        if 'full_graph_texts' in batch and 'query_texts' in batch:
            # 新架构：使用 graph embedding + prefix injection
            outputs = model(
                full_graph_texts=batch['full_graph_texts'],
                query_texts=batch['query_texts'],
                labels=batch['labels']
            )
            # CE（论文里的 hard/CE 部分）
            ce_loss = outputs['loss']
            kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)

            # KL（论文里的 token-level distribution distillation）
            if args.distill_mode == 'kl' and 'teacher_logprobs_list' in batch:
                kl_loss = compute_kl_loss_from_logprobs(
                    student_logits=outputs['logits'],
                    teacher_logprobs_list=batch['teacher_logprobs_list'],
                    labels=batch['labels'],
                    temperature=args.temperature,
                )

            # paper: L = CE + kl_weight * KL
            loss = ce_loss + (args.kl_weight * kl_loss if args.distill_mode == 'kl' else 0.0)
            losses = {'loss': loss, 'ce_loss': ce_loss, 'kl_loss': kl_loss}
        else:
            # 兼容旧架构（如果数据没有提供 full_graph_texts）
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            ce_loss = outputs.loss if hasattr(outputs, "loss") else outputs['loss']
            kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
            loss = ce_loss
            losses = {'loss': loss, 'ce_loss': ce_loss, 'kl_loss': kl_loss}
        
        # 反向传播（使用 gradient accumulation 才能正确实现大 batch）
        with accelerator.accumulate(model):
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                # 梯度裁剪
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # 更新参数（只在累积结束时执行）
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        
        # 记录
        total_loss += loss.item()
        total_kl_loss += losses['kl_loss'].item()
        total_ce_loss += losses['ce_loss'].item()
        num_batches += 1
        
        # 更新进度条
        if accelerator.is_local_main_process:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })
        
        # 日志
        if (step + 1) % args.logging_steps == 0 and accelerator.is_local_main_process:
            avg_loss = total_loss / num_batches
            log_dict = {
                'epoch': epoch,
                'step': step + 1,
                'loss': avg_loss,
                'learning_rate': lr_scheduler.get_last_lr()[0]
            }
            
            if args.distill_mode == 'kl':
                log_dict['kl_loss'] = total_kl_loss / num_batches
            log_dict['ce_loss'] = total_ce_loss / num_batches
            
            print(f"\n[Epoch {epoch} Step {step + 1}] Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches


def evaluate(model, dataloader, accelerator: Accelerator, args):
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            # 使用新架构
            if 'full_graph_texts' in batch and 'query_texts' in batch:
                outputs = model(
                    full_graph_texts=batch['full_graph_texts'],
                    query_texts=batch['query_texts'],
                    labels=batch['labels']
                )
                ce_loss = outputs['loss']
                kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
                
                # 如果需要 KL 损失
                if args.distill_mode == 'kl' and 'teacher_logprobs_list' in batch:
                    kl_loss = compute_kl_loss_from_logprobs(
                        student_logits=outputs['logits'],
                        teacher_logprobs_list=batch['teacher_logprobs_list'],
                        labels=batch['labels'],
                        temperature=args.temperature,
                    )
                loss = ce_loss + (args.kl_weight * kl_loss if args.distill_mode == 'kl' else 0.0)
            else:
                # 兼容旧架构
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'eval_loss': avg_loss}


def main():
    parser = argparse.ArgumentParser(description="Stage I 训练脚本")
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None, help='YAML 配置文件路径')
    
    # 数据
    parser.add_argument('--train', type=str, help='训练数据路径')
    parser.add_argument('--eval', type=str, default=None, help='评估数据路径')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    
    # 模型
    parser.add_argument('--student_model', type=str, default='Qwen/Qwen2.5-1.5B')
    
    # 蒸馏
    parser.add_argument('--distill_mode', type=str, default='kl', choices=['kl', 'ce'], 
                       help='Distillation mode: kl (KL divergence, Stage I default) or ce (cross-entropy)')
    # 论文：KL temperature=2.0, KL weight=0.5（CE weight 1.0）
    parser.add_argument('--temperature', type=float, default=2.0, help='KL temperature')
    parser.add_argument('--kl_weight', type=float, default=0.5, help='KL weight (CE weight fixed to 1.0)')
    
    # LoRA配置
    parser.add_argument('--use_lora', action='store_true', help='使用LoRA进行训练（而不是全量finetune）')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', 
                       default=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                       help='LoRA目标模块')
    
    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.9, 
                       help='训练集比例（默认0.9，即90%训练，10%测试）')
    parser.add_argument('--split_seed', type=int, default=42, help='数据划分的随机种子')
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # 序列长度
    parser.add_argument('--max_input_len', type=int, default=4096)
    parser.add_argument('--max_output_len', type=int, default=512)
    
    # 精度
    parser.add_argument('--bf16', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=False)
    
    # 输出
    parser.add_argument('--out_dir', type=str, default='outputs/stage1')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 用配置文件更新参数 (命令行参数优先)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision='bf16' if args.bf16 else ('fp16' if args.fp16 else 'no')
    )
    
    if accelerator.is_local_main_process:
        print("=== Stage I 训练配置 ===")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print()
    
    # 加载 tokenizer 和模型
    if accelerator.is_local_main_process:
        print("正在加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用新的 StudentRetriever 架构
    from src.model.student import StudentRetriever
    
    model = StudentRetriever(
        model_name_or_path=args.student_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        frozen_anchor_align=False,  # Stage I: trainable (same as Stage II, initialized as identity)
        prefix_length=8,  # 可以配置
        d_model=1536  # Qwen2.5-1.5B 的 hidden size
    )
    
    # 如果使用LoRA，应用LoRA适配器
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ValueError("LoRA训练需要安装PEFT库: pip install peft")
        
        if accelerator.is_local_main_process:
            print("=" * 50)
            print("使用LoRA进行训练")
            print("=" * 50)
            print(f"LoRA rank (r): {args.lora_r}")
            print(f"LoRA alpha: {args.lora_alpha}")
            print(f"LoRA dropout: {args.lora_dropout}")
            print(f"目标模块: {args.lora_target_modules}")
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA到base model
        model.model = get_peft_model(model.model, lora_config)
        
        if accelerator.is_local_main_process:
            model.model.print_trainable_parameters()
    else:
        if accelerator.is_local_main_process:
            print("=" * 50)
            print("使用全量finetune")
            print("=" * 50)
    
    # 准备数据
    if accelerator.is_local_main_process:
        print("正在加载和划分数据...")
    
    # 划分数据集
    train_examples, test_examples = split_dataset(
        args.train, 
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        max_samples=args.max_train_samples
    )
    
    # 创建数据集
    class ListDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]
    
    train_dataset = ListDataset(train_examples)
    test_dataset = ListDataset(test_examples)
    
    collator = Stage1DataCollator(
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        distill_mode=args.distill_mode
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # 使用 0 避免多进程问题
    )
    
    # 使用测试集作为评估集
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # 如果提供了外部评估集，也加载它
    eval_dataloader = None
    if args.eval:
        eval_dataset = Stage1Dataset(args.eval, max_samples=args.max_eval_samples)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0
        )
    
    # 准备优化器和调度器
    # 注意：Stage I 时 anchor_align 是可训练的（论文 Eq.(4)）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    if accelerator.is_local_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        anchor_align_params = sum(p.numel() for p in model.anchor_align.parameters())
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params_count:,} ({100*trainable_params_count/total_params:.2f}%)")
        print(f"Anchor Align Module 参数: {anchor_align_params:,} ({'Frozen' if model.anchor_align.frozen else 'Trainable'})")
        print(f"Prefix Projection 参数: {sum(p.numel() for p in model.prefix_projection.parameters()):,}")
    
    num_training_steps = len(train_dataloader) * args.num_epochs // args.grad_accum
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 使用 Accelerator 准备
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    test_dataloader = accelerator.prepare(test_dataloader)
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    
    if accelerator.is_local_main_process:
        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"训练步数: {num_training_steps}")
        print(f"Warmup 步数: {num_warmup_steps}")
        print()
    
    # 训练循环
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        if accelerator.is_local_main_process:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.num_epochs}")
            print(f"{'='*50}")
        
        # 训练
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            accelerator,
            epoch,
            args
        )
        
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch} - 平均训练损失: {train_loss:.4f}")
        
        # 评估测试集（每个epoch都评估）
        if accelerator.is_local_main_process:
            print(f"\n评估测试集...")
        test_metrics = evaluate(model, test_dataloader, accelerator, args)
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch} - 测试集损失: {test_metrics['eval_loss']:.4f}")
        
        # 如果提供了外部评估集，也评估它
        if eval_dataloader and epoch % 1 == 0:
            eval_metrics = evaluate(model, eval_dataloader, accelerator, args)
            if accelerator.is_local_main_process:
                print(f"Epoch {epoch} - 外部评估集损失: {eval_metrics['eval_loss']:.4f}")
        
        # 保存检查点
        if epoch % 1 == 0 and accelerator.is_local_main_process:
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            unwrapped_model = accelerator.unwrap_model(model)
            
            # 如果使用LoRA，保存LoRA权重
            if args.use_lora:
                unwrapped_model.model.save_pretrained(checkpoint_dir)
            else:
                unwrapped_model.save_pretrained(checkpoint_dir)
            
            tokenizer.save_pretrained(checkpoint_dir)
            
            # 保存anchor components（anchor_align和prefix_projection）
            torch.save({
                'anchor_align': unwrapped_model.anchor_align.state_dict(),
                'prefix_projection': unwrapped_model.prefix_projection.state_dict(),
                'prefix_length': unwrapped_model.prefix_length,
                'd_model': unwrapped_model.d_model
            }, checkpoint_dir / "anchor_components.pt")
            
            print(f"已保存检查点: {checkpoint_dir}")
    
    # 保存最终模型
    if accelerator.is_local_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 如果使用LoRA，保存LoRA权重
        if args.use_lora:
            unwrapped_model.model.save_pretrained(final_dir)
        else:
            unwrapped_model.save_pretrained(final_dir)
        
        tokenizer.save_pretrained(final_dir)
        
        # 保存anchor components（anchor_align和prefix_projection）
        torch.save({
            'anchor_align': unwrapped_model.anchor_align.state_dict(),
            'prefix_projection': unwrapped_model.prefix_projection.state_dict(),
            'prefix_length': unwrapped_model.prefix_length,
            'd_model': unwrapped_model.d_model
        }, final_dir / "anchor_components.pt")
        
        print(f"\n训练完成！最终模型保存在: {final_dir}")
        if args.use_lora:
            print("注意：这是LoRA适配器权重，需要与base model一起加载使用")


if __name__ == "__main__":
    main()
