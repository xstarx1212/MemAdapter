"""
Stage I 增强版训练脚本

参考 MemGen 的设计，增强点:
1. LoRA 只训练 q_proj, v_proj (与 MemGen 一致)
2. 真实的 KL 蒸馏 (使用 teacher logprobs)
3. 可选的 QA Loss
4. 温度缩放
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

from src.train.collator_nl import Stage1DataCollatorNL

# LoRA支持
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] PEFT not available. LoRA training will be disabled.")


def load_teacher_logprobs(logprobs_path: str, base_dir: str) -> Optional[Dict]:
    """加载 teacher logprobs"""
    if not logprobs_path:
        return None
    
    full_path = os.path.join(base_dir, logprobs_path)
    if not os.path.exists(full_path):
        return None
    
    try:
        data = torch.load(full_path, map_location='cpu', weights_only=False)
        return {
            'token_logprobs': data['logprobs']['token_logprobs'],
            'tokens': data['logprobs']['tokens'],
            'generated_token_ids': data.get('generated_token_ids', []),
            'vocab_size': data.get('vocab_size', 151643),
        }
    except Exception as e:
        print(f"[WARNING] Failed to load logprobs: {e}")
        return None


class Stage1DatasetEnhanced(Dataset):
    """增强版数据集，支持 teacher logprobs"""
    
    def __init__(self, data_path: str, base_dir: str, max_samples: int = None, load_logprobs: bool = True):
        self.examples = []
        self.base_dir = base_dir
        self.load_logprobs = load_logprobs
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line)
                
                if 'documents' in data and 'gold_subgraph' in data:
                    self.examples.append(data)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx].copy()
        
        # 加载 teacher logprobs (如果启用)
        if self.load_logprobs and 'teacher_logprobs_path' in example:
            logprobs = load_teacher_logprobs(example['teacher_logprobs_path'], self.base_dir)
            example['teacher_logprobs'] = logprobs
        
        return example


class EnhancedCollator(Stage1DataCollatorNL):
    """增强版 Collator，处理 teacher logprobs"""
    
    def __init__(self, *args, base_dir: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        
        # 收集 teacher logprobs
        teacher_logprobs_list = []
        for ex in examples:
            if 'teacher_logprobs' in ex and ex['teacher_logprobs'] is not None:
                teacher_logprobs_list.append(ex['teacher_logprobs'])
            else:
                teacher_logprobs_list.append(None)
        
        batch['teacher_logprobs_list'] = teacher_logprobs_list
        
        return batch


def compute_kl_loss_from_logprobs(
    student_logits: torch.Tensor,
    teacher_logprobs: List[Optional[Dict]],
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    使用 teacher token logprobs 计算 KL 散度损失
    
    由于我们只有 teacher 的 token-level logprobs (而不是完整的 vocab logits),
    我们使用一种近似方法：
    - 对于每个生成位置，计算 student 在 teacher token 上的 log prob
    - 与 teacher 的 log prob 对比
    - 这相当于一种 soft label 损失
    """
    device = student_logits.device
    dtype = student_logits.dtype
    batch_size, seq_len, vocab_size = student_logits.shape
    
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    valid_count = 0
    
    for b in range(batch_size):
        if teacher_logprobs[b] is None:
            continue
        
        teacher_token_logprobs = teacher_logprobs[b].get('token_logprobs', [])
        teacher_token_ids = teacher_logprobs[b].get('generated_token_ids', [])
        
        if not teacher_token_ids or not teacher_token_logprobs:
            continue
        
        # 找到 labels 中非 -100 的位置（即需要预测的位置）
        valid_positions = (labels[b] != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_positions) == 0:
            continue
        
        # 限制到 teacher 数据的长度
        num_teacher_tokens = min(len(teacher_token_ids), len(teacher_token_logprobs))
        num_to_process = min(len(valid_positions), num_teacher_tokens)
        
        if num_to_process == 0:
            continue
        
        # 批量处理以提高效率
        for i in range(num_to_process):
            pos = valid_positions[i].item()
            
            if pos >= seq_len:
                continue
            
            teacher_token_id = teacher_token_ids[i]
            teacher_log_prob = teacher_token_logprobs[i]
            
            # 确保 teacher_token_id 在合法范围内
            if teacher_token_id >= vocab_size or teacher_token_id < 0:
                continue
            
            # Student 在位置 pos 的 log softmax
            student_log_probs = F.log_softmax(student_logits[b, pos] / temperature, dim=-1)
            
            # Student 在 teacher token 上的 log prob
            student_log_prob_at_teacher_token = student_log_probs[teacher_token_id]
            
            # 计算损失：负对数似然 (cross entropy with soft label)
            # 由于 teacher_log_prob 是 teacher 的 log probability，
            # 我们使用 -student_log_prob 作为损失
            # 但为了更好的蒸馏效果，我们使用加权的损失
            teacher_prob = torch.exp(torch.tensor(teacher_log_prob, device=device, dtype=dtype))
            teacher_prob = teacher_prob.clamp(min=1e-8, max=1.0)
            
            # 损失 = teacher_prob * (-student_log_prob)
            # 这相当于 KL 散度的一部分，因为 teacher_prob * log(teacher_prob) 是常数
            loss_at_pos = teacher_prob * (-student_log_prob_at_teacher_token)
            
            total_loss += loss_at_pos
            valid_count += 1
    
    if valid_count > 0:
        return (total_loss / valid_count) * (temperature ** 2)
    else:
        return torch.tensor(0.0, device=device, dtype=dtype)


def train_epoch_enhanced(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    epoch: int,
    args
):
    """增强版训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(progress_bar):
        # Forward pass
        if 'context_texts' in batch and 'query_texts' in batch:
            outputs = model(
                full_graph_texts=batch['context_texts'],
                query_texts=batch['query_texts'],
                labels=batch['labels']
            )
            ce_loss = outputs['loss']
            logits = outputs.get('logits', None)
        else:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            ce_loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.get('logits', None)
        
        # KL Loss (if enabled and logits available)
        kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        if args.use_kl and logits is not None and 'teacher_logprobs_list' in batch:
            try:
                kl_loss = compute_kl_loss_from_logprobs(
                    student_logits=logits,
                    teacher_logprobs=batch['teacher_logprobs_list'],
                    labels=batch['labels'],
                    temperature=args.kl_temperature,
                )
            except Exception as e:
                if accelerator.is_local_main_process and step % 100 == 0:
                    print(f"[WARNING] KL loss failed: {e}")
        
        # 总损失
        loss = args.ce_weight * ce_loss + args.kl_weight * kl_loss
        
        # 反向传播
        accelerator.backward(loss)
        
        # 梯度裁剪
        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # 记录
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        num_batches += 1
        
        # 更新进度条
        if accelerator.is_local_main_process:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'kl': f"{kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss:.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })
        
        # 日志
        if (step + 1) % args.logging_steps == 0 and accelerator.is_local_main_process:
            avg_loss = total_loss / num_batches
            avg_ce = total_ce_loss / num_batches
            avg_kl = total_kl_loss / num_batches
            print(f"\n[Epoch {epoch} Step {step + 1}] Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KL: {avg_kl:.4f})")
    
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
    }


def evaluate_enhanced(model, dataloader, accelerator: Accelerator, args):
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            if 'context_texts' in batch and 'query_texts' in batch:
                outputs = model(
                    full_graph_texts=batch['context_texts'],
                    query_texts=batch['query_texts'],
                    labels=batch['labels']
                )
                loss = outputs['loss']
            else:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'eval_loss': avg_loss}


def split_dataset(data_path: str, base_dir: str, train_ratio: float = 0.9, 
                  seed: int = 42, max_samples: int = None, load_logprobs: bool = True):
    """划分数据集"""
    examples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            if 'documents' in data and 'gold_subgraph' in data:
                examples.append(data)
    
    import random
    random.seed(seed)
    random.shuffle(examples)
    
    train_size = int(len(examples) * train_ratio)
    train_examples = examples[:train_size]
    test_examples = examples[train_size:]
    
    print(f"[数据划分] 总样本数: {len(examples)}")
    print(f"[数据划分] 训练集: {len(train_examples)} ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"[数据划分] 测试集: {len(test_examples)} ({len(test_examples)/len(examples)*100:.1f}%)")
    
    # 统计有 logprobs 的样本数
    if load_logprobs:
        has_logprobs = sum(1 for ex in train_examples if 'teacher_logprobs_path' in ex)
        print(f"[数据划分] 训练集中有 logprobs 的样本: {has_logprobs}")
    
    return train_examples, test_examples


def main():
    parser = argparse.ArgumentParser(description="Stage I 增强版训练脚本")
    
    # 数据
    parser.add_argument('--train', type=str, required=True, help='训练数据路径')
    parser.add_argument('--base_dir', type=str, default=None, help='数据基目录 (用于加载 logprobs)')
    parser.add_argument('--max_train_samples', type=int, default=None)
    
    # 模型
    parser.add_argument('--student_model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    
    # 损失权重
    parser.add_argument('--ce_weight', type=float, default=1.0, help='CE Loss 权重')
    parser.add_argument('--kl_weight', type=float, default=0.5, help='KL Loss 权重')
    parser.add_argument('--use_kl', action='store_true', help='是否使用 KL 蒸馏')
    parser.add_argument('--kl_temperature', type=float, default=2.0, help='KL 蒸馏温度')
    
    # LoRA配置 (参考 MemGen: 只训练 q_proj, v_proj)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, nargs='+', 
                       default=['q_proj', 'v_proj'],  # 与 MemGen 一致
                       help='LoRA 目标模块 (MemGen 风格: q_proj, v_proj)')
    
    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_seed', type=int, default=42)
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (MemGen 使用 1e-5)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # 序列长度
    parser.add_argument('--max_input_len', type=int, default=4096)
    parser.add_argument('--max_output_len', type=int, default=512)
    
    # 精度
    parser.add_argument('--bf16', type=bool, default=True)
    
    # 输出
    parser.add_argument('--out_dir', type=str, default='outputs/stage1_enhanced')
    parser.add_argument('--logging_steps', type=int, default=10)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 推断 base_dir (应该是 stage1 目录)
    if args.base_dir is None:
        # train 路径是 data/teacher_supervision_smart_40k/xxx.jsonl
        # base_dir 应该是 stage1 目录（包含 data 文件夹的父目录）
        train_path = Path(args.train)
        if train_path.is_absolute():
            # 绝对路径: /xxx/stage1/data/xxx/xxx.jsonl -> base_dir = /xxx/stage1
            args.base_dir = str(train_path.parent.parent.parent)
        else:
            # 相对路径: data/xxx/xxx.jsonl -> base_dir = 当前工作目录
            args.base_dir = os.getcwd()
    
    set_seed(args.seed)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision='bf16' if args.bf16 else 'no'
    )
    
    if accelerator.is_local_main_process:
        print("=" * 70)
        print("Stage I 增强版训练 (参考 MemGen 设计)")
        print("=" * 70)
        print("增强点:")
        print("  1. LoRA 只训练 q_proj, v_proj (与 MemGen 一致)")
        print("  2. 真实的 KL 蒸馏 (使用 teacher logprobs)")
        print("  3. 学习率 1e-5 (与 MemGen 一致)")
        print("=" * 70)
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用 StudentRetriever
    from src.model.student import StudentRetriever
    
    model = StudentRetriever(
        model_name_or_path=args.student_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        frozen_anchor_align=False,
        prefix_length=8,
        d_model=1536
    )
    
    # 应用 LoRA (MemGen 风格: 只训练 q_proj, v_proj)
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ValueError("LoRA训练需要安装PEFT库")
        
        if accelerator.is_local_main_process:
            print(f"使用 LoRA 进行训练 (target: {args.lora_target_modules})")
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model.model = get_peft_model(model.model, lora_config)
        
        if accelerator.is_local_main_process:
            model.model.print_trainable_parameters()
    
    # 加载数据
    if accelerator.is_local_main_process:
        print("正在加载数据...")
    
    train_examples, test_examples = split_dataset(
        args.train,
        args.base_dir,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        max_samples=args.max_train_samples,
        load_logprobs=args.use_kl
    )
    
    class ListDataset(Dataset):
        def __init__(self, examples, base_dir, load_logprobs=False):
            self.examples = examples
            self.base_dir = base_dir
            self.load_logprobs = load_logprobs
            
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx].copy()
            
            # 加载 teacher logprobs (如果启用)
            if self.load_logprobs and 'teacher_logprobs_path' in example:
                logprobs = load_teacher_logprobs(example['teacher_logprobs_path'], self.base_dir)
                example['teacher_logprobs'] = logprobs
            
            return example
    
    train_dataset = ListDataset(train_examples, args.base_dir, load_logprobs=args.use_kl)
    test_dataset = ListDataset(test_examples, args.base_dir, load_logprobs=False)
    
    # 使用增强版 collator
    collator = EnhancedCollator(
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        distill_mode='ce',
        base_dir=args.base_dir
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    if accelerator.is_local_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params_count:,} ({100*trainable_params_count/total_params:.2f}%)")
    
    num_training_steps = len(train_dataloader) * args.num_epochs // args.grad_accum
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Accelerator prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    test_dataloader = accelerator.prepare(test_dataloader)
    
    # 创建输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / f"stage1_enhanced_{timestamp}"
    
    if accelerator.is_local_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # 训练循环
    best_eval_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch_enhanced(
            model, train_dataloader, optimizer, lr_scheduler,
            accelerator, epoch, args
        )
        
        if accelerator.is_local_main_process:
            print(f"\n[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f})")
        
        # 评估
        eval_metrics = evaluate_enhanced(model, test_dataloader, accelerator, args)
        eval_loss = eval_metrics['eval_loss']
        
        if accelerator.is_local_main_process:
            print(f"[Epoch {epoch}] Eval Loss: {eval_loss:.4f}")
        
        # 保存最佳模型
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if accelerator.is_local_main_process:
                print(f"  -> New best! Saving to {output_dir / 'best'}")
                save_dir = output_dir / "best"
                save_dir.mkdir(exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(str(save_dir))
    
    # 保存最终模型
    if accelerator.is_local_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(str(final_dir))
        
        print(f"\n训练完成！模型保存到: {output_dir}")


if __name__ == "__main__":
    main()
