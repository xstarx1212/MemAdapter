"""
损失函数: KL 蒸馏和交叉熵
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    计算 KL 散度损失用于蒸馏
    
    Args:
        student_logits: (batch_size, seq_len, vocab_size)
        teacher_logits: (batch_size, seq_len, vocab_size)
        temperature: 温度参数，用于软化分布
        reduction: 'batchmean', 'mean', 'sum', 'none'
    
    Returns:
        KL divergence loss
    """
    # 使用温度软化 logits
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 计算 KL(teacher || student)
    # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction,
        log_target=False
    )
    
    # 根据温度调整损失 (标准做法)
    kl_div = kl_div * (temperature ** 2)
    
    return kl_div


def masked_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    带 mask 的 KL 散度损失
    
    Args:
        student_logits: (batch_size, seq_len, vocab_size)
        teacher_logits: (batch_size, seq_len, vocab_size)
        attention_mask: (batch_size, seq_len)
        temperature: 温度参数
    
    Returns:
        Masked KL divergence loss
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # 计算每个位置的 KL
    kl_per_token = kl_divergence_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        reduction='none'
    )  # (batch_size, seq_len, vocab_size)
    
    # 对 vocab 维度求和
    kl_per_token = kl_per_token.sum(dim=-1)  # (batch_size, seq_len)
    
    # 应用 mask
    mask = attention_mask.float()
    kl_per_token = kl_per_token * mask
    
    # 计算平均损失 (只考虑非 padding 位置)
    num_valid_tokens = mask.sum()
    if num_valid_tokens > 0:
        loss = kl_per_token.sum() / num_valid_tokens
    else:
        loss = kl_per_token.sum()
    
    return loss


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    标准交叉熵损失 (用于 Mode B)
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        ignore_index: 忽略的标签索引 (通常是 padding)
    
    Returns:
        Cross entropy loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for loss calculation
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # 计算损失
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss


class DistillationLoss(nn.Module):
    """
    蒸馏损失模块
    支持两种模式:
    - Mode A: KL 蒸馏 (使用教师 logits)
    - Mode B: 交叉熵 (使用硬标签)
    """
    
    def __init__(
        self,
        mode: str = 'kl',
        temperature: float = 1.0,
        kl_weight: float = 1.0,
        ce_weight: float = 0.0
    ):
        """
        Args:
            mode: 'kl' 或 'ce'
            temperature: KL 蒸馏的温度
            kl_weight: KL 损失权重
            ce_weight: 交叉熵损失权重 (可以同时使用两种损失)
        """
        super().__init__()
        
        assert mode in ['kl', 'ce'], f"mode 必须是 'kl' 或 'ce', 得到: {mode}"
        
        self.mode = mode
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
    
    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> dict:
        """
        计算损失
        
        Args:
            student_logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            teacher_logits: (batch_size, seq_len, vocab_size), 仅 KL 模式需要
        
        Returns:
            {
                "loss": total loss,
                "kl_loss": kl loss (if applicable),
                "ce_loss": ce loss (if applicable)
            }
        """
        losses = {}
        total_loss = 0.0
        
        if self.mode == 'kl':
            if teacher_logits is None:
                # Fallback: 如果没有 teacher_logits，使用 CE 损失
                # 这样可以避免训练失败，但效果可能不如真正的 KL 蒸馏
                print("警告: KL 模式需要 teacher_logits，但未提供。回退到 CE 损失。")
                print("建议: 使用 --save_teacher_logprobs 生成 logprobs，或使用 HuggingFace 模型获取完整 logits")
                ce_loss = cross_entropy_loss(student_logits, labels, ignore_index=-100)
                losses['ce_loss'] = ce_loss
                losses['kl_loss'] = torch.tensor(0.0, device=student_logits.device)
                losses['loss'] = ce_loss
                return losses
            
            # KL 损失
            kl_loss = masked_kl_loss(
                student_logits,
                teacher_logits,
                attention_mask,
                temperature=self.temperature
            )
            losses['kl_loss'] = kl_loss
            total_loss += self.kl_weight * kl_loss
        
        if self.mode == 'ce' or self.ce_weight > 0:
            # 交叉熵损失
            ce_loss = cross_entropy_loss(student_logits, labels, ignore_index=-100)
            losses['ce_loss'] = ce_loss
            
            if self.mode == 'ce':
                total_loss = ce_loss
            else:
                total_loss += self.ce_weight * ce_loss
        
        losses['loss'] = total_loss
        
        return losses


def test_losses():
    """测试损失函数"""
    print("测试损失函数...")
    
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    # 模拟 logits
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 测试 KL 损失
    print("\n=== KL 蒸馏 ===")
    kl_loss_fn = DistillationLoss(mode='kl', temperature=1.0)
    kl_losses = kl_loss_fn(student_logits, labels, attention_mask, teacher_logits)
    print(f"KL Loss: {kl_losses['loss'].item():.4f}")
    
    # 测试 CE 损失
    print("\n=== 交叉熵 ===")
    ce_loss_fn = DistillationLoss(mode='ce')
    ce_losses = ce_loss_fn(student_logits, labels, attention_mask)
    print(f"CE Loss: {ce_losses['loss'].item():.4f}")
    
    # 测试混合损失
    print("\n=== 混合损失 ===")
    mixed_loss_fn = DistillationLoss(mode='kl', kl_weight=0.5, ce_weight=0.5)
    mixed_losses = mixed_loss_fn(student_logits, labels, attention_mask, teacher_logits)
    print(f"Total Loss: {mixed_losses['loss'].item():.4f}")
    print(f"  - KL: {mixed_losses['kl_loss'].item():.4f}")
    print(f"  - CE: {mixed_losses['ce_loss'].item():.4f}")


if __name__ == "__main__":
    test_losses()
