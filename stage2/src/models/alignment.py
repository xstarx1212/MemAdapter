"""
Stage2 Alignment Model

This module combines:
1. Stage1 pre-trained AnchorAlignModule (anchor encoder)
2. Paradigm-specific MemoryProjection modules

Training objective: Contrastive learning to align each paradigm's projected memory
with the anchor representation (evidence subgraph embedding from Stage1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import os
import sys

# Add stage1 src to path for importing AnchorAlignModule (repo-root relative)
_here = os.path.dirname(os.path.abspath(__file__))
_stage1_src = os.path.normpath(os.path.join(_here, "..", "..", "..", "stage1", "src"))
if _stage1_src not in sys.path:
    sys.path.insert(0, _stage1_src)

from model.anchor_align import AnchorAlignModule
from .projection import MemoryProjection, create_paradigm_projections, DEFAULT_PARADIGM_DIMS


class AlignmentModel(nn.Module):
    """
    Stage2 Alignment Model
    
    Components:
    1. anchor_align: Pre-trained from Stage1, maps evidence subgraph to anchor space
    2. projections: Paradigm-specific projections for each memory source
    
    Training:
    - anchor_align can be frozen or fine-tuned (configurable)
    - projections are always trainable
    """
    
    def __init__(
        self,
        anchor_dim: int = 1536,
        paradigm_dims: Optional[Dict[str, int]] = None,
        freeze_anchor: bool = True,
        stage1_checkpoint: Optional[str] = None,
    ):
        """
        Args:
            anchor_dim: Anchor space dimension (should match Stage1)
            paradigm_dims: Dict mapping paradigm names to their memory dimensions
            freeze_anchor: Whether to freeze the anchor_align module
            stage1_checkpoint: Path to Stage1 checkpoint containing anchor_components.pt
        """
        super().__init__()
        self.anchor_dim = anchor_dim
        self.freeze_anchor = freeze_anchor
        
        # Initialize anchor align module
        self.anchor_align = AnchorAlignModule(d_model=anchor_dim, frozen=freeze_anchor)
        
        # Load Stage1 weights if provided
        if stage1_checkpoint:
            self._load_stage1_checkpoint(stage1_checkpoint)
        
        # Initialize paradigm projections
        if paradigm_dims is None:
            paradigm_dims = DEFAULT_PARADIGM_DIMS
        
        self.projections = create_paradigm_projections(
            paradigm_dims=paradigm_dims,
            anchor_dim=anchor_dim,
        )
        
        # Paradigm name mapping for flexible access
        self.paradigm_names = list(paradigm_dims.keys())
    
    def _load_stage1_checkpoint(self, checkpoint_path: str):
        """Load anchor_align weights from Stage1 checkpoint."""
        if os.path.isdir(checkpoint_path):
            # If directory, look for anchor_components.pt
            ckpt_file = os.path.join(checkpoint_path, "anchor_components.pt")
        else:
            ckpt_file = checkpoint_path
        
        if not os.path.exists(ckpt_file):
            print(f"Warning: Stage1 checkpoint not found at {ckpt_file}")
            return
        
        print(f"Loading Stage1 anchor weights from {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        
        if 'anchor_align' in checkpoint:
            self.anchor_align.load_state_dict(checkpoint['anchor_align'])
            print(f"  Loaded anchor_align weights (d_model={checkpoint.get('d_model', 'unknown')})")
        else:
            print("  Warning: 'anchor_align' key not found in checkpoint")
        
        # Apply freeze setting after loading
        if self.freeze_anchor:
            self.anchor_align.freeze()
            print("  Anchor align module frozen")
        else:
            self.anchor_align.unfreeze()
            print("  Anchor align module unfrozen (will be fine-tuned)")
    
    def forward_anchor(self, evidence_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for anchor (evidence) side.

        Args:
            evidence_embedding: Evidence embedding [batch, anchor_dim].
                If loaded from .npy produced by extract_stage1_memory.py,
                this is *already* anchor_align(g) — i.e. h^(a).
                Applying anchor_align again would be a double-application bug.

        Returns:
            Anchor representation [batch, anchor_dim]
        """
        # The .npy evidence embeddings are already h^(a) = anchor_align(g),
        # so we pass them through as-is.  Do NOT re-apply anchor_align here.
        return evidence_embedding
    
    def forward_paradigm(
        self,
        paradigm: str,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through a specific paradigm's projection.
        
        Args:
            paradigm: Paradigm name (e.g., 'amem', 'streaming_1_5B', 'streaming')
            memory: Memory tensor [batch, paradigm_dim]
        
        Returns:
            Projected memory [batch, anchor_dim]
        """
        # Normalize paradigm name
        key = paradigm.replace('.', '_').replace('-', '_')
        
        # Try to find matching projection (allow partial match for paradigm names)
        if key not in self.projections:
            # Try to find a key that starts with the paradigm name
            matched_key = None
            for proj_key in self.projections.keys():
                if proj_key.startswith(key) or key.startswith(proj_key.split('_')[0]):
                    matched_key = proj_key
                    break
            
            if matched_key is None:
                raise ValueError(f"Unknown paradigm: {paradigm}. Available: {list(self.projections.keys())}")
            key = matched_key
        
        return self.projections[key](memory)
    
    def forward(
        self,
        evidence_embedding: torch.Tensor,
        paradigm_memories: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass.
        
        Args:
            evidence_embedding: Evidence subgraph embedding [batch, anchor_dim]
            paradigm_memories: Dict of {paradigm_name: memory_tensor}
        
        Returns:
            anchor: Anchor representation [batch, anchor_dim]
            projected: Dict of {paradigm_name: projected_memory}
        """
        # Anchor representation
        anchor = self.forward_anchor(evidence_embedding)
        
        # Project each paradigm's memory
        projected = {}
        for paradigm, memory in paradigm_memories.items():
            projected[paradigm] = self.forward_paradigm(paradigm, memory)
        
        return anchor, projected
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = []

        # Projections are always trainable
        for proj in self.projections.values():
            params.extend(proj.parameters())

        # Note: anchor_align is NOT included because forward_anchor()
        # passes evidence embeddings through as-is (they are already h^(a)).
        # The module is retained only for checkpoint save/load compatibility.

        return params
    
    def save(self, save_path: str):
        """Save model state."""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        state = {
            'anchor_align': self.anchor_align.state_dict(),
            'projections': {k: v.state_dict() for k, v in self.projections.items()},
            'anchor_dim': self.anchor_dim,
            'paradigm_names': self.paradigm_names,
            'freeze_anchor': self.freeze_anchor,
        }
        torch.save(state, save_path)
        print(f"Saved alignment model to {save_path}")
    
    def load(self, load_path: str):
        """Load model state."""
        state = torch.load(load_path, map_location='cpu', weights_only=False)
        
        self.anchor_align.load_state_dict(state['anchor_align'])
        
        for k, v in state['projections'].items():
            if k in self.projections:
                self.projections[k].load_state_dict(v)
        
        print(f"Loaded alignment model from {load_path}")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for aligning paradigm projections with anchor.
    
    Uses InfoNCE-style loss:
    - Positive: projected memory from same sample
    - Negative: projected memories from other samples in batch
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        projected_memories: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive loss.
        
        Args:
            anchor: Anchor representations [batch, dim]
            projected_memories: List of projected memories, each [batch, dim]
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual losses
        """
        batch_size = anchor.shape[0]
        device = anchor.device
        
        # Normalize
        anchor_norm = F.normalize(anchor, dim=-1)
        
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        for i, projected in enumerate(projected_memories):
            proj_norm = F.normalize(projected, dim=-1)
            
            # Similarity matrix [batch, batch]
            sim_matrix = torch.matmul(anchor_norm, proj_norm.T) / self.temperature
            
            # Labels: diagonal elements are positives
            labels = torch.arange(batch_size, device=device)
            
            # Cross entropy loss (InfoNCE)
            loss = F.cross_entropy(sim_matrix, labels)
            
            loss_dict[f'contrastive_loss_{i}'] = loss
            total_loss = total_loss + loss
        
        # Average over paradigms
        total_loss = total_loss / len(projected_memories)
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class AlignmentLoss(nn.Module):
    """
    Combined loss for Stage2 alignment training.
    
    Components:
    1. Contrastive loss: Align projected memories with anchor (InfoNCE)
    2. MSE loss (optional): Direct regression to anchor representation
    3. Consistency loss (optional): Ensure different paradigms agree
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_mse: bool = False,
        mse_weight: float = 0.1,
        use_consistency: bool = False,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.contrastive = ContrastiveLoss(temperature=temperature)
        self.use_mse = use_mse
        self.mse_weight = mse_weight
        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        anchor: torch.Tensor,
        projected_memories: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined alignment loss.
        
        Args:
            anchor: Anchor representations [batch, dim]
            projected_memories: Dict of {paradigm: projected_memory [batch, dim]}
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual losses
        """
        projected_list = list(projected_memories.values())
        
        # Contrastive loss
        contrastive_loss, loss_dict = self.contrastive(anchor, projected_list)
        total_loss = contrastive_loss
        
        # MSE loss (optional)
        if self.use_mse:
            mse_loss = torch.tensor(0.0, device=anchor.device)
            for proj in projected_list:
                mse_loss = mse_loss + F.mse_loss(proj, anchor)
            mse_loss = mse_loss / len(projected_list)
            loss_dict['mse_loss'] = mse_loss
            total_loss = total_loss + self.mse_weight * mse_loss
        
        # Consistency loss (optional): projected memories should be similar
        if self.use_consistency and len(projected_list) > 1:
            consistency_loss = torch.tensor(0.0, device=anchor.device)
            count = 0
            for i in range(len(projected_list)):
                for j in range(i + 1, len(projected_list)):
                    consistency_loss = consistency_loss + F.mse_loss(
                        projected_list[i], projected_list[j]
                    )
                    count += 1
            consistency_loss = consistency_loss / count
            loss_dict['consistency_loss'] = consistency_loss
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing AlignmentModel...")
    
    # Test with default dimensions
    model = AlignmentModel(
        anchor_dim=1536,
        paradigm_dims={'amem': 384, 'streaming_1_5B': 1536},
        freeze_anchor=True,
    )
    
    batch_size = 4
    
    # Simulate inputs
    evidence_emb = torch.randn(batch_size, 1536)
    memories = {
        'amem': torch.randn(batch_size, 384),
        'streaming_1_5B': torch.randn(batch_size, 1536),
    }
    
    # Forward
    anchor, projected = model(evidence_emb, memories)
    print(f"  Anchor shape: {anchor.shape}")
    for k, v in projected.items():
        print(f"  {k} projected shape: {v.shape}")
    
    # Test loss
    print("\nTesting AlignmentLoss...")
    loss_fn = AlignmentLoss(temperature=0.07, use_mse=True, use_consistency=True)
    loss, loss_dict = loss_fn(anchor, projected)
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
