"""
Anchor Alignment Module for MemAdapter

This module defines the unified retrieval representation space.
- Stage I: Trainable alignment module (weights initialized as identity for stable start)
- Stage II: Same trainable alignment module (continues training from Stage I)

Architecture:
- Input: g ∈ R^{d_model} (graph embedding from full graph G*)
- Output: h^(a) ∈ R^{d_model} (aligned representation)
- Structure: Linear(d_model, d_model) -> GELU -> Linear(d_model, d_model) -> LayerNorm

Note: Stage I and Stage II use the SAME module structure. The module is trainable
      in both stages. Initialization as identity is only for stable training start.
"""
import torch
import torch.nn as nn
from typing import Optional


class AnchorAlignModule(nn.Module):
    """
    Anchor Alignment Module
    
    Purpose:
    - Defines the unified retrieval representation space
    - Stage I: Trainable alignment module (same as Stage II)
    - Stage II: Continues training the same module
    
    Input: g ∈ R^{d_model} (graph embedding from full graph G*)
    Output: h^(a) ∈ R^{d_model} (aligned representation)
    
    Important: Stage I and Stage II use the SAME module structure.
               The module is trainable in both stages.
               Initialization as identity is only for stable training start.
    """
    
    def __init__(self, d_model: int = 1536, frozen: bool = False):
        """
        Initialize Anchor Alignment Module
        
        Args:
            d_model: Hidden dimension (default: 1536 for Qwen2.5-1.5B)
            frozen: If True, module is frozen (not used in normal training)
                    If False, trainable (default for both Stage I and Stage II)
        
        Note: Stage I and Stage II use the SAME trainable module.
              Default frozen=False ensures the module is trainable in both stages.
              Initialization as identity is only for stable training start.
        """
        super().__init__()
        self.d_model = d_model
        self.frozen = frozen
        
        # Two-layer MLP + LayerNorm
        # Linear(d_model, d_model) -> GELU -> Linear(d_model, d_model) -> LayerNorm
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize as identity mapping for stable training start
        self._init_as_identity()
        
        if frozen:
            # Freeze all parameters (Stage I 结束后使用)
            self.freeze()
    
    def _init_as_identity(self):
        """
        Initialize weights as identity mapping
        This ensures that forward(x) ≈ x when frozen
        """
        # Initialize first linear layer as identity
        nn.init.eye_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        
        # Initialize second linear layer as identity
        nn.init.eye_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
        # LayerNorm: initialize as identity (scale=1, bias=0)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
    
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Align graph embedding to unified space
        
        Args:
            g: Graph embedding g ∈ R^{d_model} or (batch_size, d_model)
        
        Returns:
            h^(a): Aligned representation h^(a) ∈ R^{d_model} or (batch_size, d_model)
        """
        # Two-layer MLP
        x = self.linear1(g)
        x = self.activation(x)
        x = self.linear2(x)
        
        # LayerNorm
        h_a = self.layer_norm(x)
        
        return h_a
    
    def freeze(self):
        """Freeze the module (rarely used, only for special cases)"""
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze the module (default state for both Stage I and Stage II)"""
        self.frozen = False
        for param in self.parameters():
            param.requires_grad = True


def create_anchor_align_module(d_model: int = 1536, frozen: bool = False) -> AnchorAlignModule:
    """
    Factory function to create Anchor Alignment Module
    
    Args:
        d_model: Hidden dimension (default: 1536 for Qwen2.5-1.5B)
        frozen: If True, creates frozen module (rarely used)
                If False, creates trainable module (default for both Stage I and Stage II)
    
    Returns:
        AnchorAlignModule instance
    """
    return AnchorAlignModule(d_model=d_model, frozen=frozen)
