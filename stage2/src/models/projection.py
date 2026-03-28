"""
Memory Projection Modules for Stage2 Alignment

Each paradigm has different memory dimensions:
- A-mem: 384 (MiniLM fixed)
- StreamingLLM: 1536/2048/3584 (depends on LLM backbone)
- MemoryLLM: 1536/2048/3584 (depends on LLM backbone)

These projection modules map each paradigm's memory to the unified anchor space (1536).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MemoryProjection(nn.Module):
    """
    Projects memory from a specific paradigm to the unified anchor space.
    
    Architecture: Linear -> GELU -> Linear -> LayerNorm
    Similar to the AnchorAlignModule for consistency.
    """
    
    def __init__(self, input_dim: int, anchor_dim: int = 1536, hidden_mult: float = 2.0):
        """
        Args:
            input_dim: Input memory dimension (e.g., 384 for A-mem, 1536 for StreamingLLM 1.5B)
            anchor_dim: Target anchor dimension (default: 1536 to match Stage1)
            hidden_mult: Hidden layer multiplier (default: 2.0)
        """
        super().__init__()
        self.input_dim = input_dim
        self.anchor_dim = anchor_dim
        hidden_dim = int(anchor_dim * hidden_mult)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, anchor_dim),
            nn.LayerNorm(anchor_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project memory to anchor space.
        
        Args:
            x: Input memory tensor, shape [batch_size, input_dim]
        
        Returns:
            Projected memory, shape [batch_size, anchor_dim]
        """
        return self.projection(x)


def create_paradigm_projections(
    paradigm_dims: Dict[str, int],
    anchor_dim: int = 1536,
) -> nn.ModuleDict:
    """
    Create projection modules for each paradigm.
    
    Args:
        paradigm_dims: Dictionary mapping paradigm name to its memory dimension
            Example: {
                'amem': 384,
                'streaming_1.5B': 1536,
                'streaming_3B': 2048,
                'memoryllm_1.5B': 1536,
            }
        anchor_dim: Target anchor dimension
    
    Returns:
        nn.ModuleDict with paradigm-specific projections
    """
    projections = nn.ModuleDict()
    
    for paradigm, dim in paradigm_dims.items():
        # Normalize paradigm name for valid Python identifiers
        key = paradigm.replace('.', '_').replace('-', '_')
        projections[key] = MemoryProjection(
            input_dim=dim,
            anchor_dim=anchor_dim,
        )
    
    return projections


# Default paradigm dimensions based on our memory extraction
DEFAULT_PARADIGM_DIMS = {
    'amem': 384,           # MiniLM, fixed regardless of LLM backbone
    'streaming_1_5B': 1536,  # Qwen2.5-1.5B hidden_dim
    'streaming_3B': 2048,    # Qwen2.5-3B hidden_dim
    'streaming_7B': 3584,    # Qwen2.5-7B hidden_dim
    'memoryllm_1_5B': 1536,
    'memoryllm_3B': 2048,
    'memoryllm_7B': 3584,
}


if __name__ == "__main__":
    # Test
    print("Testing MemoryProjection...")
    
    for name, dim in DEFAULT_PARADIGM_DIMS.items():
        proj = MemoryProjection(input_dim=dim, anchor_dim=1536)
        x = torch.randn(4, dim)
        y = proj(x)
        print(f"  {name}: {dim} -> {y.shape[-1]}")
    
    print("\nTesting create_paradigm_projections...")
    projections = create_paradigm_projections(DEFAULT_PARADIGM_DIMS)
    print(f"  Created {len(projections)} projections: {list(projections.keys())}")
