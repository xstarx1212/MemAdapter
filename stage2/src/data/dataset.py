"""
Stage2 Alignment Dataset

Loads memory outputs from different paradigms and aligns them by sample ID.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class Stage2AlignmentDataset(Dataset):
    """
    Dataset for Stage2 alignment training.
    
    Loads:
    1. Evidence subgraph embeddings (from Stage1 or computed)
    2. Memory outputs from each paradigm (A-mem, StreamingLLM, MemoryLLM)
    
    Aligns samples by ID to ensure correct pairing.
    """
    
    def __init__(
        self,
        memory_outputs_dir: str,
        evidence_embeddings_path: Optional[str] = None,
        paradigms: Optional[List[str]] = None,
        model_size: str = "1.5B",
    ):
        """
        Args:
            memory_outputs_dir: Directory containing memory outputs from baselines
            evidence_embeddings_path: Path to evidence embeddings from Stage1 (optional)
            paradigms: List of paradigms to include (default: all available)
            model_size: Model size for paradigms that depend on it (e.g., "1.5B", "3B", "7B")
        """
        self.memory_outputs_dir = memory_outputs_dir
        self.model_size = model_size
        
        # Default paradigms
        if paradigms is None:
            paradigms = ['amem', 'streaming', 'memoryllm']
        self.paradigms = paradigms
        
        # Load data
        self.samples = []
        self.paradigm_memories = {}
        self.evidence_embeddings = None
        
        self._load_data(evidence_embeddings_path)
    
    def _load_data(self, evidence_path: Optional[str] = None):
        """Load and align all data sources."""
        
        # Step 1: Load results.jsonl from each paradigm to get IDs
        paradigm_ids = {}
        paradigm_explicit = {}
        
        for paradigm in self.paradigms:
            paradigm_dir = self._get_paradigm_dir(paradigm)
            if not os.path.exists(paradigm_dir):
                print(f"Warning: Paradigm directory not found: {paradigm_dir}")
                continue
            
            # Find results file
            results_file = self._find_results_file(paradigm_dir, paradigm)
            if results_file is None:
                print(f"Warning: Results file not found for {paradigm}")
                continue
            
            # Load results
            with open(results_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            paradigm_ids[paradigm] = [r['id'] for r in results]
            paradigm_explicit[paradigm] = results
            print(f"Loaded {len(results)} samples from {paradigm}")
        
        if not paradigm_ids:
            raise ValueError("No paradigm data loaded!")
        
        # Step 2: Find common IDs across all paradigms
        common_ids = set(paradigm_ids[list(paradigm_ids.keys())[0]])
        for ids in paradigm_ids.values():
            common_ids &= set(ids)
        
        common_ids = sorted(common_ids)
        print(f"Common samples across all paradigms: {len(common_ids)}")
        
        # Step 3: Create ID to index mapping for each paradigm
        id_to_idx = {}
        for paradigm, ids in paradigm_ids.items():
            id_to_idx[paradigm] = {id_: i for i, id_ in enumerate(ids)}
        
        # Step 4: Load implicit memories (numpy arrays)
        paradigm_implicit = {}
        for paradigm in paradigm_ids.keys():
            paradigm_dir = self._get_paradigm_dir(paradigm)
            implicit_file = self._find_implicit_file(paradigm_dir, paradigm)
            
            if implicit_file and os.path.exists(implicit_file):
                arr = np.load(implicit_file)
                paradigm_implicit[paradigm] = arr
                print(f"Loaded implicit memory for {paradigm}: shape={arr.shape}")
            else:
                print(f"Warning: Implicit memory not found for {paradigm}")
        
        # Step 5: Build aligned samples
        self.samples = []
        for sample_id in common_ids:
            sample = {'id': sample_id}
            
            # Get explicit and implicit memory for each paradigm
            for paradigm in paradigm_ids.keys():
                idx = id_to_idx[paradigm][sample_id]
                
                # Explicit memory (context text)
                explicit_data = paradigm_explicit[paradigm][idx]
                sample[f'{paradigm}_explicit'] = explicit_data.get('context', '')[:2000]
                sample[f'{paradigm}_question'] = explicit_data.get('question', '')
                sample[f'{paradigm}_answer'] = explicit_data.get('gold_answer', explicit_data.get('answer', ''))
                
                # Implicit memory (embedding)
                if paradigm in paradigm_implicit:
                    sample[f'{paradigm}_implicit_idx'] = idx
            
            self.samples.append(sample)
        
        # Store implicit arrays for index-based access
        self.paradigm_implicit = paradigm_implicit
        
        # Step 6: Load evidence embeddings if provided
        if evidence_path and os.path.exists(evidence_path):
            self.evidence_embeddings = np.load(evidence_path)
            print(f"Loaded evidence embeddings: shape={self.evidence_embeddings.shape}")
        else:
            # Use average of implicit memories as pseudo-evidence
            print("No evidence embeddings provided, will use paradigm average as pseudo-anchor")
            self.evidence_embeddings = None
        
        print(f"Total aligned samples: {len(self.samples)}")
    
    def _get_paradigm_dir(self, paradigm: str) -> str:
        """Get directory for a paradigm."""
        if paradigm == 'amem':
            return os.path.join(self.memory_outputs_dir, 'amem')
        elif paradigm in ['streaming', 'memoryllm']:
            # These have model-size subdirectories
            size_dir = f"Qwen2.5-{self.model_size}"
            return os.path.join(self.memory_outputs_dir, paradigm, size_dir)
        else:
            return os.path.join(self.memory_outputs_dir, paradigm)
    
    def _find_results_file(self, paradigm_dir: str, paradigm: str) -> Optional[str]:
        """Find the results JSONL file in paradigm directory."""
        for filename in os.listdir(paradigm_dir):
            # Exclude memory export files like *_memory_explicit.jsonl
            # But allow files with 'memoryllm' in the name
            if filename.endswith('_results.jsonl') and '_memory_' not in filename:
                return os.path.join(paradigm_dir, filename)
        return None
    
    def _find_implicit_file(self, paradigm_dir: str, paradigm: str) -> Optional[str]:
        """Find the implicit memory NPY file in paradigm directory."""
        for filename in os.listdir(paradigm_dir):
            if filename.endswith('_memory_implicit.npy'):
                return os.path.join(paradigm_dir, filename)
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict containing:
            - 'id': Sample ID
            - 'evidence': Evidence embedding (or pseudo-anchor)
            - '{paradigm}_implicit': Implicit memory for each paradigm
            - '{paradigm}_explicit': Explicit memory text (optional)
        """
        sample = self.samples[idx]
        
        result = {'id': sample['id']}
        
        # Get implicit memories for each paradigm
        paradigm_embeds = []
        for paradigm in self.paradigm_implicit.keys():
            key = f'{paradigm}_implicit_idx'
            if key in sample:
                embed_idx = sample[key]
                embed = self.paradigm_implicit[paradigm][embed_idx]
                result[f'{paradigm}_implicit'] = torch.from_numpy(embed).float()
                paradigm_embeds.append(embed)
        
        # Evidence embedding
        if self.evidence_embeddings is not None:
            result['evidence'] = torch.from_numpy(self.evidence_embeddings[idx]).float()
        else:
            # Use first paradigm's embedding as pseudo-evidence
            # (In actual training, this should come from Stage1)
            if paradigm_embeds:
                # Find the paradigm with matching anchor_dim (1536)
                for paradigm, arr in self.paradigm_implicit.items():
                    if arr.shape[1] == 1536:  # Match anchor_dim
                        result['evidence'] = result.get(f'{paradigm}_implicit', 
                                                        torch.zeros(1536))
                        break
                else:
                    # Default to zeros if no matching paradigm
                    result['evidence'] = torch.zeros(1536)
        
        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    result = {}
    
    # Collect all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'id':
            result[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    
    return result


def create_dataloaders(
    memory_outputs_dir: str,
    evidence_embeddings_path: Optional[str] = None,
    paradigms: Optional[List[str]] = None,
    model_size: str = "1.5B",
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        memory_outputs_dir: Directory containing memory outputs
        evidence_embeddings_path: Path to evidence embeddings (optional)
        paradigms: List of paradigms to include
        model_size: Model size for paradigms
        batch_size: Batch size
        num_workers: Number of dataloader workers
        train_split: Fraction for training (rest is validation)
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    dataset = Stage2AlignmentDataset(
        memory_outputs_dir=memory_outputs_dir,
        evidence_embeddings_path=evidence_embeddings_path,
        paradigms=paradigms,
        model_size=model_size,
    )
    
    # Split into train/val
    total = len(dataset)
    train_size = int(total * train_split)
    val_size = total - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test
    print("Testing Stage2AlignmentDataset...")
    
    memory_dir = "/mnt/iusers01/nactem01/e33794xz/scratch/KDD-Memadapter/stage2/memory_outputs"
    
    if os.path.exists(memory_dir):
        dataset = Stage2AlignmentDataset(
            memory_outputs_dir=memory_dir,
            paradigms=['amem', 'streaming'],
            model_size="1.5B",
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample keys:")
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v).__name__}")
    else:
        print(f"Memory output directory not found: {memory_dir}")
