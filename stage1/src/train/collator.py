"""
数据 collator
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


@dataclass
class Stage1DataCollator:
    """
    Stage I 数据 collator
    
    处理:
    1. 格式化输入: [QUERY] {q} [FULL_GRAPH] {G*} [GENERATE_EVIDENCE_SUBGRAPH]
    2. 目标输出: {r} (evidence subgraph)
    3. Tokenization 和 padding
    4. 可选: 加载教师 logits (如果使用 KL 蒸馏)
    """
    
    tokenizer: PreTrainedTokenizer
    max_input_len: int = 4096
    max_output_len: int = 512
    padding: str = 'longest'
    distill_mode: str = 'kl'  # 'kl' (Stage I default: KL + CE) or 'ce' (CE only)
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate 一个 batch 的样本
        
        输入 examples 格式（论文 Stage I）:
        [
            {
                "id": str,
                "query": str,
                "gold_full_graph": str,
                "gold_subgraph": str,
                "teacher_logprobs_path": str (optional, for KL mode; saved by `teacher_generate_vllm_safe.py --save_teacher_logprobs`)
            },
            ...
        ]
        
        输出格式（batch 内只对 teacher logprobs 做“按样本对齐”的打包）:
        {
            "input_ids": (batch_size, input_len),
            "attention_mask": (batch_size, input_len),
            "labels": (batch_size, output_len),
            "teacher_logprobs_list": List[Optional[Dict]] [optional]
        }
        """
        batch_size = len(examples)
        
        # 构建输入文本
        input_texts = []
        target_texts = []
        
        for ex in examples:
            # 输入: [QUERY] ... [FULL_GRAPH] ... [GENERATE_EVIDENCE_SUBGRAPH]
            input_text = self._format_input(ex['query'], ex['gold_full_graph'])
            input_texts.append(input_text)
            
            # 目标: 证据子图
            target_texts.append(ex['gold_subgraph'])
        
        # Tokenize 输入
        input_encodings = self.tokenizer(
            input_texts,
            max_length=self.max_input_len,
            truncation=True,
            padding=self.padding,
            return_tensors='pt'
        )
        
        # Tokenize 目标
        target_encodings = self.tokenizer(
            target_texts,
            max_length=self.max_output_len,
            truncation=True,
            padding=self.padding,
            return_tensors='pt'
        )
        
        # 准备 labels (将 pad token 替换为 -100)
        labels = target_encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 构建 batch
        batch = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
            'target_attention_mask': target_encodings['attention_mask'],
            # New architecture: provide full_graph_texts and query_texts separately
            'full_graph_texts': [ex['gold_full_graph'] for ex in examples],
            'query_texts': [ex['query'] for ex in examples],
            'target_texts': target_texts  # For debugging
        }
        
        # 如果是 KL 模式，加载教师 logprobs（token-level）
        if self.distill_mode == 'kl':
            teacher_logprobs_list = []
            for ex in examples:
                if 'teacher_logprobs_path' in ex:
                    logprobs = self._load_teacher_logprobs(ex['teacher_logprobs_path'])
                    teacher_logprobs_list.append(logprobs)
                else:
                    teacher_logprobs_list.append(None)

            # 允许部分样本缺少 logprobs：训练时会跳过对应样本的 KL
            batch['teacher_logprobs_list'] = teacher_logprobs_list
        
        return batch
    
    def _format_input(self, query: str, full_graph: str) -> str:
        """
        格式化输入
        
        注意：这个函数在新架构中不再使用（因为使用 prefix injection）
        但保留用于兼容性检查和调试
        """
        return f"""[QUERY]
{query}

[FULL_GRAPH]
{full_graph}

[GENERATE_EVIDENCE_SUBGRAPH]
"""
    
    def _load_teacher_logprobs(self, logprobs_path: str) -> Optional[Dict]:
        """
        加载教师 logprobs（token-level 蒸馏用）。

        该文件由 `stage1/src/data/teacher_generate_vllm_safe.py` 在
        `--save_teacher_logprobs` 开启时保存，包含:
        - data['logprobs']['token_logprobs']
        - data['logprobs']['tokens']
        - data['generated_token_ids']
        """
        try:
            data = torch.load(logprobs_path, map_location='cpu', weights_only=False)
            logprobs = data.get('logprobs', None)
            if logprobs is None:
                return None

            token_logprobs = logprobs.get('token_logprobs', [])
            tokens = logprobs.get('tokens', [])
            generated_token_ids = data.get('generated_token_ids', [])
            vocab_size = data.get('vocab_size', 0)

            if not token_logprobs or not generated_token_ids:
                # 少量数据可能缺失，训练跳过该样本的 KL
                return None

            return {
                'token_logprobs': token_logprobs,
                'tokens': tokens,
                'generated_token_ids': generated_token_ids,
                'vocab_size': vocab_size,
            }
        except Exception as e:
            print(f"警告: 加载 teacher logprobs 失败 ({logprobs_path}): {e}")
            return None


def test_collator():
    """测试 collator"""
    from transformers import AutoTokenizer
    
    print("测试 collator...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = Stage1DataCollator(
        tokenizer=tokenizer,
        max_input_len=512,
        max_output_len=256,
        distill_mode='kl'  # Stage I default
    )
    
    # 测试样本
    examples = [
        {
            "id": "test1",
            "query": "Who proposed the Theory of Relativity?",
            "gold_full_graph": """[FULL_GRAPH]
<NODES>
N1: Albert Einstein
N2: Theory of Relativity

<EDGES>
N1 -> N2: proposed
""",
            "gold_subgraph": """[EVIDENCE_SUBGRAPH]
<NODES>
N1: Albert Einstein
N2: Theory of Relativity

<EDGES>
N1 -> N2: proposed
"""
        },
        {
            "id": "test2",
            "query": "What is the capital of France?",
            "gold_full_graph": """[FULL_GRAPH]
<NODES>
N1: France
N2: Paris

<EDGES>
N1 -> N2: capital
""",
            "gold_subgraph": """[EVIDENCE_SUBGRAPH]
<NODES>
N1: France
N2: Paris

<EDGES>
N1 -> N2: capital
"""
        }
    ]
    
    batch = collator(examples)
    
    print("\nBatch keys:", batch.keys())
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # 解码第一个样本
    print("\n第一个样本:")
    print("Input:", tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
    print("\nTarget:", tokenizer.decode(
        batch['labels'][0][batch['labels'][0] != -100],
        skip_special_tokens=False
    ))


if __name__ == "__main__":
    test_collator()
