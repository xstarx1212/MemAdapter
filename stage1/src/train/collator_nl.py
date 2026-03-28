"""
自然语言输入的数据 collator

与 collator.py 的区别:
- 输入使用 documents (自然语言) 而不是 gold_full_graph (结构化图)
- 这样与 baseline (StreamingLLM/MemoryLLM) 的输入格式一致
- Stage2 alignment 更有意义
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


@dataclass
class Stage1DataCollatorNL:
    """
    Stage I 数据 collator - 自然语言版本
    
    处理:
    1. 格式化输入: [QUERY] {q} [CONTEXT] {documents} [GENERATE_EVIDENCE]
    2. 目标输出: {r} (evidence subgraph)
    3. Tokenization 和 padding
    """
    
    tokenizer: PreTrainedTokenizer
    max_input_len: int = 4096
    max_output_len: int = 512
    padding: str = 'longest'
    distill_mode: str = 'ce'  # 'kl' or 'ce'
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate 一个 batch 的样本
        
        输入 examples 格式:
        [
            {
                "id": str,
                "query": str,
                "documents": List[str],  # 自然语言文档列表
                "gold_subgraph": str,    # 目标证据子图
                "teacher_logits_path": str (optional, for KL mode)
            },
            ...
        ]
        """
        batch_size = len(examples)
        
        # 构建输入文本
        input_texts = []
        target_texts = []
        context_texts = []  # 用于 prefix injection 的 context
        
        for ex in examples:
            # 将 documents 列表拼接成自然语言 context
            context = self._format_documents(ex.get('documents', []))
            context_texts.append(context)
            
            # 输入: [QUERY] ... [CONTEXT] ... [GENERATE_EVIDENCE]
            input_text = self._format_input(ex['query'], context)
            input_texts.append(input_text)
            
            # 目标: 优先使用 nl_evidence（如果存在），否则使用 gold_subgraph
            # 这样支持两种训练模式：
            # 1. 使用转换后的 NL evidence（新训练）
            # 2. 使用原始 gold_subgraph（兼容旧训练）
            if 'nl_evidence' in ex and ex['nl_evidence']:
                target_texts.append(ex['nl_evidence'])
            else:
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
            # 新架构: 使用自然语言 context (而不是 full_graph)
            'context_texts': context_texts,  # ← 这是关键区别
            'query_texts': [ex['query'] for ex in examples],
            'target_texts': target_texts
        }
        
        return batch
    
    def _format_documents(self, documents: List[str]) -> str:
        """
        将文档列表格式化为自然语言 context
        
        格式: 
        ### Document 1: {title}\n{content}
        ### Document 2: {title}\n{content}
        ...
        """
        if not documents:
            return ""
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            # 文档可能已经有 "Title: xxx\nContent..." 格式
            # 或者是纯文本
            if doc.startswith("Title:"):
                formatted_docs.append(f"### Document {i}: {doc}")
            else:
                formatted_docs.append(f"### Document {i}:\n{doc}")
        
        return "\n\n".join(formatted_docs)
    
    def _format_input(self, query: str, context: str) -> str:
        """
        格式化输入
        """
        return f"""[QUERY]
{query}

[CONTEXT]
{context}

[GENERATE_EVIDENCE]
"""


def test_collator_nl():
    """测试 collator"""
    from transformers import AutoTokenizer
    
    print("测试自然语言 collator...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = Stage1DataCollatorNL(
        tokenizer=tokenizer,
        max_input_len=512,
        max_output_len=256,
        distill_mode='ce'
    )
    
    # 测试样本
    examples = [
        {
            "id": "test1",
            "query": "Who proposed the Theory of Relativity?",
            "documents": [
                "Title: Albert Einstein\nAlbert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
                "Title: Theory of Relativity\nThe theory of relativity was proposed by Albert Einstein in 1905 and 1915."
            ],
            "gold_subgraph": """[EVIDENCE_SUBGRAPH]
<NODES>
N1: Albert Einstein
N2: Theory of Relativity

<EDGES>
N1 -> N2: proposed
"""
        },
    ]
    
    batch = collator(examples)
    
    print("\nBatch keys:", batch.keys())
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("\n第一个样本 Input:")
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)[:500])
    
    print("\n第一个样本 Context:")
    print(batch['context_texts'][0][:300])


if __name__ == "__main__":
    test_collator_nl()
