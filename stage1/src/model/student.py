"""
学生检索器模型 - 完整架构实现

架构流程:
1. 整图 G* -> 编码器 -> g ∈ R^{d_model}
2. g -> Anchor Align Module -> h^(a) ∈ R^{d_model}
3. h^(a) -> Prefix Injection -> prefix tokens
4. [PREFIX] + [QUERY] + [EVIDENCE_PREFIX] -> 子图生成模块 -> r*
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple
from .anchor_align import AnchorAlignModule


class StudentRetriever(nn.Module):
    """
    学生检索器: 给定查询 q 和完整图 G*, 生成证据子图 r
    
    完整架构:
    G* (text) -> Encoder -> g (embedding) -> Anchor Align -> h^(a) -> Prefix -> Subgraph Generator -> r*
    
    Stage I:
    - Anchor Align Module: Trainable (same as Stage II, initialized as identity)
    - 训练目标: CE loss (或 KL)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda',
        frozen_anchor_align: bool = False,
        prefix_length: int = 8,
        d_model: int = 1536
    ):
        """
        Initialize Student Retriever
        
        Args:
            model_name_or_path: Path to the base model (e.g., Qwen2.5-1.5B)
            device: Device to use
            frozen_anchor_align: If True, anchor align is frozen (rarely used)
                                 If False, trainable (default for both Stage I and Stage II)
            prefix_length: Length of prefix token sequence (m)
            d_model: Hidden dimension (default: 1536 for Qwen2.5-1.5B)
        
        Note: Stage I and Stage II use the SAME trainable align module.
              Default frozen_anchor_align=False ensures the module is trainable in both stages.
        """
        super().__init__()
        
        self.device = device
        self.prefix_length = prefix_length
        self.d_model = d_model
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Base model (used for both encoding and generation)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.to(device)
        
        # Get model's dtype (for consistency)
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Get model's hidden dimension
        if hasattr(self.model.config, 'hidden_size'):
            self.d_model = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.d_model = self.model.config.d_model
        
        # Anchor Alignment Module
        self.anchor_align = AnchorAlignModule(
            d_model=self.d_model,
            frozen=frozen_anchor_align
        )
        # 确保anchor_align使用与base model相同的数据类型
        self.anchor_align = self.anchor_align.to(dtype=self.model_dtype)
        
        # Prefix projection: h^(a) -> prefix tokens
        # W_prefix: maps h^(a) ∈ R^{d_model} to m prefix tokens ∈ R^{m × d_model}
        self.prefix_projection = nn.Linear(self.d_model, self.prefix_length * self.d_model)
        # 确保prefix_projection使用与base model相同的数据类型
        self.prefix_projection = self.prefix_projection.to(dtype=self.model_dtype)
        
        # Initialize prefix projection
        nn.init.normal_(self.prefix_projection.weight, std=0.02)
        nn.init.zeros_(self.prefix_projection.bias)
    
    def _get_base_model(self):
        """
        获取base model，处理PEFT包装的情况
        
        Returns:
            base_model: 实际的transformer base model (without lm_head)
            例如: Qwen2Model (有 embed_tokens 属性)
        """
        # Step 1: Check if self.model is PEFT wrapped
        if hasattr(self.model, 'get_base_model'):
            # PEFT model: use get_base_model() method
            # This returns Qwen2ForCausalLM (or similar)
            base_causal_lm = self.model.get_base_model()
        elif hasattr(self.model, 'model'):
            # Not PEFT wrapped, or already unwrapped
            # self.model is Qwen2ForCausalLM, access .model to get Qwen2Model
            base_causal_lm = self.model
        else:
            # Fallback: self.model might be the base model itself
            base_causal_lm = self.model
        
        # Step 2: Get the actual transformer model (Qwen2Model)
        # Qwen2ForCausalLM has a .model attribute that is Qwen2Model
        if hasattr(base_causal_lm, 'model'):
            # This is Qwen2Model (or similar), which has embed_tokens
            return base_causal_lm.model
        else:
            # Fallback: return as is (shouldn't happen normally)
            return base_causal_lm
    
    def encode_full_graph(self, full_graph_text: str) -> torch.Tensor:
        """
        编码整图 G* 得到 graph embedding g
        
        Args:
            full_graph_text: Serialized full graph text
        
        Returns:
            g: Graph embedding g ∈ R^{d_model} or (batch_size, d_model)
        """
        # Format input: 直接使用 full_graph_text（它已经包含 [FULL_GRAPH] 标记）
        # 训练时 full_graph_texts 已经是完整的 [FULL_GRAPH] 格式
        graph_input = full_graph_text
        
        # Tokenize
        graph_tokens = self.tokenizer(
            graph_input,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Get hidden states from encoder
        # Note: We need gradients for training, so don't use no_grad()
        # Access base model (without lm_head) and get hidden states
        actual_base_model = self._get_base_model()
        
        # Call the base model to get hidden states
        # Use output_hidden_states=True to ensure we can get hidden states
        # This works for both regular models and PEFT-wrapped models
        base_model_outputs = actual_base_model(
            input_ids=graph_tokens['input_ids'],
            attention_mask=graph_tokens['attention_mask'],
            output_hidden_states=True,  # Get all hidden states (we'll use the last one)
            return_dict=True
        )
        
        # Get hidden states H_G ∈ R^{L × d_model}
        # Following MemGen's approach: use hidden_states[-1] for compatibility with PEFT
        if hasattr(base_model_outputs, 'hidden_states') and base_model_outputs.hidden_states is not None:
            # If hidden_states is a tuple/list, get the last one (final layer)
            if isinstance(base_model_outputs.hidden_states, (tuple, list)) and len(base_model_outputs.hidden_states) > 0:
                hidden_states = base_model_outputs.hidden_states[-1]  # (batch_size, seq_len, d_model)
            else:
                hidden_states = base_model_outputs.hidden_states
        elif hasattr(base_model_outputs, 'last_hidden_state'):
            # Fallback: if it has last_hidden_state (BaseModelOutputWithPast)
            hidden_states = base_model_outputs.last_hidden_state  # (batch_size, seq_len, d_model)
        elif isinstance(base_model_outputs, tuple):
            # Fallback: if it's a tuple, the first element is hidden_states
            hidden_states = base_model_outputs[0]
        else:
            raise AttributeError(
                f"Unable to get hidden states from base model outputs. "
                f"Output type: {type(base_model_outputs)}, "
                f"Available attributes: {[attr for attr in dir(base_model_outputs) if not attr.startswith('_')][:15]}"
            )
        
        # Pooling: mean pooling over sequence length
        # Alternative: use special token [GRAPH]'s hidden state
        attention_mask = graph_tokens['attention_mask']
        
        # Mean pooling (masked)
        # 使用与hidden_states相同的数据类型
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(dtype=hidden_states.dtype)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        g = sum_hidden / sum_mask  # (batch_size, d_model)
        
        # 确保返回的张量在正确的设备上
        g = g.to(self.device)
        # 确保数据类型与模型一致
        if hasattr(self, 'model_dtype'):
            g = g.to(dtype=self.model_dtype)
        
        return g
    
    def encode_context(self, context_text: str) -> torch.Tensor:
        """
        编码自然语言 context 得到 context embedding
        
        与 encode_full_graph 的区别:
        - 输入是自然语言 context (文档拼接) 而不是结构化图
        - 这样与 baseline (StreamingLLM/MemoryLLM) 的输入格式一致
        - Stage2 alignment 更有意义
        
        Args:
            context_text: 自然语言 context (如: "### Document 1: ...\n### Document 2: ...")
        
        Returns:
            c: Context embedding c ∈ R^{d_model} or (batch_size, d_model)
        """
        # Tokenize
        context_tokens = self.tokenizer(
            context_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Get hidden states from encoder
        actual_base_model = self._get_base_model()
        
        base_model_outputs = actual_base_model(
            input_ids=context_tokens['input_ids'],
            attention_mask=context_tokens['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states
        if hasattr(base_model_outputs, 'hidden_states') and base_model_outputs.hidden_states is not None:
            if isinstance(base_model_outputs.hidden_states, (tuple, list)) and len(base_model_outputs.hidden_states) > 0:
                hidden_states = base_model_outputs.hidden_states[-1]
            else:
                hidden_states = base_model_outputs.hidden_states
        elif hasattr(base_model_outputs, 'last_hidden_state'):
            hidden_states = base_model_outputs.last_hidden_state
        elif isinstance(base_model_outputs, tuple):
            hidden_states = base_model_outputs[0]
        else:
            raise AttributeError(f"Unable to get hidden states from base model outputs.")
        
        # Mean pooling
        attention_mask = context_tokens['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(dtype=hidden_states.dtype)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        c = sum_hidden / sum_mask
        
        c = c.to(self.device)
        if hasattr(self, 'model_dtype'):
            c = c.to(dtype=self.model_dtype)
        
        return c
    
    def get_prefix_embeddings(self, h_a: torch.Tensor) -> torch.Tensor:
        """
        将 h^(a) 转换为 prefix token embeddings
        
        Args:
            h_a: Aligned representation h^(a) ∈ R^{d_model} or (batch_size, d_model)
        
        Returns:
            prefix_embeddings: Prefix token embeddings ∈ R^{m × d_model} or (batch_size, m, d_model)
        """
        # Project h^(a) to prefix tokens
        # W_prefix h^(a) -> (batch_size, m * d_model)
        prefix_flat = self.prefix_projection(h_a)
        
        # Reshape to (batch_size, m, d_model)
        batch_size = h_a.shape[0] if len(h_a.shape) > 1 else 1
        if len(h_a.shape) == 1:
            h_a = h_a.unsqueeze(0)
            prefix_flat = self.prefix_projection(h_a)
        
        prefix_embeddings = prefix_flat.view(batch_size, self.prefix_length, self.d_model)
        
        # 确保返回的张量在正确的设备上
        prefix_embeddings = prefix_embeddings.to(self.device)
        
        return prefix_embeddings
    
    def forward(
        self,
        full_graph_texts: list,
        query_texts: list,
        labels: Optional[torch.Tensor] = None,
        return_graph_embedding: bool = False
    ) -> Dict:
        """
        前向传播
        
        Args:
            full_graph_texts: List of full graph texts (G*)
            query_texts: List of query texts (q)
            labels: Optional labels for training (batch_size, seq_len)
            return_graph_embedding: If True, return graph embedding for debugging
        
        Returns:
            {
                "logits": (batch_size, seq_len, vocab_size),
                "loss": scalar (if labels provided),
                "graph_embedding": g (if return_graph_embedding=True),
                "aligned_embedding": h^(a) (if return_graph_embedding=True)
            }
        """
        batch_size = len(full_graph_texts)
        
        # Step 1: Encode full graphs -> g
        graph_embeddings = []
        for graph_text in full_graph_texts:
            g = self.encode_full_graph(graph_text)
            graph_embeddings.append(g)
        
        g = torch.stack(graph_embeddings)  # (batch_size, d_model)
        # 确保数据类型一致
        if hasattr(self, 'model_dtype'):
            g = g.to(dtype=self.model_dtype)
        
        # Step 2: Anchor alignment -> h^(a)
        h_a = self.anchor_align(g)  # (batch_size, d_model)
        # 确保数据类型一致
        if hasattr(self, 'model_dtype'):
            h_a = h_a.to(dtype=self.model_dtype)
        
        # Step 3: Get prefix embeddings
        prefix_embeddings = self.get_prefix_embeddings(h_a)  # (batch_size, m, d_model)
        
        # Step 4: Prepare query tokens
        # 输入格式：P = [PREFIX] + [QUERY]
        # - PREFIX: 通过 prefix_embeddings 注入（来自 graph encoding）
        # - QUERY: 原始 query 文本
        # 注意：不需要添加 [GENERATE_EVIDENCE_SUBGRAPH] 等 trigger token
        # 因为模型应该学会在 query 之后直接生成 target
        query_inputs = self.tokenizer(
            query_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embedding layer - handle PEFT wrapped models
        embed_model = self._get_base_model()
        query_embeddings = embed_model.embed_tokens(query_inputs['input_ids'])
        # (batch_size, query_len, d_model)
        # 确保数据类型一致
        if hasattr(self, 'model_dtype'):
            query_embeddings = query_embeddings.to(dtype=self.model_dtype)
        
        # Step 5: Build full input sequence X = P + Y
        # P = [PREFIX] + [QUERY]
        # Y = target tokens (labels)
        # 标准 CausalLM 范式：
        #   inputs_embeds -> X = P + Y
        #   labels       -> [-100] * len(P) + Y
        target_len = labels.shape[1] if labels is not None else 0
        target_embeddings = None
        target_attention_mask = None

        if labels is not None:
            # 将 labels 中的 -100（忽略位置）替换为 pad_token_id，以便做 embedding 查表
            target_ids_for_embed = labels.clone()
            target_ids_for_embed[target_ids_for_embed == -100] = self.tokenizer.pad_token_id

            target_embeddings = embed_model.embed_tokens(target_ids_for_embed)
            if hasattr(self, "model_dtype"):
                target_embeddings = target_embeddings.to(dtype=self.model_dtype)

            # target 部分的 attention_mask：非 -100 的位置为 1，其余为 0
            target_attention_mask = (labels != -100).long()

        # 拼接 embeddings: [PREFIX] + [QUERY] (+ [TARGET] 如果有 labels)
        combined_embeddings = torch.cat([prefix_embeddings, query_embeddings], dim=1)
        if target_embeddings is not None:
            combined_embeddings = torch.cat([combined_embeddings, target_embeddings], dim=1)
        # (batch_size, prefix_len + query_len + target_len, d_model)

        # 构建 attention_mask
        prefix_mask = torch.ones(
            (batch_size, self.prefix_length),
            device=self.device,
            dtype=query_inputs["attention_mask"].dtype,
        )
        combined_attention_mask = torch.cat([prefix_mask, query_inputs["attention_mask"]], dim=1)
        if target_attention_mask is not None:
            combined_attention_mask = torch.cat([combined_attention_mask, target_attention_mask], dim=1)

        # Step 6: Prepare labels for loss computation
        # - Prompt 部分（PREFIX + QUERY）：全部设为 -100（不计算 loss）
        # - Target 部分（Y）：真实 token ids（已包含 -100 作为 padding）
        # - 最终 labels = [-100] * len(P) + Y
        if labels is not None:
            prompt_len = self.prefix_length + query_embeddings.shape[1]
            prompt_labels = torch.full(
                (batch_size, prompt_len),
                -100,
                device=self.device,
                dtype=labels.dtype,
            )
            adjusted_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            adjusted_labels = None
        
        # Step 8: Forward through model
        # We need to use the model's forward with inputs_embeds
        outputs = self.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=adjusted_labels
        )
        
        result = {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None
        }
        
        if return_graph_embedding:
            result["graph_embedding"] = g
            result["aligned_embedding"] = h_a
        
        return result
    
    def forward_with_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        full_graph_texts: list,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        前向传播（使用预处理的 input_ids）
        
        这个版本用于与现有训练代码兼容，但需要提供 full_graph_texts 来生成 prefix
        
        Args:
            input_ids: Pre-tokenized input IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            full_graph_texts: List of full graph texts (needed for prefix generation)
            labels: Optional labels (batch_size, seq_len)
        
        Returns:
            {
                "logits": (batch_size, seq_len, vocab_size),
                "loss": scalar (if labels provided)
            }
        """
        # Extract query from input_ids (assuming format: [QUERY] q [FULL_GRAPH] G* ...)
        # For now, we'll use the full_graph_texts to generate prefix
        # and combine with the rest of the input
        
        # Encode graphs and get prefix
        graph_embeddings = []
        for graph_text in full_graph_texts:
            g = self.encode_full_graph(graph_text)
            graph_embeddings.append(g)
        
        g = torch.stack(graph_embeddings)
        h_a = self.anchor_align(g)
        prefix_embeddings = self.get_prefix_embeddings(h_a)
        
        # Get embeddings for the rest of input_ids
        # We need to split: prefix part vs query+graph part
        # For simplicity, assume input_ids contains query + graph text
        # We'll replace the graph part with prefix
        
        # Get embeddings for input_ids - handle PEFT wrapped models
        embed_model = self._get_base_model()
        input_embeddings = embed_model.embed_tokens(input_ids)
        
        # For now, prepend prefix to input_embeddings
        # This is a simplified version - in practice, you'd need to properly
        # identify and replace the graph portion
        combined_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)
        
        # Update attention mask
        batch_size = input_ids.shape[0]
        prefix_mask = torch.ones(
            (batch_size, self.prefix_length),
            device=self.device,
            dtype=attention_mask.dtype
        )
        combined_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Forward
        outputs = self.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None
        }
    
    def generate(
        self,
        query: str,
        full_graph: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成证据子图
        
        Args:
            query: 查询文本
            full_graph: 完整图的序列化文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 参数
            do_sample: 是否采样
        
        Returns:
            生成的证据子图文本
        """
        self.eval()
        
        # Encode graph and get prefix
        g = self.encode_full_graph(full_graph)
        # 确保g在正确的设备上和数据类型
        g = g.to(self.device)
        if hasattr(self, 'model_dtype'):
            g = g.to(dtype=self.model_dtype)
        h_a = self.anchor_align(g)
        # 确保h_a在正确的设备上和数据类型
        h_a = h_a.to(self.device)
        if hasattr(self, 'model_dtype'):
            h_a = h_a.to(dtype=self.model_dtype)
        prefix_embeddings = self.get_prefix_embeddings(h_a.unsqueeze(0))  # (1, m, d_model)
        # 确保prefix_embeddings在正确的设备上和数据类型
        prefix_embeddings = prefix_embeddings.to(self.device)
        if hasattr(self, 'model_dtype'):
            prefix_embeddings = prefix_embeddings.to(dtype=self.model_dtype)
        
        # Tokenize query
        # 训练时：query_texts 是原始 query，没有格式化标记
        # 新架构中，full_graph通过prefix注入，所以只需要query本身
        # 保持与训练时一致，不添加额外标记
        query_inputs = self.tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embedding layer - handle PEFT wrapped models
        embed_model = self._get_base_model()
        query_embeddings = embed_model.embed_tokens(query_inputs['input_ids'])
        # 确保query_embeddings在正确的设备上和数据类型
        query_embeddings = query_embeddings.to(self.device)
        if hasattr(self, 'model_dtype'):
            query_embeddings = query_embeddings.to(dtype=self.model_dtype)
        
        # Combine prefix + query
        combined_embeddings = torch.cat([prefix_embeddings, query_embeddings], dim=1)
        # 确保combined_embeddings在正确的设备上
        combined_embeddings = combined_embeddings.to(self.device)
        
        prefix_mask = torch.ones(
            (1, self.prefix_length),
            device=self.device,
            dtype=query_inputs['attention_mask'].dtype
        )
        combined_attention_mask = torch.cat([prefix_mask, query_inputs['attention_mask']], dim=1)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (only generated part)
        generated_ids = outputs[0][combined_embeddings.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def generate_from_aligned_memory(
        self,
        aligned_memory: torch.Tensor,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        从已对齐的 memory 直接生成证据子图
        
        用于增强测试：aligned_memory 来自 Stage2 的 alignment projection，
        已经对齐到 anchor space，可以直接作为 prefix 的输入。
        
        Args:
            aligned_memory: 对齐后的 memory [1536] 或 [batch, 1536]
                            来自 Stage2 alignment projection 的输出
            query: 查询文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 参数
            do_sample: 是否采样
        
        Returns:
            生成的证据子图文本
        """
        self.eval()
        
        # Ensure aligned_memory is on correct device and dtype
        if len(aligned_memory.shape) == 1:
            aligned_memory = aligned_memory.unsqueeze(0)  # [1, 1536]
        aligned_memory = aligned_memory.to(self.device)
        if hasattr(self, 'model_dtype'):
            aligned_memory = aligned_memory.to(dtype=self.model_dtype)
        
        # aligned_memory 已经是 h^(a)，直接生成 prefix
        prefix_embeddings = self.get_prefix_embeddings(aligned_memory)  # (1, m, d_model)
        prefix_embeddings = prefix_embeddings.to(self.device)
        if hasattr(self, 'model_dtype'):
            prefix_embeddings = prefix_embeddings.to(dtype=self.model_dtype)
        
        # Tokenize query
        query_inputs = self.tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embedding layer
        embed_model = self._get_base_model()
        query_embeddings = embed_model.embed_tokens(query_inputs['input_ids'])
        query_embeddings = query_embeddings.to(self.device)
        if hasattr(self, 'model_dtype'):
            query_embeddings = query_embeddings.to(dtype=self.model_dtype)
        
        # Combine prefix + query
        combined_embeddings = torch.cat([prefix_embeddings, query_embeddings], dim=1)
        combined_embeddings = combined_embeddings.to(self.device)
        
        prefix_mask = torch.ones(
            (1, self.prefix_length),
            device=self.device,
            dtype=query_inputs['attention_mask'].dtype
        )
        combined_attention_mask = torch.cat([prefix_mask, query_inputs['attention_mask']], dim=1)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (only generated part)
        generated_ids = outputs[0][combined_embeddings.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def save_pretrained(self, output_dir: str):
        """保存模型和 tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save anchor align and prefix projection
        torch.save({
            'anchor_align': self.anchor_align.state_dict(),
            'prefix_projection': self.prefix_projection.state_dict(),
            'prefix_length': self.prefix_length,
            'd_model': self.d_model
        }, f"{output_dir}/anchor_components.pt")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = 'cuda',
        frozen_anchor_align: bool = True
    ):
        """从保存的检查点加载"""
        model = cls(model_path, device=device, frozen_anchor_align=frozen_anchor_align)
        
        # Load anchor components if available
        try:
            components = torch.load(f"{model_path}/anchor_components.pt", map_location=device)
            model.anchor_align.load_state_dict(components['anchor_align'])
            model.prefix_projection.load_state_dict(components['prefix_projection'])
        except:
            pass  # Use default initialization
        
        return model
    
    def get_anchor_align_module(self) -> AnchorAlignModule:
        """获取 Anchor Alignment Module（用于 Stage II 训练）"""
        return self.anchor_align


def test_student_model():
    """测试学生模型"""
    print("测试学生模型...")
    
    # 使用较小的模型进行测试
    model = StudentRetriever("Qwen/Qwen2.5-1.5B", device='cpu')
    
    # 测试输入
    query = "Who proposed the Theory of Relativity?"
    full_graph = """[FULL_GRAPH]
<NODES>
N1: Albert Einstein
N2: Theory of Relativity
N3: Nobel Prize

<EDGES>
N1 -> N2: proposed
N1 -> N3: awarded
"""
    
    # 测试生成
    output = model.generate(query, full_graph, max_new_tokens=128, do_sample=False)
    
    print("Query:", query)
    print("\nFull Graph:", full_graph)
    print("\nGenerated Subgraph:", output)


if __name__ == "__main__":
    test_student_model()
