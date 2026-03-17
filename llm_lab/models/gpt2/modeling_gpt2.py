from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

from .configuration_gpt2 import GPT2Config

@dataclass
class GPT2ModelOutputWithPast:
    last_hidden_state: torch.Tensor
    past_key_values: "GPT2DynamicCache | None" = None


@dataclass
class GPT2CausalLMOutputWithPast:
    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: "GPT2DynamicCache | None" = None


class GPT2DynamicCache:
    def __init__(self, config: GPT2Config):
        self.n_layers = config.n_layer
        self.key_cache: list[torch.Tensor | None] = [None for _ in range(config.n_layer)]
        self.value_cache: list[torch.Tensor | None] = [None for _ in range(config.n_layer)]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for i in range(self.n_layers):
            if self.key_cache[i] is not None:
                device = self.key_cache[i].device
                local_beam_idx = beam_idx.to(device)
                self.key_cache[i] = self.key_cache[i].index_select(0, local_beam_idx)
                self.value_cache[i] = self.value_cache[i].index_select(0, local_beam_idx)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


def _has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [1, 1, T, D/2]
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [B, kvH, T, D] -> [B, H, T, D]
    if n_rep == 1:
        return hidden_states
    bsz, kv_h, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, kv_h, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, kv_h * n_rep, seq_len, head_dim)


def _make_attn_bias(
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    q_offset: int,
    window_size: tuple[int, int],
) -> torch.Tensor:
    q_pos = torch.arange(q_offset, q_offset + q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    allow = k_pos[None, :] <= q_pos[:, None]
    lef_window = window_size[0]
    if lef_window > 0:
        allow = allow & ((q_pos[:, None] - k_pos[None, :]) < window_size)
    min_value = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros(1, device=device, dtype=dtype), min_value)

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim**-0.5
        self.ve_gate_channels = 12

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if _has_ve(layer_idx, config.n_layer) else None

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: tuple[int, int],
        attention_mask: torch.Tensor | None,
        past_key_values: GPT2DynamicCache | None,
    ) -> torch.Tensor:

        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
        q, k = _rms_norm(q), _rms_norm(k) # QK norm
        q = q * 1.15  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.15

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if past_key_values is None or past_key_values.key_cache[self.layer_idx] is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = past_key_values.key_cache[self.layer_idx], past_key_values.value_cache[self.layer_idx]
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=past_key_values.get_seq_length(),
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == past_key_values.n_layers - 1:
                past_key_values.update(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.attn = GPT2Attention(config, layer_idx)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: tuple[int, int],
        attention_mask: torch.Tensor | None,
        past_key_values: GPT2DynamicCache | None,
    ) -> torch.Tensor:
        x = x + self.attn(_rms_norm(x), ve, cos, sin, window_size, attention_mask, past_key_values)
        x = x + self.mlp(_rms_norm(x))
        return x


class GPT2TextModel(nn.Module):
    def __init__(self, config: GPT2Config, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size

        self.embed_tokens = nn.Embedding(padded_vocab_size, config.n_embd)
        self.layers = nn.ModuleList([GPT2Block(config, i) for i in range(config.n_layer)])
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if _has_ve(i, config.n_layer)}
        )

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.8)
        s = (3.0**0.5) * (self.config.n_embd**-0.5)
        for block in self.layers:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -0.5 * s, 0.5 * s)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.embed_tokens.to(COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(COMPUTE_DTYPE)

    @staticmethod
    def _precompute_rotary_embeddings(seq_len: int, head_dim: int, base: float = 100000.0) -> tuple[torch.Tensor, torch.Tensor]:
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (ch / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    @staticmethod
    def _compute_window_sizes(config: GPT2Config) -> list[tuple[int, int]]:
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 3 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: GPT2DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> GPT2ModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = GPT2DynamicCache(self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but activae for fp16 code path)

        bsz, seq_len, _ = inputs_embeds.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_len + seq_len > self.cos.shape[1]:
            raise ValueError("Sequence length exceeded precomputed rotary cache.")
        cos = self.cos[:, past_len : past_len + seq_len, ].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        sin = self.sin[:, past_len : past_len + seq_len, ].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        x = _rms_norm(inputs_embeds)
        x0 = x
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_ids) if (input_ids is not None and str(i) in self.value_embeds) else None
            if ve is not None:
                ve = ve.to(dtype=x.dtype)
            layer_attention_mask = None
            if attention_mask is not None:
                if past_len > 0:
                    past_mask = attention_mask.new_ones((bsz, past_len))
                    layer_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
                else:
                    layer_attention_mask = attention_mask
            x = layer(
                x=x,
                ve=ve,
                cos=cos,
                sin=sin,
                window_size=self.window_sizes[i],
                attention_mask=layer_attention_mask,
                past_key_values=past_key_values,
            )
        x = _rms_norm(x)
        return GPT2ModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)


class GPT2ForCausalLM(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.model = GPT2TextModel(config)
        self.lm_head = Linear(config.n_embd, self.model.padded_vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    @torch.no_grad()
    def init_weights(self):
        self.model._init_weights()
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def estimate_flops(self) -> float:
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.model.value_embeds.values())
        nparams_exclude = (
            self.model.embed_tokens.weight.numel()
            + value_embeds_numel
            + self.model.resid_lambdas.numel()
            + self.model.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window in self.model.window_sizes:
            window = window[0] # (left, right) tuple, we use left
            effective_seq = min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self) -> dict[str, int]:
        wte = sum(p.numel() for p in self.model.embed_tokens.parameters())
        value_embeds = sum(p.numel() for p in self.model.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.model.layers.parameters())
        scalars = self.model.resid_lambdas.numel() + self.model.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        scalar_lr: float = 0.5,
    ):
        from llm_lab.nanochat.nanochat.common import get_dist_info
        from llm_lab.nanochat.nanochat.optim import DistMuonAdamW, MuonAdamW

        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        del rank, local_rank, world_size

        matrix_params = list(self.model.layers.parameters())
        value_embeds_params = list(self.model.value_embeds.parameters())
        embedding_params = list(self.model.embed_tokens.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.model.resid_lambdas]
        x0_params = [self.model.x0_lambdas]

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )

        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @staticmethod
    def _apply_penalties(
        logits: torch.Tensor,
        generated: torch.LongTensor,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> torch.Tensor:
        if repetition_penalty == 1.0 and presence_penalty == 0.0 and frequency_penalty == 0.0:
            return logits
        adjusted = logits.clone()
        batch_size = adjusted.shape[0]
        vocab_size = adjusted.shape[-1]
        for i in range(batch_size):
            token_counts = torch.bincount(generated[i], minlength=vocab_size).to(adjusted.device)
            if repetition_penalty != 1.0:
                seen = token_counts > 0
                seen_logits = adjusted[i, seen]
                adjusted[i, seen] = torch.where(
                    seen_logits < 0,
                    seen_logits * repetition_penalty,
                    seen_logits / repetition_penalty,
                )
            if presence_penalty != 0.0:
                adjusted[i, token_counts > 0] -= presence_penalty
            if frequency_penalty != 0.0:
                adjusted[i] -= frequency_penalty * token_counts.to(adjusted.dtype)
        return adjusted

    @staticmethod
    def _filter_top_k_top_p(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
        filtered = logits
        vocab_size = filtered.shape[-1]
        if top_k is not None and 0 < top_k < vocab_size:
            kth = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
            filtered = filtered.masked_fill(filtered < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            filtered = torch.full_like(filtered, float("-inf"))
            filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return filtered

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated: torch.LongTensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> torch.LongTensor:
        next_token_logits = self._apply_penalties(
            logits,
            generated=generated,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if not do_sample or temperature <= 0.0:
            return next_token_logits.argmax(dim=-1, keepdim=True)
        next_token_logits = next_token_logits / max(temperature, 1e-5)
        next_token_logits = self._filter_top_k_top_p(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: GPT2DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        targets: torch.LongTensor | None = None,
        loss_reduction: str | None = 'mean',
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> GPT2CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = slice(None)
        
        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(hidden_states[:, slice_indices, :]) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        loss = None
        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

        return GPT2CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        num_return_sequences: int = 1,
        seed: int | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.LongTensor:
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        generated = input_ids
        generated_attention_mask = attention_mask
        cache = None
        for _ in range(max_new_tokens):
            if cache is None:
                model_input_ids = generated
                model_attention_mask = generated_attention_mask
            else:
                model_input_ids = generated[:, -1:]
                model_attention_mask = torch.ones_like(model_input_ids)
            outputs = self.forward(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                past_key_values=cache,
                use_cache=use_cache,
            )
            cache = outputs.past_key_values
            next_token = self._sample_next_token(
                outputs.logits[:, -1, :],
                generated=generated,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            generated = torch.cat([generated, next_token], dim=-1)
            generated_attention_mask = torch.cat(
                [generated_attention_mask, torch.ones_like(next_token, dtype=generated_attention_mask.dtype)], dim=-1
            )
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated
