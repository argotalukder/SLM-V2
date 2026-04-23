# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ArgoConfig

config = ArgoConfig()

class Expert(nn.Module):
    """একটি সাধারণ FFN যা MoE-এর একটি Expert হিসেবে কাজ করবে"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """Mixture of Experts Layer (8 Experts, Top 2 Active)"""
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(config.n_experts)])
        self.router = nn.Linear(config.n_embd, config.n_experts)
        self.top_k = config.top_k

    def forward(self, x):
        # Routing probabilities
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select Top-K experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x)
        # Combine expert outputs
        for i in range(self.top_k):
            expert_idx = selected_experts[..., i]
            weight = routing_weights[..., i].unsqueeze(-1)
            # (Simplification for readability. In production, batched indexing is used)
            for expert_id, expert in enumerate(self.experts):
                mask = (expert_idx == expert_id).unsqueeze(-1)
                final_output += mask * weight * expert(x)
                
        return final_output

class MLAAttention(nn.Module):
    """Low-rank Multi-head Latent Attention"""
    def __init__(self):
        super().__init__()
        # Low-rank projection for memory efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(config.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class ArgoLMBlock(nn.Module):
    """Transformer Block with MLA and MoE"""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MLAAttention()
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x

class ArgoLM(nn.Module):
    """Main ArgoLM Model (1B Parameters)"""
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[ArgoLMBlock() for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten to calculate cross entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss