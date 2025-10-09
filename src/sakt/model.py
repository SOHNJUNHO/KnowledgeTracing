import torch
import torch.nn as nn
import torch.nn.functional as F

class SAKT(nn.Module):  
    def __init__(self, skill_nums, seq_len, dim, num_heads, dropout, device):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.dim = dim

        # Embeddings
        self.embd_skill_inter = nn.Embedding(2*skill_nums+1, dim, padding_idx=0)  
        self.embd_skill = nn.Embedding(skill_nums+1, dim, padding_idx=0)
        self.embd_pos = nn.Embedding(seq_len, dim)

        self.transformer_block = TransformerBlock(dim, num_heads, dropout)

        # self.transformer_blocks = nn.ModuleList([
        #     TransformerBlock(dim, num_heads, dropout) 
        #     for _ in range(num_layers)
        # ])

        self.linear_out = nn.Linear(dim, 1)

        
    def forward(self, skill_ids, skill_inter_ids):
        # Positional embeddings
        input_len = skill_inter_ids.size(1)
        pos_emb = self.embd_pos(torch.arange(input_len, device=self.device))
        
        # Skill interaction embeddings
        skill_inter_emb = self.embd_skill_inter(skill_inter_ids) + pos_emb
        query_emb = self.embd_skill(skill_ids)
        
        # x = query_emb
        # for block in self.transformer_blocks:
        #     x = block(query=x, key_value=skill_inter_emb)

        transformer_out = self.transformer_block(
            query=query_emb,
            key_value=skill_inter_emb
            # Note: attn_mask argument is gone
        )
        
        return self.linear_out(transformer_out)
        
        # x = self.final_norm(x)
        # x = self.dropout(x)
        
        # return self.linear_out(x)

    
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        norm_query = self.norm1(query)
        
        # IMPROVEMENT: Use the is_causal flag instead of passing a mask
        attn_out, _ = self.attn(
            norm_query,
            key_value,
            key_value,
            is_causal=True  # This does all the work for you!
        )

        # Register causal mask
        #self.register_buffer("causal_mask", 
        #    torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        #)

        # Pre Normalization
        x = query + self.dropout(attn_out)

        # Feed Forward Network
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        
        return x