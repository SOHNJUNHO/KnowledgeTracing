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

        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Output
        self.drop = nn.Dropout(dropout)
        self.linear_out = nn.Linear(dim, 1)

        # Register causal mask
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        )
        
    def forward(self, skill_ids, skill_inter_ids):
        # Positional embeddings
        input_len = skill_inter_ids.size(1)
        pos_emb = self.embd_pos(torch.arange(input_len, device=self.device))
        
        # Skill interaction embeddings
        skill_inter_embedding = self.embd_skill_inter(skill_inter_ids) + pos_emb
        query = self.embd_skill(skill_ids)
        
        # 1. Normalize before attention
        attn_input = self.norm1(query)
        
        # 2. Apply attention with causal masking
        attn_out, _ = self.attn(
            attn_input,
            skill_inter_embedding,
            skill_inter_embedding,
            attn_mask=self.causal_mask[:input_len, :input_len]
        )
        
        # 3. Dropout + Residual connection
        attn_out = query + self.drop(attn_out)

        # 4. Normalize before FFN
        ffn_input = self.norm2(attn_out)
        
        # 5. Apply FFN
        ffn_out = self.ffn(ffn_input)
        
        # 6. Dropout + Residual connection
        out = attn_out + self.drop(ffn_out)
        
        return self.linear_out(out)