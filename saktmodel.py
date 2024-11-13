import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SAKT(nn.Module):  
    def __init__(self , problem_num , seq_len, dim , num_heads, dropout ):
        super().__init__()
        self.seq_len = seq_len  #
        self.dim = dim  # 256

        self.embd_inter = nn.Embedding( 2 * problem_num+1, embedding_dim  = dim, padding_idx = 0 )         # Interaction embedding
        self.embd_problem = nn.Embedding( problem_num+1 , embedding_dim = dim, padding_idx = 0 )         # Interaction embedding )       # Excercise embedding  
        
        self.embd_pos = nn.Embedding(seq_len , embedding_dim = dim)       # positional_embedding

        # Would it be better to add Layer Normalizaion and Dropout layer for each Embedding layer?

        #kaiming_normal_(self.P) --> line for Noh


        self.linear = nn.ModuleList( [nn.Linear(in_features= dim , out_features= dim ) for i in range(3)] )   # Linear projection for each embedding 
        self.attn = nn.MultiheadAttention(embed_dim= dim , num_heads= num_heads, dropout= dropout)                                   
        self.ffn = nn.ModuleList([nn.Linear(in_features= dim , out_features=dim) for i in range(2)])  # feed forward layers post attention

        self.layer_norm1 = nn.LayerNorm(dim)                       
        self.drop = nn.Dropout(dropout)

        # Prediction Layer
        self.linear_out = nn.Linear(in_features= dim , out_features= 1) 


    def forward( self, input_ex, input_in):

        #pos_emb = self.embd_pos(torch.arange(self.seq_len).unsqueeze(0).cuda())       ## positional embedding    
        input_len = input_in.shape[1]  
        pos_indices = torch.arange(input_len).cuda()
        pos_emb = self.embd_pos(pos_indices) 
        
        #posemb = self.embd_pos(input_in.shape[1])

        interaction_embedding = self.embd_inter(input_in)   ## get the interaction embedding output     
        ## (b, n) --> (b,n,d)
        interaction_embedding = interaction_embedding + pos_emb

        value_in = interaction_embedding
        key_in = interaction_embedding
        
        query_ex = self.embd_problem(input_ex)   ## get the excercise embedding output   ## (batch_size,seq_len) --> (batch_size, seq_len, dim)

        
        ## Linearly project all the embedings
        value_in = self.linear[0](value_in).permute(1,0,2)        # (b,n,d) --> (n,b,d) 
        key_in = self.linear[1](key_in).permute(1,0,2)         # batch_first --> Default(False), if set True, input and output tensors are provided as (batch, seq, dim)
        query_ex =  self.linear[2](query_ex).permute(1,0,2)

        # Pre-Layer Normalization not exectuted before Attention and FFN
        # "... normalizing INPUTs across features can help in stabilizing and accelerating neural network..."

        ## pass through multihead attention
        atn_out , _ = self.attn(query_ex, key_in, value_in , attn_mask= torch.from_numpy( np.triu(np.ones((query_ex.shape[0] ,query_ex.shape[0])), k=1).astype('bool')).cuda())      # lower triangular mask, bool, torch    (n,b,d)
        
        # attn_emb = self.attn_dropout(attn_emb) -> Exist in pyKT version code

        # outputs = self.dropout(outputs + F.relu(residual)) --> Mistrall version code used attention 
        # "... residual connection can help propagating the embeddings of the RECENTLY SOLVED EXERCISES to the final layer..."

        atn_out = query_ex + atn_out            # Residual connection
        atn_out = self.layer_norm1( atn_out )         # Post-Layer normalization     #print('atn',atn_out.shape) #n,b,d = atn_out.shape

        #take batch on first axis 
        atn_out = atn_out.permute(1,0,2)                              #  (n,b,d) --> (b,n,d)
        
        ## FFN 2 layers
        ffn_out = self.ffn[1](F.relu(self.ffn[0]( atn_out )))   # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.drop( ffn_out + atn_out )     # Residual connection

        ## dropout layer skipped

        return self.linear_out(ffn_out) ## 1차원으로