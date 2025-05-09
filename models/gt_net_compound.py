import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
import dgl
from models.graph_transformer_edge_layer import GraphTransformerLayer


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads, in_feat_dropout, dropout, pos_enc_dim):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.linear_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

    def forward(self, g):
        # input embedding
        g = g.to(self.device)
        h = g.ndata['atom'].float().to(self.device)
        h_lap_pos_enc = g.ndata['lap_pos_enc'].to(self.device)
        e = g.edata['bond'].float().to(self.device)

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc
        h = self.in_feat_dropout(h)

        e = self.linear_e(e)
        
        # 모든 레이어의 어텐션 스코어를 저장할 리스트
        all_attn_scores = []

        # convnets
        for i, conv in enumerate(self.layers):
            h, e, attn_scores = conv(g, h, e)
            all_attn_scores.append(attn_scores)

        g.ndata['h'] = h
        
        # 모든 레이어의 어텐션 스코어 반환
        return h, e, all_attn_scores