# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch.glob as dgl_nn_glob
from . import gt_net_compound

class DrugDrugSynergy(nn.Module):
    def __init__(self, device, hidden_dim=128, gt_layers=3, gt_heads=4, in_feat_dropout=0.1, dropout=0.3, node_dim=24, edge_dim=10, pos_enc_dim=8):
        super(DrugDrugSynergy, self).__init__()
        
        self.device = device
        
        # GraphTransformer ���� - �Ķ���� ���� ����
        self.graph_transformer = gt_net_compound.GraphTransformer(
            device=device,
            n_layers=gt_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_heads=gt_heads,
            in_feat_dropout=in_feat_dropout,
            dropout=dropout,
            pos_enc_dim=pos_enc_dim
        )
        
        # Ǯ�� ���̾� (DGL ���)
        self.pool = dgl_nn_glob.AvgPooling()
        
        # ������ ���� ó�� ���̾�
        self.gene_shared = nn.Sequential(
            nn.Linear(978, 978),
            nn.ELU(),
            nn.BatchNorm1d(978)
        )
        
        self.glu1 = nn.Linear(978 * 2, 978)
        self.glu2 = nn.Linear(978 * 2, 978)
        
        self.bn_gene1 = nn.BatchNorm1d(978)
        self.bn_gene2 = nn.BatchNorm1d(978)
        
        self.gene_shared2 = nn.Sequential(
            nn.Linear(978, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # �з���
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.analysis_mode = False
        
    def process_gene_expression(self, gene_expr_a, gene_expr_b):
        gene_expr_a = gene_expr_a.view(-1, 978)
        gene_expr_b = gene_expr_b.view(-1, 978)
        
        drug1 = self.gene_shared(gene_expr_a)
        drug2 = self.gene_shared(gene_expr_b)
        
        concat = torch.cat([drug1, drug2], dim=1)
        
        glu1 = torch.sigmoid(self.glu1(concat))
        glu2 = torch.sigmoid(self.glu2(concat))
        
        # GLU ����ġ ���� (�м���)
        self.glu1_weights = glu1
        self.glu2_weights = glu2
        
        drug1_selected = drug1 * glu1
        drug2_selected = drug2 * glu2
        
        drug1_selected = self.bn_gene1(drug1_selected)
        drug2_selected = self.bn_gene2(drug2_selected)
        
        drug1_emb = self.gene_shared2(drug1_selected)
        drug2_emb = self.gene_shared2(drug2_selected)
        
        return drug1_emb, drug2_emb
    
    def set_analysis_mode(self, mode=True):
        self.analysis_mode = mode
        return self
    
    def forward(self, data_a, data_b, expr_a, expr_b):
        # GraphTransformer�� ��� Ư��, ���� Ư�� �� ��� ���̾��� ���ټ� ���ھ� ��ȯ
        node_feat_a, edge_feat_a, attn_scores_all_layers_a = self.graph_transformer(data_a)
        node_feat_b, edge_feat_b, attn_scores_all_layers_b = self.graph_transformer(data_b)
        
        # �׷��� Ǯ�� (DGL ���)
        data_a.ndata['h'] = node_feat_a
        data_b.ndata['h'] = node_feat_b
        
        graph_feat_a = self.pool(data_a, node_feat_a)
        graph_feat_b = self.pool(data_b, node_feat_b)
        
        # ������ ���� ������ ó��
        gene_feat_a, gene_feat_b = self.process_gene_expression(expr_a, expr_b)
        
        # ��� Ư�� ����
        combined_feats = torch.cat([
            graph_feat_a, graph_feat_b,
            gene_feat_a, gene_feat_b
        ], dim=1)
        
        # ���� ����
        output = self.classifier(combined_feats)
        
        # �м� ����� ��쿡�� �߰� Ư���� ��ȯ
        if self.analysis_mode:
            return output, {
                'attn_scores_all_layers_a': attn_scores_all_layers_a,  # ��� ���̾��� ���ټ� ���ھ�
                'attn_scores_all_layers_b': attn_scores_all_layers_b,  # ��� ���̾��� ���ټ� ���ھ�
                'node_feat_a': node_feat_a,
                'node_feat_b': node_feat_b,
                'edge_feat_a': edge_feat_a,
                'edge_feat_b': edge_feat_b,
                'graph_feat_a': graph_feat_a,
                'graph_feat_b': graph_feat_b,
                'gene_feat_a': gene_feat_a,
                'gene_feat_b': gene_feat_b,
                'glu1_weights': self.glu1_weights,
                'glu2_weights': self.glu2_weights
            }
        
        # �н� ��忡���� ��¸� ��ȯ
        return output