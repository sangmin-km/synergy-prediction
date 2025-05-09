import os
import json
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
import dgl

def smiles_features(mol):
    # 원자 타입 -> 9차원
    symbols = ['B', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
    
    # 혼성화 상태 -> 3차원
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    
    # Bond stereochemistry
    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    
    xs = []  # Store node features here
    
    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol_idx = symbols.index(atom.GetSymbol()) if atom.GetSymbol() in symbols else -1
        if symbol_idx >= 0:
            symbol[symbol_idx] = 1.

        # Atom degree (0~4)
        degree = [0.] * 5
        degree[min(atom.GetDegree(), 4)] = 1.
        
        # Hybridization state (3D: SP, SP2, SP3)
        hybridization = [0.] * len(hybridizations)
        hyb_type = atom.GetHybridization()
        if hyb_type in hybridizations:
            hyb_idx = hybridizations.index(hyb_type)
            hybridization[hyb_idx] = 1.
        
        # Atom feature vector construction
        features = (
            symbol +                     # Atom symbol (9)
            degree +                     # Atom degree (5)
            [atom.GetFormalCharge()] +   # Formal charge (1)
            [atom.GetIsAromatic()] +     # Aromaticity (1)
            hybridization +              # Hybridization (3)
            [float(i == atom.GetTotalNumHs()) for i in range(4)] +  # Number of hydrogens (4)
            [atom.HasProp('_ChiralityPossible')]  # Chirality (1)
        )
        xs.append(features)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i))  # Add bidirectional edges

        # Bond features
        edge_attr = [
            # Single/double/triple/aromatic bond
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE,  
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
            bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
            bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
            # Conjugation
            bond.GetIsConjugated(),
            # Ring structure
            bond.IsInRing()
        ] + [float(bond.GetStereo() == s) for s in stereos]
        
        edge_attrs.append(edge_attr)
        edge_attrs.append(edge_attr)  # Add same features for bidirectional edges

    x = torch.tensor(xs, dtype=torch.float)
    
    if edge_indices:
        edge_index = edge_indices
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = []
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    
    return x, edge_index, edge_attr

def compute_laplacian_pe(adj_matrix, k=8):
    # Calculate Laplacian positional encoding
    N = adj_matrix.shape[0]
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + np.eye(N) * 1e-8))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    try:
        eigval, eigvec = np.linalg.eigh(L_norm)
        idx = eigval.argsort()[:k]
        return torch.from_numpy(eigvec[:, idx]).float()
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Laplacian matrix eigendecomposition failed: {str(e)}")

def smiles_to_dgl_graph(smiles):
    # Convert SMILES string to DGL graph
    mol = Chem.MolFromSmiles(smiles)   
    if mol is None:
        return None
    
    # Extract node features, edge indices, edge features
    node_feats, edge_indices, edge_feats = smiles_features(mol)
    
    # Create DGL graph
    g = dgl.DGLGraph()
    g.add_nodes(node_feats.size(0))
    g.ndata['atom'] = node_feats  
    
    # Process edges if they exist
    if edge_indices:
        src, dst = zip(*edge_indices)
        g.add_edges(src, dst)
        g.edata['bond'] = edge_feats
        
        # Calculate adjacency matrix and add Laplacian positional encoding
        adj_matrix = torch.zeros((node_feats.size(0), node_feats.size(0)))
        for i, j in edge_indices:
            adj_matrix[i, j] = 1 
        try:
            pos_enc = compute_laplacian_pe(adj_matrix.numpy())
            g.ndata['lap_pos_enc'] = pos_enc
        except Exception as e:
            g.ndata['lap_pos_enc'] = torch.zeros((node_feats.size(0), 8), dtype=torch.float)
    return g

class DrugDatasetDGL(torch.utils.data.Dataset):
    # Drug dataset class (DGL graph format)
    def __init__(self, data_list=None):
        super(DrugDatasetDGL, self).__init__()
        self.graphs = []  # DGL graphs
        self.express = []  # Expression data
        self.labels = []  # Labels
        
        if data_list is not None:
            for data in data_list:
                self.graphs.append(data['graph'])
                self.express.append(data['express'])
                self.labels.append(data['label'])
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.express[idx], self.labels[idx]
    
    def collate(self, samples):
        # Collate function for batch creation
        graphs, express, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_express = torch.stack(express)
        batched_labels = torch.stack(labels)
        
        return batched_graph, batched_express, batched_labels

def load_integrated_data_dgl(extract_file, smiles_file, cell_line_files, cell_line_path, score="S"):
    # Load necessary data
    extract = pd.read_csv(extract_file, usecols=[3,4,5,6,7,8,9,10])
    smiles_data = pd.read_csv(smiles_file)
    
    # Load cell line expression data
    all_express = {}
    for cell in cell_line_files:
        cell_data = pd.read_csv(f"{cell_line_path}{cell}.csv")
        all_express[cell] = cell_data
    
    drug_a_data = []
    drug_b_data = []
    
    # Process each drug pair
    for idx, row in extract.iterrows():
        try:
            drug_a_id = row['drug_row_cid']
            drug_b_id = row['drug_col_cid']
            
            # Find SMILES strings
            smiles_a = smiles_data.loc[smiles_data['drug_id'] == drug_a_id, 'smiles'].values
            smiles_b = smiles_data.loc[smiles_data['drug_id'] == drug_b_id, 'smiles'].values
            
            if len(smiles_a) == 0 or len(smiles_b) == 0:
                continue
            
            # Convert SMILES to DGL graphs
            graph_a = smiles_to_dgl_graph(smiles_a[0])
            graph_b = smiles_to_dgl_graph(smiles_b[0])
            
            if graph_a is None or graph_b is None:
                continue
            
            cell_line = row['cell_line_name']
            if cell_line not in all_express:
                continue
                
            expr_data = all_express[cell_line]
            
            if str(drug_a_id) not in expr_data.columns or str(drug_b_id) not in expr_data.columns:
                continue
            
            # Extract expression data and labels
            expr_a = torch.tensor(expr_data[str(drug_a_id)].values, dtype=torch.float32)
            expr_b = torch.tensor(expr_data[str(drug_b_id)].values, dtype=torch.float32)
            label = torch.tensor([row[score]], dtype=torch.float32)
            
            # Check for NaN values
            if torch.isnan(expr_a).any() or torch.isnan(expr_b).any() or torch.isnan(label).any():
                continue
            
            # Store data as dictionaries
            drug_a_data.append({
                'graph': graph_a,
                'express': expr_a,
                'label': label
            })
            
            drug_b_data.append({
                'graph': graph_b,
                'express': expr_b,
                'label': label
            })
            
        except Exception as e:
            continue
    
    return DrugDatasetDGL(drug_a_data), DrugDatasetDGL(drug_b_data)

def preprocess_and_save():
    paths = {
        'extract': "data/data/drugdrug_extract.csv",
        'cell_line': "data/data/gene_expression/",
        'smiles': "data/data/smiles.csv"
    }
    
    # Cell line 
    cell_lines = ["A375", "A549", "HCT116", "HS 578T", "HT29", "LNCAP", "LOVO",
                 "MCF7", "PC-3", "RKO", "SK-MEL-28", "SW-620", "VCAP"]
    
    print("Starting data preprocessing...")
    
    # Load and preprocess data
    print("Converting SMILES data to graphs and integrating expression data...")
    drug_a_dataset, drug_b_dataset = load_integrated_data_dgl(
        paths['extract'], 
        paths['smiles'], 
        cell_lines, 
        paths['cell_line']
    )
    
    # Save datasets
    output_dir = "drug_datasets_dgl"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generated datasets - Drug A: {len(drug_a_dataset)} samples, Drug B: {len(drug_b_dataset)} samples")
    print(f"Saving datasets to {output_dir} directory...")
    
    torch.save(drug_a_dataset, f"{output_dir}/drug_a_dataset_dgl.pt")
    torch.save(drug_b_dataset, f"{output_dir}/drug_b_dataset_dgl.pt")
    
    print("Data preprocessing and saving completed!")

if __name__ == '__main__':
    preprocess_and_save()
