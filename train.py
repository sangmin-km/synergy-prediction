import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import os
import warnings
import dgl

from models.net import DrugDrugSynergy

warnings.filterwarnings('ignore')

# CUDA 설정
torch.backends.cuda.matmul.allow_tf32 = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# DrugDatasetDGL 클래스 정의 (데이터셋 로드를 위해)
class DrugDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, data_list=None):
        super(DrugDatasetDGL, self).__init__()
        self.graphs = []  # DGL 그래프
        self.express = []  # 발현 데이터
        self.labels = []  # 레이블
        
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
        # 배치를 생성하기 위한 collate 함수
        graphs, express, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_express = torch.stack(express)
        batched_labels = torch.stack(labels)
        
        return batched_graph, batched_express, batched_labels

# DGL 데이터셋에 맞는 커스텀 배치 함수
def collate_dgl(samples):
    # samples는 (drug_a, drug_b) 튜플의 리스트
    drug_a_samples = [item[0] for item in samples]
    drug_b_samples = [item[1] for item in samples]
    
    # 각 drug_a, drug_b는 (graph, express, label) 튜플
    graphs_a, express_a, labels_a = map(list, zip(*drug_a_samples))
    graphs_b, express_b, labels_b = map(list, zip(*drug_b_samples))
    
    # DGL 그래프 배치화
    batched_graph_a = dgl.batch(graphs_a)
    batched_graph_b = dgl.batch(graphs_b)
    
    # 텐서 스택
    batched_express_a = torch.stack(express_a)
    batched_express_b = torch.stack(express_b)
    batched_labels = torch.stack(labels_a)  # a와 b는 동일한 레이블을 공유
    
    return (batched_graph_a, batched_express_a, batched_graph_b, batched_express_b, batched_labels)

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for graph_a, expr_a, graph_b, expr_b, labels in loader:
        # 데이터를 디바이스로 이동
        graph_a = graph_a.to(device)
        graph_b = graph_b.to(device)
        expr_a = expr_a.to(device)
        expr_b = expr_b.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(graph_a, graph_b, expr_a, expr_b)
        output = output.view(-1)
        loss = criterion(output, labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for graph_a, expr_a, graph_b, expr_b, labels in loader:
            # 데이터를 디바이스로 이동
            graph_a = graph_a.to(device)
            graph_b = graph_b.to(device)
            expr_a = expr_a.to(device)
            expr_b = expr_b.to(device)
            labels = labels.to(device)
            
            output = model(graph_a, graph_b, expr_a, expr_b)
            output = output.view(-1).cpu().numpy()
            y = labels.view(-1).cpu().numpy()
            predictions.extend(output)
            actuals.extend(y)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return predictions, actuals

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    return mse, rmse, mae, r2, pearson

if __name__ == "__main__":
    # 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 200
    batch_size = 32
    lr = 0.0001
    k_folds = 10
    
    # 데이터 로드 - DGL 데이터셋 경로로 변경
    print('Loading datasets...')
    drug_a_dataset = torch.load('drug_datasets_dgl/drug_a_dataset_dgl.pt')
    drug_b_dataset = torch.load('drug_datasets_dgl/drug_b_dataset_dgl.pt')
    all_data = list(zip(
        [(g, e, l) for g, e, l in zip(drug_a_dataset.graphs, drug_a_dataset.express, drug_a_dataset.labels)],
        [(g, e, l) for g, e, l in zip(drug_b_dataset.graphs, drug_b_dataset.express, drug_b_dataset.labels)]
    ))
    print(f'Total samples: {len(all_data)}')
    
    # 샘플 그래프 정보 출력
    sample_graph_a = drug_a_dataset.graphs[0]
    print(f"Sample node feature dim: {sample_graph_a.ndata['atom'].shape[1]}")
    print(f"Sample edge feature dim: {sample_graph_a.edata['bond'].shape[1]}")
    print(f"Sample pos encoding dim: {sample_graph_a.ndata['lap_pos_enc'].shape[1]}")
    print(f"Sample expr dim: {drug_a_dataset.express[0].shape[0]}")
    
    # K-Fold Cross Validation
    print(f'\nStarting {k_folds}-fold cross validation')
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data), 1):
        print(f"\nFold {fold}/{k_folds}")
        
        # 데이터 분할
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        
        # DataLoader 생성 (DGL 용 collate 함수 사용)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_dgl)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_dgl)
        
        # 모델 초기화 (수정된 인터페이스 사용)
        model = DrugDrugSynergy(
            device=device,
            node_dim=sample_graph_a.ndata['atom'].shape[1],
            edge_dim=sample_graph_a.edata['bond'].shape[1],
            pos_enc_dim=sample_graph_a.ndata['lap_pos_enc'].shape[1],
            hidden_dim=128,
            gt_layers=3,
            gt_heads=4,
            in_feat_dropout=0.1,
            dropout=0.2
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 최적 모델 저장을 위한 변수
        best_test_loss = float('inf')
        
        # 학습 진행
        pbar = tqdm(range(num_epochs), desc=f"Fold {fold}")
        for epoch in pbar:
            # 학습
            train_loss = train(model, device, train_loader, optimizer, criterion)
            
            # 테스트
            test_predictions, test_actuals = evaluate(model, device, test_loader)
            test_loss = mean_squared_error(test_actuals, test_predictions)
            
            # 최적 모델 저장
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pt')
            
            # 진행 상황 출력
            pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}', 
                'Test Loss': f'{test_loss:.4f}'
            })
            
            if epoch == num_epochs - 1:
                print(f'Final Epoch: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # 최고 성능 모델로 평가
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pt'))
        test_predictions, test_actuals = evaluate(model, device, test_loader)
        metrics = calculate_metrics(test_actuals, test_predictions)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold} Results:")
        print(f"MSE: {metrics[0]:.4f}")
        print(f"RMSE: {metrics[1]:.4f}")
        print(f"MAE: {metrics[2]:.4f}")
        print(f"R2: {metrics[3]:.4f}")
        print(f"Pearson: {metrics[4]:.4f}")
    
    # 전체 fold 결과의 평균과 표준편차 계산
    fold_metrics = np.array(fold_metrics)
    mean_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)
    
    print("\nOverall Results:")
    print(f"MSE: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"RMSE: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"MAE: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"R2: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"Pearson: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
