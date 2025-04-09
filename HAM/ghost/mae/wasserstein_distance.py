import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import os
from datetime import datetime
from util.models_mae import mae_vit_huge_patch14
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
from dataset import getHAM10000Dataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Classification용 모델 클래스 정의
class MAEForClassification(nn.Module):
    def __init__(self, mae_model, num_classes=6):
        super(MAEForClassification, self).__init__()
        self.mae = mae_model
        self.head = nn.Linear(1280, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        x = self.mae.forward_encoder(x, mask_ratio=0)[0]
        cls_token = x[:, 0]
        return self.head(cls_token)

# 기본 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
args = {
    'image_size': 224,
    'random_seed': 42,
    'batch_size': 16,
    'num_workers': 8,
    'output_dir': '/nahcooy/OSR/HAM/ghost/mae/checkpoint',
}

# Wasserstein 거리 계산 함수
def wasserstein_distance_torch(features, centers):
    batch_size, feature_dim = features.size()
    num_classes = centers.size(0)
    features_exp = features.unsqueeze(1).expand(-1, num_classes, -1)
    centers_exp = centers.unsqueeze(0).expand(batch_size, -1, -1)
    distances = torch.abs(features_exp - centers_exp).mean(dim=-1)
    return distances

# OSR 모델 클래스 정의
class MAEForOSR(nn.Module):
    def __init__(self, finetuned_model, num_classes=6, feature_dim=1280):
        super(MAEForOSR, self).__init__()
        self.encoder = finetuned_model.mae
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Fine-tuned 모델의 head를 OSR용 cls_head로 초기화
        self.cls_head = nn.Linear(feature_dim, num_classes)
        with torch.no_grad():
            self.cls_head.weight.copy_(finetuned_model.head.weight)
            self.cls_head.bias.copy_(finetuned_model.head.bias)

        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.centers)

        self.reciprocal_points = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.reciprocal_points)

        self.threshold = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.encoder.forward_encoder(x, mask_ratio=0)[0]  # torch.no_grad() 제거
        features = x[:, 0]
        logits = self.cls_head(features)
        return features, logits

    def compute_losses(self, features, targets):
        batch_size = features.size(0)
        distances = wasserstein_distance_torch(features, self.centers)  # [batch_size, num_classes]
        
        # w_loss: 논문의 L_known (cross-entropy 기반)
        logits = -distances  # 거리를 음수로 변환해 logits로 사용
        w_loss = F.cross_entropy(logits, targets, reduction='mean')  # Known 클래스 포함 모든 샘플 대상
        
        # o_loss: Unknown 샘플에 대한 open space risk
        o_loss = torch.zeros(1, device=features.device)
        for i in range(batch_size):
            target = targets[i].item()
            if target >= self.num_classes:  # Unknown 샘플만
                min_dist = distances[i].min()
                o_loss += F.relu(self.threshold - min_dist)
        o_loss = o_loss / (batch_size if batch_size > 0 else 1)

        # c_loss: Center Loss
        c_loss = torch.zeros(1, device=features.device)
        for i in range(batch_size):
            target = targets[i].item()
            if target < self.num_classes:
                c_loss += F.mse_loss(features[i], self.centers[target])
        c_loss = c_loss / batch_size

        # r_loss: Reciprocal Points Loss
        r_loss = torch.zeros(1, device=features.device)
        for i in range(batch_size):
            target = targets[i].item()
            if target < self.num_classes:
                r_loss += F.mse_loss(features[i], self.reciprocal_points[target])
        r_loss = -r_loss / batch_size

        total_loss = w_loss + o_loss + 0.5 * c_loss + 0.1 * r_loss
        return total_loss, {'w_loss': w_loss, 'o_loss': o_loss, 'c_loss': c_loss, 'r_loss': r_loss}

    def inference(self, features):
        distances = wasserstein_distance_torch(features, self.centers)
        min_dists, preds = distances.min(dim=1)
        is_unknown = min_dists > self.threshold
        return preds, is_unknown, min_dists

# OSR 학습 및 평가 함수
def train_and_evaluate_osr(model, train_loader, val_loader, epochs=200):
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + 
        [model.cls_head.weight, model.cls_head.bias, model.centers, model.reciprocal_points, model.threshold], 
        lr=1e-3
    )
    device = next(model.parameters()).device

    best_train_loss = float('inf')
    best_auroc = 0.0
    best_train_loss_path = os.path.join(args['output_dir'], 'best_train_loss_osr.pth')
    best_auroc_path = os.path.join(args['output_dir'], 'best_auroc_osr.pth')

    for epoch in range(epochs):
        # 학습
        model.train()
        total_loss = 0
        loss_dict_sum = {'w_loss': 0, 'o_loss': 0, 'c_loss': 0, 'r_loss': 0}

        for batch_idx, (samples, targets) in enumerate(train_loader):
            samples, targets = samples.to(device), targets.to(device)
            features, logits = model(samples)

            total_loss_batch, loss_dict = model.compute_losses(features, targets)
            total_loss += total_loss_batch.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v.item()

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(train_loader):
                print(f"[{datetime.now()}] Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}")
                print(f"  Total Loss: {total_loss_batch.item():.4f}")
                print(f"  Loss Breakdown: {', '.join(f'{k}: {v.item():.4f}' for k, v in loss_dict.items())}")

        avg_loss = total_loss / len(train_loader)
        avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
        print(f"[{datetime.now()}] Epoch {epoch + 1}/{epochs}, Avg Train Loss: {avg_loss:.4f}")
        print(f"Loss Breakdown: {avg_loss_dict}")

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            torch.save(model.state_dict(), best_train_loss_path)
            print(f"Saved best train loss model (Loss: {best_train_loss:.4f}) to {best_train_loss_path}")

        # 10 에포크마다 검증 및 추론
        if (epoch + 1) % 10 == 1:
            model.eval()
            all_preds = []
            all_is_unknown = []
            all_targets = []
            all_dists = []
            with torch.no_grad():
                for samples, targets in val_loader:
                    samples, targets = samples.to(device), targets.to(device)
                    features, logits = model(samples)
                    preds, is_unknown, min_dists = model.inference(features)
                    all_preds.extend(preds.cpu().numpy())
                    all_is_unknown.extend(is_unknown.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_dists.extend(min_dists.cpu().numpy())

            all_preds = np.array(all_preds)
            all_is_unknown = np.array(all_is_unknown)
            all_targets = np.array(all_targets)
            all_dists = np.array(all_dists)

            known_mask = all_targets < 6
            unknown_mask = ~known_mask

            # Known 클래스 평가: Known Accuracy와 F1 Score만
            known_acc = accuracy_score(all_targets[known_mask], all_preds[known_mask]) if known_mask.sum() > 0 else 0
            f1 = f1_score(all_targets[known_mask], all_preds[known_mask], average='macro', zero_division=0)

            # Unknown/Known 평가: AUROC와 Unknown Detection Rate
            true_labels = np.zeros_like(all_targets)
            true_labels[unknown_mask] = 1
            auroc = roc_auc_score(true_labels, all_dists) if len(np.unique(true_labels)) > 1 else 0
            unknown_det = np.mean(all_is_unknown[unknown_mask]) if unknown_mask.sum() > 0 else 0

            # 혼동 행렬: Unknown Detection Rate 기반 (Known vs Unknown)
            conf_matrix = confusion_matrix(true_labels, all_is_unknown)

            # 출력
            print(f"[{datetime.now()}] Validation - Epoch {epoch + 1}:")
            print(f"Known Metrics:")
            print(f"  Known Accuracy: {known_acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"Unknown/Known Metrics:")
            print(f"  Unknown Detection Rate: {unknown_det:.4f}")
            print(f"  AUROC: {auroc:.4f}")
            print(f"Confusion Matrix (Known vs Unknown):\n{conf_matrix}")
            print(f"[[True Negative, False Positive], [False Negative, True Positive]]")

            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(model.state_dict(), best_auroc_path)
                print(f"Saved best AUROC model (AUROC: {best_auroc:.4f}) to {best_auroc_path}")

# 메인 실행
if __name__ == "__main__":
    finetuned_model = MAEForClassification(mae_vit_huge_patch14())
    finetuned_checkpoint = torch.load(os.path.join(args['output_dir'], 'best_finetune_checkpoint.pth'), map_location='cpu')
    finetuned_model.load_state_dict(finetuned_checkpoint['model'])
    finetuned_model.to(device)

    osr_model = MAEForOSR(finetuned_model, num_classes=6, feature_dim=1280)
    osr_model.to(device)

    train_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='train', **args)
    val_known_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_known', **args)
    val_unknown_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_unknown', **args)
    val_dataset = ConcatDataset([val_known_dataset, val_unknown_dataset])

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

    train_and_evaluate_osr(osr_model, train_loader, val_loader, epochs=200)