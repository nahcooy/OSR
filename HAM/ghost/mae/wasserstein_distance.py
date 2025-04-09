import torch
import torch.nn as nn
import torch.nn.functional as F  # 추가
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import os
from datetime import datetime
from util.models_mae import mae_vit_huge_patch14
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
import numpy as np
from dataset import getHAM10000Dataset
from geomloss import SamplesLoss

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 기본 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = {
    'image_size': 224,
    'random_seed': 42,
    'batch_size': 16,
    'num_workers': 8,
    'output_dir': '/nahcooy/OSR/HAM/ghost/mae/checkpoint',
    'data_augmentation': True
}

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

# Wasserstein 거리 계산 (Sinkhorn 알고리즘 사용)
def wasserstein_distance_torch(features, centers):
    """
    features: [batch_size, feature_dim]
    centers: [num_classes, feature_dim]
    """
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.8)
    batch_size, feature_dim = features.size()
    num_classes = centers.size(0)
    
    features_exp = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
    centers_exp = centers.unsqueeze(0)    # [1, num_classes, feature_dim]
    
    distances = []
    for i in range(batch_size):
        feat = features_exp[i]
        dist = loss(feat, centers_exp.squeeze(0))
        distances.append(dist)
    
    distances = torch.stack(distances)  # [batch_size, num_classes]
    return distances.float()  # Float 타입 보장

# OSR 모델 정의
class MAEForOSR(nn.Module):
    def __init__(self, finetuned_model, num_classes=6, feature_dim=1280):
        super(MAEForOSR, self).__init__()
        self.encoder = finetuned_model.mae
        self.num_classes = num_classes
        self.feature_dim = feature_dim

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
        x = self.encoder.forward_encoder(x, mask_ratio=0)[0]
        features = x[:, 0]
        logits = self.cls_head(features)
        return features, logits

    def compute_losses(self, features, targets):
        batch_size = features.size(0)
        distances = wasserstein_distance_torch(features, self.centers)  # [batch_size, num_classes]
        
        if targets.dim() > 1:
            targets = targets.squeeze()
        assert targets.dim() == 1, f"Targets should be 1D, got {targets.shape}"
        assert torch.all(targets < self.num_classes), f"Targets contain values >= {self.num_classes}: {targets}"

        logits = -distances  # Wasserstein 거리를 logits로 변환
        w_loss = F.cross_entropy(logits, targets, reduction='mean')

        c_loss = torch.zeros(1, device=features.device)
        for i in range(batch_size):
            target = targets[i].item()
            c_loss += F.mse_loss(features[i], self.centers[target])
        c_loss = c_loss / batch_size

        ac_loss = torch.zeros(1, device=features.device)
        for target in range(self.num_classes):
            other_centers = self.centers[torch.arange(self.num_classes) != target]
            reciprocal_point = self.reciprocal_points[target]
            ac_loss += F.mse_loss(reciprocal_point, other_centers.mean(dim=0))
        ac_loss = ac_loss / self.num_classes

        o_loss = torch.zeros(1, device=features.device)
        for i in range(batch_size):
            target = targets[i].item()
            reciprocal_dist = wasserstein_distance_torch(features[i].unsqueeze(0), self.reciprocal_points)[0, target]
            o_loss -= reciprocal_dist
        o_loss = o_loss / batch_size

        total_loss = w_loss + 0.5 * c_loss + 0.1 * o_loss + 0.1 * ac_loss
        return total_loss, {
            'w_loss': w_loss,
            'o_loss': o_loss,
            'c_loss': c_loss,
            'ac_loss': ac_loss
        }

    def inference(self, features):
        distances = wasserstein_distance_torch(features, self.centers)
        min_dists, preds = distances.min(dim=1)
        is_unknown = min_dists > self.threshold
        return preds, is_unknown, min_dists

# OSR 학습 및 평가 함수
def train_and_evaluate_osr(model, train_loader, val_loader, args, epochs=200):
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + 
        [model.cls_head.weight, model.cls_head.bias, model.centers, model.reciprocal_points, model.threshold], 
        lr=1e-3, weight_decay=0.01
    )
    loss_scaler = NativeScaler()
    device = next(model.parameters()).device

    best_train_loss = float('inf')
    best_auroc = 0.0
    best_train_loss_path = os.path.join(args['output_dir'], 'best_train_loss_osr.pth')
    best_auroc_path = os.path.join(args['output_dir'], 'best_auroc_osr.pth')

    os.makedirs(args['output_dir'], exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loss_dict_sum = {'w_loss': 0, 'o_loss': 0, 'c_loss': 0, 'ac_loss': 0}

        for batch_idx, (samples, targets) in enumerate(train_loader):
            samples, targets = samples.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                features, logits = model(samples)
                total_loss_batch, loss_dict = model.compute_losses(features, targets)
            total_loss += total_loss_batch.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v.item()

            optimizer.zero_grad()
            loss_scaler(total_loss_batch, optimizer, parameters=model.parameters())

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

        if (epoch + 1) % 10 == 0:
            model.eval()
            all_preds = []
            all_is_unknown = []
            all_targets = []
            all_dists = []
            with torch.no_grad():
                for samples, targets in val_loader:
                    samples, targets = samples.to(device), targets.to(device)
                    with torch.cuda.amp.autocast():
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

            known_mask = all_targets < model.num_classes
            unknown_mask = ~known_mask

            known_acc = accuracy_score(all_targets[known_mask], all_preds[known_mask]) if known_mask.sum() > 0 else 0
            f1 = f1_score(all_targets[known_mask], all_preds[known_mask], average='macro', zero_division=0)

            true_labels = np.zeros_like(all_targets)
            true_labels[unknown_mask] = 1
            auroc = roc_auc_score(true_labels, all_dists) if len(np.unique(true_labels)) > 1 else 0
            unknown_det = np.mean(all_is_unknown[unknown_mask]) if unknown_mask.sum() > 0 else 0
            conf_matrix = confusion_matrix(true_labels, all_is_unknown)

            print(f"[{datetime.now()}] Validation - Epoch {epoch + 1}:")
            print(f"Known Metrics: Accuracy: {known_acc:.4f}, F1: {f1:.4f}")
            print(f"Unknown/Known Metrics: Detection Rate: {unknown_det:.4f}, AUROC: {auroc:.4f}")
            print(f"Confusion Matrix (Known vs Unknown):\n{conf_matrix}")

            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(model.state_dict(), best_auroc_path)
                print(f"Saved best AUROC model (AUROC: {best_auroc:.4f}) to {best_auroc_path}")

# 메인 실행
if __name__ == "__main__":
    # 1. Fine-tuning된 모델 로드
    finetuned_model = MAEForClassification(mae_vit_huge_patch14())
    finetuned_checkpoint_path = os.path.join(args['output_dir'], 'best_finetune_checkpoint.pth')
    if os.path.exists(finetuned_checkpoint_path):
        checkpoint = torch.load(finetuned_checkpoint_path, map_location='cpu')
        finetuned_model.load_state_dict(checkpoint['model'])
        print(f"Loaded fine-tuned model from {finetuned_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Finetuned checkpoint not found at {finetuned_checkpoint_path}")
    finetuned_model.to(device)

    # 2. OSR 모델 초기화
    osr_model = MAEForOSR(finetuned_model, num_classes=6, feature_dim=1280)
    osr_model.to(device)

    # 3. 데이터셋 로드
    train_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='train', **args)
    val_known_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_known', **args)
    val_unknown_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_unknown', **args)
    val_dataset = ConcatDataset([val_known_dataset, val_unknown_dataset])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args['batch_size'], 
        shuffle=True, 
        num_workers=args['num_workers'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args['batch_size'], 
        shuffle=False, 
        num_workers=args['num_workers'],
        drop_last=False
    )

    # 4. OSR 학습 및 평가
    print(f"Starting OSR training with train: {len(train_dataset)}, val: {len(val_dataset)} samples")
    train_and_evaluate_osr(osr_model, train_loader, val_loader, args, epochs=200)