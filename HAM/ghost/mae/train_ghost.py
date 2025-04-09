import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from util.models_mae import mae_vit_huge_patch14
from dataset import getHAM10000Dataset
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

class GHOST:
    def __init__(self, data, FVs, gt):
        self.Gaus_dict = self.Gaus_gen(data, FVs, gt)

    def norm(self, logits, FVs):
        pred = torch.max(logits, dim=1).indices
        normalized_logits = torch.zeros(logits.shape)
        print("Iterating classes for Gaussian scoring")
        for c in tqdm(self.Gaus_dict.keys()):
            class_mask = pred == c
            mean_vector, std_vector = self.Gaus_dict[c]
            FV_Z_Score = torch.abs((FVs[class_mask] - mean_vector) / std_vector)
            diff_score = torch.sum(FV_Z_Score, dim=1)  # 논문대로 sum 사용
            normalized_logits[class_mask, c] = logits[class_mask, c] / diff_score
            if torch.isnan(normalized_logits).any() or torch.isinf(normalized_logits).any():
                print(f"NaN/Inf detected at class {c}")
                normalized_logits[torch.isnan(normalized_logits) | torch.isinf(normalized_logits)] = 0
        return normalized_logits

    def Gaus_gen(self, logits, FV, gt):
        classes = torch.unique(gt).long().tolist()  # Ground truth 기준으로 클래스 추출
        class_models = {}
        print("Generating Gaussian models")
        for c in tqdm(classes):
            select_class_FVs = FV[gt == c]  # Ground truth로 필터링
            mean = torch.mean(select_class_FVs, dim=0)
            std = torch.std(select_class_FVs, dim=0)
            std[std == 0] = 1e-6  # inf 방지
            class_models[c] = (mean, std)
        return class_models

    def ReScore(self, data, FVs, known_mask=None):
        normd_data = self.norm(data, FVs)
        max_data, pred = torch.max(data, dim=1)
        rescored_data = normd_data[range(len(pred)), pred]  # 예측 클래스의 γ 추출
        # Known 기반 정규화 (선택적)
        if known_mask is not None:
            known_scores = rescored_data[known_mask]
            min_score = known_scores.min()
            max_score = known_scores.max()
            rescored_data = (rescored_data - min_score) / (max_score - min_score + 1e-6)
            rescored_data = torch.clamp(rescored_data, 0, 1)
        return rescored_data

# 기본 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = {
    'image_size': 224,
    'random_seed': 42,
    'batch_size': 16,
    'num_workers': 8,
    'output_dir': '/nahcooy/OSR/HAM/ghost/mae/checkpoint',
}

class MAEForClassification(nn.Module):
    def __init__(self, mae_model, num_classes=6):
        super().__init__()
        self.mae = mae_model
        self.head = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.mae.forward_encoder(x, mask_ratio=0)[0]
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return cls_token, logits

# 데이터 로드
train_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='train', **args)
val_known_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_known', **args)
val_unknown_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_unknown', **args)
val_dataset = ConcatDataset([val_known_dataset, val_unknown_dataset])

train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

# 모델 로드
mae_model = mae_vit_huge_patch14()
model = MAEForClassification(mae_model, num_classes=6)
checkpoint = torch.load(os.path.join(args['output_dir'], 'best_finetune_checkpoint.pth'), map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# 학습 데이터 추출
print("Extracting training data...")
train_logits, train_fvs, train_gt = [], [], []
with torch.no_grad():
    for samples, targets in tqdm(train_loader):
        samples, targets = samples.to(device), targets.to(device)
        fvs, logits = model(samples)
        train_logits.append(logits.cpu())
        train_fvs.append(fvs.cpu())
        train_gt.append(targets.cpu())
train_logits = torch.cat(train_logits)
train_fvs = torch.cat(train_fvs)
train_gt = torch.cat(train_gt)

print(f"Train Logits - Min: {train_logits.min().item():.4f}, Max: {train_logits.max().item():.4f}, Mean: {train_logits.mean().item():.4f}")
print(f"Train FVs - Min: {train_fvs.min().item():.4f}, Max: {train_fvs.max().item():.4f}, Mean: {train_fvs.mean().item():.4f}")

# 필터링
train_preds = train_logits.argmax(dim=1)
filtered_mask = train_preds == train_gt
filtered_train_logits = train_logits[filtered_mask]
filtered_train_fvs = train_fvs[filtered_mask]
filtered_train_gt = train_gt[filtered_mask]

# GHOST 피팅 (ground truth 전달)
print("Fitting GHOST model...")
GHOST_model = GHOST(filtered_train_logits, filtered_train_fvs, filtered_train_gt)

# 테스트 데이터 추출
print("Processing validation data...")
val_logits, val_fvs, val_gt = [], [], []
with torch.no_grad():
    for samples, targets in tqdm(val_loader):
        samples, targets = samples.to(device), targets.to(device)
        fvs, logits = model(samples)
        val_logits.append(logits.cpu())
        val_fvs.append(fvs.cpu())
        val_gt.append(targets.cpu())
val_logits = torch.cat(val_logits)
val_fvs = torch.cat(val_fvs)
val_gt = torch.cat(val_gt)

# val_unknown을 OOD로 처리
val_gt[len(val_known_dataset):] = -1

# Known 마스크 생성
known_mask = val_gt >= 0

# GHOST 스코어 계산
probs = GHOST_model.ReScore(val_logits, val_fvs, known_mask=known_mask)
val_preds = val_logits.argmax(dim=1)

# 스코어 분포 확인
print("\nGHOST Score Statistics:")
print(f"Min: {probs.min().item():.4f}, Max: {probs.max().item():.4f}, Mean: {probs.mean().item():.4f}, Std: {probs.std().item():.4f}")
known_scores = probs[val_gt >= 0]
unknown_scores = probs[val_gt == -1]
print(f"Known Scores - Min: {known_scores.min().item():.4f}, Max: {known_scores.max().item():.4f}, Mean: {known_scores.mean().item():.4f}")
print(f"Unknown Scores - Min: {unknown_scores.min().item():.4f}, Max: {unknown_scores.max().item():.4f}, Mean: {unknown_scores.mean().item():.4f}")

# 결과 저장
run_folder = os.path.join(args['output_dir'], 'GHOST_Runs')
os.makedirs(run_folder, exist_ok=True)
output = torch.cat((val_gt.view(-1, 1), val_preds.view(-1, 1), probs.view(-1, 1)), dim=1)
np.save(os.path.join(run_folder, "GHOST_val_preds.npy"), output.numpy())
print(f"Saved predictions to {run_folder}/GHOST_val_preds.npy")

# 지표 계산
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
val_gt_binary = (val_gt >= 0).numpy().astype(int)
probs_np = probs.numpy()

for threshold in thresholds:
    print(f"\n--- Threshold: {threshold} ---")
    pred_known = probs >= threshold
    pred_unknown = probs < threshold
    pred_binary = pred_known.numpy().astype(int)
    
    unknown_mask = val_gt == -1
    unknown_correct = (pred_unknown & unknown_mask).sum().item()
    unknown_total = unknown_mask.sum().item()
    unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0
    print(f"Unknown Accuracy: {unknown_accuracy:.4f} ({unknown_correct}/{unknown_total})")
    
    auroc = roc_auc_score(val_gt_binary, probs_np)
    print(f"AUROC: {auroc:.4f}")
    
    cm = confusion_matrix(val_gt_binary, pred_binary, labels=[0, 1])
    print("Confusion Matrix:")
    print(f"[[TN: {cm[0,0]}, FP: {cm[0,1]}]")
    print(f" [FN: {cm[1,0]}, TP: {cm[1,1]}]]")