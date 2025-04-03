import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from scipy.stats import weibull_min
from data_loader import load_data

# OpenMax 클래스
class OpenMax:
    def __init__(self, model, tailsize=20, alpha=1.0, threshold=0.5):
        self.model = model
        self.tailsize = tailsize
        self.alpha = alpha
        self.threshold = threshold
        self.weibull_models = {}
        self.classid_list = model.classid_list

    def fit_weibull(self, train_loader):
        """Known 클래스에 대해 Weibull 분포 피팅"""
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in train_loader:
                images = images.to(self.model.device)
                outputs = self.model(images)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        for cls_idx, cls in enumerate(self.classid_list):
            cls_mask = all_targets[:, cls_idx] == 1
            cls_outputs = all_outputs[cls_mask, cls_idx]
            sorted_scores = np.sort(cls_outputs)[::-1][:self.tailsize]
            if len(sorted_scores) > 0:
                weibull_fit = weibull_min.fit(sorted_scores, floc=0)
                self.weibull_models[cls] = weibull_fit

    def openmax_recalibrate(self, logits):
        """OpenMax로 확률 재조정"""
        scores = torch.sigmoid(logits).cpu().numpy()  # Sigmoid 확률
        batch_size = scores.shape[0]
        openmax_scores = np.zeros(batch_size)  # Unknown 확률
        recalibrated_scores = np.zeros_like(scores)  # 재조정된 Known 확률

        for i in range(batch_size):
            v = scores[i]
            w = np.zeros_like(v)

            for j, cls in enumerate(self.classid_list):
                if cls in self.weibull_models:
                    weibull_params = self.weibull_models[cls]
                    w[j] = 1 - weibull_min.cdf(v[j], *weibull_params)

            v_open = np.sum(v * (1 - w))  # Unknown에 기여하는 부분
            v_known = v * w  # Known 클래스 확률 재조정
            total = np.sum(v_known) + v_open

            if total > 0:
                recalibrated_scores[i] = v_known / total
                openmax_scores[i] = v_open / total
            else:
                recalibrated_scores[i] = v
                openmax_scores[i] = 0.0

        return recalibrated_scores, openmax_scores

# 모델 클래스 (ResNet50 + OpenMax)
class OSRClassifier(nn.Module):
    def __init__(self, classid_list, feature_dim=2048):
        super(OSRClassifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda:0')
        self.classid_list = classid_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classid_list)}

        # ResNet50 백본
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 흑백 이미지용
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(feature_dim, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# 학습 및 평가 함수 (OpenMax 적용)
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    openmax = OpenMax(model, tailsize=20, alpha=1.0, threshold=0.5)
    best_train_loss = float('inf')
    best_val_loss_unknown = float('inf')
    best_val_loss_multiclass = float('inf')
    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        # 학습
        model.train()
        train_loss = 0.0
        batch_count = 0
        running_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            batch_count += 1
            images, targets = images.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(torch.sigmoid(logits), targets[:, :len(model.classid_list)])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            running_loss += loss.item() * images.size(0)
            if batch_count % 50 == 0:
                avg_running_loss = running_loss / (50 * images.size(0))
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] Epoch {epoch+1}, Batch {batch_count}/{total_batches}: Running Train Loss: {avg_running_loss:.4f}")
                running_loss = 0.0
        train_loss /= len(train_loader.dataset)

        # Weibull 피팅 (첫 에포크에서만)
        if epoch == 0:
            openmax.fit_weibull(train_loader)

        # 검증
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        all_openmax_preds = []
        all_openmax_unknown = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(model.device), targets.to(model.device)
                logits = model(images)
                loss = criterion(torch.sigmoid(logits), targets[:, :len(model.classid_list)])
                val_loss += loss.item() * images.size(0)
                recalibrated, unknown_scores = openmax.openmax_recalibrate(logits)
                all_outputs.append(logits.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_openmax_preds.append(recalibrated)
                all_openmax_unknown.append(unknown_scores)

        val_loss /= len(val_loader.dataset)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_openmax_preds = np.concatenate(all_openmax_preds, axis=0)
        all_openmax_unknown = np.concatenate(all_openmax_unknown, axis=0)

        # 1. Binary Metric (Unknown vs Known)
        binary_gt = np.any(all_targets[:, len(model.classid_list):], axis=1).astype(int)
        binary_pred = (all_openmax_unknown > openmax.threshold).astype(int)
        binary_auroc = roc_auc_score(binary_gt, all_openmax_unknown)
        binary_f1 = f1_score(binary_gt, binary_pred)

        # 2. Multiclass Metric (Known Only)
        known_mask = binary_pred == 0
        if np.sum(known_mask) > 0:
            known_targets = all_targets[known_mask, :len(model.classid_list)]
            known_preds = (all_openmax_preds[known_mask] > 0.5).astype(int)
            multiclass_auroc = roc_auc_score(known_targets, all_openmax_preds[known_mask], average='macro')
            multiclass_f1 = f1_score(known_targets, known_preds, average='macro')
            val_loss_multiclass = criterion(
                torch.tensor(all_openmax_preds[known_mask], device=model.device),
                torch.tensor(known_targets, device=model.device)
            ).item()
        else:
            multiclass_auroc, multiclass_f1, val_loss_multiclass = 0.0, 0.0, float('inf')

        # 3. Closed Set Metric
        closed_targets = all_targets[:, :len(model.classid_list)]
        closed_outputs = all_openmax_preds
        closed_pred = (closed_outputs > 0.5).astype(int)
        closed_auroc = roc_auc_score(closed_targets, closed_outputs, average='macro')
        closed_f1 = f1_score(closed_targets, closed_pred, average='macro')
        val_loss_unknown = criterion(
            torch.tensor(closed_outputs, device=model.device),
            torch.tensor(closed_targets, device=model.device)
        ).item()

        # 메트릭 출력
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{current_time}] Epoch {epoch+1}:")
        print(f"[{current_time}]   Train Loss: {train_loss:.4f}")
        print(f"[{current_time}]   Val Loss: {val_loss:.4f}")
        print(f"[{current_time}]   Binary (Unknown vs Known) - AUROC: {binary_auroc:.4f}, F1: {binary_f1:.4f}")
        print(f"[{current_time}]   Multiclass (Known Only) - AUROC: {multiclass_auroc:.4f}, F1: {multiclass_f1:.4f}")
        print(f"[{current_time}]   Closed Set - AUROC: {closed_auroc:.4f}, F1: {closed_f1:.4f}")

        # Best 모델 저장
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), '/nahcooy/OSR/real_open_max/checkpoints/best_train_loss.pt')
            print(f"[{current_time}]   Saved: best_train_loss.pt")
        if val_loss_unknown < best_val_loss_unknown:
            best_val_loss_unknown = val_loss_unknown
            torch.save(model.state_dict(), '/nahcooy/OSR/real_open_max/checkpoints/best_val_loss_unknown.pt')
            print(f"[{current_time}]   Saved: best_val_loss_unknown.pt")
        if val_loss_multiclass < best_val_loss_multiclass:
            best_val_loss_multiclass = val_loss_multiclass
            torch.save(model.state_dict(), '/nahcooy/OSR/real_open_max/checkpoints/best_val_loss_multiclass.pt')
            print(f"[{current_time}]   Saved: best_val_loss_multiclass.pt")

# 메인 함수
def main():
    train_csv = '/dataset/nahcooy/CXR8/images/train.csv'
    train_dir = '/dataset/nahcooy/CXR8/images/train'
    val_csv = '/dataset/nahcooy/CXR8/images/val.csv'
    val_dir = '/dataset/nahcooy/CXR8/images/val'
    known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                    "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding",
                    "Pneumonia", "Edema", "Emphysema", "Fibrosis"]
    unknown_labels = ["Nodule"]
    all_labels = known_labels + unknown_labels

    train_loader, val_loader = load_data(
        train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels,
        batch_size=128, train_data_ratio=0.3, val_data_ratio=1.0
    )

    model = OSRClassifier(classid_list=known_labels)
    model.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    num_epochs = 100
    train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()