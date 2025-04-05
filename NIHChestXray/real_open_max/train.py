import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from datetime import datetime
from scipy.stats import weibull_min
from data_loader import load_data
from torchmetrics import AUROC, F1Score
from torch.utils.data import DataLoader

# OpenMax 클래스 (GPU 병렬화)
class OpenMax:
    def __init__(self, model, tailsize=20, alpha=1.5, threshold=0.5):
        self.model = model
        self.tailsize = tailsize
        self.alpha = alpha
        self.threshold = threshold
        self.weibull_models = {}
        self.classid_list = model.classid_list
        self.device = model.device

    def fit_weibull(self, train_loader):
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in train_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        for cls_idx, cls in enumerate(self.classid_list):
            cls_mask = all_targets[:, cls_idx] == 1
            cls_outputs = all_outputs[cls_mask, cls_idx]
            sorted_scores = np.sort(cls_outputs)[::-1][:self.tailsize]
            if len(sorted_scores) >= self.tailsize // 2:
                weibull_fit = weibull_min.fit(sorted_scores)
                self.weibull_models[cls] = weibull_fit

    def openmax_recalibrate(self, logits):
        scores = torch.sigmoid(logits)
        batch_size, num_classes = scores.shape
        w = torch.ones_like(scores, device=self.device)

        for j, cls in enumerate(self.classid_list):
            if cls in self.weibull_models:
                weibull_params = self.weibull_models[cls]
                shape, loc, scale = weibull_params
                cdf = 1 - torch.exp(-((scores[:, j] - loc) / scale) ** shape)
                w[:, j] = 1 - self.alpha * cdf

        v_open = torch.sum(scores * (1 - w), dim=1)
        v_known = scores * w
        total = torch.sum(v_known, dim=1) + v_open

        recalibrated_scores = v_known / total.unsqueeze(1)
        openmax_scores = v_open / total
        return recalibrated_scores, openmax_scores

# 모델 클래스
class OSRClassifier(nn.Module):
    def __init__(self, classid_list, feature_dim=2048):
        super(OSRClassifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda:1')
        self.classid_list = classid_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classid_list)}

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feature_dim, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# 학습 및 평가 함수 (검증 부분만 수정)
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    openmax = OpenMax(model)
    best_train_loss = float('inf')
    best_val_loss_unknown = float('inf')
    best_val_loss_multiclass = float('inf')
    total_batches = len(train_loader)
    checkpoint_dir = '/nahcooy/OSR/real_open_max/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    binary_auroc_metric = AUROC(task="binary").to(model.device)
    multiclass_auroc_metric = AUROC(task="multilabel", num_labels=len(model.classid_list), average="macro").to(model.device)
    multiclass_f1_metric = F1Score(task="multilabel", num_labels=len(model.classid_list), average="macro").to(model.device)

    for epoch in range(num_epochs):
        # 학습 (생략)
        model.train()
        train_loss = 0.0
        batch_count = 0
        running_loss = 0.0
        sample_count = 0
        for i, (images, targets) in enumerate(train_loader):
            batch_count += 1
            sample_count += images.size(0)
            images, targets = images.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, targets[:, :len(model.classid_list)])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            running_loss += loss.item() * images.size(0)
            if batch_count % 50 == 0:
                avg_running_loss = running_loss / sample_count
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] Epoch {epoch+1}, Batch {batch_count}/{total_batches}: Running Train Loss: {avg_running_loss:.4f}")
                running_loss = 0.0
                sample_count = 0
        train_loss /= len(train_loader.dataset)

        if epoch % 5 == 0:
            openmax.fit_weibull(train_loader)

        # 검증
        model.eval()
        val_loss = 0.0
        num_val_samples = len(val_loader.dataset)
        all_targets = torch.zeros((num_val_samples, len(model.classid_list) + 1), device=model.device)
        all_openmax_preds = torch.zeros((num_val_samples, len(model.classid_list)), device=model.device)
        all_openmax_unknown = torch.zeros(num_val_samples, device=model.device)
        idx = 0

        with torch.no_grad():
            for images, targets in val_loader:
                batch_size = images.size(0)
                images, targets = images.to(model.device), targets.to(model.device)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, targets[:, :len(model.classid_list)])
                    recalibrated, unknown_scores = openmax.openmax_recalibrate(logits)
                val_loss += loss.item() * batch_size
                all_targets[idx:idx+batch_size] = targets
                all_openmax_preds[idx:idx+batch_size] = recalibrated
                all_openmax_unknown[idx:idx+batch_size] = unknown_scores
                idx += batch_size

        val_loss /= num_val_samples

        # 메트릭 계산 (5 에포크마다 또는 마지막 에포크)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            binary_gt = all_targets[:, len(model.classid_list):].any(dim=1).float()
            binary_pred = (all_openmax_unknown > openmax.threshold).float()
            binary_auroc = binary_auroc_metric(all_openmax_unknown, binary_gt).item()
            binary_f1 = F1Score(task="binary").to(model.device)(binary_pred, binary_gt).item()

            known_mask = (all_openmax_unknown <= openmax.threshold)
            if known_mask.sum() > 0:
                known_targets = all_targets[known_mask, :len(model.classid_list)]
                known_preds = (all_openmax_preds[known_mask] > 0.5).float()
                known_targets_long = known_targets.long()  # torch.float -> torch.long 변환
                multiclass_auroc = multiclass_auroc_metric(all_openmax_preds[known_mask], known_targets_long).item()
                multiclass_f1 = multiclass_f1_metric(known_preds, known_targets_long).item()
                val_loss_multiclass = criterion(all_openmax_preds[known_mask], known_targets).item()  # 손실은 float 유지
            else:
                multiclass_auroc, multiclass_f1, val_loss_multiclass = 0.0, 0.0, float('inf')

            closed_targets = all_targets[:, :len(model.classid_list)]
            closed_preds = (all_openmax_preds > 0.5).float()
            closed_targets_long = closed_targets.long()  # torch.float -> torch.long 변환
            closed_auroc = multiclass_auroc_metric(all_openmax_preds, closed_targets_long).item()
            closed_f1 = multiclass_f1_metric(closed_preds, closed_targets_long).item()
            val_loss_unknown = criterion(all_openmax_preds, closed_targets).item()  # 손실은 float 유지

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] Epoch {epoch+1}:")
            print(f"[{current_time}]   Train Loss: {train_loss:.4f}")
            print(f"[{current_time}]   Val Loss: {val_loss:.4f}")
            print(f"[{current_time}]   Binary (Unknown vs Known) - AUROC: {binary_auroc:.4f}, F1: {binary_f1:.4f}")
            print(f"[{current_time}]   Multiclass (Known Only) - AUROC: {multiclass_auroc:.4f}, F1: {multiclass_f1:.4f}")
            print(f"[{current_time}]   Closed Set - AUROC: {closed_auroc:.4f}, F1: {closed_f1:.4f}")
        else:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] Epoch {epoch+1}:")
            print(f"[{current_time}]   Train Loss: {train_loss:.4f}")
            print(f"[{current_time}]   Val Loss: {val_loss:.4f}")

        # 모델 저장 (생략)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_train_loss.pt'))
            print(f"[{current_time}]   Saved: best_train_loss.pt")
        if val_loss_unknown < best_val_loss_unknown:
            best_val_loss_unknown = val_loss_unknown
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_loss_unknown.pt'))
            print(f"[{current_time}]   Saved: best_val_loss_unknown.pt")
        if val_loss_multiclass < best_val_loss_multiclass:
            best_val_loss_multiclass = val_loss_multiclass
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_loss_multiclass.pt'))
            print(f"[{current_time}]   Saved: best_val_loss_multiclass.pt")

def main():
    train_csv = '/dataset/nahcooy/CXR8/images/train.csv'
    train_dir = '/dataset/nahcooy/CXR8/images/train'
    val_csv = '/dataset/nahcooy/CXR8/images/val.csv'
    val_dir = '/dataset/nahcooy/CXR8/images/val'
    known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                    "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding",
                    "Pneumonia", "Edema", "Emphysema", "Fibrosis"]
    unknown_labels = ["Nodule"]

    train_loader, val_loader = load_data(
        train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels,
        batch_size=256, train_data_ratio=0.5, val_data_ratio=1.0
    )
    train_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_loader.dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)

    model = OSRClassifier(classid_list=known_labels)
    model.to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # 수정: BCEWithLogitsLoss로 변경

    num_epochs = 200
    train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    main()