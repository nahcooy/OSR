import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tqdm import tqdm

# 시드 고정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Known과 Unknown 라벨 정의
known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
                "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding"]
unknown_labels = ["Pneumonia", "Edema", "Emphysema", "Fibrosis"]
all_labels = known_labels + unknown_labels

# Copycat 네트워크
class Copycat(nn.Module):
    def __init__(self, feature_dim=1536):
        super(Copycat, self).__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1536, feature_dim)

    def forward(self, x):
        intermediate_features = []
        x1 = self.features[0](x)
        intermediate_features.append(x1)
        x2 = self.features[1](x1)
        intermediate_features.append(x2)
        x3 = self.features[2](x2)
        intermediate_features.append(x3)
        x4 = self.features[3](x3)
        intermediate_features.append(x4)
        x5 = self.features[4](x4)
        intermediate_features.append(x5)
        x = x5
        for layer in self.features[5:]:
            x = layer(x)
        x6 = x
        intermediate_features.append(x6)
        x = self.avgpool(x6)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, intermediate_features

# Classifier 네트워크
class Classifier(nn.Module):
    def __init__(self, classid_list, feature_dim=1536):
        super(Classifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classid_list = classid_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classid_list)}
        
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1536, self.num_classes)

    def forward(self, x, copycat_features=None):
        intermediate_features = []
        x1 = self.features[0](x)
        intermediate_features.append(x1)
        x2 = self.features[1](x1)
        intermediate_features.append(x2)
        x3 = self.features[2](x2)
        intermediate_features.append(x3)
        x4 = self.features[3](x3)
        if copycat_features is not None:
            x4 = x4 + copycat_features[3]  # Feature injection
        intermediate_features.append(x4)
        x5 = self.features[4](x4)
        intermediate_features.append(x5)
        x = x5
        for layer in self.features[5:]:
            x = layer(x)
        x6 = x
        intermediate_features.append(x6)
        x = self.avgpool(x6)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, intermediate_features

    def compute_joint_energy(self, logits):
        label_wise_energy = -torch.log(1 + torch.exp(logits))
        joint_energy = -torch.sum(label_wise_energy, dim=1)
        return joint_energy

    def predict(self, logits, tau_nor, tau_unk):
        joint_energy = self.compute_joint_energy(logits)
        probs = torch.sigmoid(logits)
        preds = torch.zeros_like(probs, device=self.device)
        for i in range(logits.shape[0]):
            if joint_energy[i] < tau_nor:
                preds[i, self.class_to_idx["No Finding"]] = 1  # Normal
            elif joint_energy[i] > tau_unk:
                preds[i] = (probs[i] > 0.5).float()  # Unknown은 Known처럼 예측 후 후처리
                if not preds[i].any():  # 예측 없으면 Unknown
                    preds[i] = -1
            else:
                preds[i] = (probs[i] > 0.5).float()  # Known
        return preds, joint_energy

# 학습 함수 정의
def train(net1, net2, criterion, optimizer1, optimizer2, trainloader, **options):
    lsr_criterion = SmoothCrossEntropy(options['smoothing'])
    lsr_criterion2 = SmoothCrossEntropy(options['smoothing2'])
    l1_loss = nn.L1Loss()

    torch.cuda.empty_cache()

    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.cuda(), labels.cuda()
        bsz = labels.size(0)

        # Copycat 네트워크 훈련
        net1.train()
        net2.eval()
        optimizer1.zero_grad()

        feat11, feat12, feat1, out1 = net1(data)
        feat21, feat22, feat2, out2 = net2(data)
        loss1 = criterion(out1, labels)
        pullloss1 = l1_loss(feat11.reshape(bsz, -1), feat21.reshape(bsz, -1).detach())
        pullloss2 = l1_loss(feat12.reshape(bsz, -1), feat22.reshape(bsz, -1).detach())
        pullloss = (pullloss1 + pullloss2) / 2
        loss1 = loss1 + options['pull_ratio'] * pullloss
        loss1.backward()
        optimizer1.step()

        # Classifier 네트워크 훈련
        net1.eval()
        net2.train()
        optimizer2.zero_grad()

        out21 = net2(feat11.detach())
        out22 = net2(feat12.detach())
        out20 = net2(feat1.clone().detach())
        klu0 = lsr_criterion2(out20, labels)
        klu1 = lsr_criterion(out21, labels)
        klu2 = lsr_criterion(out22, labels)
        klu = (klu0 + klu1 + klu2) / 3
        loss2 = criterion(out2, labels)
        loss2 = loss2 + klu * options['fake_ratio']
        loss2.backward()
        optimizer2.step()

# 평가 함수
def evaluate_multilabel(predictions, true_labels, all_labels, known_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    normal_true = (true_labels[:, all_labels.index("No Finding")] == 1) & (true_labels.sum(axis=1) == 1)
    normal_pred = (predictions[:, all_labels.index("No Finding")] == 1) & (predictions.sum(axis=1) == 1)

    known_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]).issubset(known_labels) and l.sum() > 0 else 0 for l in true_labels])
    unknown_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) - set(known_labels) and l.sum() > 0 else 0 for l in true_labels])

    known_pred = np.zeros(len(predictions))
    unknown_pred = np.zeros(len(predictions))
    for i, pred in enumerate(predictions):
        pred_labels = set([all_labels[idx] for idx, val in enumerate(pred) if val > 0])
        if pred_labels.issubset(known_labels) and len(pred_labels) > 0:
            known_pred[i] = 1
        elif pred_labels - set(known_labels) or (pred == -1).any():
            unknown_pred[i] = 1

    # AUROC 계산
    if len(np.unique(normal_true)) > 1:
        auroc_normal = roc_auc_score(normal_true, normal_pred)
        print(f'AUROC Normal: {auroc_normal:.4f}')
    if len(np.unique(known_true)) > 1:
        auroc_known = roc_auc_score(known_true, known_pred)
        print(f'AUROC Known: {auroc_known:.4f}')
    if len(np.unique(unknown_true)) > 1:
        auroc_unknown = roc_auc_score(unknown_true, unknown_pred)
        print(f'AUROC Unknown: {auroc_unknown:.4f}')

    # 라벨별 성능
    for i, label in enumerate(all_labels):
        if len(np.unique(true_labels[:, i])) > 1:
            auroc = roc_auc_score(true_labels[:, i], predictions[:, i])
            precision = precision_score(true_labels[:, i], (predictions[:, i] > 0).astype(int), zero_division=0)
            recall = recall_score(true_labels[:, i], (predictions[:, i] > 0).astype(int), zero_division=0)
            print(f'{label}: AUROC={auroc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')
        else:
            print(f'{label}: 평가 불가 (단일 클래스)')

# 평가 실행
evaluate_multilabel(predictions, true_labels, all_labels, known_labels)
