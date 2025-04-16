import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from datetime import datetime
import numpy as np
from torchvision import models  # 🔹 torchvision에서 ResNet 사용

from dataset import getHAM10000Dataset  # 사용자 정의 데이터셋 로더

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 기본 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
args = {
    'data_path': '/dataset/nahcooy/HAM_split',
    'image_size': 224,
    'batch_size': 64,
    'num_workers': 8,
    'epochs': 400,
    'output_dir': './checkpoint_no_aug',
    'num_classes': 7,
    'lr': 2e-4,
    'seed': 42,
    'pretrained': True  # ✅ 사전학습 가중치 사용 권장
}
os.makedirs(args['output_dir'], exist_ok=True)

# 🔹 ResNet-152 기반 모델 정의
class ResNet152ForClassification(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ResNet152ForClassification, self).__init__()
        self.model = models.resnet152(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 🔹 모델 로드
def load_model():
    model = ResNet152ForClassification(num_classes=args['num_classes'], pretrained=args['pretrained'])
    return model

# 🔹 학습 루프
def train():
    model = load_model().to(device)

    # 데이터 로딩
    train_dataset = getHAM10000Dataset(data_path=args['data_path'], split='train', image_size=args['image_size'])
    val_dataset = getHAM10000Dataset(data_path=args['data_path'], split='val', image_size=args['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

    best_val_loss = float('inf')
    best_val_acc = 0.0
    print_freq = 50

    for epoch in range(args['epochs']):
        model.train()
        train_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (step + 1) % print_freq == 0 or (step + 1) == len(train_loader):
                print(f"[{datetime.now()}] Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"[{datetime.now()}] Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f}")

        # 🔸 검증
        model.eval()
        val_loss = 0.0
        correct_top1, correct_top3, total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds_top1 = torch.argmax(probs, dim=1)
                preds_top3 = torch.topk(probs, k=3, dim=1).indices

                correct_top1 += (preds_top1 == labels).sum().item()
                correct_top3 += sum([labels[i] in preds_top3[i] for i in range(len(labels))])
                total += labels.size(0)

                all_preds.extend(preds_top1.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        top1_acc = correct_top1 / total
        top3_acc = correct_top3 / total

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        f1 = f1_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        all_labels_oh = label_binarize(all_labels, classes=np.arange(args['num_classes']))
        auroc = roc_auc_score(all_labels_oh, all_probs, average='macro', multi_class='ovr')

        cm = confusion_matrix(all_labels, all_preds)

        # 🔸 검증 메트릭 출력
        print(f"\n[{datetime.now()}] Epoch [{epoch+1}/{args['epochs']}] Validation:")
        print(f"  🔸 Val Loss: {avg_val_loss:.4f}, Top-1 Acc: {top1_acc:.4f}, Top-3 Acc: {top3_acc:.4f}")
        print(f"  🔸 F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUROC: {auroc:.4f}")
        print(f"  🔸 Confusion Matrix:\n{cm}")
        print(f"  🔸 Classification Report:\n{classification_report(all_labels, all_preds, digits=4)}")

        # 🔸 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = os.path.join(args['output_dir'], 'best_val_loss.pth')
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'val_loss': best_val_loss}, path)
            print(f"✅ Best (Val Loss) model saved at epoch {epoch+1} → {path}")

        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            path = os.path.join(args['output_dir'], 'best_val_acc.pth')
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'val_acc': best_val_acc}, path)
            print(f"✅ Best (Accuracy) model saved at epoch {epoch+1} → {path}")

        scheduler.step()

    # 🔸 최종 리포트
    print("\n📊 Final Classification Report (Validation Set):")
    print(classification_report(all_labels, all_preds, digits=4))
    print("📊 Final Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    train()
