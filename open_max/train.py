import os
import random
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# 데이터셋 클래스
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', 
                 known_labels=None, unknown_labels=None, data_ratio=1.0, extra_unknown_data=None, train_dir=None):
        print(f"Loading CSV: {csv_file}")
        self.data = pd.read_csv(csv_file)
        print(f"Initial data size from {csv_file}: {len(self.data)}")
        self.root_dir = root_dir
        self.train_dir = train_dir if train_dir else root_dir
        self.transform = transform
        self.mode = mode
        self.known_labels = known_labels if known_labels is not None else []
        self.unknown_labels = unknown_labels if unknown_labels is not None else []
        self.data_ratio = max(0.0, min(1.0, data_ratio))
        self.extra_unknown_data = extra_unknown_data

        if mode == 'train':
            self.data = self.data[self.data['Finding Labels'].apply(
                lambda x: all(label in self.known_labels + ['No Finding'] for label in x.split('|'))
            )].reset_index(drop=True)
            print(f"Train mode filtered data size: {len(self.data)}")
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels)}
        elif mode == 'val':
            if self.extra_unknown_data is not None:
                print(f"Extra unknown data size: {len(self.extra_unknown_data)}")
                self.data = pd.concat([self.data, self.extra_unknown_data]).drop_duplicates(subset=['Image Index']).reset_index(drop=True)
                print(f"Val mode after concat and dedup: {len(self.data)}")
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels + self.unknown_labels)}
            self.unknown_data_only = self.data[self.data['Finding Labels'].apply(
                lambda x: any(label in self.unknown_labels for label in x.split('|')) and 
                all(label not in self.known_labels for label in x.split('|'))
            )]
            self.unknown_and_known_data = self.data[self.data['Finding Labels'].apply(
                lambda x: any(label in self.unknown_labels for label in x.split('|')) and 
                any(label in self.known_labels for label in x.split('|'))
            )]
            self.known_data = self.data[self.data['Finding Labels'].apply(
                lambda x: all(label in self.known_labels + ['No Finding'] for label in x.split('|'))
            )]
            print(f"Val mode - unknown only: {len(self.unknown_data_only)}, mixed: {len(self.unknown_and_known_data)}, known: {len(self.known_data)}")
        elif mode == 'test':
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels + self.unknown_labels)}

        if self.data_ratio < 1.0:
            label_counts = {label: [] for label in self.class_to_idx.keys()}
            normal_indices = []
            for idx in range(len(self.data)):
                labels = self.data.iloc[idx]['Finding Labels'].split('|')
                if "No Finding" in labels:
                    normal_indices.append(idx)
                else:
                    for label in labels:
                        if label in self.class_to_idx:
                            label_counts[label].append(idx)

            sampled_indices = set()
            for label, indices in label_counts.items():
                if indices and label != "No Finding":
                    num_samples = max(1, int(len(indices) * self.data_ratio))
                    sampled_indices.update(random.sample(indices, min(num_samples, len(indices))))
            if normal_indices:
                num_normal = max(1, int(len(normal_indices) * self.data_ratio))
                sampled_indices.update(random.sample(normal_indices, min(num_normal, len(normal_indices))))

            self.data = self.data.iloc[list(sampled_indices)].reset_index(drop=True)
            print(f"Data size after sampling: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_idx = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.train_dir if self.extra_unknown_data is not None and img_idx in self.extra_unknown_data['Image Index'].values else self.root_dir, img_idx)
        try:
            image = Image.open(img_path).convert('L')
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        
        labels = self.data.iloc[idx]['Finding Labels'].split('|')
        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# 데이터 로드 함수
def load_data(train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels, 
              batch_size=64, train_data_ratio=1.0, val_data_ratio=1.0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_data = pd.read_csv(train_csv)
    print(f"Full train data size: {len(full_train_data)}")
    train_known_data = full_train_data[full_train_data['Finding Labels'].apply(
        lambda x: all(label in known_labels + ['No Finding'] for label in x.split('|'))
    )]
    train_unknown_data = full_train_data[~full_train_data.index.isin(train_known_data.index)]
    val_data = pd.read_csv(val_csv)
    train_unknown_data = train_unknown_data[~train_unknown_data['Image Index'].isin(val_data['Image Index'])]

    train_dataset = NIHChestXrayDataset(
        csv_file=train_csv, root_dir=train_dir, transform=transform, mode='train', 
        known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=train_data_ratio
    )
    val_dataset = NIHChestXrayDataset(
        csv_file=val_csv, root_dir=val_dir, transform=transform, mode='val', 
        known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=val_data_ratio,
        extra_unknown_data=train_unknown_data, train_dir=train_dir
    )

    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return torch.zeros(1, 1, 224, 224), torch.zeros(1, len(known_labels) + len(unknown_labels))
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.stack(targets)
        return images, targets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader

# 모델 클래스 (ResNet50 기반)
class OSRClassifier(nn.Module):
    def __init__(self, classid_list, feature_dim=2048):  # ResNet50의 기본 출력 차원
        super(OSRClassifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda:0')
        self.classid_list = classid_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classid_list)}

        # ResNet50 백본
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 흑백 이미지용
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])  # 마지막 FC 레이어 제거
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.heads = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(self.num_classes)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        outputs = [self.sigmoid(head(x)) for head in self.heads]
        return torch.cat(outputs, dim=1)

    def predict(self, outputs, threshold=0.5):
        preds = outputs > threshold
        pred_labels = torch.argmax(outputs, dim=1)
        unknown_mask = preds.sum(dim=1) == 0
        pred_labels[unknown_mask] = -1
        return pred_labels

# 학습 및 평가 함수
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs):
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
            outputs = model(images)
            loss = criterion(outputs, targets[:, :len(model.classid_list)])
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

        # 검증
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(model.device), targets.to(model.device)
                outputs = model(images)
                loss = criterion(outputs, targets[:, :len(model.classid_list)])
                val_loss += loss.item() * images.size(0)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 1. Binary Metric (Unknown vs Known)
        binary_gt = np.any(all_targets[:, len(model.classid_list):], axis=1).astype(int)
        binary_pred = (np.max(all_outputs, axis=1) < 0.5).astype(int)
        binary_auroc = roc_auc_score(binary_gt, 1 - np.max(all_outputs, axis=1))
        binary_f1 = f1_score(binary_gt, binary_pred)

        # 2. Multiclass Metric
        multiclass_gt = all_targets
        multiclass_pred = np.zeros_like(multiclass_gt)
        multiclass_pred[:, :len(model.classid_list)] = (all_outputs > 0.5).astype(int)
        multiclass_pred[:, len(model.classid_list):] = (np.max(all_outputs, axis=1, keepdims=True) < 0.5).astype(int)
        multiclass_auroc = roc_auc_score(multiclass_gt, multiclass_pred, average='macro', multi_class='ovr')
        multiclass_f1 = f1_score(multiclass_gt, multiclass_pred, average='macro')
        val_loss_multiclass = criterion(torch.tensor(all_outputs, device=model.device), 
                                        torch.tensor(all_targets[:, :len(model.classid_list)], device=model.device)).item()

        # 3. Closed Set Metric
        closed_targets = all_targets[:, :len(model.classid_list)]
        closed_outputs = all_outputs
        closed_pred = (closed_outputs > 0.5).astype(int)
        closed_auroc = roc_auc_score(closed_targets, closed_outputs, average='macro')
        closed_f1 = f1_score(closed_targets, closed_pred, average='macro')
        val_loss_unknown = criterion(torch.tensor(all_outputs, device=model.device), 
                                     torch.tensor(all_targets[:, :len(model.classid_list)], device=model.device)).item()

        # 메트릭 출력
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{current_time}] Epoch {epoch+1}:")
        print(f"[{current_time}]   Train Loss: {train_loss:.4f}")
        print(f"[{current_time}]   Val Loss: {val_loss:.4f}")
        print(f"[{current_time}]   Binary (Unknown vs Known) - AUROC: {binary_auroc:.4f}, F1: {binary_f1:.4f}")
        print(f"[{current_time}]   Multiclass - AUROC: {multiclass_auroc:.4f}, F1: {multiclass_f1:.4f}")
        print(f"[{current_time}]   Closed Set - AUROC: {closed_auroc:.4f}, F1: {closed_f1:.4f}")

        # Best 모델 저장
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), '/nahcooy/OSR/multi_head/checkpoints/best_train_loss.pt')
            print(f"[{current_time}]   Saved: best_train_loss.pt")
        if val_loss_unknown < best_val_loss_unknown:
            best_val_loss_unknown = val_loss_unknown
            torch.save(model.state_dict(), '/nahcooy/OSR/multi_head/checkpoints/best_val_loss_unknown.pt')
            print(f"[{current_time}]   Saved: best_val_loss_unknown.pt")
        if val_loss_multiclass < best_val_loss_multiclass:
            best_val_loss_multiclass = val_loss_multiclass
            torch.save(model.state_dict(), '/nahcooy/OSR/multi_head/checkpoints/best_val_loss_multiclass.pt')
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