import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from data_loader import load_data  # 이전에 제공된 load_data 가정
from datetime import datetime  # 시간 출력을 위한 모듈 추가

class OSRClassifier(nn.Module):
    def __init__(self, classid_list, feature_dim=1536):
        super(OSRClassifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda:1')
        self.classid_list = classid_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classid_list)}

        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = self.backbone.features
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
        binary_pred = (np.max(all_outputs,軸=1) < 0.5).astype(int)
        binary_auroc = roc_auc_score(binary_gt, 1 - np.max(all_outputs, axis=1))
        binary_f1 = f1_score(binary_gt, binary_pred)

        # 2. Multiclass Metric (다중 레이블)
        multiclass_gt = all_targets
        multiclass_pred = np.zeros_like(multiclass_gt)
        multiclass_pred[:, :len(model.classid_list)] = (all_outputs > 0.5).astype(int)
        multiclass_pred[:, len(model.classid_list):] = (np.max(all_outputs, axis=1, keepdims=True) < 0.5).astype(int)
        multiclass_auroc = roc_auc_score(multiclass_gt, multiclass_pred, average='macro', multi_class='ovr')
        multiclass_f1 = f1_score(multiclass_gt, multiclass_pred, average='macro')
        val_loss_multiclass = criterion(torch.tensor(all_outputs, device=model.device), 
                                        torch.tensor(all_targets[:, :len(model.classid_list)], device=model.device)).item()

        # 3. Closed Set Metric (known_labels만)
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
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}]   Saved: best_train_loss.pt")
        if val_loss_unknown < best_val_loss_unknown:
            best_val_loss_unknown = val_loss_unknown
            torch.save(model.state_dict(), '/nahcooy/OSR/multi_head/checkpoints/best_val_loss_unknown.pt')
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}]   Saved: best_val_loss_unknown.pt")
        if val_loss_multiclass < best_val_loss_multiclass:
            best_val_loss_multiclass = val_loss_multiclass
            torch.save(model.state_dict(), '/nahcooy/OSR/multi_head/checkpoints/best_val_loss_multiclass.pt')
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    torch.cuda.set_device(1)
    main()