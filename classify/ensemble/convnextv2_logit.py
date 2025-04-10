import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from dataset import getHAM10000Dataset  # 너의 기존 코드 기반
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_CLASSES = 7

class ConvNeXtV2ForClassification(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super().__init__()
        self.convnext = models.convnext_base(weights=None)
        in_features = self.convnext.classifier[-1].in_features
        self.convnext.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.convnext(x)


### 🔸 PTH 로드 함수
def load_model(weight_path):
    model = ConvNeXtV2ForClassification(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

### 🔸 로짓 추출 및 저장 함수
def extract_logits(model, split, save_dir):
    dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM_total', split=split, image_size=224)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Extracting {split}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # softmax로 확률로 바꿈
            all_logits.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{split}_logits.npy'), logits)
    np.save(os.path.join(save_dir, f'{split}_labels.npy'), labels)
    print(f"✅ {split} 저장 완료 → {save_dir}")

### 🔸 실행
if __name__ == '__main__':
    model = load_model('/nahcooy/OSR/classify/ConvNeXt/checkpoint/best_val_f1.pth')  # 수정 가능
    extract_logits(model, 'train', '/nahcooy/OSR/classify/ensemble/logits/convnext')   # 또는 'swin' 등
    extract_logits(model, 'val', '/nahcooy/OSR/classify/ensemble/logits/convnext')
