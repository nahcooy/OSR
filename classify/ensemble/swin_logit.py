import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from dataset import getHAM10000Dataset
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_CLASSES = 7

# 🔧 Swin Transformer 기반 Classification 모델 정의
class SwinForClassification(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.swin = models.swin_v2_s(weights='IMAGENET1K_V1' if pretrained else None)  # 사전학습 선택 가능
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.swin(x)

# 🔸 PTH 로드 함수
def load_model(weight_path):
    model = SwinForClassification().to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt['model'])  # 'model' key 기준 저장된 경우
    model.eval()
    return model

# 🔸 로짓 추출 및 저장 함수
def extract_logits(model, split, save_dir):
    dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM_total', split=split, image_size=224)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Extracting {split}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # 확률로 변환
            all_logits.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{split}_logits.npy'), logits)
    np.save(os.path.join(save_dir, f'{split}_labels.npy'), labels)
    print(f"✅ {split} 저장 완료 → {save_dir}")

# 🔸 실행
if __name__ == '__main__':
    weight_path = '/nahcooy/OSR/classify/swin/checkpoint/best_val_f1.pth'
    save_path = 'logits/swin'
    model = load_model(weight_path)
    extract_logits(model, 'train', save_path)
    extract_logits(model, 'val', save_path)
