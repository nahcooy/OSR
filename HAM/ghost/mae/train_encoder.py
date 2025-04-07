import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime
from util.models_mae import mae_vit_huge_patch14
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from dataset import getHAM10000Dataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 기본 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = './checkpoint'  # 저장 경로
os.makedirs(output_dir, exist_ok=True)

# 학습에 필요한 설정만 남김
settings = {
    'image_size': 224,
    'random_seed': 42,
    'batch_size': 32,
    'num_workers': 8,
    'split': 'total_known',
}

def pretrain_mae():
    # 데이터셋 로딩
    dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', **settings)

    data_loader = DataLoader(
        dataset,
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        shuffle=True,
        drop_last=True,
    )

    # 모델 로딩
    model = mae_vit_huge_patch14()
    checkpoint = torch.load('/nahcooy/OSR/HAM/ghost/mae/mae_pretrain_vit_huge.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    # Optimizer 및 Loss Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_scaler = NativeScaler()

    # 학습 설정
    epochs = 500
    mask_ratio = 0.75
    print_freq = 50

    # Best 모델 추적
    best_loss = float('inf')  # 초기값을 무한대로 설정
    best_epoch = 0
    save_path = os.path.join(output_dir, 'best_checkpoint.pth')

    print(f"Starting MAE pre-training with {len(dataset)} samples")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (samples, _) in enumerate(data_loader):
            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=mask_ratio)
            loss_value = loss.item()
            total_loss += loss_value

            loss_scaler(loss, optimizer, parameters=model.parameters())
            optimizer.zero_grad()

            if (step + 1) % print_freq == 0 or (step + 1) == len(data_loader):
                print(f"[{datetime.now()}] Epoch {epoch + 1}, Batch {(step + 1)}/{len(data_loader)}: Loss: {loss_value:.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"[{datetime.now()}] Epoch {epoch + 1} Avg Loss: {avg_loss:.4f}")

        # Best 모델인지 확인하고 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {best_epoch} with loss {best_loss:.4f} to {save_path}")

    print(f"Training completed. Best model was at epoch {best_epoch} with loss {best_loss:.4f}")

if __name__ == '__main__':
    pretrain_mae()