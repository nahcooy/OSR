import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from datetime import datetime
from util.models_mae import mae_vit_huge_patch14
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import numpy as np
from dataset import getHAM10000Dataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 기본 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
args = {
    'image_size': 224,
    'random_seed': 42,
    'batch_size': 8,
    'num_workers': 8,
    'output_dir': '/home/nikhil/nahcooy/mae/checkpoint',
}

# Classification용 모델 클래스 정의
class MAEForClassification(nn.Module):
    def __init__(self, mae_model, num_classes=6):
        super(MAEForClassification, self).__init__()
        self.mae = mae_model
        self.head = nn.Linear(1280, num_classes)  # ViT-Huge의 embed_dim은 1280
        nn.init.trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        x = self.mae.forward_encoder(x, mask_ratio=0)[0]  # mask_ratio=0으로 인코딩만 수행
        cls_token = x[:, 0]  # [CLS] 토큰
        return self.head(cls_token)  # logits 반환

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else torch.ones(6).to(device)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# Fine-tuning 함수
def finetune_mae(resume='', start_epoch=None):
    train_dataset = getHAM10000Dataset(data_path='/home/nikhil/nahcooy/HAM', split='train', **args)
    val_dataset = getHAM10000Dataset(data_path='/home/nikhil/nahcooy/HAM', split='val_known', **args)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                              num_workers=args['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False,
                            num_workers=args['num_workers'], drop_last=False)

    # Pretrain 모델 로드
    mae_model = mae_vit_huge_patch14()
    checkpoint_path = os.path.join(args['output_dir'], 'best_mae_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint  # 'model' 키가 없는 경우 가정
    # Decoder 관련 파라미터 제거
    for k in list(checkpoint_model.keys()):
        if 'decoder' in k:
            del checkpoint_model[k]
    mae_model.load_state_dict(checkpoint_model, strict=False)

    # Classification 모델로 변환
    model = MAEForClassification(mae_model, num_classes=6)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    loss_scaler = NativeScaler()
    # Focal Loss 적용 (클래스 가중치 추가)
    alpha = torch.tensor([0.25, 1.0, 1.0, 1.5, 2.0, 2.0]).to(device)  # 소수 클래스에 높은 가중치
    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    # 체크포인트 로드 (resume)
    if resume and os.path.exists(resume):
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))  # 옵티마이저 상태 로드 (선택적)
        loaded_epoch = checkpoint.get('epoch', 0)  # 저장된 epoch 가져오기, 없으면 0
        print(f"Loaded checkpoint from epoch {loaded_epoch}")
    else:
        loaded_epoch = 0

    # 시작 epoch 설정
    if start_epoch is not None:
        initial_epoch = start_epoch
        print(f"Starting from user-specified epoch: {start_epoch}")
    else:
        initial_epoch = loaded_epoch
        print(f"Starting from loaded epoch: {loaded_epoch}")

    epochs = 200
    print_freq = 50
    best_val_loss = float('inf')
    best_epoch = 0
    save_path = os.path.join(args['output_dir'], 'best_finetune_checkpoint_focal.pth')

    os.makedirs(args['output_dir'], exist_ok=True)
    print(f"Starting fine-tuning with train: {len(train_dataset)}, val: {len(val_dataset)} samples")
    for epoch in range(initial_epoch, epochs):
        # 학습
        model.train()
        total_loss = 0.0
        for step, (samples, targets) in enumerate(train_loader):
            samples, targets = samples.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss_value = loss.item()
            total_loss += loss_value

            optimizer.zero_grad()
            loss_scaler(loss, optimizer, parameters=model.parameters())

            if (step + 1) % print_freq == 0 or (step + 1) == len(train_loader):
                print(f"[{datetime.now()}] Epoch {epoch + 1}, Batch {(step + 1)}/{len(train_loader)}: Loss: {loss_value:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"[{datetime.now()}] Epoch {epoch + 1} Avg Train Loss: {avg_train_loss:.4f}")

        # 검증
        model.eval()
        val_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []
        top1_correct, top5_correct, total = 0, 0, 0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples, targets = samples.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                outputs_topk = torch.topk(probs, k=5, dim=1)
                top1_preds = outputs_topk.indices[:, 0]
                top5_preds = outputs_topk.indices

                targets_cpu = targets.cpu()
                top1_correct += (top1_preds.cpu() == targets_cpu).sum().item()
                top5_correct += sum([targets_cpu[i] in top5_preds[i] for i in range(len(targets_cpu))])
                total += targets.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets_cpu.numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # 원-핫 인코딩으로 변환
        all_targets_onehot = np.eye(6)[all_targets]  # (1901,) -> (1901, 6)

        # AUROC 계산
        auroc = roc_auc_score(all_targets_onehot, all_probs, multi_class='ovr', average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        precision = precision_score(all_targets, all_preds, average='macro')
        top1_acc = top1_correct / total
        top5_acc = top5_correct / total
        cm = confusion_matrix(all_targets, all_preds)

        print(f"[{datetime.now()}] Epoch {epoch + 1} Validation:")
        print(f"Loss: {avg_val_loss:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Top-1 Acc: {top1_acc:.4f}, Top-5 Acc: {top5_acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # Best 모델 저장 (Validation Loss 기준)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),  # 옵티마이저 상태 저장
                'epoch': epoch + 1,  # 현재 epoch 저장
                'val_loss': best_val_loss
            }, save_path)
            print(f"New best model saved at epoch {best_epoch} with Val Loss {best_val_loss:.4f} to {save_path}")

    print(f"Fine-tuning completed. Best model was at epoch {best_epoch} with Val Loss {best_val_loss:.4f}")

if __name__ == '__main__':
    # 예시: 특정 체크포인트에서 재개하고 시작 epoch을 설정
    finetune_mae(resume='/home/nikhil/nahcooy/mae/checkpoint/best_finetune_checkpoint_focal.pth', start_epoch=30)
    # 또는 체크포인트 없이 처음부터 시작
    # finetune_mae()