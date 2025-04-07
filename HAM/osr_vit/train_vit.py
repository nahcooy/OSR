import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from utils import TensorboardWriter, MetricTracker, accuracy, write_json
from model import VisionTransformer, load_checkpoint
from dataset import getHAM10000Dataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import os

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, device):
    print(f"[{datetime.now()}] Starting Epoch {epoch}")
    model.train()
    running_loss = 0.0
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        running_loss += loss.item()
        if batch_idx % 50 == 49:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"[{datetime.now()}] Epoch {epoch}, Batch {batch_idx + 1}/{len(data_loader)}: Loss: {avg_loss:.4f}")
    return avg_loss

def valid_epoch(epoch, model, data_loader, criterion, device):
    print(f"[{datetime.now()}] Starting Validation for Epoch {epoch}")
    model.eval()
    losses, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch_data, batch_target in data_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            losses.append(loss.item())
            all_preds.append(torch.softmax(batch_pred, dim=1).cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # 메트릭 계산 (Known 클래스 6개만 평가)
    auroc = roc_auc_score(np.eye(6)[all_targets], all_preds[:, :6], multi_class='ovr', average='macro')
    preds = np.argmax(all_preds[:, :6], axis=1)
    f1 = f1_score(all_targets, preds, average='macro')
    recall = recall_score(all_targets, preds, average='macro')
    precision = precision_score(all_targets, preds, average='macro')
    cm = confusion_matrix(all_targets, preds)
    
    print(f"[{datetime.now()}] Epoch {epoch} Validation:")
    print(f"Loss: {avg_loss:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return {'loss': avg_loss, 'auroc': auroc, 'f1': f1, 'recall': recall, 'precision': precision}

def train_vit(config, device, device_ids):
    print(f"[{datetime.now()}] train vit 시작")

    print(f"[{datetime.now()}] 모델 초기화 시작")

    # 모델 초기화
    vit_model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate,
    ).to(device)
    
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path, new_img=config.image_size, emb_dim=config.emb_dim, layers=config.num_layers, patch=config.patch_size)
        vit_model.load_state_dict(state_dict, strict=False)

    if len(device_ids) > 1:
        vit_model = nn.DataParallel(vit_model, device_ids=device_ids)

    print(f"[{datetime.now()}] 데이터셋 로드 시작")
    # 데이터셋 로드
    train_dataset = getHAM10000Dataset(image_size=config.image_size, split='train', data_path=config.data_dir, random_seed=config.random_seed)
    val_known_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_known', data_path=config.data_dir, random_seed=config.random_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_known_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print(f"[{datetime.now()}] 학습 설정 시작")

    # 학습 설정
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(params=vit_model.parameters(), lr=config.lr, weight_decay=config.wd)
    total_steps = config.epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, pct_start=config.warmup_steps/total_steps, total_steps=total_steps)

    print(f"[{datetime.now()}] stage 1 학습 시작")
    # 학습 루프
    best_auroc = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(epoch, vit_model, train_dataloader, criterion, optimizer, lr_scheduler, device)
        val_metrics = valid_epoch(epoch, vit_model, val_dataloader, criterion, device)
        
        # 최고 AUROC 모델 저장
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': vit_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'auroc': best_auroc
            }, os.path.join(config.checkpoint_dir, 'vit_best.pt'))
            print(f"[{datetime.now()}] Saved best ViT model with AUROC: {best_auroc:.4f}")
    
    return vit_model

if __name__ == "__main__":
    class Config:
        summary_dir = "experiments/tb"
        tensorboard = True
        image_size = 224
        patch_size = 16
        emb_dim = 768
        mlp_dim = 3072
        num_heads = 12
        num_layers = 12
        num_classes = 7
        attn_dropout_rate = 0.0
        dropout_rate = 0.1
        checkpoint_path = None
        data_dir = "/dataset/nahcooy/HAM"
        batch_size = 32
        num_workers = 4
        label_smoothing = 0.1
        lr = 1e-3
        wd = 0.01
        epochs = 200
        warmup_steps = 500
        random_seed = 42
        checkpoint_dir = "/nahcooy/OSR/HAM/osr_vit/checkpoints/0404"

    config = Config()
    device = torch.device("cuda:1")
    device_ids = [1]
    vit_model = train_vit(config, device, device_ids)