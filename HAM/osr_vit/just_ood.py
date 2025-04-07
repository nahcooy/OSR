import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from datetime import datetime
from utils import TensorboardWriter, MetricTracker, accuracy, write_json
from model import VisionTransformer, OODTransformer, load_checkpoint  # 코드 A의 모델 사용
from dataset import getHAM10000Dataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import os

def run_model(model, loader, device):
    model.eval()
    out_list, tgt_list = [], []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            output = model(images)
            out_list.append(output.data)
            tgt_list.append(target)
    return torch.cat(out_list), torch.cat(tgt_list)

def train_ood_epoch(epoch, vit_model, ood_model, data_loader, criterion, optimizer, classes_mean, device):
    print(f"[{datetime.now()}] Starting OOD Epoch {epoch}")
    ood_model.train()
    vit_model.eval()
    running_loss = 0.0
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        optimizer.zero_grad()
        emb = ood_model(batch_data)
        loss = 0
        for i in range(batch_data.size(0)):
            class_center = classes_mean[batch_target[i]]
            loss += criterion(emb[i], class_center)
        loss = loss / batch_data.size(0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 50 == 49:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"[{datetime.now()}] Epoch {epoch}, Batch {batch_idx + 1}/{len(data_loader)}: Loss: {avg_loss:.4f}")
    avg_loss = running_loss / len(data_loader)
    return avg_loss

def valid_ood_epoch(epoch, vit_model, ood_model, val_loader, classes_mean, device, criterion):
    print(f"[{datetime.now()}] Starting OOD Validation for Epoch {epoch} (Known Only)")
    vit_model.eval()
    ood_model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_data, batch_target in val_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = ood_model(batch_data)
            loss = 0
            for i in range(batch_data.size(0)):
                class_center = classes_mean[batch_target[i]]
                loss += criterion(batch_pred[i], class_center)
            loss = loss / batch_data.size(0)
            total_loss += loss.item() * batch_data.size(0)
            num_samples += batch_data.size(0)
            
            all_preds.append(batch_pred.cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    
    avg_val_loss = total_loss / num_samples
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    distances = np.array([np.min([np.linalg.norm(pred - mean) for mean in classes_mean.cpu().numpy()]) for pred in all_preds])
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print(f"[{datetime.now()}] Epoch {epoch} Validation (Known Only):")
    print(f"Average Distance: {avg_distance:.4f}, Std Distance: {std_distance:.4f}, Validation Loss: {avg_val_loss:.4f}")
    return {'avg_distance': avg_distance, 'std_distance': std_distance, 'avg_val_loss': avg_val_loss}

def inference_ood(config, device, device_ids, classes_mean):
    print(f"[{datetime.now()}] Starting OOD Inference on Known + Unknown Data")
    
    ood_model = OODTransformer(
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
    
    if len(device_ids) > 1:
        ood_model = nn.DataParallel(ood_model, device_ids=device_ids)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, 'ood_best.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ood_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[{datetime.now()}] Loaded OOD model from {checkpoint_path} with Avg Distance: {checkpoint.get('avg_distance', 'N/A')}")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    
    val_known_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_known', data_path=config.data_dir, random_seed=config.random_seed)
    val_unknown_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_unknown', data_path=config.data_dir, random_seed=config.random_seed)
    full_valid_dataset = ConcatDataset([val_known_dataset, val_unknown_dataset])
    full_valid_dataloader = DataLoader(full_valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    ood_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_data, batch_target in full_valid_dataloader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = ood_model(batch_data)
            all_preds.append(batch_pred.cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    binary_targets = (all_targets == 6).astype(int)
    
    distances = np.array([np.min([np.linalg.norm(pred - mean) for mean in classes_mean.cpu().numpy()]) for pred in all_preds])
    binary_preds = (distances > np.median(distances)).astype(int)
    
    auroc = roc_auc_score(binary_targets, distances)
    f1 = f1_score(binary_targets, binary_preds, average='binary')
    recall = recall_score(binary_targets, binary_preds, average='binary')
    precision = precision_score(binary_targets, binary_preds, average='binary')
    cm = confusion_matrix(binary_targets, binary_preds)
    
    print(f"[{datetime.now()}] OOD Inference Results (Known + Unknown):")
    print(f"AUROC: {auroc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return {'auroc': auroc, 'f1': f1, 'recall': recall, 'precision': precision}

def train_ood(config, device, device_ids):
    print(f"[{datetime.now()}] train ood 시작")

    print(f"[{datetime.now()}] vit 모델 초기화 시작")
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
    
    vit_checkpoint_path = os.path.join(config.checkpoint_dir, 'vit_best.pt')
    if os.path.exists(vit_checkpoint_path):
        checkpoint = torch.load(vit_checkpoint_path, map_location=device, weights_only=False)
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[{datetime.now()}] Loaded ViT model from {vit_checkpoint_path} with AUROC: {checkpoint['auroc']:.4f}")
    else:
        raise FileNotFoundError(f"Checkpoint {vit_checkpoint_path} not found.")

    for param in vit_model.parameters():
        param.requires_grad = False

    print(f"[{datetime.now()}] ood 모델 초기화 시작")
    ood_model = OODTransformer(
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
    
    if len(device_ids) > 1:
        vit_model = nn.DataParallel(vit_model, device_ids=device_ids)
        ood_model = nn.DataParallel(ood_model, device_ids=device_ids)

    print(f"[{datetime.now()}] 데이터셋 로드 시작")
    train_dataset = getHAM10000Dataset(image_size=config.image_size, split='train', data_path=config.data_dir, random_seed=config.random_seed)
    val_known_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_known', data_path=config.data_dir, random_seed=config.random_seed)
    val_unknown_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_unknown', data_path=config.data_dir, random_seed=config.random_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_known_dataloader = DataLoader(val_known_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print(f"[{datetime.now()}] 클래스 평균 계산 시작")
    train_emb, train_targets = run_model(ood_model, train_dataloader, device)
    in_classes = torch.unique(train_targets)
    class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
    classes_feats = [train_emb[idx] for idx in class_idx]
    classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats], dim=0)
    print(f"[{datetime.now()}] classes_mean shape: {classes_mean.shape}")

    ood_checkpoint_path = os.path.join(config.checkpoint_dir, 'ood_best.pt')
    if os.path.exists(ood_checkpoint_path):
        print(f"[{datetime.now()}] OOD checkpoint found at {ood_checkpoint_path}. Skipping training.")
        checkpoint = torch.load(ood_checkpoint_path, map_location=device, weights_only=False)
        ood_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[{datetime.now()}] Loaded OOD model from {ood_checkpoint_path} with Avg Distance: {checkpoint.get('avg_distance', 'N/A')}")
    else:
        print(f"[{datetime.now()}] No OOD checkpoint found. Starting Stage 2 training.")
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.AdamW(params=ood_model.parameters(), lr=config.lr, weight_decay=config.wd)

        print(f"[{datetime.now()}] stage 2 학습 시작")
        best_avg_distance = float('inf')
        for epoch in range(1, config.epochs + 1):
            train_loss = train_ood_epoch(epoch, vit_model, ood_model, train_dataloader, criterion, optimizer, classes_mean, device)
            val_metrics = valid_ood_epoch(epoch, vit_model, ood_model, val_known_dataloader, classes_mean, device, criterion)
            
            print(f"[{datetime.now()}] Epoch {epoch} Training Loss: {train_loss:.4f}")
            
            if val_metrics['avg_distance'] < best_avg_distance:
                best_avg_distance = val_metrics['avg_distance']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ood_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_distance': best_avg_distance
                }, ood_checkpoint_path)
                print(f"[{datetime.now()}] Saved best OOD model with Avg Distance: {best_avg_distance:.4f}")
    
    return classes_mean

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
        checkpoint_dir = "/nahcooy/OSR/HAM/osr_vit/checkpoints/0406"

    config = Config()
    device = torch.device("cuda:0")
    device_ids = [0]
    
    print(f"[{datetime.now()}] Starting from OOD training/inference")
    classes_mean = train_ood(config, device, device_ids)
    
    print(f"[{datetime.now()}] Starting OOD Inference")
    inference_ood(config, device, device_ids, classes_mean)