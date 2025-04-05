import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
from .utils import TensorboardWriter, MetricTracker, accuracy, write_json
from .model import VisionTransformer, OODTransformer, load_checkpoint  # OODTransformer 추가
from .dataset import getHAM10000Dataset
from sklearn.metrics import roc_auc_score, f1_score
import os

# 클래스 평균 계산 함수
def run_model(model, loader, device):
    model.eval()
    out_list = []
    tgt_list = []
    with torch.no_grad():
        for images, target in tqdm(loader, desc="Computing class means"):
            images = images.to(device)
            output = model(images)  # ViT의 임베딩 출력
            out_list.append(output.data)
            tgt_list.append(target)
    return torch.cat(out_list), torch.cat(tgt_list)

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, config, device):
    print(f"[{datetime.now()}] Starting Epoch {epoch}")
    metrics.reset()
    running_loss = 0.0
    for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Training")):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data)  # ViT로 로짓 출력
        loss = criterion(batch_pred, batch_target)  # CrossEntropyLoss
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if metrics.writer is not None:
            metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        running_loss += loss.item()
        metrics.update('loss', loss.item())
        if batch_idx % 50 == 49:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"[{datetime.now()}] Epoch {epoch}, Batch {batch_idx + 1}/{len(data_loader)}: Running Train Loss: {avg_loss:.4f}")
        if batch_idx % 100 == 10:
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
            metrics.update('acc1', acc1.item())
            metrics.update('acc5', acc5.item())
    return metrics.result()

def valid_epoch(epoch, vit_model, ood_model, val_known_loader, full_loader, criterion, metrics, config, classes_mean, device):
    vit_model.eval()
    ood_model.eval()
    print(f"[{datetime.now()}] Starting Validation for Epoch {epoch}")

    # 1. Known 10% 평가 (ViT 기반)
    print(f"[{datetime.now()}] Evaluating Known 10%")
    metrics.reset()
    losses, acc1s, acc5s, all_preds, all_targets = [], [], [], [], []
    with torch.no_grad():
        for batch_data, batch_target in tqdm(val_known_loader, desc="Known 10% Validation"):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = vit_model(batch_data)  # ViT로 로짓 출력
            loss = criterion(batch_pred, batch_target)
            losses.append(loss.item())
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())
            all_preds.append(torch.softmax(batch_pred, dim=1).cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    known_loss = np.mean(losses)
    known_acc1 = np.mean(acc1s)
    known_acc5 = np.mean(acc5s)
    all_preds = np.concatenate(all_preds)
    all_targets = np.array(all_targets)
    known_auroc = roc_auc_score(np.eye(6)[all_targets], all_preds[:, :6], multi_class='ovr', average='macro')
    known_f1 = f1_score(all_targets, np.argmax(all_preds[:, :6], axis=1), average='macro')
    metrics.update('known_loss', known_loss)
    metrics.update('known_acc1', known_acc1)
    metrics.update('known_acc5', known_acc5)
    metrics.update('known_auroc', known_auroc)
    metrics.update('known_f1', known_f1)

    # 2. Binary (Known vs Unknown) 평가 (OODTransformer 기반)
    print(f"[{datetime.now()}] Evaluating Binary (Unknown vs Known)")
    losses, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch_data, batch_target in tqdm(full_loader, desc="Full Set Validation"):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = ood_model(batch_data)  # OODTransformer로 임베딩 출력
            # ViT로 계산한 손실 (참고용)
            vit_pred = vit_model(batch_data)
            loss = criterion(vit_pred, batch_target)
            losses.append(loss.item())
            all_preds.append(batch_pred.cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    binary_loss = np.mean(losses)
    all_preds = np.concatenate(all_preds)
    all_targets = np.array(all_targets)
    binary_targets = (all_targets == 6).astype(int)  # Known: 0~5 (0), Unknown: 6 (1)
    # 클래스 평균과의 거리 계산
    distances = np.array([np.min([np.linalg.norm(pred - mean) for mean in classes_mean.cpu().numpy()]) for pred in all_preds])
    binary_auroc = roc_auc_score(binary_targets, distances)
    binary_f1 = f1_score(binary_targets, (distances > np.median(distances)).astype(int), average='binary')
    metrics.update('binary_loss', binary_loss)
    metrics.update('binary_auroc', binary_auroc)
    metrics.update('binary_f1', binary_f1)

    # 3. Multiclass 평가 (ViT 기반)
    print(f"[{datetime.now()}] Evaluating Multiclass")
    losses, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch_data, batch_target in tqdm(full_loader, desc="Full Set Validation"):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = vit_model(batch_data)  # ViT로 로짓 출력
            loss = criterion(batch_pred, batch_target)
            losses.append(loss.item())
            all_preds.append(torch.softmax(batch_pred, dim=1).cpu().numpy())
            all_targets.append(batch_target.cpu().numpy())
    multi_loss = np.mean(losses)
    all_preds = np.concatenate(all_preds)
    all_targets = np.array(all_targets)
    multi_auroc = roc_auc_score(np.eye(7)[all_targets], all_preds, multi_class='ovr', average='macro')
    multi_f1 = f1_score(all_targets, np.argmax(all_preds, axis=1), average='macro')
    metrics.update('multi_loss', multi_loss)
    metrics.update('multi_auroc', multi_auroc)
    metrics.update('multi_f1', multi_f1)

    # 결과 출력
    print(f"[{datetime.now()}] Epoch {epoch}:")
    print(f"[{datetime.now()}]   Train Loss: {metrics.result()['loss']:.4f}")
    print(f"[{datetime.now()}]   Known 10% - Loss: {known_loss:.4f}, Acc1: {known_acc1:.4f}, Acc5: {known_acc5:.4f}, AUROC: {known_auroc:.4f}, F1: {known_f1:.4f}")
    print(f"[{datetime.now()}]   Binary (Unknown vs Known) - Loss: {binary_loss:.4f}, AUROC: {binary_auroc:.4f}, F1: {binary_f1:.4f}")
    print(f"[{datetime.now()}]   Multiclass - Loss: {multi_loss:.4f}, AUROC: {multi_auroc:.4f}, F1: {multi_f1:.4f}")
    return metrics.result()

def save_best_models(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, metrics):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
    }
    known_auroc = metrics.result().get('known_auroc', 0.0)
    if known_auroc > getattr(save_best_models, 'best_known_auroc', 0.0):
        save_best_models.best_known_auroc = known_auroc
        torch.save(state, str(save_dir + 'best_known_set.pt'))
        print(f"[{datetime.now()}]   Saved: best_known_set.pt (AUROC: {known_auroc:.4f})")
    
    binary_auroc = metrics.result().get('binary_auroc', 0.0)
    if binary_auroc > getattr(save_best_models, 'best_binary_auroc', 0.0):
        save_best_models.best_binary_auroc = binary_auroc
        torch.save(state, str(save_dir + 'best_unknown_detection.pt'))
        print(f"[{datetime.now()}]   Saved: best_unknown_detection.pt (AUROC: {binary_auroc:.4f})")
    
    multi_auroc = metrics.result().get('multi_auroc', 0.0)
    if multi_auroc > getattr(save_best_models, 'best_multi_auroc', 0.0):
        save_best_models.best_multi_auroc = multi_auroc
        torch.save(state, str(save_dir + 'best_multiclass.pt'))
        print(f"[{datetime.now()}]   Saved: best_multiclass.pt (AUROC: {multi_auroc:.4f})")

def train(config, device, device_ids):
    step = 0
    print(f"[{datetime.now()}] Step {step}: Starting Tensorboard setup")
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)
    metric_names = ['loss', 'acc1', 'acc5', 'known_loss', 'known_acc1', 'known_acc5', 'known_auroc', 'known_f1', 
                    'binary_loss', 'binary_auroc', 'binary_f1', 'multi_loss', 'multi_auroc', 'multi_f1']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    step += 1
    print(f"[{datetime.now()}] Step {step}: Starting model creation")
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
    )
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
    )

    if config.checkpoint_path:
        step += 1
        print(f"[{datetime.now()}] Step {step}: Loading pretrained weights from {config.checkpoint_path}")
        state_dict = load_checkpoint(config.checkpoint_path, new_img=config.image_size, emb_dim=config.emb_dim, layers=config.num_layers, patch=config.patch_size)
        if config.num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print("Re-initializing fc layer")
            vit_model.load_state_dict(state_dict, strict=False)
            ood_model.load_state_dict(state_dict, strict=False)  # OODTransformer에도 동일하게 로드
        else:
            vit_model.load_state_dict(state_dict, strict=False)
            ood_model.load_state_dict(state_dict, strict=False)
        print("Missing keys from checkpoint:", vit_model.load_state_dict(state_dict, strict=False).missing_keys)
        print("Unexpected keys in network:", vit_model.load_state_dict(state_dict, strict=False).unexpected_keys)

    step += 1
    print(f"[{datetime.now()}] Step {step}: Moving model to {device}")
    vit_model = vit_model.to(device)
    ood_model = ood_model.to(device)
    if len(device_ids) > 1:
        vit_model = nn.DataParallel(vit_model, device_ids=device_ids)
        ood_model = nn.DataParallel(ood_model, device_ids=device_ids)

    config.model = 'vit'
    step += 1
    print(f"[{datetime.now()}] Step {step}: Setting up dataset for HAM10000")
    total_classes = 7
    import random
    random.seed(config.random_seed)
    close_classes = [0, 1, 2, 3, 4, 5]

    step += 1
    print(f"[{datetime.now()}] Step {step}: Loading datasets")
    train_dataset = getHAM10000Dataset(image_size=config.image_size, split='train', data_path=config.data_dir, random_seed=config.random_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_known_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_known', data_path=config.data_dir, random_seed=config.random_seed)
    val_unknown_dataset = getHAM10000Dataset(image_size=config.image_size, split='val_unknown', data_path=config.data_dir, random_seed=config.random_seed)
    val_known_dataloader = DataLoader(val_known_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    full_valid_dataset = ConcatDataset([val_known_dataset, val_unknown_dataset])
    full_valid_dataloader = DataLoader(full_valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 클래스 평균 계산 (ViT로 학습된 임베딩 사용)
    step += 1
    print(f"[{datetime.now()}] Step {step}: Computing class means")
    train_emb, train_targets = run_model(vit_model, train_dataloader, device)  # ViT로 임베딩 추출
    in_classes = torch.unique(train_targets)  # Known 클래스 (0~5)
    class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
    classes_feats = [train_emb[idx] for idx in class_idx]
    classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats], dim=0)

    step += 1
    print(f"[{datetime.now()}] Step {step}: Creating criterion and optimizer")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(params=vit_model.parameters(), lr=config.lr, weight_decay=config.wd)  # ViT만 학습
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=config.lr, pct_start=config.warmup_steps / config.train_steps, total_steps=config.train_steps)

    step += 1
    print(f"[{datetime.now()}] Step {step}: Starting training")
    config.epochs = config.train_steps // len(train_dataloader)
    print(f"Length of train loader: {len(train_dataloader)}, total epochs: {config.epochs}")
    for epoch in range(1, config.epochs + 1):
        for param_group in optimizer.param_groups:
            print(f"[{datetime.now()}] Learning rate at epoch {epoch} is {param_group['lr']}")
        log = {'epoch': epoch}
        vit_model.train()
        result = train_epoch(epoch, vit_model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, config, device)
        log.update(result)
        result = valid_epoch(epoch, vit_model, ood_model, val_known_dataloader, full_valid_dataloader, criterion, valid_metrics, config, classes_mean, device)
        log.update(**{'val_' + k: v for k, v in result.items()})
        save_best_models(config.checkpoint_dir, epoch, vit_model, optimizer, lr_scheduler, device_ids, valid_metrics)  # ViT만 저장

    print(f"[{datetime.now()}] Training completed")
    best_curr_acc = {'best_known_auroc': getattr(save_best_models, 'best_known_auroc', 0.0),
                     'best_binary_auroc': getattr(save_best_models, 'best_binary_auroc', 0.0),
                     'best_multi_auroc': getattr(save_best_models, 'best_multi_auroc', 0.0),
                     'curr_epoch': epoch}
    write_json(best_curr_acc, os.path.join(config.checkpoint_dir, 'acc.json'))

if __name__ == "__main__":
    class Config:
        summary_dir = "experiments/tb"
        tensorboard = False
        image_size = 224
        patch_size = 16
        emb_dim = 768
        mlp_dim = 3072
        num_heads = 12
        num_layers = 12
        num_classes = 7  # HAM10000: 6 closed + 1 unknown
        attn_dropout_rate = 0.0
        dropout_rate = 0.1
        checkpoint_path = None
        dataset = "HAM10000"
        data_dir = "/dataset/nahcooy/HAM"
        batch_size = 32
        num_workers = 4
        label_smoothing = 0.1
        opt = "AdamW"
        lr = 1e-3
        wd = 0.01
        train_steps = 10000
        warmup_steps = 500
        random_seed = 42
        checkpoint_dir = "/nahcooy/OSR/HAM/osr_vit/checkpoints/0404"
        save_freq = 100

    config = Config()
    device = torch.device("cuda:1")
    device_ids = [1]
    train(config, device, device_ids)