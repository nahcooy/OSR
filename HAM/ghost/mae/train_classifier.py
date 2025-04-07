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

# 기본 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = {
    'image_size': 224,
    'random_seed': 42,
    'batch_size': 32,
    'num_workers': 4,
    'output_dir': './output_finetune',
}
os.makedirs(args['output_dir'], exist_ok=True)

def finetune_mae():
    train_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='train', **args)
    val_dataset = getHAM10000Dataset(data_path='/dataset/nahcooy/HAM', split='val_known', **args)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                              num_workers=args['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False,
                            num_workers=args['num_workers'], drop_last=False)

    model = mae_vit_huge_patch14()
    checkpoint = torch.load('pretrain.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    for k in list(checkpoint_model.keys()):
        if 'decoder' in k:
            del checkpoint_model[k]
    model.load_state_dict(checkpoint_model, strict=False)

    embed_dim = model.cls_token.shape[-1]
    model.head = nn.Linear(embed_dim, 6)
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    loss_scaler = NativeScaler()
    criterion = nn.CrossEntropyLoss()

    epochs = 50
    print_freq = 50
    print(f"Starting fine-tuning with train: {len(train_dataset)}, val: {len(val_dataset)} samples")

    for epoch in range(epochs):
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

        # Validation
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
        all_preds, all_targets = np.array(all_preds), np.array(all_targets)
        all_probs = np.array(all_probs)

        auroc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
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

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

if __name__ == '__main__':
    finetune_mae()
