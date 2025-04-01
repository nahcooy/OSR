import torch
import torch.optim as optim
from tqdm import tqdm
from model import Classifier, Copycat
from data_loader import load_data
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
def evaluate_metrics(true_labels, predictions, all_labels, known_labels, unknown_labels):
    # Normal 계산
    normal_true = (true_labels[:, all_labels.index("No Finding")] == 1) & (true_labels.sum(axis=1) == 1)
    normal_pred = (predictions[:, all_labels.index("No Finding")] == 1) & (predictions.sum(axis=1) == 1)
    
    # Known 계산
    known_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]).issubset(known_labels) and l.sum() > 0 else 0 for l in true_labels])
    known_pred = ((predictions != 0) & (predictions != -1)).any(axis=1).astype(int)
    
    # Unknown 계산
    unknown_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) - set(known_labels) and l.sum() > 0 else 0 for l in true_labels])
    unknown_pred = (predictions == -1).all(axis=1).astype(int)

    # Normal 메트릭
    if len(np.unique(normal_true)) > 1:
        auroc_normal = roc_auc_score(normal_true, normal_pred)
        f1_normal = f1_score(normal_true, normal_pred)
        precision_normal = precision_score(normal_true, normal_pred, zero_division=0)
        recall_normal = recall_score(normal_true, normal_pred, zero_division=0)
        print(f'Normal - AUROC: {auroc_normal:.4f}, F1: {f1_normal:.4f}, Precision: {precision_normal:.4f}, Recall: {recall_normal:.4f}')

    # Known 메트릭
    if len(np.unique(known_true)) > 1:
        auroc_known = roc_auc_score(known_true, known_pred)
        f1_known = f1_score(known_true, known_pred)
        precision_known = precision_score(known_true, known_pred, zero_division=0)
        recall_known = recall_score(known_true, known_pred, zero_division=0)
        print(f'Known - AUROC: {auroc_known:.4f}, F1: {f1_known:.4f}, Precision: {precision_known:.4f}, Recall: {recall_known:.4f}')

    # Unknown 메트릭
    if len(np.unique(unknown_true)) > 1:
        auroc_unknown = roc_auc_score(unknown_true, unknown_pred)
        f1_unknown = f1_score(unknown_true, unknown_pred)
        precision_unknown = precision_score(unknown_true, unknown_pred, zero_division=0)
        recall_unknown = recall_score(unknown_true, unknown_pred, zero_division=0)
        print(f'Unknown - AUROC: {auroc_unknown:.4f}, F1: {f1_unknown:.4f}, Precision: {precision_unknown:.4f}, Recall: {recall_unknown:.4f}')

def main(epoch=10, tau_nor=0.5, tau_unk=1.5, lambda_kd=1.0, lambda_ent=0.1):
    # 데이터 경로
    train_csv = '/dataset/nahcooy/CXR8/images/train.csv'
    train_dir = '/dataset/nahcooy/CXR8/images/train'
    val_csv = '/dataset/nahcooy/CXR8/images/val.csv'
    val_dir = '/dataset/nahcooy/CXR8/images/val'
    known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                    "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding",
                    "Pneumonia", "Edema", "Emphysema", "Fibrosis"]
    unknown_labels = ["Nodule"]

    # 데이터 로드
    try:
        train_loader, val_loader = load_data(train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels)
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("DataLoader is empty!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    all_labels = known_labels + unknown_labels

    # 모델 초기화
    classifier = Classifier(classid_list=all_labels).cuda()
    copycat = Copycat(num_classes=len(all_labels)).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.0001)
    optimizer_copycat = optim.Adam(copycat.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    best_epoch = 0

    # 학습 루프
    for current_epoch in range(epoch):
        classifier.train()
        copycat.train()
        classifier_loss_total = 0.0
        copycat_loss_total = 0.0
        val_loss_total = 0.0
        predictions, true_labels = [], []

        # Train Loop
        with tqdm(train_loader, desc=f'Epoch {current_epoch+1}/{epoch}', unit='batch') as t:
            for images, labels in t:
                images, labels = images.cuda(), labels.cuda()

                # 1. Copycat 학습
                optimizer_copycat.zero_grad()
                copycat_logits, copycat_features = copycat(images)
                l_nll = criterion(copycat_logits, labels)
                l_kd = 0
                with torch.no_grad():
                    _, classifier_features = classifier(images)
                for i in [1, 2, 3, 4]:  # conv2, conv3, conv4, conv5
                    l_kd += F.l1_loss(copycat_features[i], classifier_features[i])
                l_kd /= 4  # 평균
                copycat_loss = l_nll + lambda_kd * l_kd
                copycat_loss.backward()
                optimizer_copycat.step()

                # 2. Classifier 학습
                optimizer_classifier.zero_grad()
                outputs, _ = classifier(images, copycat_features)  # conv2, conv4, conv6 대체
                logits = outputs['base']
                logits_conv2 = outputs['conv2']
                logits_conv4 = outputs['conv4']
                logits_conv6 = outputs['conv6']

                l_close = criterion(logits, labels)
                # Entropy Loss
                probs_conv2 = torch.sigmoid(logits_conv2)
                probs_conv4 = torch.sigmoid(logits_conv4)
                probs_conv6 = torch.sigmoid(logits_conv6)
                alpha_easy = 0.5  # Easy 난이도
                alpha_hard = 1.0  # Hard 난이도
                y_tilde_easy = (1 - alpha_easy) * labels + alpha_easy / len(all_labels)
                y_tilde_hard = (1 - alpha_hard) * labels + alpha_hard / len(all_labels)
                l_ent_easy2 = -torch.mean(torch.sum(y_tilde_easy * torch.log(probs_conv2 + 1e-10), dim=1))
                l_ent_easy4 = -torch.mean(torch.sum(y_tilde_easy * torch.log(probs_conv4 + 1e-10), dim=1))
                l_ent_hard6 = -torch.mean(torch.sum(y_tilde_hard * torch.log(probs_conv6 + 1e-10), dim=1))
                classifier_loss = l_close + lambda_ent * (l_ent_easy2 + l_ent_easy4 + l_ent_hard6)
                classifier_loss.backward()
                optimizer_classifier.step()

                classifier_loss_total += classifier_loss.item()
                copycat_loss_total += copycat_loss.item()
                t.set_postfix(clf_loss=classifier_loss.item(), cpy_loss=copycat_loss.item())

        # Validation Loop
        classifier.eval()
        copycat.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                _, copycat_features = copycat(images)
                outputs, _ = classifier(images, copycat_features)
                logits = outputs['base']

                val_loss = criterion(logits, labels)
                val_loss_total += val_loss.item()

                preds, _ = classifier.predict(logits, tau_nor, tau_unk)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 메트릭 출력
        val_loss_avg = val_loss_total / len(val_loader)
        print(f'Epoch {current_epoch+1}/{epoch}, Classifier Loss: {classifier_loss_total/len(train_loader):.4f}, '
              f'Copycat Loss: {copycat_loss_total/len(train_loader):.4f}, Val Loss: {val_loss_avg:.4f}')
        evaluate_metrics(np.array(true_labels), np.array(predictions), all_labels, known_labels, unknown_labels)

        # Best 모델 저장
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = current_epoch
            torch.save(classifier.state_dict(), f'best_classifier_epoch_{best_epoch+1}.pth')
            torch.save(copycat.state_dict(), f'best_copycat_epoch_{best_epoch+1}.pth')

        # Learning Rate Decay
        if (current_epoch + 1) % 60 == 0:
            for param_group in optimizer_classifier.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_copycat.param_groups:
                param_group['lr'] *= 0.1

    print(f"Best model saved at epoch {best_epoch+1}")

if __name__ == "__main__":
    main(epoch=10, tau_nor=0.5, tau_unk=1.5, lambda_kd=1.0, lambda_ent=0.1)