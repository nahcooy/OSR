import torch
import torch.optim as optim
from tqdm import tqdm
from model import Classifier, Copycat
from data_loader import load_data
from torch import nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np


def evaluate_metrics(true_labels, predictions, all_labels, known_labels, unknown_labels):
    # Known/Unknown 분류 메트릭
    normal_true = (true_labels[:, all_labels.index("No Finding")] == 1) & (true_labels.sum(axis=1) == 1)
    normal_pred = (predictions[:, all_labels.index("No Finding")] == 1) & (predictions.sum(axis=1) == 1)

    known_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]).issubset(known_labels) and l.sum() > 0 else 0 for l in true_labels])
    unknown_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) - set(known_labels) and l.sum() > 0 else 0 for l in true_labels])

    known_pred = ((predictions != 0) & (predictions != -1)).any(axis=1).astype(int)
    unknown_pred = (predictions == -1).all(axis=1).astype(int)

    # AUROC 계산
    if len(np.unique(normal_true)) > 1:
        auroc_normal = roc_auc_score(normal_true, normal_pred)
        print(f'AUROC Normal: {auroc_normal:.4f}')
    if len(np.unique(known_true)) > 1:
        auroc_known = roc_auc_score(known_true, known_pred)
        print(f'AUROC Known: {auroc_known:.4f}')
    if len(np.unique(unknown_true)) > 1:
        auroc_unknown = roc_auc_score(unknown_true, unknown_pred)
        print(f'AUROC Unknown: {auroc_unknown:.4f}')


def main(epoch=10):
    # 학습 파라미터 설정
    train_csv = '/dataset/nahcooy/CXR8/images/train.csv'
    train_dir = '/dataset/nahcooy/CXR8/images/train'
    val_csv = '/dataset/nahcooy/CXR8/images/val.csv'
    val_dir = '/dataset/nahcooy/CXR8/images/val'

    # Known과 Unknown 라벨 정의 (train.py에서 정의)
    known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
                    "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding"]
    unknown_labels = ["Pneumonia", "Edema", "Emphysema", "Fibrosis"]

    # DataLoader 불러오기 (test_loader는 사용하지 않음)
    train_loader, val_loader, _ = load_data(train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels)

    all_labels = known_labels + unknown_labels

    # 모델 초기화
    classifier = Classifier(classid_list=all_labels).cuda()
    copycat = Copycat().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.0001)
    optimizer_copycat = optim.Adam(copycat.parameters(), lr=0.0001)

    # 하이퍼파라미터
    lambda_kd = 1.0
    lambda_ent = 0.1

    # Best 모델 저장을 위한 변수 초기화
    best_val_loss = float('inf')
    best_epoch = 0
    best_classifier = None
    best_copycat = None

    # 학습 루프
    for current_epoch in range(epoch):
        classifier.train()
        copycat.eval()
        classifier_loss_total = 0.0
        val_loss_total = 0.0
        predictions, true_labels = [], []

        # Train Loop
        with tqdm(train_loader, desc=f'Epoch {current_epoch+1}/{epoch} - Classifier', unit='batch') as t:
            for images, labels in t:
                images, labels = images.cuda(), labels.cuda()
                optimizer_classifier.zero_grad()

                # Forward pass through Copycat
                _, copycat_features = copycat(images)
                logits, _ = classifier(images, copycat_features)

                # Loss calculation
                loss = criterion(logits, labels)
                loss.backward()
                optimizer_classifier.step()

                classifier_loss_total += loss.item()
                t.set_postfix(loss=loss.item())

        # Validation Loss and Metrics
        classifier.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                logits, _ = classifier(images)

                # Calculate validation loss
                val_loss = criterion(logits, labels)
                val_loss_total += val_loss.item()

                # Collect predictions and true labels for metrics calculation
                preds = torch.sigmoid(logits).round()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Metrics Calculation
        val_loss_avg = val_loss_total / len(val_loader)
        print(f'Epoch {current_epoch+1}/{epoch}, Classifier Loss: {classifier_loss_total/len(train_loader):.4f}, Val Loss: {val_loss_avg:.4f}')

        # Evaluate metrics (AUROC, Precision, Recall)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        evaluate_metrics(true_labels, predictions, all_labels, known_labels, unknown_labels)

        # 모델 성능 개선 시 모델 저장
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = current_epoch
            best_classifier = classifier.state_dict()
            best_copycat = copycat.state_dict()

        # Learning Rate Decay
        if (current_epoch + 1) % 60 == 0:
            for param_group in optimizer_classifier.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_copycat.param_groups:
                param_group['lr'] *= 0.1

    # 모델 저장 (최고 성능 모델)
    if best_classifier is not None and best_copycat is not None:
        torch.save(best_classifier, f'best_classifier_epoch_{best_epoch+1}.pth')
        torch.save(best_copycat, f'best_copycat_epoch_{best_epoch+1}.pth')
        print(f"Best model saved at epoch {best_epoch+1}")

if __name__ == '__main__':
    main(epoch=10)  # 원하는 epoch 수를 인자로 전달
