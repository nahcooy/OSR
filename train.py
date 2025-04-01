import torch
import torch.optim as optim
from tqdm import tqdm
from model import Classifier, Copycat
from data_loader import load_data
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_metrics(true_labels, predictions, joint_energies, all_labels, known_labels, unknown_labels, tau_nor=0.5, tau_unk=1.5):
    # True 라벨 정의
    known_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]).issubset(known_labels) and l.sum() > 0 else 0 for l in true_labels])
    unknown_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) == set(unknown_labels) and l.sum() > 0 else 0 for l in true_labels])
    unknown_with_other_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) - set(known_labels) and not set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) == set(unknown_labels) and l.sum() > 0 else 0 for l in true_labels])

    # 여러 tau 값 정의
    tau_versions = [
        {"tau_nor": 0.3, "tau_unk": 1.0, "name": "Low Tau"},
        {"tau_nor": 0.5, "tau_unk": 1.5, "name": "Default Tau"},
        {"tau_nor": 0.7, "tau_unk": 2.0, "name": "High Tau"}
    ]

    for tau_version in tau_versions:
        tau_nor = tau_version["tau_nor"]
        tau_unk = tau_version["tau_unk"]
        version_name = tau_version["name"]
        print(f"\n=== Evaluating with {version_name} (tau_nor={tau_nor}, tau_unk={tau_unk}) ===")

        # 예측 정의
        known_pred = (joint_energies <= tau_unk).astype(int)
        unknown_pred = (joint_energies > tau_unk).astype(int)
        unknown_with_other_pred = (joint_energies > tau_unk).astype(int)  # 동일한 tau_unk 사용

        # 개별 메트릭 출력
        for name, true, pred in [("Known", known_true, known_pred),
                                 ("Unknown", unknown_true, unknown_pred),
                                 ("Unknown with Other", unknown_with_other_true, unknown_with_other_pred)]:
            if len(np.unique(true)) > 1:
                auroc = roc_auc_score(true, pred)
                f1 = f1_score(true, pred)
                precision = precision_score(true, pred, zero_division=0)
                recall = recall_score(true, pred, zero_division=0)
                print(f"{name} - AUROC: {auroc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            else:
                print(f"{name} - No variation in true labels, skipping metrics")

        # 전체 메트릭
        combined_true = known_true + unknown_true + unknown_with_other_true
        combined_true = (combined_true > 0).astype(int)
        combined_pred = (joint_energies > tau_nor).astype(int)
        if len(np.unique(combined_true)) > 1:
            auroc = roc_auc_score(combined_true, combined_pred)
            f1 = f1_score(combined_true, combined_pred)
            precision = precision_score(combined_true, combined_pred, zero_division=0)
            recall = recall_score(combined_true, combined_pred, zero_division=0)
            print(f"Combined - AUROC: {auroc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        else:
            print("Combined - No variation in true labels, skipping metrics")

def main(epoch=10, tau_nor=0.5, tau_unk=1.5, lambda_kd=1.0, lambda_ent=0.1):
    train_csv = '/dataset/nahcooy/CXR8/images/train.csv'
    train_dir = '/dataset/nahcooy/CXR8/images/train'
    val_csv = '/dataset/nahcooy/CXR8/images/val.csv'
    val_dir = '/dataset/nahcooy/CXR8/images/val'
    known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                    "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding",
                    "Pneumonia", "Edema", "Emphysema", "Fibrosis"]
    unknown_labels = ["Nodule"]
    all_labels = known_labels + unknown_labels

    print("1. Loading train and val data...")
    try:
        train_loader, val_loader = load_data(train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels, 
                                             batch_size=32, train_data_ratio=0.3, val_data_ratio=0.3)
        print(f"2. Train loader size: {len(train_loader.dataset)}, batches: {len(train_loader)}")
        print(f"3. Val loader size: {len(val_loader.dataset)}, batches: {len(val_loader)}")
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("DataLoader is empty!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("4. Initializing models...")
    classifier = Classifier(classid_list=known_labels).cuda()
    copycat = Copycat(num_classes=len(known_labels)).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.0001)
    optimizer_copycat = optim.Adam(copycat.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    best_epoch = 0

    for current_epoch in range(epoch):
        copycat.train()
        classifier.train()
        copycat_loss_total = 0.0
        classifier_loss_total = 0.0
        with tqdm(train_loader, desc=f'Epoch {current_epoch+1}/{epoch} - Training', unit='batch') as t:
            for batch_idx, (images, labels) in enumerate(t):
                images, labels = images.cuda(), labels.cuda()
                optimizer_copycat.zero_grad()
                copycat_logits, copycat_features = copycat(images)
                l_nll = criterion(copycat_logits, labels)
                l_kd = 0
                with torch.no_grad():
                    _, classifier_features = classifier(images)
                for i in [1, 2, 3, 4]:
                    l_kd += F.l1_loss(copycat_features[i], classifier_features[i])
                l_kd /= 4
                copycat_loss = l_nll + lambda_kd * l_kd
                copycat_loss.backward()
                optimizer_copycat.step()
                copycat_loss_total += copycat_loss.item()

                optimizer_classifier.zero_grad()
                with torch.no_grad():
                    _, copycat_features = copycat(images)
                outputs, _ = classifier(images, copycat_features)
                logits = outputs['base']
                logits_conv2 = outputs['conv2']
                logits_conv4 = outputs['conv4']
                logits_conv6 = outputs['conv6']
                l_close = criterion(logits, labels)
                probs_conv2 = torch.sigmoid(logits_conv2)
                probs_conv4 = torch.sigmoid(logits_conv4)
                probs_conv6 = torch.sigmoid(logits_conv6)
                alpha_easy = 0.5
                alpha_hard = 1.0
                y_tilde_easy = (1 - alpha_easy) * labels + alpha_easy / len(known_labels)
                y_tilde_hard = (1 - alpha_hard) * labels + alpha_hard / len(known_labels)
                l_ent_easy2 = -torch.mean(torch.sum(y_tilde_easy * torch.log(probs_conv2 + 1e-10), dim=1))
                l_ent_easy4 = -torch.mean(torch.sum(y_tilde_easy * torch.log(probs_conv4 + 1e-10), dim=1))
                l_ent_hard6 = -torch.mean(torch.sum(y_tilde_hard * torch.log(probs_conv6 + 1e-10), dim=1))
                classifier_loss = l_close + lambda_ent * (l_ent_easy2 + l_ent_easy4 + l_ent_hard6)
                classifier_loss.backward()
                optimizer_classifier.step()
                classifier_loss_total += classifier_loss.item()
                t.set_postfix(cpy_loss=copycat_loss.item(), clf_loss=classifier_loss.item())

        print(f"Epoch {current_epoch+1}/{epoch}, Copycat Loss: {copycat_loss_total/len(train_loader):.4f}, "
              f"Classifier Loss: {classifier_loss_total/len(train_loader):.4f}")

        # Validation
        classifier.eval()
        copycat.eval()
        val_loss_total = 0.0
        predictions, true_labels, joint_energies = [], [], []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.cuda(), labels.cuda()
                _, copycat_features = copycat(images)
                outputs, _ = classifier(images, copycat_features)
                logits = outputs['base']
                labels = labels[:, :len(known_labels)]
                val_loss = criterion(logits, labels)
                val_loss_total += val_loss.item()

                joint_energy = classifier.compute_joint_energy(logits)
                joint_energies.extend(joint_energy.cpu().numpy())
                predictions.extend(torch.sigmoid(logits).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        val_loss_avg = val_loss_total / len(val_loader)
        print(f"8. Epoch {current_epoch+1}/{epoch}, Val Loss: {val_loss_avg:.4f}")
        evaluate_metrics(np.array(true_labels), np.array(predictions), np.array(joint_energies), all_labels, known_labels, unknown_labels, tau_nor, tau_unk)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = current_epoch
            torch.save(classifier.state_dict(), f'best_classifier_epoch_{best_epoch+1}.pth')
            torch.save(copycat.state_dict(), f'best_copycat_epoch_{best_epoch+1}.pth')

        if (current_epoch + 1) % 60 == 0:
            for param_group in optimizer_classifier.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_copycat.param_groups:
                param_group['lr'] *= 0.1

    print(f"9. Best model saved at epoch {best_epoch+1}")

if __name__ == "__main__":
    main(epoch=10, tau_nor=0.5, tau_unk=1.5, lambda_kd=1.0, lambda_ent=0.1)