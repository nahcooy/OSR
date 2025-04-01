import torch
import torch.optim as optim
from tqdm import tqdm
from model import Classifier, Copycat
from data_loader import load_data, NIHChestXrayDataset
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def evaluate_metrics(true_labels, predictions, joint_energies, all_labels, known_labels, unknown_labels, tau_nor=0.5, tau_unk=1.5, category_name=""):
    tau_versions = [
        {"tau_nor": 0.3, "tau_unk": 1.0, "name": "Low Tau"},
        {"tau_nor": 0.5, "tau_unk": 1.5, "name": "Default Tau"},
        {"tau_nor": 0.7, "tau_unk": 2.0, "name": "High Tau"}
    ]

    metrics_dict = {}
    for tau_version in tau_versions:
        tau_nor = tau_version["tau_nor"]
        tau_unk = tau_version["tau_unk"]
        version_name = tau_version["name"]
        print(f"\n=== {category_name} - Evaluating with {version_name} (tau_nor={tau_nor}, tau_unk={tau_unk}) ===")

        # 예측 생성
        preds = np.zeros_like(predictions)
        for i in range(len(joint_energies)):
            if joint_energies[i] < tau_nor:
                preds[i, all_labels.index("No Finding")] = 1  # Normal
            elif joint_energies[i] < tau_unk:
                preds[i] = -1  # Unknown
            else:
                preds[i] = (predictions[i] > 0.5).astype(float)  # Known

        # Unknown 비율 계산
        unknown_count = np.sum([1 for pred in preds if np.all(pred == -1)])
        total_samples = len(preds)
        unknown_ratio = unknown_count / total_samples if total_samples > 0 else 0
        print(f"Unknown Ratio: {unknown_ratio:.4f} ({unknown_count}/{total_samples})")

        # 1. Unknown 탐지 (Binary)
        unknown_true = np.array([1 if set([all_labels[idx] for idx, val in enumerate(l) if val == 1]) == set(unknown_labels) and l.sum() > 0 else 0 for l in true_labels])
        unknown_pred = np.array([1 if np.all(pred == -1) else 0 for pred in preds])
        if len(np.unique(unknown_true)) > 1:
            unknown_auroc = roc_auc_score(unknown_true, unknown_pred)
            unknown_f1 = f1_score(unknown_true, unknown_pred)
            print(f"Unknown Detection - AUROC: {unknown_auroc:.4f}, F1: {unknown_f1:.4f}")
        else:
            print("Unknown Detection - No variation in true labels, skipping metrics")
            unknown_auroc, unknown_f1 = None, None

        # 2. Multi-class Classification
        multi_true = true_labels  # [num_samples, num_known_classes]
        multi_pred = preds  # [num_samples, num_known_classes]
        multi_auroc = []
        multi_f1 = []
        for idx in range(len(known_labels)):
            if len(np.unique(multi_true[:, idx])) > 1 and not np.all(multi_pred[:, idx] == -1):
                auroc = roc_auc_score(multi_true[:, idx], multi_pred[:, idx])
                f1 = f1_score(multi_true[:, idx], multi_pred[:, idx], zero_division=0)
                multi_auroc.append(auroc)
                multi_f1.append(f1)
        if multi_auroc:
            avg_multi_auroc = np.mean(multi_auroc)
            avg_multi_f1 = np.mean(multi_f1)
            print(f"Multi-class Classification - Avg AUROC: {avg_multi_auroc:.4f}, Avg F1: {avg_multi_f1:.4f}")
        else:
            print("Multi-class Classification - No valid classes for metrics")
            avg_multi_auroc, avg_multi_f1 = None, None

        metrics_dict[version_name] = {
            "Unknown": {"AUROC": unknown_auroc, "F1": unknown_f1},
            "Multi-class": {"AUROC": avg_multi_auroc, "F1": avg_multi_f1}
        }
    
    return metrics_dict

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
                                             batch_size=32, train_data_ratio=0.3, val_data_ratio=1.0)
        val_dataset = val_loader.dataset
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    val_unknown_only = NIHChestXrayDataset(csv_file=val_csv, root_dir=val_dir, transform=val_dataset.transform, mode='val', 
                                           known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=1.0)
    val_unknown_only.data = val_unknown_only.unknown_data_only
    val_unknown_and_known = NIHChestXrayDataset(csv_file=val_csv, root_dir=val_dir, transform=val_dataset.transform, mode='val', 
                                                known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=1.0)
    val_unknown_and_known.data = val_unknown_and_known.unknown_and_known_data
    val_known_only = NIHChestXrayDataset(csv_file=val_csv, root_dir=val_dir, transform=val_dataset.transform, mode='val', 
                                         known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=1.0)
    val_known_only.data = val_known_only.known_data

    val_loaders = {
        "Unknown Only": DataLoader(val_unknown_only, batch_size=32, shuffle=False),
        "Mixed (Unknown + Known)": DataLoader(val_unknown_and_known, batch_size=32, shuffle=False),
        "Known Only": DataLoader(val_known_only, batch_size=32, shuffle=False)
    }

    print("4. Initializing models...")
    classifier = Classifier(classid_list=known_labels).to(device)
    copycat = Copycat(num_classes=len(known_labels)).to(device)

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
                images, labels = images.to(device), labels.to(device)
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
                alpha_easy = 1.0
                alpha_hard = 0.5
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
        all_predictions, all_true_labels, all_joint_energies = [], [], []
        category_metrics = {}

        for category, loader in val_loaders.items():
            predictions, true_labels, joint_energies = [], [], []
            val_loss_category = 0.0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(loader):
                    if images is None or labels is None:
                        continue
                    images, labels = images.cuda(), labels.cuda()
                    _, copycat_features = copycat(images)
                    outputs, _ = classifier(images, copycat_features)
                    logits = outputs['base']
                    labels = labels[:, :len(known_labels)]
                    val_loss = criterion(logits, labels)
                    val_loss_category += val_loss.item()

                    joint_energy = classifier.compute_joint_energy(logits)
                    joint_energies.extend(joint_energy.cpu().numpy())
                    predictions.extend(torch.sigmoid(logits).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            val_loss_avg = val_loss_category / len(loader) if len(loader) > 0 else 0
            print(f"8. {category} - Epoch {current_epoch+1}/{epoch}, Val Loss: {val_loss_avg:.4f}")
            val_loss_total += val_loss_avg

            if len(true_labels) > 0:
                metrics = evaluate_metrics(np.array(true_labels), np.array(predictions), np.array(joint_energies), 
                                           all_labels, known_labels, unknown_labels, tau_nor, tau_unk, category)
                category_metrics[category] = metrics

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)
            all_joint_energies.extend(joint_energies)

        val_loss_avg_total = val_loss_total / len(val_loaders)
        print(f"8. Total - Epoch {current_epoch+1}/{epoch}, Val Loss: {val_loss_avg_total:.4f}")
        total_metrics = evaluate_metrics(np.array(all_true_labels), np.array(all_predictions), np.array(all_joint_energies), 
                                         all_labels, known_labels, unknown_labels, tau_nor, tau_unk, "Total")

        print("\n=== Average Metrics Across Categories ===")
        for tau_version in ["Low Tau", "Default Tau", "High Tau"]:
            print(f"\nTau Version: {tau_version}")
            for metric_name in ["Unknown", "Multi-class"]:
                auroc_vals, f1_vals = [], []
                for category in category_metrics:
                    metrics = category_metrics[category][tau_version][metric_name]
                    if metrics["AUROC"] is not None:
                        auroc_vals.append(metrics["AUROC"])
                        f1_vals.append(metrics["F1"])
                if auroc_vals:
                    print(f"{metric_name} - Avg AUROC: {np.mean(auroc_vals):.4f}, Avg F1: {np.mean(f1_vals):.4f}")

        if val_loss_avg_total < best_val_loss:
            best_val_loss = val_loss_avg_total
            best_epoch = current_epoch

    print(f"9. Best model saved at epoch {best_epoch+1}")

if __name__ == "__main__":
    main(epoch=10, tau_nor=0.5, tau_unk=1.5, lambda_kd=1.0, lambda_ent=0.1)