import torch
import torch.optim as optim
from model import Classifier, Copycat
from data_loader import load_data, NIHChestXrayDataset
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from torch.utils.data import DataLoader
import time

def print_with_time(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def evaluate_metrics(true_labels, predictions, joint_energies, all_labels, known_labels, unknown_labels, tau_nor=1.0, tau_unk=3.0, category_name=""):
    tau_versions = [
        {"tau_nor": 0.5, "tau_unk": 2.0, "name": "Low Tau"},
        {"tau_nor": 1.0, "tau_unk": 3.0, "name": "Default Tau"},
        {"tau_nor": 1.5, "tau_unk": 4.0, "name": "High Tau"}
    ]

    metrics_dict = {}
    for tau_version in tau_versions:
        tau_nor = tau_version["tau_nor"]
        tau_unk = tau_version["tau_unk"]
        version_name = tau_version["name"]
        print_with_time(f"\n=== {category_name} - Evaluating with {version_name} (tau_nor={tau_nor}, tau_unk={tau_unk}) ===")

        print_with_time(f"Joint Energies - Min: {np.min(joint_energies):.4f}, Max: {np.max(joint_energies):.4f}, Mean: {np.mean(joint_energies):.4f}")

        preds = np.zeros((len(predictions), len(all_labels)))
        binary_pred = []
        binary_true = []

        for i in range(len(joint_energies)):
            if joint_energies[i] < tau_nor:
                preds[i, all_labels.index("No Finding")] = 1
                binary_pred.append(0)
            elif joint_energies[i] < tau_unk:
                for unk in unknown_labels:
                    preds[i, all_labels.index(unk)] = 1
                binary_pred.append(1)
            else:
                preds[i, :len(predictions[i])] = (predictions[i] > 0.5).astype(float)
                binary_pred.append(0)

            label_names = [all_labels[j] for j, val in enumerate(true_labels[i]) if val == 1]
            binary_true.append(1 if set(label_names).issubset(set(unknown_labels)) else 0)

        if len(np.unique(binary_true)) > 1:
            unknown_auroc = roc_auc_score(binary_true, binary_pred)
            unknown_f1 = f1_score(binary_true, binary_pred)
            print_with_time(f"Binary Unknown Detection - AUROC: {unknown_auroc:.4f}, F1: {unknown_f1:.4f}")
        else:
            unknown_auroc, unknown_f1 = None, None
            print_with_time("Binary Unknown Detection - Skipped due to label imbalance")

        multi_auroc = []
        multi_f1 = []
        for idx in range(len(all_labels)):
            if idx >= true_labels.shape[1]:
                continue
            y_true = true_labels[:, idx]
            y_pred = preds[:, idx]
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                multi_auroc.append(auroc)
                multi_f1.append(f1)

        if multi_auroc:
            avg_multi_auroc = np.mean(multi_auroc)
            avg_multi_f1 = np.mean(multi_f1)
            print_with_time(f"Multi-label Classification (with Unknown) - AUROC: {avg_multi_auroc:.4f}, F1: {avg_multi_f1:.4f}")
        else:
            avg_multi_auroc, avg_multi_f1 = None, None
            print_with_time("Multi-label Classification - Skipped due to lack of variation")

        metrics_dict[version_name] = {
            "Binary Unknown Detection": {"AUROC": unknown_auroc, "F1": unknown_f1},
            "Multi-label with Unknown": {"AUROC": avg_multi_auroc, "F1": avg_multi_f1}
        }

    return metrics_dict

def main(epoch=10, tau_nor=1.0, tau_unk=3.0, lambda_kd=1.0, lambda_ent=0.1, save_path_prefix=""):
    train_csv = '/dataset/nahcooy/CXR8/images/train.csv'
    train_dir = '/dataset/nahcooy/CXR8/images/train'
    val_csv = '/dataset/nahcooy/CXR8/images/val.csv'
    val_dir = '/dataset/nahcooy/CXR8/images/val'
    known_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                    "Pneumothorax", "Consolidation", "Pleural_Thickening", "Hernia", "No Finding",
                    "Pneumonia", "Edema", "Emphysema", "Fibrosis"]
    unknown_labels = ["Nodule"]
    all_labels = known_labels + unknown_labels

    print_with_time("1. Loading train and val data...")
    try:
        train_loader, val_loader = load_data(train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels,
                                             batch_size=32, train_data_ratio=0.3, val_data_ratio=1.0)
        val_dataset = val_loader.dataset
    except Exception as e:
        print_with_time(f"Error loading data: {e}")
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

    print_with_time("4. Initializing models...")
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
        print_with_time(f"Epoch {current_epoch+1}/{epoch} - Training started")
        for batch_idx, (images, labels) in enumerate(train_loader):
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

            if (batch_idx + 1) % 50 == 0:
                print_with_time(f"Batch {batch_idx+1}/{len(train_loader)}: Copycat Loss = {copycat_loss.item():.4f}, Classifier Loss = {classifier_loss.item():.4f}")

        print_with_time(f"Epoch {current_epoch+1}/{epoch}, Copycat Loss: {copycat_loss_total/len(train_loader):.4f}, Classifier Loss: {classifier_loss_total/len(train_loader):.4f}")

        classifier.eval()
        copycat.eval()
        val_loss_total = 0.0
        all_predictions, all_true_labels, all_joint_energies = [], [], []
        category_metrics = {}

        for category, loader in val_loaders.items():
            print_with_time(f"Validation for {category} started")
            predictions, true_labels, joint_energies = [], [], []
            val_loss_category = 0.0
            for batch_idx, (images, labels) in enumerate(loader):
                if images is None or labels is None:
                    continue
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, copycat_features = copycat(images)
                    outputs, _ = classifier(images, copycat_features)
                    logits = outputs['base']
                    val_loss = criterion(logits, labels[:, :len(known_labels)])
                    val_loss_category += val_loss.item()

                    joint_energy = classifier.compute_joint_energy(logits)
                    joint_energies.extend(joint_energy.cpu().numpy())
                    predictions.extend(torch.sigmoid(logits).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            val_loss_avg = val_loss_category / len(loader) if len(loader) > 0 else 0
            print_with_time(f"8. {category} - Epoch {current_epoch+1}/{epoch}, Val Loss: {val_loss_avg:.4f}")
            val_loss_total += val_loss_avg

            if len(true_labels) > 0:
                metrics = evaluate_metrics(np.array(true_labels), np.array(predictions), np.array(joint_energies),
                                           all_labels, known_labels, unknown_labels, tau_nor, tau_unk, category)
                category_metrics[category] = metrics

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)
            all_joint_energies.extend(joint_energies)
            print_with_time(f"Validation for {category} ended")

        val_loss_avg_total = val_loss_total / len(val_loaders)
        print_with_time(f"8. Total - Epoch {current_epoch+1}/{epoch}, VaQl Loss: {val_loss_avg_total:.4f}")
        total_metrics = evaluate_metrics(np.array(all_true_labels), np.array(all_predictions), np.array(all_joint_energies),
                                         all_labels, known_labels, unknown_labels, tau_nor, tau_unk, "Total")

        print_with_time("\n=== Average Metrics Across Categories ===")
        for tau_version in ["Low Tau", "Default Tau", "High Tau"]:
            print_with_time(f"\nTau Version: {tau_version}")
            for metric_name in ["Binary Unknown Detection", "Multi-label with Unknown"]:
                auroc_vals, f1_vals = [], []
                for category in category_metrics:
                    metrics = category_metrics[category][tau_version][metric_name]
                    if metrics["AUROC"] is not None:
                        auroc_vals.append(metrics["AUROC"])
                        f1_vals.append(metrics["F1"])
                if auroc_vals:
                    print_with_time(f"{metric_name} - Avg AUROC: {np.mean(auroc_vals):.4f}, Avg F1: {np.mean(f1_vals):.4f}")

        if val_loss_avg_total < best_val_loss:
            best_val_loss = val_loss_avg_total
            best_epoch = current_epoch
            torch.save(classifier.state_dict(), f"{save_path_prefix}best_classifier_epoch_{best_epoch+1}.pth")
            torch.save(copycat.state_dict(), f"{save_path_prefix}best_copycat_epoch_{best_epoch+1}.pth")
            print_with_time(f"Best model saved at epoch {best_epoch+1} with Val Loss: {best_val_loss:.4f}")

    print_with_time(f"Training completed. Best model was at epoch {best_epoch+1}")

if __name__ == "__main__":
    main(epoch=200, tau_nor=1.0, tau_unk=3.0, lambda_kd=1.0, lambda_ent=0.1, save_path_prefix="./checkpoints/train_ver2_0402")
