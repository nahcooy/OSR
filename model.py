# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class Copycat(nn.Module):
    def __init__(self, feature_dim=1536):
        super(Copycat, self).__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1536, feature_dim)

    def forward(self, x):
        intermediate_features = []
        x1 = self.features[0](x)
        intermediate_features.append(x1)
        x2 = self.features[1](x1)
        intermediate_features.append(x2)
        x3 = self.features[2](x2)
        intermediate_features.append(x3)
        x4 = self.features[3](x3)
        intermediate_features.append(x4)
        x5 = self.features[4](x4)
        intermediate_features.append(x5)
        x = x5
        for layer in self.features[5:]:
            x = layer(x)
        x6 = x
        intermediate_features.append(x6)
        x = self.avgpool(x6)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, intermediate_features


class Classifier(nn.Module):
    def __init__(self, classid_list, feature_dim=1536):
        super(Classifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classid_list = classid_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classid_list)}
        
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1536, self.num_classes)

    def forward(self, x, copycat_features=None):
        intermediate_features = []
        x1 = self.features[0](x)
        intermediate_features.append(x1)
        x2 = self.features[1](x1)
        intermediate_features.append(x2)
        x3 = self.features[2](x2)
        intermediate_features.append(x3)
        x4 = self.features[3](x3)
        if copycat_features is not None:
            x4 = x4 + copycat_features[3]  # Feature injection
        intermediate_features.append(x4)
        x5 = self.features[4](x4)
        intermediate_features.append(x5)
        x = x5
        for layer in self.features[5:]:
            x = layer(x)
        x6 = x
        intermediate_features.append(x6)
        x = self.avgpool(x6)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, intermediate_features

    def compute_joint_energy(self, logits):
        label_wise_energy = -torch.log(1 + torch.exp(logits))
        joint_energy = -torch.sum(label_wise_energy, dim=1)
        return joint_energy

    def predict(self, logits, tau_nor, tau_unk):
        joint_energy = self.compute_joint_energy(logits)
        probs = torch.sigmoid(logits)
        preds = torch.zeros_like(probs, device=self.device)
        for i in range(logits.shape[0]):
            if joint_energy[i] < tau_nor:
                preds[i, self.class_to_idx["No Finding"]] = 1  # Normal
            elif joint_energy[i] > tau_unk:
                preds[i] = (probs[i] > 0.5).float()  # Unknown은 Known처럼 예측 후 후처리
                if not preds[i].any():  # 예측 없으면 Unknown
                    preds[i] = -1
            else:
                preds[i] = (probs[i] > 0.5).float()  # Known
        return preds, joint_energy
