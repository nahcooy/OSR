import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class Copycat(nn.Module):
    def __init__(self, feature_dim=1536, num_classes=None):
        super(Copycat, self).__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1536, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes) if num_classes else None

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
        x6 = self.features[5](x5)
        intermediate_features.append(x6)
        x = x6
        for layer in self.features[6:]:
            x = layer(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        logits = self.classifier(features) if self.classifier else features
        return logits, intermediate_features


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
        outputs = {}

        # 기본 경로 (주입 없이)
        x_base = x
        for i, layer in enumerate(self.features):
            x_base = layer(x_base)
            intermediate_features.append(x_base)
        x_base = self.avgpool(x_base)
        x_base = self.dropout(x_base)
        x_base = torch.flatten(x_base, 1)
        logits_base = self.fc(x_base)
        outputs['base'] = logits_base

        # 주입 경로 (conv2, conv4, conv6 대체)
        if copycat_features is not None:
            # conv2 대체
            x_conv2 = x
            for i, layer in enumerate(self.features):
                if i == 1:  # conv2
                    x_conv2 = copycat_features[1]  # 대체
                else:
                    x_conv2 = layer(x_conv2)
            x_conv2 = self.avgpool(x_conv2)
            x_conv2 = self.dropout(x_conv2)
            x_conv2 = torch.flatten(x_conv2, 1)
            outputs['conv2'] = self.fc(x_conv2)

            # conv4 대체
            x_conv4 = x
            for i, layer in enumerate(self.features):
                if i == 3:  # conv4
                    x_conv4 = copycat_features[3]  # 대체
                else:
                    x_conv4 = layer(x_conv4)
            x_conv4 = self.avgpool(x_conv4)
            x_conv4 = self.dropout(x_conv4)
            x_conv4 = torch.flatten(x_conv4, 1)
            outputs['conv4'] = self.fc(x_conv4)

            # conv6 대체
            x_conv6 = x
            for i, layer in enumerate(self.features):
                if i == 5:  # conv6
                    x_conv6 = copycat_features[5]  # 대체
                else:
                    x_conv6 = layer(x_conv6)
            x_conv6 = self.avgpool(x_conv6)
            x_conv6 = self.dropout(x_conv6)
            x_conv6 = torch.flatten(x_conv6, 1)
            outputs['conv6'] = self.fc(x_conv6)

        return outputs, intermediate_features

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
                preds[i] = (probs[i] > 0.5).float()
                if not preds[i].any():
                    preds[i] = -1
            else:
                preds[i] = (probs[i] > 0.5).float()
        return preds, joint_energy