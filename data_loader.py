import os
import random
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', 
                 known_labels=None, unknown_labels=None, data_ratio=1.0, extra_unknown_data=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.known_labels = known_labels if known_labels is not None else []
        self.unknown_labels = unknown_labels if unknown_labels is not None else []
        self.data_ratio = max(0.0, min(1.0, data_ratio))
        self.extra_unknown_data = extra_unknown_data

        if mode == 'train':
            self.data = self.data[self.data['Finding Labels'].apply(
                lambda x: all(label in self.known_labels + ['No Finding'] for label in x.split('|'))
            )]
        elif mode == 'val':
            if self.extra_unknown_data is not None:
                # 중복 제거 후 병합
                self.data = pd.concat([self.data, self.extra_unknown_data]).drop_duplicates(subset=['Image Index']).reset_index(drop=True)
            self.unknown_data_only = self.data[self.data['Finding Labels'].apply(
                lambda x: any(label in self.unknown_labels for label in x.split('|')) and 
                all(label not in self.known_labels for label in x.split('|'))
            )]
            self.unknown_and_known_data = self.data[self.data['Finding Labels'].apply(
                lambda x: any(label in self.unknown_labels for label in x.split('|')) and 
                any(label in self.known_labels for label in x.split('|'))
            )]
            self.known_data = self.data[self.data['Finding Labels'].apply(
                lambda x: all(label in self.known_labels + ['No Finding'] for label in x.split('|'))
            )]
        elif mode == 'test':
            pass

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels + self.unknown_labels)}
        
        # 샘플링
        label_counts = {label: [] for label in self.class_to_idx.keys()}
        normal_indices = []
        for idx, row in self.data.iterrows():
            labels = row['Finding Labels'].split('|')
            if "No Finding" in labels:
                normal_indices.append(idx)
            else:
                for label in labels:
                    label_counts[label].append(idx)

        sampled_indices = []
        for label, indices in label_counts.items():
            if label != "No Finding" and indices:
                num_samples = int(len(indices) * self.data_ratio)
                sampled_indices.extend(random.sample(indices, max(1, min(num_samples, len(indices)))))
        if normal_indices:
            num_normal_samples = int(len(normal_indices) * self.data_ratio)
            sampled_indices.extend(random.sample(normal_indices, max(1, min(num_normal_samples, len(normal_indices)))))

        self.data = self.data.iloc[list(set(sampled_indices))].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['Image Index'])
        try:
            image = Image.open(img_path).convert('L')
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  # 필요 시 대체값 반환
        labels = self.data.iloc[idx]['Finding Labels'].split('|')
        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in labels:
            target[self.class_to_idx[label]] = 1.0
        if self.transform:
            image = self.transform(image)
        return image, target

    def get_unknown_data_only(self):
        return getattr(self, 'unknown_data_only', pd.DataFrame())

    def get_unknown_and_known_data(self):
        return getattr(self, 'unknown_and_known_data', pd.DataFrame())

    def get_known_data(self):
        return getattr(self, 'known_data', pd.DataFrame())


def load_data(train_csv, train_dir, val_csv, val_dir, known_labels, unknown_labels, 
              batch_size=64, train_data_ratio=1.0, val_data_ratio=1.0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_data = pd.read_csv(train_csv)
    train_known_data = full_train_data[full_train_data['Finding Labels'].apply(
        lambda x: all(label in known_labels + ['No Finding'] for label in x.split('|'))
    )]
    train_unknown_data = full_train_data[~full_train_data.index.isin(train_known_data.index)]

    # val 데이터와 중복 제거
    val_data = pd.read_csv(val_csv)
    train_unknown_data = train_unknown_data[~train_unknown_data['Image Index'].isin(val_data['Image Index'])]

    train_dataset = NIHChestXrayDataset(
        csv_file=train_csv, root_dir=train_dir, transform=transform, mode='train', 
        known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=train_data_ratio
    )
    val_dataset = NIHChestXrayDataset(
        csv_file=val_csv, root_dir=val_dir, transform=transform, mode='val', 
        known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=val_data_ratio,
        extra_unknown_data=train_unknown_data
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    known_labels = ["Pneumonia", "Effusion"]
    unknown_labels = ["Cardiomegaly", "Mass"]
    train_loader, val_loader = load_data(
        train_csv="train.csv", train_dir="train_images/",
        val_csv="val.csv", val_dir="val_images/",
        known_labels=known_labels, unknown_labels=unknown_labels,
        batch_size=32, train_data_ratio=0.5, val_data_ratio=0.3
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")