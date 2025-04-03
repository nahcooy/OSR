import os
import random
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', 
                 known_labels=None, unknown_labels=None, data_ratio=1.0, extra_unknown_data=None, train_dir=None):
        print(f"Loading CSV: {csv_file}")
        self.data = pd.read_csv(csv_file)
        print(f"Initial data size from {csv_file}: {len(self.data)}")
        self.root_dir = root_dir
        self.train_dir = train_dir if train_dir else root_dir
        self.transform = transform
        self.mode = mode
        self.known_labels = known_labels if known_labels is not None else []
        self.unknown_labels = unknown_labels if unknown_labels is not None else []
        self.data_ratio = max(0.0, min(1.0, data_ratio))
        self.extra_unknown_data = extra_unknown_data

        if mode == 'train':
            self.data = self.data[self.data['Finding Labels'].apply(
                lambda x: all(label in self.known_labels + ['No Finding'] for label in x.split('|'))
            )].reset_index(drop=True)
            print(f"Train mode filtered data size: {len(self.data)}")
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels)}
        elif mode == 'val':
            if self.extra_unknown_data is not None:
                print(f"Extra unknown data size: {len(self.extra_unknown_data)}")
                self.data = pd.concat([self.data, self.extra_unknown_data]).drop_duplicates(subset=['Image Index']).reset_index(drop=True)
                print(f"Val mode after concat and dedup: {len(self.data)}")
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels + self.unknown_labels)}
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
            print(f"Val mode - unknown only: {len(self.unknown_data_only)}, mixed: {len(self.unknown_and_known_data)}, known: {len(self.known_data)}")
        elif mode == 'test':
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.known_labels + self.unknown_labels)}

        if self.data_ratio < 1.0:
            label_counts = {label: [] for label in self.class_to_idx.keys()}
            normal_indices = []
            for idx in range(len(self.data)):
                labels = self.data.iloc[idx]['Finding Labels'].split('|')
                if "No Finding" in labels:
                    normal_indices.append(idx)
                else:
                    for label in labels:
                        if label in self.class_to_idx:
                            label_counts[label].append(idx)

            sampled_indices = set()
            for label, indices in label_counts.items():
                if indices and label != "No Finding":
                    num_samples = max(1, int(len(indices) * self.data_ratio))
                    sampled_indices.update(random.sample(indices, min(num_samples, len(indices))))
            if normal_indices:
                num_normal = max(1, int(len(normal_indices) * self.data_ratio))
                sampled_indices.update(random.sample(normal_indices, min(num_normal, len(normal_indices))))

            self.data = self.data.iloc[list(sampled_indices)].reset_index(drop=True)
            print(f"Data size after sampling: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_idx = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.train_dir if self.extra_unknown_data is not None and img_idx in self.extra_unknown_data['Image Index'].values else self.root_dir, img_idx)
        try:
            image = Image.open(img_path).convert('L')
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  # 문제가 있는 경우 None 반환
        
        labels = self.data.iloc[idx]['Finding Labels'].split('|')
        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
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
    print(f"Full train data size: {len(full_train_data)}")
    train_known_data = full_train_data[full_train_data['Finding Labels'].apply(
        lambda x: all(label in known_labels + ['No Finding'] for label in x.split('|'))
    )]
    train_unknown_data = full_train_data[~full_train_data.index.isin(train_known_data.index)]
    val_data = pd.read_csv(val_csv)
    train_unknown_data = train_unknown_data[~train_unknown_data['Image Index'].isin(val_data['Image Index'])]

    train_dataset = NIHChestXrayDataset(
        csv_file=train_csv, root_dir=train_dir, transform=transform, mode='train', 
        known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=train_data_ratio
    )
    val_dataset = NIHChestXrayDataset(
        csv_file=val_csv, root_dir=val_dir, transform=transform, mode='val', 
        known_labels=known_labels, unknown_labels=unknown_labels, data_ratio=val_data_ratio,
        extra_unknown_data=train_unknown_data, train_dir=train_dir
    )

    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            # 빈 배치 대신 더미 데이터 반환
            return torch.zeros(1, 1, 224, 224), torch.zeros(1, len(known_labels) + len(unknown_labels))
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.stack(targets)  # 전체 클래스 유지
        return images, targets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
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