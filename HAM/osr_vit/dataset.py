import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def getHAM10000Dataset(data_path='./data', **args):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']
    metadata_path = os.path.join(data_path, 'HAM10000_metadata.csv')
    images_dir = os.path.join(data_path, 'HAM10000_images')
    metadata = pd.read_csv(metadata_path)
    
    # 전체 메타데이터 출력
    print(f"Loaded metadata from {metadata_path}:")
    print(metadata.head())  # 처음 5행 출력
    print(f"Total samples: {len(metadata)}")
    print(f"Class distribution:\n{metadata['dx'].value_counts()}\n")

    all_classes = ['nv', 'mel', 'bkl', 'akiec', 'vasc', 'df', 'bcc']
    close_classes = ['nv', 'mel', 'bkl', 'akiec', 'vasc', 'df']
    unknown_classes = ['bcc']

    close_mapping = {cls: idx for idx, cls in enumerate(close_classes)}
    unknown_mapping = {cls: 6 for cls in unknown_classes}  # Unknown을 6으로 매핑

    # Known 클래스 데이터 분할
    close_idx = metadata['dx'].isin(close_classes)
    close_metadata = metadata[close_idx].reset_index(drop=True)
    train_meta, val_meta = train_test_split(close_metadata, test_size=0.1, random_state=args.get('random_seed', 42))

    if split == 'train':
        filtered_metadata = train_meta  # Known 90%
        mapping = close_mapping
    elif split == 'val_known':
        filtered_metadata = val_meta  # Known 10%
        mapping = close_mapping
    elif split == 'val_unknown':
        filtered_metadata = metadata[metadata['dx'].isin(unknown_classes)].reset_index(drop=True)  # Unknown 전체
        mapping = unknown_mapping
    else:
        raise ValueError(f"Unknown split: {split}")

    # 필터링된 메타데이터 출력
    print(f"Filtered metadata for split '{split}':")
    print(filtered_metadata.head())  # 처음 5행 출력
    print(f"Total samples in split: {len(filtered_metadata)}")
    print(f"Class distribution in split:\n{filtered_metadata['dx'].value_counts()}\n")

    class CustomHAM10000(Dataset):
        def __init__(self, metadata, images_dir, transform=None, target_transform=None):
            self.metadata = metadata
            self.images_dir = images_dir
            self.transform = transform
            self.target_transform = target_transform
            self.loader = default_loader

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, idx):
            row = self.metadata.iloc[idx]
            img_path = os.path.join(self.images_dir, f"{row['image_id']}.jpg")
            target = mapping[row['dx']]
            img = self.loader(img_path)
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                target = self.target_transform(target)
            return img, target

    dataset = CustomHAM10000(filtered_metadata, images_dir, transform=transform)
    return dataset