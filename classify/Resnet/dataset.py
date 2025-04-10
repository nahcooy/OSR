import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import pandas as pd
import os

def getHAM10000Dataset(data_path='./data', split='train', image_size=224, **kwargs):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 메타데이터 및 이미지 경로 설정
    metadata_path = os.path.join(data_path, 'HAM10000_metadata_augmented.csv')
    image_dir = os.path.join(data_path, split)  # train/ 또는 val/
    metadata = pd.read_csv(metadata_path)

    # 클래스 매핑 (전체 7개 클래스)
    all_classes = ['nv', 'mel', 'bkl', 'akiec', 'vasc', 'df', 'bcc']
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    # split 디렉토리에 있는 이미지만 필터링
    available_images = set([fname.replace('.jpg', '') for fname in os.listdir(image_dir)])
    metadata = metadata[metadata['image_id'].isin(available_images)].reset_index(drop=True)

    print(f"✅ Split: '{split}' | Samples: {len(metadata)}")
    print(f"Class distribution:\n{metadata['dx'].value_counts()}\n")

    class HAMDataset(Dataset):
        def __init__(self, metadata, image_dir, transform=None):
            self.metadata = metadata
            self.image_dir = image_dir
            self.transform = transform
            self.loader = default_loader

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, idx):
            row = self.metadata.iloc[idx]
            image_id = row['image_id']
            label_name = row['dx']
            label = class_to_idx[label_name]

            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            img = self.loader(img_path)

            if self.transform:
                img = self.transform(img)

            return img, label

    return HAMDataset(metadata, image_dir, transform)
