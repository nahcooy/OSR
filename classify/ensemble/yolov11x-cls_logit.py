import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import cv2
from sklearn.preprocessing import LabelEncoder

device = 'cuda:1'  # 또는 'cpu'
IMAGE_SIZE = 224
DATA_DIR = '/dataset/nahcooy/HAM_yolo'  # train/val 구조
SAVE_DIR = 'logits/yolo'
CLASSES = ['nv', 'mel', 'bkl', 'akiec', 'vasc', 'df', 'bcc']
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)

model = YOLO('saver/yolo11-ham_0410/best.pt')

def extract_yolo_logits(split):
    image_dir = os.path.join(DATA_DIR, split)
    image_paths = []
    labels = []

    for cls in os.listdir(image_dir):
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for img_name in os.listdir(cls_dir):
            if img_name.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(cls)

    logits_list = []
    labels_encoded = label_encoder.transform(labels)

    for path in tqdm(image_paths, desc=f"YOLOv11 {split}"):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = model.predict(img_rgb, imgsz=IMAGE_SIZE, device=device, verbose=False)[0]
        probs = result.probs.cpu().numpy()
        logits_list.append(probs)

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, f'{split}_logits.npy'), np.stack(logits_list))
    np.save(os.path.join(SAVE_DIR, f'{split}_labels.npy'), labels_encoded)
    print(f"✅ YOLO {split} 저장 완료 → {SAVE_DIR}")

if __name__ == '__main__':
    extract_yolo_logits('train')
    extract_yolo_logits('val')
