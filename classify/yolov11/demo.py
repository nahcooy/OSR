from ultralytics import YOLO
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from pathlib import Path
import pandas as pd

# 모델 로드
model = YOLO("/nahcooy/OSR/classify/yolov11/saver/yolo11-ham_no_aug/weights/best.pt")

# 검증 데이터 경로
val_path = "/dataset/nahcooy/HAM_yolo/val/total"

# 메타데이터 CSV 파일 로드
metadata_path = "/dataset/nahcooy/HAM_aug/HAM10000_metadata_augmented.csv"
metadata = pd.read_csv(metadata_path)

# 모델이 기대하는 클래스 (7개)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# CSV에서 유효한 클래스만 필터링
valid_classes = set(class_names)
metadata = metadata[metadata['dx'].isin(valid_classes)]
print(f"Filtered metadata rows: {len(metadata)}")

# 이미지 ID와 레이블 매핑
image_to_label = {row['image_id']: class_to_idx[row['dx']] for _, row in metadata.iterrows()}

# total 디렉토리의 이미지 파일 확인
image_files = list(Path(val_path).glob('*.jpg'))  # 확장자에 따라 수정
print(f"Total images in {val_path}: {len(image_files)}")

# 예측 수행
results = model.predict(source=val_path, save=False, stream=True)

# 실제 레이블, 예측 레이블, 예측 확률 저장
true_labels = []
pred_labels = []
pred_probs = []

for result in results:
    image_id = Path(result.path).stem  # 예: 'ISIC_0027419'
    if image_id in image_to_label:
        label = image_to_label[image_id]
        true_labels.append(label)
        
        # 예측 레이블 (가장 높은 확률의 클래스)
        probs = result.probs.data.cpu().numpy()
        pred_label = np.argmax(probs)
        pred_labels.append(pred_label)
        
        pred_probs.append(probs)
    else:
        print(f"Warning: {image_id} not found in metadata")

# 리스트를 NumPy 배열로 변환
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
pred_probs = np.array(pred_probs)
print(f"Processed images: {len(true_labels)}")

# 클래스 수 확인
num_classes = pred_probs.shape[1]
if num_classes != len(class_names):
    raise ValueError(f"Model expects {len(class_names)} classes, but got {num_classes}")

# Confusion Matrix 직접 계산
cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
print("Confusion Matrix:")
print(cm)

# One-vs-Rest 방식으로 AUROC 계산
auroc_scores = []
for i in range(num_classes):
    true_binary = (true_labels == i).astype(int)
    pred_prob_class = pred_probs[:, i]
    auroc = roc_auc_score(true_binary, pred_prob_class)
    auroc_scores.append(auroc)

# 평균 AUROC
macro_auroc = np.mean(auroc_scores)

# AUROC 출력
print("\nClass-wise AUROC:")
for i, score in enumerate(auroc_scores):
    print(f"{class_names[i]}: {score:.4f}")
print(f"Macro AUROC: {macro_auroc:.4f}")