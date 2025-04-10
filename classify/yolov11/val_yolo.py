# from ultralytics import YOLO

# model = YOLO("/nahcooy/OSR/classify/yolov11/saver/yolo11-ham_0410/weights/best.pt")

# metrics = model.val()

# # Confusion Matrix 출력
# print("Confusion Matrix:")
# print(metrics.confusion_matrix.matrix)

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Confusion Matrix
cm = np.array([
    [65, 0, 0, 0, 0, 0, 0],
    [0, 101, 0, 0, 0, 2, 0],
    [0, 0, 202, 0, 4, 7, 0],
    [0, 0, 0, 23, 0, 0, 0],
    [0, 0, 3, 0, 178, 12, 0],
    [0, 2, 15, 0, 41, 1318, 0],
    [0, 0, 0, 0, 0, 2, 28],
])

# 각 샘플에 대한 예측값과 정답값 복원
y_true = []
y_pred = []

for true_class, row in enumerate(cm):
    for pred_class, count in enumerate(row):
        y_true += [true_class] * count
        y_pred += [pred_class] * count

# Accuracy
acc = accuracy_score(y_true, y_pred)

# F1-score (macro)
f1 = f1_score(y_true, y_pred, average='macro')

# AUROC는 생략 (confusion matrix만으로는 계산 불가능)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Macro F1-score: {f1:.4f}")
print(f"⚠️ AUROC: 계산 불가 (확률 값이 필요함)")
