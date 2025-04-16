from ultralytics import YOLO

# 🔹 YOLOv11 분류 모델 불러오기 (사전학습 모델)
model = YOLO('yolo11x-cls.pt')  # 또는 yolo11s-cls.pt, yolo11m-cls.pt 등

# 🔹 학습
model.train(
    data='/dataset/nahcooy/HAM_yolo_no_aug',  # train/val 구조를 가진 디렉토리
    epochs=400,
    imgsz=224,
    batch=128,
    project='saver',
    name='yolo11-ham_no_aug',
    pretrained=True,
    plots = True,
    device=1  # GPU ID, 여러 개면 '0,1' 등 지정 가능
)
