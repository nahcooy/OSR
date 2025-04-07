import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# CSV 파일 읽기
df = pd.read_csv('/dataset/nahcooy/CXR8/Data_Entry_2017_v2020.csv')

# 이미지 디렉토리 경로
image_dir = '/dataset/nahcooy/CXR8/images/original_images'

# 저장할 디렉토리 경로
train_image_dir = '/dataset/nahcooy/CXR8/images/train'
val_image_dir = '/dataset/nahcooy/CXR8/images/val'

# 디렉토리가 없으면 생성
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

# 각 Finding Labels에 대해 데이터를 분할
train_data = []
val_data = []

# 각 클래스별로 데이터를 나누기 (class 비율 유지)
for label in df['Finding Labels'].unique():
    class_data = df[df['Finding Labels'] == label]  # 해당 라벨에 맞는 데이터 필터링
    
    if len(class_data) > 1:  # 클래스별 샘플 수가 2개 이상인 경우
        # 샘플을 랜덤하게 분할 (stratify 없이)
        train_class, val_class = train_test_split(class_data, test_size=0.2, random_state=42)
    else:  # 클래스별 샘플 수가 1개인 경우
        print(f"Class with only one sample: {label}")
        # 1개의 샘플은 검증 세트로 바로 이동시키고, 훈련 세트는 빈 데이터로 설정
        train_class = pd.DataFrame(columns=class_data.columns)  # 빈 DataFrame 생성
        val_class = class_data  # 해당 샘플을 검증 세트로

    # 나눈 데이터를 train_data와 val_data에 추가
    train_data.append(train_class)
    val_data.append(val_class)

    # 이미지 파일 경로 추출 및 이동
    for idx, row in train_class.iterrows():
        image_name = row['Image Index']
        src_image_path = os.path.join(image_dir, image_name)
        dst_image_path = os.path.join(train_image_dir, image_name)
        shutil.copy(src_image_path, dst_image_path)

    for idx, row in val_class.iterrows():
        image_name = row['Image Index']
        src_image_path = os.path.join(image_dir, image_name)
        dst_image_path = os.path.join(val_image_dir, image_name)
        shutil.copy(src_image_path, dst_image_path)

# 전체 데이터를 합치기
train_df = pd.concat(train_data)
val_df = pd.concat(val_data)

# CSV 파일로 저장 (train과 val 데이터)
train_df.to_csv('/dataset/nahcooy/CXR8/images/train.csv', index=False)
val_df.to_csv('/dataset/nahcooy/CXR8/images/val.csv', index=False)

# 데이터셋 크기 출력
print(f"Train dataset size: {len(train_df)}")
print(f"Val dataset size: {len(val_df)}")
