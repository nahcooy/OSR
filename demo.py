import os
import random
from PIL import Image

# 특정 디렉토리 경로
directory = '/dataset/nahcooy/CXR8/images/original_images'  # 디렉토리 경로를 적어주세요

# 디렉토리 내 PNG 파일 리스트 가져오기
png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

# 랜덤으로 100개 샘플 선택
sample_files = random.sample(png_files, 100)

# 변수 초기화
total_width = 0
total_height = 0

# 각 이미지의 width, height 합산
for file in sample_files:
    img_path = os.path.join(directory, file)
    with Image.open(img_path) as img:
        width, height = img.size
        total_width += width
        total_height += height

# 평균 계산
average_width = total_width / 100
average_height = total_height / 100

# 평균 출력
print(f"Average Width: {average_width:.2f}")
print(f"Average Height: {average_height:.2f}")
