import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('/dataset/nahcooy/CXR8/Data_Entry_2017_v2020.csv')

# 'Finding Labels'의 고유한 값과 각 값의 개수 출력
label_counts = df['Finding Labels'].value_counts()

# 각 라벨의 비율 계산
label_percentages = df['Finding Labels'].value_counts(normalize=True) * 100

# 결과 출력
print("Unique Finding Labels, their counts, and percentages:")
for label, count, percentage in zip(label_counts.index, label_counts.values, label_percentages.values):
    print(f"{label}: Count = {count}, Percentage = {percentage:.2f}%")
