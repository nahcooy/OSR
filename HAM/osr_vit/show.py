import os

def print_python_files(directory):
    # 디렉토리 내의 모든 파일들 반복
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # .py 파일만 선택
                file_path = os.path.join(root, file)
                print(f"파일 이름: {file_path}")
                
                # 파일 내용 출력
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    print(file_content)
                
                # 구분선
                print("\n~~~~~~~~~~\n")

# 사용 예시
directory = '/nahcooy/OSR/HAM/osr_vit'  # 여기서 경로를 원하는 디렉토리로 바꿔주세요.
print_python_files(directory)
