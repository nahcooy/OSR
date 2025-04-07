import os

def save_all_python_files_to_txt(directory, output_txt_file):
    # .py 파일들의 내용을 하나의 txt 파일로 합침
    with open(output_txt_file, 'w', encoding='utf-8') as output_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):  # .py 파일만 선택
                    file_path = os.path.abspath(os.path.join(root, file))  # 절대 경로로 변환
                    print(f"파일 이름: {file_path}")
                    
                    # 각 파일의 내용 읽어서 output_txt_file에 추가
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        output_file.write(f"파일 이름: {file_path}\n")
                        output_file.write(file_content)
                        output_file.write("\n~~~~~~~~~~\n")  # 구분선 추가

    print(f"모든 Python 파일 내용이 {output_txt_file}에 저장되었습니다.")

# 사용 예시
directory = '/nahcooy/COPY_ONLY/mae'  # 탐색할 디렉토리 경로
output_txt_file = '/nahcooy/OSR/HAM/osr_vit/mae.txt'  # 결과를 저장할 txt 파일 경로
save_all_python_files_to_txt(directory, output_txt_file)
