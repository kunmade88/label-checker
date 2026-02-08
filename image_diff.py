import cv2
import numpy as np
import os
import glob
from pdf2image import convert_from_path

def get_image_from_file(file_path):
    """PDF 또는 이미지를 OpenCV 이미지 객체로 변환"""
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
        # 첫 번째 페이지를 넘파이 배열로 변환
        img = np.array(pages[0])
    else:
        img = cv2.imread(file_path)
    
    # OpenCV에서 사용하기 위해 BGR 형식으로 변환
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def run_image_compare():
    files = glob.glob("*.pdf") + glob.glob("*.jpg") + glob.glob("*.png")
    if len(files) < 2:
        print("비교할 파일이 2개 필요합니다.")
        return

    files.sort(key=os.path.getmtime)
    file1, file2 = files[0], files[1]

    img1 = get_image_from_file(file1)
    img2 = get_image_from_file(file2)

    # 두 이미지 크기를 동일하게 맞춤 (비교를 위해 필수)
    height, width = img2.shape[:2]
    img1 = cv2.resize(img1, (width, height))

    # 1. 두 이미지의 차이 계산 (절대 차이)
    diff = cv2.absdiff(img1, img2)
    
    # 2. 차이점을 그레이스케일로 변환 후 이진화(흑백 처리)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 3. 차이가 있는 부분에 외곽선(박스) 그리기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 결과 표시용 이미지 복사
    result_img = img2.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # 너무 작은 노이즈는 무시
            x, y, w, h = cv2.boundingRect(contour)
            # 수정된 부분에 빨간색 사각형 그리기
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 4. 결과 저장
    cv2.imwrite("visual_diff_result.jpg", result_img)
    print(f"✅ 시각적 비교 완료! 'visual_diff_result.jpg' 확인")

if __name__ == "__main__":
    run_image_compare()