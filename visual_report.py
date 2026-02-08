import cv2
import numpy as np
import os
import glob
import base64
from pdf2image import convert_from_path

# 1. 파일에서 이미지를 가져오는 함수
def get_image_from_file(file_path):
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
        img = np.array(pages[0]) # 첫 페이지만 사용
    else:
        img = cv2.imread(file_path)
    
    # 색상 체계 변환 (RGB -> BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def run_visual_html_compare():
    # 2. 파일 목록 정렬 (시간순)
    files = glob.glob("*.pdf") + glob.glob("*.jpg") + glob.glob("*.png")
    if len(files) < 2:
        print("에러: 비교할 파일이 2개 필요합니다.")
        return

    files.sort(key=os.path.getmtime)
    file1, file2 = files[0], files[1]

    print(f"📸 이미지 비교 중: {file1} vs {file2}")

    img1 = get_image_from_file(file1)
    img2 = get_image_from_file(file2)

    # 3. 두 이미지 크기 통일
    height, width = img2.shape[:2]
    img1 = cv2.resize(img1, (width, height))

    # 4. 차이점 계산 및 빨간 박스 그리기
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img2.copy()
    diff_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 50: # 미세한 노이즈 무시
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3) # 빨간 박스
            diff_count += 1

    # 5. 이미지를 HTML에 포함하기 위해 텍스트(base64)로 변환
    _, buffer = cv2.imencode('.jpg', result_img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    # 6. HTML 리포트 내용 구성
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Apple SD Gothic Neo', sans-serif; padding: 40px; background: #f0f2f5; text-align: center; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            h2 {{ color: #333; margin-bottom: 5px; }}
            .info {{ margin-bottom: 25px; color: #666; font-size: 0.9em; }}
            .result-img {{ max-width: 100%; border: 3px solid #ff4d4d; border-radius: 8px; }}
            .badge {{ display: inline-block; padding: 8px 20px; background: #ff4d4d; color: white; border-radius: 25px; font-weight: bold; margin-top: 20px; }}
            .footer {{ margin-top: 30px; font-size: 0.85em; color: #999; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📸 이미지 시각적 비교 리포트</h2>
            <div class="info">기준: {file1} ➡️ <b>비교대상: {file2}</b></div>
            
            <img class="result-img" src="data:image/jpeg;base64,{img_str}">
            
            <br>
            <div class="badge">검출된 변경 지점: {diff_count}곳</div>
            
            <div class="footer">
                ※ 빨간색 박스는 이전 파일 대비 픽셀 변화(글자 수정, 위치 이동 등)가 감지된 구역입니다.
            </div>
        </div>
    </body>
    </html>
    """

    # 7. HTML 파일 저장
    with open("visual_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 완료! 'visual_report.html' 파일을 확인하세요.")

if __name__ == "__main__":
    run_visual_html_compare()