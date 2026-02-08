import cv2
import numpy as np
import os
import glob
import base64
import difflib
import pytesseract
import platform
from PIL import Image
from pdf2image import convert_from_path

# 1. 운영체제 확인 및 경로 자동 설정
current_os = platform.system()

if current_os == "Windows":
    # 윈도우용 설정 (나중에 윈도우에서 테서랙트 설치 후 경로 확인)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r'C:\poppler\bin' 
else:
    # 맥북용 설정
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
    POPPLER_PATH = None

def get_data(file_path):
    if file_path.lower().endswith('.pdf'):
        # 윈도우 대응을 위해 poppler_path 추가
        pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        img = np.array(pages[0])
        text = pytesseract.image_to_string(pages[0], lang='kor+eng')
    else:
        img = cv2.imread(file_path)
        text = pytesseract.image_to_string(Image.open(file_path), lang='kor+eng')
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, text

def run_final_compare():
    files = glob.glob("*.pdf") + glob.glob("*.jpg") + glob.glob("*.png")
    if len(files) < 2:
        print("에러: 비교할 파일이 2개 필요합니다.")
        return

    files.sort(key=os.path.getmtime)
    file1, file2 = files[0], files[1]

    img1, text1 = get_data(file1)
    img2, text2 = get_data(file2)

    # 이미지 크기 맞춤 및 하이라이트 처리
    height, width = img2.shape[:2]
    img1 = cv2.resize(img1, (width, height))
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img2.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

    img2_highlighted = cv2.addWeighted(overlay, 0.25, img2, 0.75, 0)

    def to_base64(img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    img1_str = to_base64(img1)
    img2_str = to_base64(img2_highlighted)

    # 텍스트 비교 (표 형식)
    list1, list2 = text1.split(), text2.split()
    d = difflib.Differ()
    diff_result = list(d.compare(list1, list2))
    
    table_rows = ""
    change_idx = 1
    i = 0
    while i < len(diff_result):
        if i + 1 < len(diff_result) and diff_result[i].startswith('- ') and diff_result[i+1].startswith('+ '):
            table_rows += f"<tr><td>{change_idx}</td><td class='del-cell'>{diff_result[i][2:]}</td><td class='add-cell'>{diff_result[i+1][2:]}</td><td>내용 수정</td></tr>"
            change_idx += 2; i += 2
        elif diff_result[i].startswith('- '):
            table_rows += f"<tr><td>{change_idx}</td><td class='del-cell'>{diff_result[i][2:]}</td><td class='empty-cell'>-</td><td>항목 삭제</td></tr>"
            change_idx += 1; i += 1
        elif diff_result[i].startswith('+ '):
            table_rows += f"<tr><td>{change_idx}</td><td class='empty-cell'>-</td><td class='add-cell'>{diff_result[i][2:]}</td><td>항목 추가</td></tr>"
            change_idx += 1; i += 1
        else: i += 1

    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #f4f7f9; }}
            .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .compare-grid {{ display: flex; gap: 20px; }}
            .compare-grid div {{ flex: 1; }}
            img {{ width: 100%; border: 1px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 10px; border: 1px solid #eee; text-align: center; }}
            .del-cell {{ background: #fff5f5; color: red; text-decoration: line-through; }}
            .add-cell {{ background: #f0fff4; color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>🧪 전성분 변경 내역 정밀 리포트</h2>
        <div class="card">
            <h3>📸 시각적 비교</h3>
            <div class="compare-grid">
                <div><p>수정 전 ({file1})</p><img src="data:image/jpeg;base64,{img1_str}"></div>
                <div><p>수정 후 ({file2})</p><img src="data:image/jpeg;base64,{img2_str}"></div>
            </div>
        </div>
        <div class="card">
            <h3>📝 텍스트 변경 상세</h3>
            <table>
                <thead><tr><th>번호</th><th>기존</th><th>변경</th><th>구분</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </body>
    </html>
    """
    with open("final_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 리포트 생성 완료!")

if __name__ == "__main__":
    run_final_compare()