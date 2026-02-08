import cv2
import numpy as np
import os
import glob
import base64
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# 1. 테서랙트 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def get_data(file_path):
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
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

    height, width = img2.shape[:2]
    img1 = cv2.resize(img1, (width, height))

    # 이미지 차이 및 투명 형광펜 처리
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img2.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

    alpha = 0.25  # 형광펜 투명도
    img2_highlighted = cv2.addWeighted(overlay, alpha, img2, 1 - alpha, 0)

    def to_base64(img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    img1_str = to_base64(img1)
    img2_str = to_base64(img2_highlighted)

    # 텍스트 비교 로직 개선 (표 형식 생성)
    list1, list2 = text1.split(), text2.split()
    d = difflib.Differ()
    diff_result = list(d.compare(list1, list2))
    
    table_rows = ""
    change_idx = 1
    i = 0
    while i < len(diff_result):
        if i + 1 < len(diff_result) and diff_result[i].startswith('- ') and diff_result[i+1].startswith('+ '):
            # '수정'인 경우
            table_rows += f"<tr><td>{change_idx}</td><td class='del-cell'>{diff_result[i][2:]}</td><td class='add-cell'>{diff_result[i+1][2:]}</td><td>내용 수정</td></tr>"
            change_idx += 1
            i += 2
        elif diff_result[i].startswith('- '):
            # '삭제'인 경우
            table_rows += f"<tr><td>{change_idx}</td><td class='del-cell'>{diff_result[i][2:]}</td><td class='empty-cell'>-</td><td>항목 삭제</td></tr>"
            change_idx += 1
            i += 1
        elif diff_result[i].startswith('+ '):
            # '추가'인 경우
            table_rows += f"<tr><td>{change_idx}</td><td class='empty-cell'>-</td><td class='add-cell'>{diff_result[i][2:]}</td><td>항목 추가</td></tr>"
            change_idx += 1
            i += 1
        else:
            i += 1

    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Apple SD Gothic Neo', sans-serif; padding: 30px; background: #f4f7f9; color: #333; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 25px; }}
            .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 30px; }}
            .compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .img-label {{ font-weight: bold; margin-bottom: 10px; display: block; color: #555; }}
            img {{ width: 100%; border: 1px solid #eee; border-radius: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background: #f8f9fa; color: #666; font-size: 0.9em; padding: 12px; border-bottom: 2px solid #dee2e6; }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: center; font-size: 0.95em; }}
            .del-cell {{ color: #d73a49; text-decoration: line-through; background: #fff5f5; }}
            .add-cell {{ color: #28a745; font-weight: bold; background: #f0fff4; }}
            .empty-cell {{ color: #ccc; font-style: italic; }}
            .status-badge {{ font-size: 0.8em; padding: 3px 8px; border-radius: 10px; background: #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin:0;">🧪 전성분 변경 내역 정밀 리포트</h1>
                <p style="color:#888;">대상 파일: <b>{file1}</b> ➡️ <b>{file2}</b></p>
            </div>

            <div class="card">
                <h3 style="margin-top:0;">📸 시각적 비교 (Side-by-Side)</h3>
                <div class="compare-grid">
                    <div>
                        <span class="img-label">● 수정 전 (Original)</span>
                        <img src="data:image/jpeg;base64,{img1_str}">
                    </div>
                    <div>
                        <span class="img-label" style="color:#d73a49;">● 수정 후 (Highlighted)</span>
                        <img src="data:image/jpeg;base64,{img2_str}">
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top:0;">📝 텍스트 변경 상세 요약</h3>
                <table>
                    <thead>
                        <tr>
                            <th width="8%">번호</th>
                            <th width="35%">수정 전 (Before)</th>
                            <th width="35%">수정 후 (After)</th>
                            <th width="22%">변경 구분</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows if table_rows else "<tr><td colspan='4'>변경된 텍스트가 없습니다.</td></tr>"}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    with open("final_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 통합 리포트 생성 완료! 'final_report.html'을 확인하세요.")

if __name__ == "__main__":
    run_final_compare()