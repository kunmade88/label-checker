import pytesseract
from PIL import Image
import difflib
import os
import glob
from pdf2image import convert_from_path

# 1. 맥북 테서랙트 경로 설정 (이전과 동일)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def get_text_from_file(file_path):
    """파일에서 글자를 추출하는 함수"""
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
        # 첫 페이지만 텍스트 추출
        text = pytesseract.image_to_string(pages[0], lang='kor+eng')
    else:
        text = pytesseract.image_to_string(Image.open(file_path), lang='kor+eng')
    return text

def run_highlight_compare():
    # 2. 파일 목록 가져오기 및 시간순 정렬 (오래된 게 1차, 최신이 2차)
    files = glob.glob("*.pdf") + glob.glob("*.jpg") + glob.glob("*.png")
    if len(files) < 2:
        print("❌ 에러: 폴더에 비교할 파일이 2개 이상 필요합니다.")
        return

    files.sort(key=os.path.getmtime)
    file1, file2 = files[0], files[1]

    print(f"🔍 형광펜 모드: [1차] {file1} 대비 [2차] {file2}의 변경사항 분석 중...")

    # 3. 텍스트 추출 및 단어 단위 분리
    text1 = get_text_from_file(file1)
    text2 = get_text_from_file(file2)

    list1 = text1.split()
    list2 = text2.split()

    # 4. 차이점 분석
    d = difflib.Differ()
    diff = list(d.compare(list1, list2))

    # 5. HTML 리포트 생성 (수정 후 파일 기준)
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Apple SD Gothic Neo', sans-serif; line-height: 2.3; padding: 40px; background: #ffffff; color: #333; }}
            .container {{ max-width: 900px; margin: auto; border: 1px solid #ddd; padding: 50px; border-radius: 4px; box-shadow: 0 0 10px rgba(0,0,0,0.05); }}
            h2 {{ text-align: center; color: #000; margin-bottom: 10px; border-bottom: 2px solid #000; padding-bottom: 15px; }}
            .file-info {{ font-size: 0.85em; color: #666; text-align: center; margin-bottom: 30px; }}
            .highlight {{ background-color: #ffcccc; border-bottom: 2px solid #ff4d4d; font-weight: bold; padding: 2px 0; }}
            .legend {{ background: #f9f9f9; padding: 15px; border-radius: 5px; font-size: 0.9em; margin-bottom: 30px; border-left: 5px solid #ff4d4d; }}
            .content {{ text-align: justify; word-break: break-all; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔎 수정본 변경사항 강조 리포트</h2>
            <div class="file-info">비교 기준: {file1} → <b>강조 대상: {file2}</b></div>
            <div class="legend">
                <b>💡 확인 방법:</b> 아래 텍스트는 <b>수정 후 파일({file2})</b>의 전체 내용입니다. <br>
                그중에서 이전 파일과 비교하여 <b>새로 추가되거나 바뀐 단어</b>만 <span class="highlight">붉은 형광펜</span>으로 표시했습니다.
            </div>
            <div class="content">
    """

    # 6. 수정 후(list2)를 기준으로 변경된 단어만 형광펜 칠하기
    for item in diff:
        word = item[2:]
        if item.startswith('+ '):
            # 새로 추가되거나 변경된 단어만 강조
            html_content += f'<span class="highlight">{word}</span> '
        elif item.startswith('  '):
            # 변동 없는 단어는 그대로 출력
            html_content += f'<span>{word}</span> '
        # '- '(삭제)는 수정 후 파일 기준이므로 여기서는 무시함

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    # 7. 파일 저장
    output_name = "highlight_report.html"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ 완료! '{output_name}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_highlight_compare()