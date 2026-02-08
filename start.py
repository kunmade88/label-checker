import pytesseract
from PIL import Image
import difflib
import os
import glob
from pdf2image import convert_from_path

# 맥북 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def get_text_from_file(file_path):
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
        text = pytesseract.image_to_string(pages[0], lang='kor+eng')
    else:
        text = pytesseract.image_to_string(Image.open(file_path), lang='kor+eng')
    return text

def run_compare():
    pdf_files = glob.glob("*.pdf")
    if len(pdf_files) < 2:
        print("에러: PDF 파일이 2개 필요합니다.")
        return

    # 파일 생성 시간순으로 정렬 (먼저 넣은 것이 before)
    pdf_files.sort(key=os.path.getmtime)
    file1, file2 = pdf_files[0], pdf_files[1]

    text1 = get_text_from_file(file1)
    text2 = get_text_from_file(file2)

    # 성분표를 띄어쓰기 단위로 쪼개서 단어별로 비교합니다. (들여쓰기 수정됨)
    list1 = text1.split() 
    list2 = text2.split()

    # 차이점 분석
    d = difflib.Differ()
    diff = list(d.compare(list1, list2))

    # HTML 리포트 생성
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: sans-serif; line-height: 1.8; padding: 40px; background: #f4f7f6; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .summary {{ margin-bottom: 20px; font-weight: bold; color: #666; }}
            .ingredient {{ display: inline-block; padding: 4px 8px; margin: 4px; border-radius: 4px; border: 1px solid #ddd; }}
            .added {{ background-color: #e6ffed; border-color: #34d058; color: #22863a; font-weight: bold; }}
            .deleted {{ background-color: #ffeef0; border-color: #f97583; color: #cb2431; text-decoration: line-through; }}
            .info {{ font-size: 0.8em; color: #999; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🧪 전성분 비교 리포트</h2>
            <div class="summary">비교 파일: {file1} ➡️ {file2}</div>
            <div class="content">
    """

    added_count = 0
    deleted_count = 0

    for item in diff:
        word = item[2:]
        if item.startswith('+ '):
            html_content += f'<span class="ingredient added">➕ {word}</span> '
            added_count += 1
        elif item.startswith('- '):
            html_content += f'<span class="ingredient deleted">➖ {word}</span> '
            deleted_count += 1
        elif item.startswith('  '):
            html_content += f'<span class="ingredient">{word}</span> '

    html_content += f"""
            </div>
            <div class="info">
                총 {added_count}개 추가됨 / {deleted_count}개 삭제됨
            </div>
        </div>
    </body>
    </html>
    """

    with open("result.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ 분석 완료! [기존] {file1} -> [수정] {file2}")
    print(f"결과: 추가 {added_count}개, 삭제 {deleted_count}개")
    print("result.html을 다시 확인해보세요.")

if __name__ == "__main__":
    run_compare()