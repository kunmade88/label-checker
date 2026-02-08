import streamlit as st
import cv2
import numpy as np
import base64
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import io
import pandas as pd

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🧪 전성분 변경 내역 정밀 분석")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_path(io.BytesIO(file_bytes))
        img = np.array(pages[0])
        text = pytesseract.image_to_string(pages[0], lang='kor+eng')
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), lang='kor+eng')
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    return img_bgr, text

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요 (PDF, JPG, PNG)", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    file1, file2 = uploaded_files[0], uploaded_files[1]
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('이미지 대조 및 OCR 분석 중...'):
            try:
                img1, text1 = get_data_from_upload(file1)
                img2, text2 = get_data_from_upload(file2)

                height, width = img2.shape[:2]
                img1_resized = cv2.resize(img1, (width, height))
                diff = cv2.absdiff(img1_resized, img2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                for contour in contours:
                    if cv2.contourArea(contour) > 50:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

                img2_highlighted = cv2.addWeighted(overlay, 0.25, img2, 0.75, 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB), caption=f"수정 전: {file1.name}")
                with col2:
                    st.image(cv2.cvtColor(img2_highlighted, cv2.COLOR_BGR2RGB), caption=f"수정 후 (변경점 하이라이트)")

                list1, list2 = text1.split(), text2.split()
                d = difflib.Differ()
                diff_result = list(d.compare(list1, list2))
                
                changes = []
                i = 0
                while i < len(diff_result):
                    if i + 1 < len(diff_result) and diff_result[i].startswith('- ') and diff_result[i+1].startswith('+ '):
                        changes.append({"구분": "내용 수정", "기존": diff_result[i][2:], "변경": diff_result[i+1][2:]})
                        i += 2
                    elif diff_result[i].startswith('- '):
                        changes.append({"구분": "항목 삭제", "기존": diff_result[i][2:], "변경": "-"})
                        i += 1
                    elif diff_result[i].startswith('+ '):
                        changes.append({"구분": "항목 추가", "기존": "-", "변경": diff_result[i][2:]})
                        i += 1
                    else: i += 1

                st.subheader("📝 상세 변경 내역")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("텍스트에서 변경된 내용이 없습니다.")
            except Exception as e:
                st.error(f"오류 발생: {e}")
else:
    st.info("파일을 2개 업로드하면 분석 버튼이 나타납니다.")