import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

st.set_page_config(page_title="라벨 체크 AI", layout="wide")
st.title("🔍 전성분 문안 확인 (모든 차이점 음영 표기)")

def clean_text(text):
    # 공백과 특수문자만 제거하여 글자 알맹이만 비교
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text)

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)

uploaded_files = st.file_uploader("파일 2개 선택 (1:수정전, 2:수정후)", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    if st.button("🚀 분석 시작"):
        try:
            img1, data1 = get_data_from_upload(uploaded_files[0])
            img2, data2 = get_data_from_upload(uploaded_files[1])
            overlay = img2.copy()
            changes = []

            # 이미지 1과 2의 모든 유효 단어를 순서대로 리스트화
            list1_clean = [clean_text(t) for t in data1['text'] if t.strip()]
            list2_all = [(i, clean_text(t), t) for i, t in enumerate(data2['text']) if t.strip() and int(data2['conf'][i]) >= 40]

            # 1:1로 엄격하게 대조 (순서/내용 하나라도 다르면 음영)
            for j, (ocr_idx, txt2_clean, txt2_raw) in enumerate(list2_all):
                is_different = False
                
                # 원본보다 단어가 많아졌거나, 해당 순서의 단어가 일치하지 않으면 무조건 표시
                if j >= len(list1_clean) or list1_clean[j] != txt2_clean:
                    is_different = True

                if is_different:
                    x, y, w, h = data2['left'][ocr_idx], data2['top'][ocr_idx], data2['width'][ocr_idx], data2['height'][ocr_idx]
                    roi = overlay[y:y+h, x:x+w]
                    red = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                    overlay[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.7, red, 0.3, 0)
                    
                    orig_txt = list1_clean[j] if j < len(list1_clean) else "(없음)"
                    changes.append({"순서": j + 1, "원본 문구": orig_txt, "수정본 문구": txt2_raw})

            col1, col2 = st.columns(2)
            with col1: st.image(img1, caption="[전] 이미지", use_container_width=True)
            with col2: st.image(overlay, caption="[후] 차이점 음영 표기 완료", use_container_width=True)
            
            if changes:
                st.subheader("📋 변경 리스트")
                st.table(pd.DataFrame(changes))
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")