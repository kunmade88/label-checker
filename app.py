import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

st.set_page_config(page_title="라벨 체크 AI", layout="wide")
st.title("🔍 전성분 정밀 분석 (모든 변경사항 음영 표기)")

def clean_text(text):
    # 특수문자나 공백 차이로 인한 오탐지를 줄이기 위해 알맹이만 추출
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

uploaded_files = st.file_uploader("파일 2개 선택 (순서대로 전/후)", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    if st.button("🚀 분석 시작"):
        img1, data1 = get_data_from_upload(uploaded_files[0])
        img2, data2 = get_data_from_upload(uploaded_files[1])
        overlay = img2.copy()
        changes = []

        # 이미지 1과 2의 유효 단어 리스트를 순서대로 생성
        list1_clean = [clean_text(t) for t in data1['text'] if t.strip()]
        # 이미지 2는 좌표값(i)을 같이 저장
        list2_info = [(i, clean_text(t), t) for i, t in enumerate(data2['text']) if t.strip() and int(data2['conf'][i]) >= 40]

        # 이미지 2의 단어를 하나씩 꺼내어 이미지 1의 같은 순서와 대조
        for j, (ocr_idx, txt2_clean, txt2_raw) in enumerate(list2_info):
            is_mismatch = False
            
            # 1. 원본보다 순서가 길어지거나
            # 2. 같은 순서(j번째)의 글자가 서로 다르면 무조건 음영 표기
            if j >= len(list1_clean) or list1_clean[j] != txt2_clean:
                is_mismatch = True

            if is_mismatch:
                x, y, w, h = data2['left'][ocr_idx], data2['top'][ocr_idx], data2['width'][ocr_idx], data2['height'][ocr_idx]
                roi = overlay[y:y+h, x:x+w]
                red = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                overlay[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.7, red, 0.3, 0)
                
                orig_txt = list1_clean[j] if j < len(list1_clean) else "없음"
                changes.append({"순서": j + 1, "원본": orig_txt, "수정본": txt2_raw})

        c1, c2 = st.columns(2)
        c1.image(img1, caption="[전] 이미지", use_container_width=True)
        c2.image(overlay, caption="[후] 모든 변경사항 하이라이트", use_container_width=True)
        if changes: st.table(pd.DataFrame(changes))