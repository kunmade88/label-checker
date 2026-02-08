import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문안 정밀 확인 용훈테스트중")

def clean_text(text):
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
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

def get_all_texts(ocr_data):
    # 신뢰도 40 이상의 모든 텍스트를 순서대로 리스트화
    return [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 40]

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('이미지 및 순서 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                # 1. 픽셀 차이 감지 (사용자님이 선호하는 방식)
                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                gray1 = cv2.cvtColor(img1_res, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                
                # 2. 텍스트 순서 및 오타 정밀 대조
                lines1 = get_all_texts(data1)
                lines2 = get_all_texts(data2)
                
                overlay = img2.copy()
                changes = []
                
                # 순차적 1:1 대조 (순서가 틀리면 여기서 걸림)
                max_len = max(len(lines1), len(lines2))
                for i in range(max_len):
                    l1 = lines1[i] if i < len(lines1) else " (항목 없음)"
                    l2 = lines2[i] if i < len(lines2) else " (항목 없음)"
                    
                    if clean_text(l1) != clean_text(l2):
                        changes.append({
                            "순서": i + 1,
                            "원본(전)": l1,
                            "수정본(후)": l2,
                            "상태": "❌ 불일치/순서오류"
                        })

                # 이미지 위에 빨간색 음영 표시 (픽셀 차이 구역)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 300:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        roi = overlay[y:y+bh, x:x+bw]
                        red = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                        overlay[y:y+bh, x:x+bw] = cv2.addWeighted(roi, 0.7, red, 0.3, 0)

                # 출력
                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="원본(수정 전)")
                with col2: st.image(overlay, caption="변경 감지(빨간 음영)")
                
                st.subheader("📋 정밀 대조 리포트")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.error("순서 혹은 내용이 일치하지 않는 구간이 발견되었습니다.")
                else:
                    st.success("모든 문구의 순서와 내용이 일치합니다.")

            except Exception as e:
                st.error(f"분석 오류: {e}")