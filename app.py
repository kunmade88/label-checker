import streamlit as st
import cv2
import numpy as np
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd
import re

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🧪 문구 내용 중심 정밀 분석 test 용훈")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 상세 OCR 데이터 추출 (좌표 포함)
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 및 가독성 최적화 시작"):
        with st.spinner('내용이 동일한 구간의 음영을 제거하는 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                
                # 1. 시각적 차이 추출 (기초 레이어)
                gray1 = cv2.cvtColor(img1_res, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                final_changes = []

                # 2. 오탐지 제거 로직: 하이라이트 영역 내 텍스트가 다를 때만 표시
                for cnt in contours:
                    if cv2.contourArea(cnt) > 300:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        
                        # 해당 영역의 텍스트 추출 (원본 vs 수정본)
                        def get_text_in_roi(data, rx, ry, rw, rh):
                            txts = []
                            for i in range(len(data['text'])):
                                tx, ty = data['left'][i], data['top'][i]
                                if rx <= tx <= rx+rw and ry <= ty <= ry+rh:
                                    if data['text'][i].strip():
                                        txts.append(data['text'][i].strip())
                            return " ".join(txts)

                        text_old = get_text_in_roi(data1, x, y, bw, bh)
                        text_new = get_text_in_roi(data2, x, y, bw, bh)

                        # 텍스트 내용이 다를 때만 하이라이트 그리기 (오탐지 방지의 핵심)
                        if text_old != text_new and (len(text_old) > 0 or len(text_new) > 0):
                            roi = overlay[y:y+bh, x:x+bw]
                            rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                            overlay[y:y+bh, x:x+bw] = cv2.addWeighted(roi, 0.7, rect, 0.3, 0)
                            
                            if text_old != text_new:
                                final_changes.append({"기존 내용": text_old if text_old else "(없음)", 
                                                      "변경 내용": text_new if text_new else "(추가됨)"})

                st.image(np.hstack([img1_res, overlay]), caption="좌: 원본 / 우: 내용 변경 구간 하이라이트", use_container_width=True)

                st.subheader("📋 정돈된 상세 변경 리포트")
                if final_changes:
                    st.table(pd.DataFrame(final_changes))
                    st.balloons()
                else:
                    st.success("내용(텍스트)이 변경된 구간이 없습니다.")

            except Exception as e:
                st.error(f"오류: {e}")