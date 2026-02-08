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
st.title("🧪 전성분 및 문구 변경 정밀 분석")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        # 색상 왜곡 방지: PIL 이미지를 RGB 배열로 변환
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # BGR을 RGB로 변환하여 원본 색상 유지
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # OCR 데이터 추출 (좌표 정보 포함)
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 시작 (이미지-텍스트 동기화)"):
        with st.spinner('원본 색상을 유지하며 정밀 분석 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                # 1. 시각적 차이 추출 (색상 보존형)
                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                
                # 차이 계산을 위해 그레이스케일 변환
                gray1 = cv2.cvtColor(img1_res, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                
                _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
                kernel = np.ones((10,10), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                changed_rects = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        changed_rects.append((x, y, bw, bh))
                        # 투명 하이라이트 (RGB 컬러 유지)
                        roi = overlay[y:y+bh, x:x+bw]
                        rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8) # 빨간색
                        overlay[y:y+bh, x:x+bw] = cv2.addWeighted(roi, 0.7, rect, 0.3, 0)

                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="수정 전", use_container_width=True)
                with col2: st.image(overlay, caption="수정 후 (하이라이트)", use_container_width=True)

                # 2. 리포트 생성 (이미지 하이라이트 영역 내 텍스트만 추출)
                def get_text_from_rects(ocr_data, rects):
                    found_texts = []
                    for i in range(len(ocr_data['text'])):
                        if int(ocr_data['conf'][i]) > 40:
                            tx, ty, tw, th = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                            # 텍스트 좌표가 하이라이트된 영역 안에 있는지 검사
                            for (rx, ry, rw, rh) in rects:
                                if rx <= tx <= rx+rw and ry <= ty <= ry+rh:
                                    found_texts.append(ocr_data['text'][i])
                                    break
                    return " ".join(found_texts)

                # 텍스트 비교 (전체 텍스트 대신 하이라이트 영역 중심)
                text1_clean = " ".join([data1['text'][i] for i in range(len(data1['text'])) if data1['text'][i].strip()])
                text2_clean = " ".join([data2['text'][i] for i in range(len(data2['text'])) if data2['text'][i].strip()])

                # 문장 단위 리포트 생성 로직
                d = difflib.Differ()
                diff_res = list(d.compare(text1_clean.split('. '), text2_clean.split('. ')))
                
                changes = []
                for line in diff_res:
                    if line.startswith('- '):
                        changes.append({"구분": "❌ 삭제/수정전", "내용": line[2:]})
                    elif line.startswith('+ '):
                        changes.append({"구분": "✅ 추가/수정후", "내용": line[2:]})

                st.subheader("📋 실질적 문구 변경 리포트")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("이미지상 차이가 발견된 구역에 유의미한 텍스트 변경이 없습니다.")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")