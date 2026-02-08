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
st.title("🔍 전성분 문구 변경 정밀 분석 test 용훈")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB')) # 원본 색상 유지
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 원본 색상 유지
    
    # OCR 데이터 및 좌표 추출
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 및 리포트 정돈"):
        with st.spinner('이미지 색상을 유지하며 가독성 리포트를 생성 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                # 1. 시각적 차이 추출
                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                gray1 = cv2.cvtColor(img1_res, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                
                _, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
                kernel = np.ones((12,12), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=1)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                highlight_rects = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 600:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        highlight_rects.append((x, y, bw, bh))
                        # 투명 빨간색 하이라이트
                        roi = overlay[y:y+bh, x:x+bw]
                        red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                        overlay[y:y+bh, x:x+bw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)

                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="수정 전 원본", use_container_width=True)
                with col2: st.image(overlay, caption="수정 후 하이라이트", use_container_width=True)

                # 2. 리포트 생성 (하이라이트 영역과 일치하는 텍스트만)
                def get_highlighted_text(ocr_data, rects):
                    lines_found = []
                    current_line = []
                    last_y = -1
                    
                    for i in range(len(ocr_data['text'])):
                        txt = ocr_data['text'][i].strip()
                        if not txt or int(ocr_data['conf'][i]) < 45: continue
                        
                        tx, ty, tw, th = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                        
                        is_in_highlight = any(rx <= tx <= rx+rw and ry <= ty <= ry+rh for (rx, ry, rw, rh) in rects)
                        
                        if is_in_highlight:
                            # 줄바꿈 감지 (Y좌표 차이 이용)
                            if last_y != -1 and abs(ty - last_y) > 15:
                                lines_found.append(" ".join(current_line))
                                current_line = []
                            current_line.append(txt)
                            last_y = ty
                            
                    if current_line: lines_found.append(" ".join(current_line))
                    return lines_found

                lines1 = get_highlighted_text(data1, highlight_rects) # 기존 이미지에서도 해당 영역 텍스트 추출
                lines2 = get_highlighted_text(data2, highlight_rects)

                # 문장 단위 대조
                d = difflib.Differ()
                diff_res = list(d.compare(lines1, lines2))
                
                changes = []
                for line in diff_res:
                    if line.startswith('- '):
                        changes.append({"상태": "❌ 삭제됨", "내용": line[2:]})
                    elif line.startswith('+ '):
                        changes.append({"상태": "✅ 변경/추가됨", "내용": line[2:]})

                st.subheader("📋 변경사항 입니다.")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("이미지상 하이라이트된 영역에 텍스트 변경이 없습니다. (디자인적 차이일 수 있습니다)")

            except Exception as e:
                st.error(f"오류: {e}")