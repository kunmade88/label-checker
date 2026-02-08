import streamlit as st
import cv2
import numpy as np
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문구 변경 정밀 분석 리포트 test 용훈")

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

def get_highlighted_text(ocr_data, rects):
    lines_found = []
    current_line = []
    last_y = -1
    
    for i in range(len(ocr_data['text'])):
        txt = ocr_data['text'][i].strip()
        if not txt or int(ocr_data['conf'][i]) < 45: continue
        
        tx, ty, tw, th = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        
        # 하이라이트 사각형 영역 내에 텍스트가 있는지 확인
        is_in_highlight = any(rx <= tx <= rx+rw and ry <= ty <= ry+rh for (rx, ry, rw, rh) in rects)
        
        if is_in_highlight:
            if last_y != -1 and abs(ty - last_y) > 15: # 줄바꿈 감지
                lines_found.append(" ".join(current_line))
                current_line = []
            current_line.append(txt)
            last_y = ty
            
    if current_line: lines_found.append(" ".join(current_line))
    return lines_found

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('이미지 및 텍스트 데이터 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                # 1. 시각적 픽셀 차이 추출 (OpenCV)
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
                        roi = overlay[y:y+bh, x:x+bw]
                        red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                        overlay[y:y+bh, x:x+bw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)

                # 시각화 출력
                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="수정 전 (원본)", use_container_width=True)
                with col2: st.image(overlay, caption="변경 감지 하이라이트 (시각적 차이)", use_container_width=True)

                # 2. 하이라이트 영역 내 텍스트 추출
                lines1 = get_highlighted_text(data1, highlight_rects)
                lines2 = get_highlighted_text(data2, highlight_rects)

                # 3. 상세 비교 알고리즘 적용
                changes = []
                max_len = max(len(lines1), len(lines2))
                
                for i in range(max_len):
                    l1 = lines1[i] if i < len(lines1) else ""
                    l2 = lines2[i] if i < len(lines2) else ""
                    
                    if not l1 and l2:
                        changes.append({"상태": "✅ 추가됨", "내용": l2, "비고": "새로운 문구 삽입"})
                    elif l1 and not l2:
                        changes.append({"상태": "❌ 삭제됨", "내용": l1, "비고": "기존 문구 삭제"})
                    elif l1 != l2:
                        # 공백 제거 후 내용 동일 여부 확인 (자간 차이 대응)
                        if l1.replace(" ", "") == l2.replace(" ", ""):
                            changes.append({
                                "상태": "⚠️ 스타일 변경", 
                                "내용": l2, 
                                "비고": "텍스트 동일 / 자간 및 공백 차이"
                            })
                        else:
                            similarity = SequenceMatcher(None, l1, l2).ratio()
                            remarks = "단어 일부 수정" if similarity > 0.8 else "내용 변경"
                            changes.append({
                                "상태": "🔄 내용 수정", 
                                "내용": f"{l1} ➔ {l2}", 
                                "비고": remarks
                            })

                # 4. 결과 리포트 출력
                st.subheader("📋 실질적 변경 내용 정밀 분석 결과")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("이미지 상의 미세한 픽셀 차이는 있으나, 텍스트 내용은 동일합니다.")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")