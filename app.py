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
st.title("🧪 전성분 문구 변경 정밀 분석_made 용훈")

def normalize_text(text):
    # 불필요한 공백 및 특수기호 정리하여 내용에만 집중
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0])
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # OCR 추출
    text = pytesseract.image_to_string(img, lang='kor+eng')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    return img_bgr, text

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 내용 중심 정밀 분석 시작"):
        with st.spinner('글자 크기 등 디자인 요소를 제외하고 내용을 분석 중입니다...'):
            try:
                img1, text1 = get_data_from_upload(uploaded_files[0])
                img2, text2 = get_data_from_upload(uploaded_files[1])

                # 1. 시각적 하이라이트 (투명도 적용)
                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                diff = cv2.absdiff(img1_res, img2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY) # 감도를 높여 큰 변화만 감지
                
                kernel = np.ones((15,15), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=1)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                for cnt in contours:
                    if cv2.contourArea(cnt) > 800: # 더 큰 영역만 하이라이트
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        roi = overlay[y:y+bh, x:x+bw]
                        rect = np.full(roi.shape, (0, 0, 255), dtype=np.uint8)
                        res = cv2.addWeighted(roi, 0.7, rect, 0.3, 0)
                        overlay[y:y+bh, x:x+bw] = res

                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="원본", use_container_width=True)
                with col2: st.image(overlay, caption="변경 확인 (시각적 변화)", use_container_width=True)

                # 2. 문장 내용 비교 (유사도 필터링 강화)
                lines1 = [normalize_text(l) for l in text1.splitlines() if len(l.strip()) > 3]
                lines2 = [normalize_text(l) for l in text2.splitlines() if len(l.strip()) > 3]
                
                d = difflib.Differ()
                diff_result = list(d.compare(lines1, lines2))
                
                changes = []
                idx = 0
                while idx < len(diff_result):
                    if idx + 1 < len(diff_result) and diff_result[idx].startswith('- ') and diff_result[idx+1].startswith('+ '):
                        old_txt = diff_result[idx][2:]
                        new_txt = diff_result[idx+1][2:]
                        
                        # 내용 유사도 검사
                        similarity = difflib.SequenceMatcher(None, old_txt, new_txt).ratio()
                        
                        # 유사도가 0.99면 거의 같은 문장이므로 무시, 그 미만일 때만 표시
                        if similarity < 0.99:
                            changes.append({"구분": "📝 문장 수정", "기존 내용": old_txt, "변경 내용": new_txt})
                        idx += 2
                    elif diff_result[idx].startswith('- '):
                        changes.append({"구분": "❌ 문장 삭제", "기존 내용": diff_result[idx][2:], "변경 내용": "-"})
                        idx += 1
                    elif diff_result[idx].startswith('+ '):
                        changes.append({"구분": "✅ 문장 추가", "기존 내용": "-", "변경 내용": diff_result[idx][2:]})
                        idx += 1
                    else:
                        idx += 1

                st.subheader("📋 내용 변경 리포트 (디자인 무시)")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("문구 내용에서 변경된 사항이 없습니다.")

            except Exception as e:
                st.error(f"오류: {e}")