import streamlit as st
import cv2
import numpy as np
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
import pandas as pd

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🧪 전성분 및 문구 변경 내역 정밀 분석_made 용훈")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0])
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # OCR 추출 (문장 단위를 위해 명확하게 추출)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    return img_bgr, text

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    file1, file2 = uploaded_files[0], uploaded_files[1]
    
    if st.button("🚀 문장 단위 정밀 분석 시작"):
        with st.spinner('문구 및 이미지 대조 중...'):
            try:
                img1, text1 = get_data_from_upload(file1)
                img2, text2 = get_data_from_upload(file2)

                # 1. 시각적 하이라이트 처리 (투명도 적용)
                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                diff = cv2.absdiff(img1_res, img2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)
                
                kernel = np.ones((10,10), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=1)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                for cnt in contours:
                    if cv2.contourArea(cnt) > 200:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        # 투명 하이라이트 효과
                        roi = overlay[y:y+bh, x:x+bw]
                        rect = np.full(roi.shape, (0, 0, 255), dtype=np.uint8) # 빨간색
                        res = cv2.addWeighted(roi, 0.7, rect, 0.3, 0) # 30% 투명도
                        overlay[y:y+bh, x:x+bw] = res

                # 이미지 출력
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(img1_res, cv2.COLOR_BGR2RGB), caption=f"수정 전 ({file1.name})", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"수정 후 하이라이트 ({file2.name})", use_container_width=True)

                # 2. 문장 단위 비교 로직 (핵심!)
                # .split() 대신 .splitlines()를 사용하여 줄/문장 단위로 비교합니다.
                lines1 = [line.strip() for line in text1.splitlines() if line.strip()]
                lines2 = [line.strip() for line in text2.strip().splitlines() if line.strip()]
                
                d = difflib.Differ()
                diff_result = list(d.compare(lines1, lines2))
                
                changes = []
                idx = 0
                while idx < len(diff_result):
                    # 문장이 수정된 경우 (기존 문장 삭제 후 새 문장 추가)
                    if idx + 1 < len(diff_result) and diff_result[idx].startswith('- ') and diff_result[idx+1].startswith('+ '):
                        changes.append({
                            "구분": "📝 문장 수정",
                            "기존 내용": diff_result[idx][2:],
                            "변경 내용": diff_result[idx+1][2:]
                        })
                        idx += 2
                    elif diff_result[idx].startswith('- '):
                        changes.append({
                            "구분": "❌ 문장 삭제",
                            "기존 내용": diff_result[idx][2:],
                            "변경 내용": "-"
                        })
                        idx += 1
                    elif diff_result[idx].startswith('+ '):
                        changes.append({
                            "구분": "✅ 문장 추가",
                            "기존 내용": "-",
                            "변경 내용": diff_result[idx][2:]
                        })
                        idx += 1
                    else:
                        idx += 1

                st.subheader("📋 상세 변경 리포트 (문장 단위)")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("발견된 문장 단위 변경 사항이 없습니다.")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")