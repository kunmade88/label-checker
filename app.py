import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="전성분 문안 확인용 테스트 용훈", layout="wide")

# --- 유틸리티 함수 ---
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

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

# --- 메인 UI ---
st.title("🔍 전성분 문안 정밀 확인 시스템")
mode = st.sidebar.radio("작업 모드 선택", ["Excel vs PDF (성분 순서 검증)", "PDF vs PDF (시각적 차이)"])

if mode == "Excel vs PDF (성분 순서 검증)":
    st.subheader("📊 엑셀 데이터 vs PDF 실물 대조")
    
    lang_choice = st.radio("검증 기준 언어", ["영문명", "한글명"], horizontal=True)
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        excel_file = st.file_uploader("표준 전성분 엑셀 업로드", type=['xlsx', 'xls', 'csv'])
    with col_u2:
        pdf_file = st.file_uploader("검토할 이미지/PDF 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        if st.button("🚀 데이터 추출 및 시각적 대조 시작"):
            try:
                # 1. 엑셀 파싱 (가변 헤더 대응)
                df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
                header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values or "영문명" in row.values), 0)
                df_clean = pd.read_excel(excel_file, skiprows=header_idx + 1)
                
                standard_list = df_clean[lang_choice].dropna().astype(str).tolist()

                # 2. PDF OCR 수행
                img, ocr_data = get_data_from_upload(pdf_file)
                
                # 3. 화면 분할 레이아웃 (이미지 비교 섹션)
                st.write("### 🖼️ 시각적 비교 영역")
                view_col1, view_col2 = st.columns(2)
                
                # 왼쪽: 엑셀 데이터를 텍스트로 시각화 (이미지처럼 출력)
                with view_col1:
                    st.info("📂 엑셀 기준 리스트")
                    excel_text_display = "\n".join([f"{i+1}. {name}" for i, name in enumerate(standard_list)])
                    st.text_area("Excel Data Extract", excel_text_display, height=400)

                # 오른쪽: PDF에서 글자가 있는 구역 표시
                with view_col2:
                    st.info("📸 PDF 추출 구역 (글자 감지)")
                    overlay = img.copy()
                    for i in range(len(ocr_data['text'])):
                        if int(ocr_data['conf'][i]) > 30:
                            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    st.image(overlay, use_container_width=True)

                # 4. 정밀 대조 테이블 결과
                st.write("---")
                st.write("### 📋 항목별 대조 리포트")
                
                # (기존 대조 로직 수행...)
                full_text = " ".join([t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 30])
                extracted_list = [t.strip() for t in full_text.split(',') if t.strip()]
                
                comparison = []
                max_len = max(len(standard_list), len(extracted_list))
                for i in range(max_len):
                    std = standard_list[i] if i < len(standard_list) else "-"
                    ext = extracted_list[i] if i < len(extracted_list) else "-"
                    ratio = SequenceMatcher(None, clean_text(std), clean_text(ext)).ratio()
                    
                    status = "✅ 일치" if ratio == 1.0 else "🔍 오타" if ratio > 0.7 else "❌ 불일치"
                    comparison.append({"순번": i+1, "엑셀(표준)": std, "이미지(OCR)": ext, "상태": status})
                
                st.table(pd.DataFrame(comparison))

            except Exception as e:
                st.error(f"오류 발생: {e}")