import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

st.set_page_config(page_title="라벨 체크 AI 통합본", layout="wide")

def get_image_and_data(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# --- 사이드바: 여기서 모드를 바꾸시면 사라진 기능이 나옵니다! ---
with st.sidebar:
    st.header("🛠️ 작업 모드 선택")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 비교)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("언어 선택", ["영문명", "한글명"])
        compare_limit = st.number_input("검증 개수", value=16)

# --- [모드 1] Excel vs PDF ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인용 테스트 용훈")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        excel_file = st.file_uploader("📂 기준 엑셀", type=['xlsx', 'csv'])
    with col_u2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        # 상단 시각화 (엑셀 표 & PDF 이미지)
        st.markdown("### 📋 업로드 데이터 실시간 확인")
        view_c1, view_c2 = st.columns(2)
        
        with view_c1:
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=350) # 엑셀 시트를 이미지처럼 확인

        with view_c2:
            img = get_image_and_data(pdf_file)
            st.image(img, width=400) # PDF 이미지를 작게 고정

        if st.button("🚀 분석 시작", use_container_width=True):
            # OCR 수행 및 '압축 텍스트' 생성 (콤마/줄바꿈 무시)
            ocr_data = pytesseract.image_to_string(img, lang='eng+kor')
            compact_ocr = re.sub(r'[^a-zA-Z0-9가-힣]', '', ocr_data) # 모든 공백/기호 제거

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            
            for i, std in enumerate(standard_list):
                clean_std = re.sub(r'[^a-zA-Z0-9가-힣]', '', std) # 엑셀 단어도 압축
                if clean_std in compact_ocr:
                    res, status = std, "✅ 일치"
                else:
                    res, status = "미검출", "❌ 오류"
                comparison.append({"No": i+1, "Excel 기준": std, "인식 결과": res, "상태": status})

            st.subheader("📊 검증 리포트")
            st.table(pd.DataFrame(comparison).style.applymap(lambda x: f'background-color: {"#d4edda" if x == "✅ 일치" else "#f8d7da"}', subset=['상태']))

# --- [모드 2] PDF vs PDF (삭제되지 않았습니다!) ---
elif mode == "PDF vs PDF (시각적 비교)":
    st.title("🖼️ PDF간 시각적 차이 분석")
    # ... (기존 시각 비교 코드 유지)