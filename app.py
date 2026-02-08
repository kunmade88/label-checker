import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

# 페이지 설정
st.set_page_config(page_title="라벨 체크 AI 통합 시스템", layout="wide")

# --- 유틸리티 함수 ---
def get_image(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        # DPI 300으로 고정하여 OCR 기본 정확도 확보
        pages = convert_from_bytes(file_bytes, dpi=300)
        return np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clean_for_match(text):
    """기호와 공백을 제거하여 순수 글자만 남김"""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)
    else:
        st.info("🖼️ 두 PDF/이미지 간의 시각적 차이를 분석합니다.")

# --- 모드 1: Excel vs PDF (성분 검증) ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인 전성분 검토용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        # 엑셀과 PDF 미리보기를 나란히 배치 (비중 1:1)
        view_c1, view_c2 = st.columns(2)
        
        with view_c1:
            st.subheader("📊 엑셀 데이터 (확대)")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            # 엑셀 창 높이를 PDF 이미지와 비슷하게 확장
            st.dataframe(df_display, height=650, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 검토 대상 이미지")
            img = get_image(pdf_file)
            st.image(img, use_container_width=True)

        if st.button("🚀 분석 시작", use_container_width=True):
            # OCR 수행 및 '글자 바다' 생성
            ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
            compact_ocr = clean_for_match(ocr_text)

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            search_area = compact_ocr

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                
                # 순차적 포함 여부 확인 (가장 깔끔한 로직)
                if clean_std and clean_std in search_area:
                    status = "✅ 일치"
                    pos = search_area.find(clean_std)
                    search_area = search_area[pos + len(clean_std):] # 다음 성분 검색을 위해 커서 이동
                else:
                    status = "❌ 오류"
                
                comparison.append({"No": i+1, "Excel 기준": std_name, "상태": status})

            st.markdown("---")
            st.subheader("📋 분석 리포트")
            res_df = pd.DataFrame(comparison)
            
            # 결과 테이블 시각화
            def color_status(val):
                color = '#d4edda' if val == "✅ 일치" else '#f8d7da'
                return f'background-color: {color}'

            st.table(res_df.style.applymap(color_status, subset=['상태']))

# --- 모드 2: PDF vs PDF (이전 코드 복구) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인 수정전/후 비교 테스트 용훈")
    st.markdown("원본과 수정본의 이미지를 대조하여 **빨간색 박스**로 차이를 표시합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        f_old = st.file_uploader("📄 원본 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        f_new = st.file_uploader("📄 수정본 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행", use_container_width=True):
            img_old = get_image(f_old)
            img_new = get_image(f_new)
            
            # 크기 맞춤 (수정본 기준)
            h, w, _ = img_new.shape
            img_old = cv2.resize(img_old, (w, h))
            
            # 차이 계산 로직
            gray_old = cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY)
            gray_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(gray_old, gray_new)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output = img_new.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, w_b, h_b = cv2.boundingRect(cnt)
                    cv2.rectangle(output, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
            
            res_c1, res_c2 = st.columns(2)
            res_c1.image(img_old, caption="원본(Base)", use_container_width=True)
            res_c2.image(output, caption="차이점 감지 결과", use_container_width=True)