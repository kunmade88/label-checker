import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

# 페이지 설정
st.set_page_config(page_title="라벨 체크 AI", layout="wide")

# --- 유틸리티 함수 ---
def get_image(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        # DPI를 300으로 고정하여 선명도 확보 (전처리보다 이게 더 중요합니다)
        pages = convert_from_bytes(file_bytes, dpi=300)
        return np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clean_for_match(text):
    """알파벳, 숫자, 한글만 남기고 모두 제거 (매칭용 알맹이)"""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

# --- 사이드바 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 개수", value=26)
    else:
        st.info("🖼️ 두 파일의 시각적 차이를 분석합니다.")

# --- 모드 1: Excel vs PDF ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 문안확인 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        view_c1, view_c2 = st.columns(2)
        
        # 1. 엑셀 로드 및 시각화
        with view_c1:
            st.subheader("📊 엑셀 데이터 미리보기")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=600, use_container_width=True)

        # 2. 이미지 로드 및 시각화
        with view_c2:
            st.subheader("🖼️ 검토 대상 이미지")
            img = get_image(pdf_file)
            st.image(img, use_container_width=True)

        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            # OCR 수행 (복잡한 전처리 없이 원본 선명도 활용)
            ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
            # [핵심] 이미지의 모든 글자를 기호 없이 '하나의 바다'로 만듦
            compact_ocr = clean_for_match(ocr_text)

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            
            # 순차 검색용 커서 (중복 성분이 있을 경우를 대비)
            current_search_area = compact_blob = compact_ocr

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                
                if clean_std and clean_std in current_search_area:
                    detected, status = std_name, "✅ 일치"
                    # 찾은 위치 이후부터 다음 성분을 찾도록 영역 제한 (오검출 방지)
                    find_pos = current_search_area.find(clean_std)
                    current_search_area = current_search_area[find_pos + len(clean_std):]
                else:
                    detected, status = "미검출", "❌ 오류"
                
                comparison.append({"No": i+1, "Excel 기준": std_name, "상태": status})

            st.markdown("---")
            st.subheader("📋 분석 결과 리포트")
            res_df = pd.DataFrame(comparison)
            st.table(res_df.style.applymap(lambda x: f'background-color: {"#d4edda" if x == "✅ 일치" else "#f8d7da"}', subset=['상태']))

# --- 모드 2: PDF vs PDF (깔끔하게 정리) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안 수정전/후 비교 테스트 용훈")
    col1, col2 = st.columns(2)
    with col1:
        f_old = st.file_uploader("원본 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        f_new = st.file_uploader("수정본 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if f_old and f_new:
        if st.button("🔍 차이점 찾기", use_container_width=True):
            img_old = get_image(f_old)
            img_new = get_image(f_new)
            
            # 사이즈 맞추기
            h, w, _ = img_new.shape
            img_old = cv2.resize(img_old, (w, h))
            
            # 차이 계산
            diff = cv2.absdiff(cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            out = img_new.copy()
            for c in contours:
                if cv2.contourArea(c) > 50:
                    x, y, w_b, h_b = cv2.boundingRect(c)
                    cv2.rectangle(out, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
            
            c1, c2 = st.columns(2)
            c1.image(img_old, caption="원본", use_container_width=True)
            c2.image(out, caption="차이 발생 (빨간 박스)", use_container_width=True)