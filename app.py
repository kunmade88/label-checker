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
def get_clean_image(uploaded_file):
    """배경은 완전 흰색, 글자는 진한 검정색으로 변환"""
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. 노이즈 제거 (글자 테두리 정리)
    dist = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 3. OTSU 이진화 (배경과 글자를 자동으로 분석해 흑백으로 나눔)
    # 배경이 어두울 경우를 대비해 반전 처리가 필요하면 자동으로 보정
    _, binary = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 만약 배경이 검정색으로 나왔다면 다시 반전 (글자가 검정색이 되도록)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def clean_for_match(text, is_ocr=False):
    if not text: return ""
    # 전성분 관련 제목 키워드 삭제 (사용자 요청)
    if is_ocr:
        text = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', str(text))
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower().strip()

# --- 사이드바 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)

# --- 모드 1: Excel vs PDF (성분 검증) ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안 전성분 검토용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        view_c1, view_c2 = st.columns(2)
        
        with view_c1:
            st.subheader("📊 엑셀 데이터 미리보기")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=750, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 가독성 최적화 (배경:흰색 / 글자:검정)")
            processed_img = get_clean_image(pdf_file)
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 분석 시작", use_container_width=True):
            ocr_text = pytesseract.image_to_string(processed_img, lang='kor+eng')
            compact_ocr = clean_for_match(ocr_text, is_ocr=True)

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            search_area = compact_ocr

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                if clean_std and clean_std in search_area:
                    status = "✅ 일치"
                    pos = search_area.find(clean_std)
                    search_area = search_area[pos + len(clean_std):]
                else:
                    status = "❌ 오류"
                comparison.append({"No": i+1, "Excel 기준": std_name, "상태": status})

            st.markdown("---")
            st.subheader("📋 분석 리포트")
            res_df = pd.DataFrame(comparison)
            st.table(res_df.style.applymap(lambda x: f'background-color: {"#d4edda" if x == "✅ 일치" else "#f8d7da"}', subset=['상태']))

# --- 모드 2: PDF vs PDF (시각적 차이) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안검토용 수정전/후 비교테스트 용훈")
    col1, col2 = st.columns(2)
    with col1:
        f_old = st.file_uploader("📄 원본 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        f_new = st.file_uploader("📄 수정본 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행", use_container_width=True):
            img_old = get_clean_image(f_old)
            img_new = get_clean_image(f_new)
            h, w, _ = img_new.shape
            img_old = cv2.resize(img_old, (w, h))
            
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