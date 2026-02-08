import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

# 페이지 설정
st.set_page_config(page_title="라벨 체크 AI 정밀 분석", layout="wide")

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

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dist = cv2.fastNlMeansDenoising(gray, h=10)
    # OTSU 이진화로 배경과 글자 분리
    _, binary = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binary) < 127: # 배경이 어두우면 반전
        binary = cv2.bitwise_not(binary)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def clean_for_match(text, is_ocr=False):
    if not text: return ""
    # 전성분/Ingredients 제목 제외
    if is_ocr:
        text = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', str(text))
    # 매칭용 알맹이 (기호 제거)
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower().strip()

# --- 사이드바 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)

# --- 모드 1: Excel vs PDF ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안 전성분 검토 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        view_c1, view_c2 = st.columns(2)
        
        with view_c1:
            st.subheader("📊 엑셀 데이터")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=600, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 가독성 최적화 이미지")
            processed_img = get_clean_image(pdf_file)
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 분석 시작", use_container_width=True):
            # OCR 수행 및 원문 데이터 보존
            ocr_text = pytesseract.image_to_string(processed_img, lang='kor+eng')
            # 쉼표(,) 기준으로 쪼개서 리스트화 (이미지의 실제 전성분 순서 추적용)
            raw_ocr_parts = [p.strip() for p in ocr_text.replace('\n', ' ').split(',') if p.strip()]

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            
            # 매칭 로직
            compact_ocr_blob = clean_for_match(ocr_text, is_ocr=True)
            search_area = compact_ocr_blob

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                detected_text = "미검출" # 기본값
                
                if clean_std and clean_std in search_area:
                    status = "✅ 일치"
                    # 실제 PDF에서 어떻게 읽혔는지 가장 유사한 조각을 찾아 기록
                    # (단순 구현을 위해 엑셀 이름과 가장 닮은 OCR 조각 추출)
                    pos = search_area.find(clean_std)
                    search_area = search_area[pos + len(clean_std):]
                    detected_text = std_name # 일치할 경우 엑셀명 표시
                else:
                    status = "❌ 오류"
                    # 오류일 경우, 현재 search_area의 앞부분 일부를 보여줌 (뭐가 있는지 확인용)
                    detected_text = f"(추정): {ocr_text.split(',')[i] if i < len(ocr_text.split(',')) else '데이터 없음'}"

                comparison.append({
                    "No": i+1,
                    "엑셀 기준 (A)": std_name,
                    "PDF 검출 내용 (B)": detected_text,
                    "상태": status
                })

            st.markdown("---")
            st.subheader("📋 성분 대조 결과 리포트")
            res_df = pd.DataFrame(comparison)
            
            # 스타일 정의: A와 B가 다를 경우 강조
            def highlight_diff(row):
                if row['상태'] == "❌ 오류":
                    return ['background-color: #f8d7da'] * len(row)
                return ['background-color: #d4edda'] * len(row)

            st.dataframe(res_df.style.apply(highlight_diff, axis=1), use_container_width=True, height=600)

# --- 모드 2: PDF vs PDF ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안검토 수정전/후 비교 테스트 용훈")
    # (이전의 시각적 차이 분석 코드와 동일하여 유지됩니다)
    f_old = st.file_uploader("원본 업로드", type=['pdf', 'jpg', 'png'], key="old")
    f_new = st.file_uploader("수정본 업로드", type=['pdf', 'jpg', 'png'], key="new")
    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행"):
            img_old = get_clean_image(f_old)
            img_new = get_clean_image(f_new)
            h, w, _ = img_new.shape
            img_old = cv2.resize(img_old, (w, h))
            diff = cv2.absdiff(cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output = img_new.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, w_b, h_b = cv2.boundingRect(cnt)
                    cv2.rectangle(output, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
            st.image(output, use_container_width=True)