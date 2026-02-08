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
def get_processed_images(uploaded_file):
    """배경 흰색, 글자 검정색의 '가독성 최적화' 이미지 생성"""
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ 배경 흰색, 글자 검정색으로 만드는 강력한 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 노이즈 제거 및 선명도 강화
    dist = cv2.fastNlMeansDenoising(gray, h=10)
    # 배경과 글자를 흑백으로 명확히 분리 (OTSU 방식)
    _, binary = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 만약 배경이 검정색으로 나왔다면 흰색으로 반전 (글자가 검정색이 되도록)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # OCR용(원본급)과 사용자 보기용(가공본) 동일하게 적용하여 정확도 통일
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return img, processed

def clean_for_match(text, is_ocr=False):
    """줄바꿈과 기호를 완전히 제거하여 매칭 오류 방지"""
    if not text: return ""
    # ✅ 1. 줄바꿈(\n)을 공백으로 치환하여 단어가 잘리는 현상 방지
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 2. 전성분/Ingredients 제목 키워드 제거
    if is_ocr:
        text = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', str(text))
    
    # 3. 모든 기호 제거 후 소문자 결합 (가장 확실한 매칭 방법)
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
    st.title("🔍 문안 전성분 확인용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 이미지/PDF 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        # 이미지 로직 복구
        raw_img, processed_img = get_processed_images(pdf_file)
        
        view_c1, view_c2 = st.columns(2)
        with view_c1:
            st.subheader("📊 엑셀 기준 데이터")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=600, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 가공된 이미지 (배경:흰색 / 글자:검정)")
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 정밀 대조 시작", use_container_width=True):
            # ✅ 줄바꿈이 제거된 OCR 텍스트 생성
            ocr_text_raw = pytesseract.image_to_string(processed_img, lang='kor+eng')
            compact_ocr_blob = clean_for_match(ocr_text_raw, is_ocr=True)
            
            # 비교용 텍스트 조각들 (오류 시 보여줄 용도)
            ocr_parts = [p.strip() for p in ocr_text_raw.replace('\n', ' ').split(',') if len(p.strip()) > 1]

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            search_area = compact_ocr_blob

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                found_pdf_text = "❌ 미검출"
                
                # ✅ 줄바꿈 상관없이 포함 여부 체크
                if clean_std and clean_std in search_area:
                    status = "✅ 일치"
                    pos = search_area.find(clean_std)
                    search_area = search_area[pos + len(clean_std):]
                    found_pdf_text = std_name
                else:
                    status = "❌ 오류"
                    if i < len(ocr_parts):
                        found_pdf_text = ocr_parts[i]

                comparison.append({
                    "No": i+1,
                    "엑셀 기준 성분명": std_name,
                    "PDF 실제 검출 내용": found_pdf_text,
                    "상태": status
                })

            st.markdown("---")
            st.subheader("📋 상세 대조 분석표")
            res_df = pd.DataFrame(comparison)
            
            # ✅ 가독성을 위해 배경은 파스텔, 글자색은 검정색(#000000)으로 고정
            def apply_final_style(row):
                bg_color = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg_color}; color: #000000; font-weight: bold;'] * len(row)

            st.dataframe(res_df.style.apply(apply_final_style, axis=1), use_container_width=True, height=600)

# --- 모드 2: PDF vs PDF ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인용 수정전/후 테스트 용훈")
    f_old = st.file_uploader("원본 업로드", key="old_file")
    f_new = st.file_uploader("수정본 업로드", key="new_file")
    if f_old and f_new:
        if st.button("🔍 차이점 분석"):
            _, img1 = get_processed_images(f_old)
            _, img2 = get_processed_images(f_new)
            h, w, _ = img2.shape
            img1 = cv2.resize(img1, (w, h))
            diff = cv2.absdiff(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = img2.copy()
            for c in contours:
                if cv2.contourArea(c) > 50:
                    x, y, wb, hb = cv2.boundingRect(c)
                    cv2.rectangle(out, (x, y), (x+wb, y+hb), (255, 0, 0), 2)
            st.image(out, use_container_width=True)