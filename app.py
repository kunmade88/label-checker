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
def get_images(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 가독성용 흑백 변환 (배경 흰색, 글자 검정)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    view_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    
    return img, cv2.cvtColor(view_img, cv2.COLOR_GRAY2RGB)

def clean_for_match(text, is_ocr=False):
    if not text: return ""
    # ✅ 줄바꿈(\n)과 공백을 모두 제거하여 한 줄로 통합 (줄바꿈 오류 방지)
    text = text.replace('\n', ' ').replace('\r', ' ')
    if is_ocr:
        text = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', str(text))
    # 기호 제거 후 소문자로 통합
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
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        ocr_img, view_img = get_images(pdf_file)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 엑셀 기준")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=500, use_container_width=True)
        with c2:
            st.subheader("🖼️ 검토 이미지 (가독성 모드)")
            st.image(view_img, use_container_width=True)

        if st.button("🚀 분석 시작", use_container_width=True):
            # OCR 및 줄바꿈 제거 처리
            ocr_raw_text = pytesseract.image_to_string(ocr_img, lang='kor+eng')
            compact_ocr = clean_for_match(ocr_raw_text, is_ocr=True)
            
            # PDF 텍스트를 쉼표 기준으로 쪼개서 대조 칸에 보여줄 준비
            ocr_parts = [p.strip() for p in ocr_raw_text.replace('\n', ' ').split(',') if len(p.strip()) > 1]

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            search_area = compact_ocr

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                found_text = "❌ 데이터 없음"
                
                if clean_std and clean_std in search_area:
                    status = "✅ 일치"
                    pos = search_area.find(clean_std)
                    search_area = search_area[pos + len(clean_std):]
                    found_text = std_name
                else:
                    status = "❌ 오류"
                    if i < len(ocr_parts):
                        found_text = ocr_parts[i]
                
                comparison.append({"No": i+1, "엑셀 기준": std_name, "PDF 검출 내용": found_text, "상태": status})

            st.markdown("---")
            st.subheader("📋 분석 결과 (글자색 검정 고정)")
            res_df = pd.DataFrame(comparison)

            # ✅ 스타일 수정: 배경색은 파스텔톤, 글자색은 검정(#000000)으로 고정
            def style_rows(row):
                bg_color = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg_color}; color: #000000; font-weight: bold;'] * len(row)

            st.dataframe(res_df.style.apply(style_rows, axis=1), use_container_width=True, height=600)

# --- 모드 2: PDF vs PDF ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인용 수정전/후 비교 테스트 용훈")
    # ... (생략 없이 이전의 안정적인 차이 분석 로직 포함)
    f_old = st.file_uploader("원본 업로드", key="o")
    f_new = st.file_uploader("수정본 업로드", key="n")
    if f_old and f_new:
        if st.button("🔍 차이점 분석"):
            img1, _ = get_images(f_old)
            img2, _ = get_images(f_new)
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