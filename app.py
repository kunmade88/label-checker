import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

# 페이지 설정
st.set_page_config(page_title="라벨 체크 AI 정밀 분석", layout="wide")

# --- 유틸리티 함수 (가공 로직 유지) ---
def get_clean_image(uploaded_file):
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
    _, binary = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binary) < 127: 
        binary = cv2.bitwise_not(binary)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def clean_for_match(text):
    if not text: return ""
    # 매칭용 순수 텍스트 (특수문자/공백 제거 후 소문자화)
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

# --- 사이드바 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["한글명", "영문명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)

# --- 모드 1: Excel vs PDF (엄격한 1:1 순서 대조) ---
if mode == "Excel vs PDF (성분 검증)":
    # 요청하신 제목 반영
    st.title("🔍 문안확인 전성분 확인용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1: excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2: pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        processed_img = get_clean_image(pdf_file)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 엑셀 기준 데이터")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=400, use_container_width=True)
        with c2:
            st.subheader("🖼️ 가공된 이미지")
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 1:1 순서 정밀 분석 시작", use_container_width=True):
            # OCR 수행 (PSM 6: 단일 텍스트 블록)
            ocr_text = pytesseract.image_to_string(processed_img, lang='kor+eng', config='--psm 6')
            # 머리말 제거
            pure_ocr = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', ocr_text)
            # 쉼표(,)를 기준으로만 정확히 쪼갬 (순서 보존의 핵심)
            pdf_parts = [p.strip() for p in pure_ocr.replace('\n', ' ').split(',') if len(p.strip()) > 0]
            
            excel_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            
            # ✅ 사용자님이 원하신 엄격한 1:1 비교 로직
            for i in range(len(excel_list)):
                std_name = excel_list[i]
                status = "❌ 오류"
                detected_text = "미검출"
                
                if i < len(pdf_parts):
                    actual_part = pdf_parts[i]
                    detected_text = actual_part # PDF 조각 내용을 가감 없이 노출
                    
                    # 엑셀 i번째와 PDF i번째 조각의 유사도 측정
                    similarity = SequenceMatcher(None, clean_for_match(std_name), clean_for_match(actual_part)).ratio()
                    
                    # 유사도가 90% 이상일 때만 일치로 간주
                    if similarity > 0.9:
                        status = "✅ 일치"
                    else:
                        status = "❌ 오류"
                
                comparison.append({
                    "No": i+1,
                    "엑셀 기준 (A)": std_name,
                    "PDF 실제 검출 내용 (B)": detected_text,
                    "상태": status
                })

            st.markdown("---")
            st.subheader("📋 성분 대조 결과 리포트")
            res_df = pd.DataFrame(comparison)
            
            # 스타일: 배경색 파스텔, 글자색 진한 검정 고정
            def style_row(row):
                bg = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg}; color: #000000; font-weight: bold; font-size: 14px;'] * len(row)

            st.table(res_df.style.apply(style_row, axis=1))

# --- 모드 2: PDF vs PDF (시각적 차이 비교) ---
elif mode == "PDF vs PDF (시각적 차이)":
    # 요청하신 제목 반영
    st.title("🖼️ 문안확인 수정전/후 확인용 테스트 용훈")
    
    f_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    f_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")
    
    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행", use_container_width=True):
            img_old = get_clean_image(f_old)
            img_new = get_clean_image(f_new)
            
            h, w, _ = img_new.shape
            img_old_res = cv2.resize(img_old, (w, h))
            
            diff = cv2.absdiff(cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY), 
                               cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output = img_new.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, wb, hb = cv2.boundingRect(cnt)
                    cv2.rectangle(output, (x, y), (x+wb, y+hb), (255, 0, 0), 2)
            
            c1, c2 = st.columns(2)
            c1.image(img_old_res, caption="원본(Base)", use_container_width=True)
            c2.image(output, caption="수정본 (차이점:빨간 박스)", use_container_width=True)