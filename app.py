import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

# 페이지 설정
st.set_page_config(page_title="라벨 체크 AI 통합 분석기", layout="wide")

# --- 유틸리티 함수 (첫 번째 가공 로직 - 유지) ---
def get_processed_images(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def clean_text(text):
    if not text: return ""
    text = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text.strip()

def get_similarity(a, b):
    a_clean = re.sub(r'[^a-zA-Z0-9가-힣]', '', str(a)).lower()
    b_clean = re.sub(r'[^a-zA-Z0-9가-힣]', '', str(b)).lower()
    return SequenceMatcher(None, a_clean, b_clean).ratio()

# --- 사이드바 메뉴 (모드 선택) ---
with st.sidebar:
    st.header("⚙️ 작업 모드")
    mode = st.radio("분석 유형 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["한글명", "영문명"], index=0)

# --- 모드 1: Excel vs PDF (성분 정밀 대조) ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인 전성분 확인용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1: excel_file = st.file_uploader("📂 엑셀 업로드", type=['xlsx', 'csv'])
    with col2: pdf_file = st.file_uploader("📄 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        raw_img, proc_img = get_processed_images(pdf_file)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 엑셀 데이터")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(50)
            st.dataframe(df_display, height=450, use_container_width=True)
        with c2:
            st.subheader("🖼️ 가공된 이미지")
            st.image(proc_img, use_container_width=True)

        if st.button("🚀 분석 시작 (순서 정밀 매칭)", use_container_width=True):
            ocr_raw = pytesseract.image_to_string(proc_img, lang='kor+eng', config='--psm 6')
            ocr_cleaned = clean_text(ocr_raw)
            # 순서 보장을 위해 구분자로 쪼개기
            pdf_ingredients = [p.strip() for p in re.split(r'[,.\n]', ocr_cleaned) if len(p.strip()) > 1]
            excel_list = df_display[lang_choice].dropna().astype(str).tolist()
            
            comparison = []
            pdf_idx = 0
            for i, excel_name in enumerate(excel_list):
                detected_name = "❌ 미검출"
                status = "❌ 오류"
                
                # 유연한 순서 매칭 (윈도우 탐색)
                search_range = pdf_ingredients[max(0, pdf_idx-1) : pdf_idx+4]
                
                for p_text in search_range:
                    if get_similarity(excel_name, p_text) > 0.8:
                        status = "✅ 일치"
                        detected_name = p_text
                        if p_text in pdf_ingredients:
                            pdf_idx = pdf_ingredients.index(p_text) + 1
                        break
                
                comparison.append({
                    "No": i+1,
                    "엑셀 기준 (A)": excel_name,
                    "PDF 검출 내용 (B)": detected_name,
                    "상태": status
                })

            st.markdown("---")
            st.subheader("📋 최종 분석 리포트")
            res_df = pd.DataFrame(comparison)
            
            # 가독성 개선 스타일 (무조건 검정 글씨)
            def style_report(row):
                bg = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg}; color: #000000; font-weight: 900; font-size: 14px;'] * len(row)

            st.table(res_df.style.apply(style_report, axis=1))

# --- 모드 2: PDF vs PDF (시각적 차이 비교) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안검토 수정전/후 비교 테스트 용훈")
    st.info("원본과 수정본의 디자인적 차이나 오타를 시각적으로 대조합니다.")
    
    col1, col2 = st.columns(2)
    with col1: f_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2: f_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행", use_container_width=True):
            img_old_raw, _ = get_processed_images(f_old)
            img_new_raw, _ = get_processed_images(f_new)
            
            # 크기 맞춤
            h, w, _ = img_new_raw.shape
            img_old_res = cv2.resize(img_old_raw, (w, h))
            
            # 차이 계산
            gray_old = cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY)
            gray_new = cv2.cvtColor(img_new_raw, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(gray_old, gray_new)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output = img_new_raw.copy()
            for c in contours:
                if cv2.contourArea(c) > 50:
                    x, y, wb, hb = cv2.boundingRect(c)
                    cv2.rectangle(output, (x, y), (x+wb, y+hb), (255, 0, 0), 2)
            
            res_c1, res_c2 = st.columns(2)
            res_c1.image(img_old_res, caption="원본 이미지", use_container_width=True)
            res_c2.image(output, caption="차이점 감지 (빨간 박스)", use_container_width=True)