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

# --- 유틸리티 함수 ---
def get_processed_images(uploaded_file):
    """배경 흰색, 글자 검정 가공 이미지 생성"""
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ㄹ/ㅁ 오독 방지를 위한 선명도 강화
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def clean_for_match(text):
    if not text: return ""
    # 줄바꿈 제거 및 기호 제거 (매칭용 정제)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower().strip()

def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# --- 사이드바 메뉴 ---
with st.sidebar:
    st.header("⚙️ 분석 메뉴")
    mode = st.radio("작업 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["한글명", "영문명"])
        sim_threshold = st.slider("유사도 보정 강도(오독 허용 범위)", 0.7, 1.0, 0.85)

# --- 모드 1: Excel vs PDF (성분 검증) ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인용 전성분 확인 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1: excel_file = st.file_uploader("📂 엑셀 업로드", type=['xlsx', 'csv'])
    with col2: pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        raw_img, proc_img = get_processed_images(pdf_file)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 엑셀 기준 데이터")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(40)
            st.dataframe(df_display, height=450, use_container_width=True)
        with c2:
            st.subheader("🖼️ 가공 이미지 (배경:흰색 / 글자:검정)")
            st.image(proc_img, use_container_width=True)

        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            # OCR 텍스트 추출
            ocr_text_raw = pytesseract.image_to_string(proc_img, lang='kor+eng', config='--psm 6')
            
            # 쉼표(,)를 기준으로 쪼개서 각 조각 리스트 생성
            ocr_parts = [p.strip() for p in ocr_text_raw.replace('\n', ' ').split(',') if len(p.strip()) > 1]
            clean_ocr_parts = [clean_for_match(p) for p in ocr_parts]

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                status = "❌ 오류"
                detected_val = "미검출" # 기본값
                
                # 유사도 기반 매칭 (실레이트 vs 심레이트 등 보정)
                for part, raw_part in zip(clean_ocr_parts, ocr_parts):
                    if get_similarity(clean_std, part) > sim_threshold:
                        status = "✅ 일치"
                        detected_val = raw_part # PDF에서 실제로 읽은 텍스트
                        break
                
                comparison.append({
                    "No": i+1, 
                    "엑셀 기준 (A)": std_name, 
                    "PDF 검출 내용 (B)": detected_val, # ✅ 요청하신 칸 다시 추가
                    "상태": status
                })

            st.markdown("---")
            st.subheader("📋 성분 대조 결과 리포트")
            res_df = pd.DataFrame(comparison)
            
            def style_row(row):
                bg = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg}; color: #000000; font-weight: bold;'] * len(row)
            
            # 결과 테이블 출력 (글자색 검정 고정)
            st.table(res_df.style.apply(style_row, axis=1))

# --- 모드 2: PDF vs PDF (파일 비교 기능) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인 수정전/후 비교 테스트 용훈")
    st.info("원본과 수정본 이미지를 대조하여 변경된 부분을 빨간 박스로 표시합니다.")
    
    col1, col2 = st.columns(2)
    with col1: f_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2: f_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행", use_container_width=True):
            img_old, _ = get_processed_images(f_old)
            img_new, _ = get_processed_images(f_new)
            
            # 두 이미지 크기 통일
            h, w, _ = img_new.shape
            img_old_res = cv2.resize(img_old, (w, h))
            
            # 이미지 차이 분석 (Pixel Difference)
            gray_old = cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY)
            gray_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(gray_old, gray_new)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output = img_new.copy()
            for c in contours:
                if cv2.contourArea(c) > 50:
                    x, y, wb, hb = cv2.boundingRect(c)
                    cv2.rectangle(output, (x, y), (x+wb, y+hb), (255, 0, 0), 2)
            
            res_c1, res_c2 = st.columns(2)
            res_c1.image(img_old_res, caption="원본(이전 버전)", use_container_width=True)
            res_c2.image(output, caption="수정본(변경 사항 감지)", use_container_width=True)