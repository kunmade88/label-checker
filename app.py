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

# --- 유틸리티 함수 ---
def get_clean_image(uploaded_file):
    """배경은 완전 흰색, 글자는 진한 검정색으로 변환 (기존 가공 로직 유지)"""
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
    """비교를 위해 특수문자 및 불필요 키워드 제거"""
    if not text: return ""
    text = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', str(text))
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower().strip()

def get_similarity(a, b):
    """글자 유사도 계산 (오독 대응용)"""
    return SequenceMatcher(None, clean_for_match(a), clean_for_match(b)).ratio()

# --- 사이드바 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["한글명", "영문명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)

# --- 모드 1: Excel vs PDF ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 문안확인용 테스트 용훈")
    
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
            st.subheader("🖼️ 가공 이미지 (배경:흰색 / 글자:검정)")
            processed_img = get_clean_image(pdf_file)
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 분석 시작", use_container_width=True):
            # OCR 수행 및 원문 데이터 보존
            ocr_text = pytesseract.image_to_string(processed_img, lang='kor+eng', config='--psm 6')
            
            # 제목(전성분 등) 제거 후 쉼표 기준으로 쪼개서 리스트화 (이미지 순서 보존)
            pure_ocr = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', ocr_text)
            pdf_parts = [p.strip() for p in pure_ocr.replace('\n', ' ').split(',') if len(p.strip()) > 1]

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []

            # ✅ 순서 매칭 로직 (C12-15, 데실글루코사이드 등 오독 발생 시 해당 위치 값 표기)
            for i, std_name in enumerate(standard_list):
                status = "❌ 오류"
                detected_text = "데이터 부족"
                
                if i < len(pdf_parts):
                    actual_pdf_text = pdf_parts[i]
                    # 유사도가 85% 이상이면 일치로 판정
                    if get_similarity(std_name, actual_pdf_text) > 0.85:
                        status = "✅ 일치"
                        detected_text = actual_pdf_text
                    else:
                        # 틀렸을 경우 PDF가 실제로 뭐라고 읽었는지 그대로 보여줌
                        status = "❌ 오류"
                        detected_text = actual_pdf_text
                
                comparison.append({
                    "No": i+1,
                    "엑셀 기준 (A)": std_name,
                    "PDF 검출 내용 (B)": detected_text,
                    "상태": status
                })

            st.markdown("---")
            st.subheader("📋 성분 대조 결과 리포트")
            res_df = pd.DataFrame(comparison)
            
            # ✅ 가독성 개선 스타일: 배경색은 유지, 글자색은 무조건 진한 검정(#000000)
            def style_row(row):
                bg = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg}; color: #000000; font-weight: bold;'] * len(row)

            # table 형식이 가독성이 가장 좋으므로 table로 출력
            st.table(res_df.style.apply(style_row, axis=1))

# --- 모드 2: PDF vs PDF ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인 수정전/후 비교 테스트(yh)")
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