import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

# 페이지 설정
st.set_page_config(page_title="라벨 체크 AI 통합 시스템", layout="wide")

# --- 유틸리티 함수 ---
def get_image(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        # ✅ DPI 300으로 폰트 뭉개짐 방지 (전처리보다 이게 훨씬 중요함)
        pages = convert_from_bytes(file_bytes, dpi=300)
        return np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clean_for_match(text):
    """비교를 위해 기호/공백 제거 후 소문자로 변환"""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

def get_similarity(a, b):
    """문자열 유사도 측정"""
    return SequenceMatcher(None, a, b).ratio()

def apply_row_style(row):
    """행 전체에 색상 적용"""
    status = row['상태']
    if status == "✅ 일치":
        return ['background-color: #d4edda'] * len(row)
    elif status == "🟡 유사(확인필요)":
        return ['background-color: #fff3cd'] * len(row)
    else:
        return ['background-color: #f8d7da'] * len(row)

# --- 사이드바 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)
        st.info("💡 90% 이상 유사하면 노란색으로 표시합니다.")
    else:
        st.info("🖼️ 원본/수정본의 시각적 차이를 빨간 박스로 표시합니다.")

# --- 모드 1: Excel vs PDF ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인 전성분 검토용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

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
            st.subheader("🖼️ 라벨 이미지")
            img = get_image(pdf_file)
            st.image(img, use_container_width=True)

        if st.button("🚀 분석 시작", use_container_width=True):
            # 1. OCR (원본의 화질을 믿고 그대로 수행)
            ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
            compact_ocr = clean_for_match(ocr_text)

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            search_area = compact_ocr

            for i, std_name in enumerate(standard_list):
                clean_std = clean_for_match(std_name)
                found_status = "❌ 미검출"
                
                if not clean_std: continue

                # [Step 1] 완전 일치 검사
                if clean_std in search_area:
                    found_status = "✅ 일치"
                    pos = search_area.find(clean_std)
                    search_area = search_area[pos + len(clean_std):] # 순차 검색 커서 이동
                
                # [Step 2] 유사도 기반 검사 (C12-15, 데실글루코사이드 핵심)
                else:
                    std_len = len(clean_std)
                    # 현재 위치 주변 500자 이내에서 가장 비슷한 문구 찾기
                    lookup_range = search_area[:500] 
                    best_sim = 0
                    best_pos = -1
                    
                    for j in range(len(lookup_range) - std_len + 1):
                        segment = lookup_range[j : j + std_len]
                        sim = get_similarity(clean_std, segment)
                        if sim > best_sim:
                            best_sim = sim
                            best_pos = j
                    
                    if best_sim >= 0.90: # 유사도 90% 임계점
                        found_status = "🟡 유사(확인필요)"
                        search_area = search_area[best_pos + std_len:]
                
                comparison.append({"No": i+1, "성분명": std_name, "상태": found_status})

            st.markdown("---")
            st.subheader("📋 분석 결과")
            res_df = pd.DataFrame(comparison)
            st.dataframe(res_df.style.apply(apply_row_style, axis=1), use_container_width=True, height=600)

# --- 모드 2: PDF vs PDF ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인 수정전/후 비교 테스트 용훈")
    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("원본 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        file_new = st.file_uploader("수정본 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if file_old and file_new:
        if st.button("🔍 차이점 찾기", use_container_width=True):
            img_old = get_image(file_old)
            img_new = get_image(file_new)
            h, w, _ = img_new.shape
            img_old = cv2.resize(img_old, (w, h))

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