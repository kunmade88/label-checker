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
def get_image_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        return np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clean_text(text):
    """기호와 공백을 모두 제거하여 순수 알맹이만 남김"""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

# --- 사이드바 모드 설정 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26) # 26번까지 확인하시므로 기본값 조정
    else:
        st.info("🖼️ 원본과 수정본 PDF/이미지를 대조하여 바뀐 부분을 표시합니다.")

# --- 모드 1: Excel vs PDF (성분 검증) ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 문안확인 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        # 엑셀과 PDF를 같은 비중으로 배치
        view_c1, view_c2 = st.columns([1, 1])
        
        with view_c1:
            st.subheader("📊 엑셀 데이터 미리보기 (확대)")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            # [수정] 높이를 600으로 늘려 PDF 이미지 크기에 맞춤
            st.dataframe(df_display, height=600, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 검토 대상 이미지")
            img = get_image_from_upload(pdf_file)
            # [수정] PDF 이미지도 시원하게 출력
            st.image(img, use_container_width=True)

        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            # [심도 있는 매칭 로직]
            ocr_data = pytesseract.image_to_string(img, lang='kor+eng')
            compact_blob = clean_text(ocr_data) # 글자 바다 생성

            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            search_area = compact_blob
            
            for i, std_name in enumerate(standard_list):
                clean_std = clean_text(std_name)
                # C12-15 및 데실글루코사이드 대응을 위한 '포함' 검사
                if clean_std in search_area:
                    detected, status = std_name, "✅ 일치"
                    find_idx = search_area.find(clean_std)
                    search_area = search_area[find_idx + len(clean_std):]
                else:
                    detected, status = "미검출 (확인 요망)", "❌ 오류"
                
                comparison.append({"No": i+1, "Excel 기준": std_name, "인식 결과": detected, "상태": status})

            st.markdown("---")
            st.subheader("📋 최종 분석 리포트")
            res_df = pd.DataFrame(comparison)
            st.dataframe(res_df.style.applymap(lambda x: f'background-color: {"#d4edda" if x == "✅ 일치" else "#f8d7da"}', subset=['상태']), use_container_width=True, height=500)

# --- 모드 2: PDF vs PDF (시각적 차이) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ PDF/이미지 시각적 차이 분석")
    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        file_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if file_old and file_new:
        if st.button("🔍 시각적 차이점 찾기", use_container_width=True):
            img_old = get_image_from_upload(file_old)
            img_new = get_image_from_upload(file_new)
            
            h, w, _ = img_new.shape
            img_old_res = cv2.resize(img_old, (w, h))
            
            diff = cv2.absdiff(cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output = img_new.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, w_b, h_b = cv2.boundingRect(cnt)
                    cv2.rectangle(output, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)

            res_col1, res_col2 = st.columns(2)
            res_col1.image(img_old_res, caption="원본", use_container_width=True)
            res_col2.image(output, caption="차이 발생 구역", use_container_width=True)