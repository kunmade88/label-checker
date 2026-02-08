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
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["한글명", "영문명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)

if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인 전성분 확인용 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1: excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2: pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        processed_img = get_clean_image(pdf_file)
        c1, c2 = st.columns(2)
        with c1:
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=400, use_container_width=True)
        with c2:
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 1:1 순서 정밀 분석 시작", use_container_width=True):
            ocr_text = pytesseract.image_to_string(processed_img, lang='kor+eng', config='--psm 6')
            protected_text = re.sub(r'(\d+),(\d+)', r'\1_DIGIT_COMMA_\2', ocr_text)
            pure_ocr = re.sub(r'전성분|Ingredients|INGREDIENTS|인그리디언트|전 성 분', '', protected_text)
            raw_parts = pure_ocr.replace('\n', ' ').split(',')
            pdf_parts = [p.replace('_DIGIT_COMMA_', ',').strip() for p in raw_parts if len(p.strip()) > 0]
            
            excel_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            
            for i in range(len(excel_list)):
                std_name = excel_list[i]
                status, detected_text = "❌ 오류", "미검출"
                
                if i < len(pdf_parts):
                    actual_part = pdf_parts[i]
                    detected_text = actual_part 
                    raw_sim = SequenceMatcher(None, std_name.lower().strip(), actual_part.lower().strip()).ratio()
                    clean_sim = SequenceMatcher(None, clean_for_match(std_name), clean_for_match(actual_part)).ratio()
                    
                    if raw_sim > 0.98: status = "✅ 일치"
                    elif clean_sim > 0.95: status = "⚠️ 주의"
                    elif clean_sim > 0.7: status = "⚠️ 주의"
                    else: status = "❌ 오류"
                
                comparison.append({"No": i+1, "엑셀 기준 (A)": std_name, "PDF 실제 검출 내용 (B)": detected_text, "상태": status})

            st.markdown("---")
            res_df = pd.DataFrame(comparison)
            def style_row(row):
                bg = '#d4edda' if row['상태'] == "✅ 일치" else '#fff3cd' if row['상태'] == "⚠️ 주의" else '#f8d7da'
                return [f'background-color: {bg}; color: #000000; font-weight: bold; font-size: 14px;'] * len(row)
            st.table(res_df.style.apply(style_row, axis=1))

elif mode == "PDF vs PDF (시각적 차이)":
    # (모드 2 로직 동일)
    st.title("🖼️ 문안확인 수정전/후 확인용 테스트 용훈")
    f_old = st.file_uploader("원본 업로드", type=['pdf', 'jpg', 'png'], key="old")
    f_new = st.file_uploader("수정본 업로드", type=['pdf', 'jpg', 'png'], key="new")
    if f_old and f_new and st.button("🔍 차이점 분석 실행"):
        img_old, img_new = get_clean_image(f_old), get_clean_image(f_new)
        img_old_res = cv2.resize(img_old, (img_new.shape[1], img_new.shape[0]))
        diff = cv2.absdiff(cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY))
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img_new.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, wb, hb = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x+wb, y+hb), (255, 0, 0), 2)
        st.image([img_old_res, output], caption=["원본", "수정본(차이점)"], use_container_width=True)