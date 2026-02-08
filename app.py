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

# --- 유틸리티 함수 (사용자님이 만족하신 가공 로직 유지) ---
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

def get_similarity(a, b):
    # 유사도 계산 시 순수 텍스트만 비교
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
    st.title("🔍 문안확인 전성분 확인용 테스트 용훈") # 제목 수정
    
    col1, col2 = st.columns(2)
    with col1: excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2: pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        processed_img = get_clean_image(pdf_file)
        
        view_c1, view_c2 = st.columns(2)
        with view_c1:
            st.subheader("📊 엑셀 기준 데이터")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=400, use_container_width=True)
        with view_c2:
            st.subheader("🖼️ 가공된 이미지")
            st.image(processed_img, use_container_width=True)

        if st.button("🚀 분석 시작 (순서 밀림 방지 적용)", use_container_width=True):
            # OCR 수행
            ocr_text = pytesseract.image_to_string(processed_img, lang='kor+eng', config='--psm 6')
            # 쉼표 인식이 안 된 경우를 대비해 줄바꿈을 공백으로 바꾸고 쉼표로 쪼갬
            pdf_parts = [p.strip() for p in ocr_text.replace('\n', ' ').split(',') if len(p.strip()) > 1]
            
            standard_list = df_display[lang_choice].dropna().astype(str).tolist()
            comparison = []
            
            pdf_ptr = 0 # PDF 리스트 내의 현재 위치 추적용 포인터
            
            for i, std_name in enumerate(standard_list):
                status = "❌ 오류"
                detected_text = "미검출"
                
                # 핵심: 8번처럼 뭉침 현상이 발생해도 포인터를 통해 다음 성분을 찾아냄
                # 현재 위치 근처 (최대 5개 조각) 탐색
                search_range = pdf_parts[pdf_ptr : pdf_ptr + 5]
                
                for offset, part in enumerate(search_range):
                    # 1. 유사도 검사 (85% 이상)
                    if get_similarity(std_name, part) > 0.85:
                        status = "✅ 일치"
                        detected_text = part
                        pdf_ptr += offset + 1 # 찾은 위치 다음으로 포인터 이동
                        break
                    # 2. 뭉침 대응: 조각 안에 성분이 포함되어 있는지 확인
                    elif clean_for_match(std_name) in clean_for_match(part):
                        status = "✅ 일치"
                        detected_text = std_name # 뭉친 것 중 해당 성분만 인정
                        # 뭉친 조각이 너무 크면 포인터를 뒤로 미루지 않고 다음 검색에서 다시 사용
                        if len(clean_for_match(part)) < len(clean_for_match(std_name)) * 2:
                            pdf_ptr += offset + 1
                        break
                
                if status == "❌ 오류" and i < len(pdf_parts):
                    # 못 찾았을 경우, 해당 순서의 실제 PDF 내용을 보여줌
                    detected_text = pdf_parts[pdf_ptr] if pdf_ptr < len(pdf_parts) else "데이터 없음"

                comparison.append({
                    "No": i+1,
                    "엑셀 기준 (A)": std_name,
                    "PDF 검출 내용 (B)": detected_text,
                    "상태": status
                })

            st.markdown("---")
            res_df = pd.DataFrame(comparison)
            def style_row(row):
                bg = '#d4edda' if row['상태'] == "✅ 일치" else '#f8d7da'
                return [f'background-color: {bg}; color: #000000; font-weight: bold;'] * len(row)
            st.table(res_df.style.apply(style_row, axis=1))

# --- 모드 2: PDF vs PDF ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 문안확인 수정전/후 확인용 테스트 용훈") # 제목 수정
    f_old = st.file_uploader("원본 업로드", key="old")
    f_new = st.file_uploader("수정본 업로드", key="new")
    if f_old and f_new:
        if st.button("🔍 차이점 분석 실행"):
            img_old = get_clean_image(f_old)
            img_new = get_clean_image(f_new)
            h, w, _ = img_new.shape
            img_old = cv2.resize(img_old, (w, h))
            diff = cv2.absdiff(cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = img_new.copy()
            for c in contours:
                if cv2.contourArea(c) > 50:
                    x, y, wb, hb = cv2.boundingRect(c)
                    cv2.rectangle(out, (x, y), (x+wb, y+hb), (255, 0, 0), 2)
            st.image(out, use_container_width=True)