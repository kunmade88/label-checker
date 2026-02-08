import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="전성분 문안확인 테스트용 용훈", layout="wide")

def clean_text(text):
    """비교를 위해 특수문자/공백 제거 및 소문자화"""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

# --- 메인 UI ---
st.title("🔍 전성분 정밀 검증 (복합 성분명 대응형)")

lang_choice = st.sidebar.radio("검증 기준 언어", ["영문명", "한글명"])
compare_limit = st.sidebar.number_input("비교할 성분 개수 (위에서부터)", value=16)

col_u1, col_u2 = st.columns(2)
with col_u1:
    excel_file = st.file_uploader("표준 전성분 엑셀 업로드", type=['xlsx', 'xls', 'csv'])
with col_u2:
    pdf_file = st.file_uploader("검토용 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

if excel_file and pdf_file:
    if st.button("🚀 분석 시작"):
        try:
            # 1. 엑셀 파싱 (상단 번호별 리스트 사용)
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_clean = pd.read_excel(excel_file, skiprows=header_idx + 1)
            
            # 엑셀 상단 리스트에서 지정된 개수만큼 가져오기
            standard_list = df_clean[lang_choice].dropna().astype(str).tolist()[:int(compare_limit)]

            # 2. PDF OCR 및 '글자 바다' 생성
            img, ocr_data = get_data_from_upload(pdf_file)
            all_words = [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 30]
            
            # Ingredients 이후 텍스트 합치기
            start_idx = 0
            for i, word in enumerate(all_words):
                if "ingredient" in word.lower():
                    start_idx = i + 1
                    break
            full_blob = " ".join(all_words[start_idx:])

            # 3. 지능형 순차 매칭 (콤마 무시 로직)
            comparison = []
            current_search_area = full_blob
            
            for i, std_name in enumerate(standard_list):
                # 정규식 에러 방지를 위해 특수문자 이스케이프 및 공백 유연화
                search_pattern = re.escape(std_name).replace(r'\ ', r'\s*')
                match = re.search(search_pattern, current_search_area, re.IGNORECASE)
                
                if match:
                    detected_text = match.group()
                    status = "✅ 일치"
                    # 찾은 위치 이후부터 다음 성분 검색 (순서 보장)
                    current_search_area = current_search_area[match.end():]
                else:
                    detected_text = "(미검출/오타)"
                    status = "❌ 불일치"
                
                comparison.append({
                    "순번": i + 1,
                    "엑셀 기준": std_name,
                    "PDF 인식결과": detected_text,
                    "상태": status
                })

            # 4. 결과 출력
            st.write(f"### 📋 리포트 (상단 {compare_limit}개 성분 대조)")
            res_df = pd.DataFrame(comparison)
            st.table(res_df)

        except Exception as e:
            st.error(f"오류 발생: {e}")