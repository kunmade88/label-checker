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

def get_filtered_ingredients(ocr_data):
    """'Ingredients' 단어 이후의 텍스트만 추출하여 리스트화"""
    all_words = [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 30]
    
    # "Ingredients" 단어 찾기 (대소문자 무시)
    start_idx = 0
    for i, word in enumerate(all_words):
        if "ingredient" in word.lower():
            start_idx = i + 1  # "Ingredients" 다음 단어부터 시작
            break
            
    full_text = " ".join(all_words[start_idx:])
    # 콤마로 분리하여 성분 리스트 생성
    return [t.strip() for t in full_text.split(',') if t.strip()]

# --- 메인 UI ---
st.title("🔍 전성분 정밀 검증 시스템 (Ingredients 제외 로직)")

mode = st.sidebar.radio("작업 모드", ["Excel vs PDF (성분 대조)", "PDF vs PDF (시각 비교)"])

if mode == "Excel vs PDF (성분 대조)":
    lang_choice = st.radio("검증 기준 언어", ["영문명", "한글명"], horizontal=True)
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        excel_file = st.file_uploader("기준 엑셀 업로드", type=['xlsx', 'xls', 'csv'])
    with col_u2:
        pdf_file = st.file_uploader("검토용 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        if st.button("🚀 분석 시작"):
            try:
                # 1. 엑셀 로드 및 헤더 찾기
                df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
                header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
                df_clean = pd.read_excel(excel_file, skiprows=header_idx + 1)
                
                # 엑셀 데이터 리스트 (끝까지 로드)
                standard_list = df_clean[lang_choice].dropna().astype(str).tolist()
                st.sidebar.info(f"💡 엑셀에서 총 {len(standard_list)}개의 성분을 찾았습니다.")

                # 2. PDF OCR 및 필터링
                img, ocr_data = get_data_from_upload(pdf_file)
                extracted_list = get_filtered_ingredients(ocr_data)

                # 3. 시각적 병렬 배치
                st.write("### 🖼️ 데이터 시각적 대조")
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.success(f"📂 엑셀 기준 ({lang_choice})")
                    st.text_area("Excel List", "\n".join([f"{i+1}. {x}" for i, x in enumerate(standard_list)]), height=300)
                
                with v_col2:
                    st.warning("📸 PDF 추출 (Ingredients 이후)")
                    overlay = img.copy()
                    # 간단하게 글자 구역 박스 표시
                    for i in range(len(ocr_data['text'])):
                        if int(ocr_data['conf'][i]) > 30:
                            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    st.image(overlay)

                # 4. 결과 테이블
                st.write("### 📋 정밀 대조 리포트")
                comparison = []
                max_len = max(len(standard_list), len(extracted_list))
                for i in range(max_len):
                    std = standard_list[i] if i < len(standard_list) else "(엑셀 없음)"
                    ext = extracted_list[i] if i < len(extracted_list) else "(이미지 없음)"
                    ratio = SequenceMatcher(None, clean_text(std), clean_text(ext)).ratio()
                    
                    status = "✅ 일치" if ratio == 1.0 else "🔍 오타 의심" if ratio > 0.7 else "❌ 순서오류"
                    comparison.append({"순번": i+1, "엑셀(표준)": std, "이미지(추출)": ext, "상태": status})
                
                st.table(pd.DataFrame(comparison))

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")