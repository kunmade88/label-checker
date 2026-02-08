import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")

def clean_text(text):
    """비교를 위해 특수문자 제거 및 소문자화 (한글 포함)"""
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
    
    # lang='kor+eng'로 설정하여 한글과 영어를 동시에 인식
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

def get_all_texts(ocr_data):
    valid_texts = [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 30]
    full_text = " ".join(valid_texts)
    # 전성분 리스트는 콤마(,)가 기준이므로 콤마로 쪼개기
    return [t.strip() for t in full_text.split(',') if t.strip()]

# --- 메인 UI ---
st.title("🔍 전성분 문안 정밀 확인 시스템 테스트 용훈")
mode = st.sidebar.radio("작업 모드 선택", ["Excel vs PDF (성분 순서 검증)", "PDF vs PDF (시각적 차이)"])

if mode == "Excel vs PDF (성분 순서 검증)":
    st.subheader("📊 엑셀-이미지 전성분 대조")
    
    # 언어 선택 추가
    check_lang = st.radio("검증할 언어 선택", ["영문명", "한글명"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("표준 전성분 엑셀 업로드", type=['xlsx', 'xls', 'csv'])
    with col2:
        pdf_file = st.file_uploader("검토할 이미지/PDF 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        if st.button("🚀 분석 시작"):
            with st.spinner(f'{check_lang} 기준으로 대조 중...'):
                try:
                    # [1] 엑셀 처리
                    df_raw = pd.read_excel(excel_file) if excel_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(excel_file)
                    header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), None)
                    
                    if header_idx is not None:
                        df_clean = pd.read_excel(excel_file, skiprows=header_idx + 1)
                    else:
                        df_clean = df_raw

                    # 사용자가 선택한 언어(영문명 또는 한글명) 컬럼 추출
                    if check_lang in df_clean.columns:
                        standard_list = df_clean[check_lang].dropna().astype(str).tolist()
                    else:
                        st.error(f"엑셀에 '{check_lang}' 컬럼이 없습니다. 컬럼명을 확인해주세요.")
                        st.stop()

                    # [2] OCR 및 대조
                    img, ocr_data = get_data_from_upload(pdf_file)
                    extracted_list = get_all_texts(ocr_data)

                    comparison = []
                    max_len = max(len(standard_list), len(extracted_list))

                    for i in range(max_len):
                        std = standard_list[i] if i < len(standard_list) else "(엑셀 없음)"
                        ext = extracted_list[i] if i < len(extracted_list) else "(이미지 없음)"
                        
                        ratio = SequenceMatcher(None, clean_text(std), clean_text(ext)).ratio()
                        
                        if clean_text(std) == clean_text(ext):
                            status = "✅ 일치"
                        elif ratio > 0.6: # 한글은 획이 복잡해 영문보다 조금 낮게 설정 가능
                            status = "🔍 오타 의심"
                        else:
                            status = "❌ 순서오류/누락"
                        
                        comparison.append({
                            "순번": i + 1,
                            "엑셀 표준": std,
                            "이미지 추출": ext,
                            "상태": status
                        })

                    st.table(pd.DataFrame(comparison))
                except Exception as e:
                    st.error(f"에러 발생: {e}")

# ... (이하 PDF vs PDF 모드 생략)