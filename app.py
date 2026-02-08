import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

st.set_page_config(page_title="라벨 체크 AI", layout="wide")

# --- 이미지/데이터 로드 함수 ---
def get_image_and_data(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ocr_data = pytesseract.image_to_data(gray, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

# --- 메인 화면 ---
st.title("🔍 문안확인용 테스트 용훈")

with st.sidebar:
    st.header("⚙️ 설정")
    lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
    compare_limit = st.number_input("비교 성분 개수", value=16)

# 파일 업로드 영역
col_u1, col_u2 = st.columns(2)
with col_u1:
    excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
with col_u2:
    pdf_file = st.file_uploader("📄 검토 PDF 업로드", type=['pdf', 'jpg', 'png'])

# [핵심] 업로드 즉시 양옆에 시각화하여 보여주기
if excel_file and pdf_file:
    st.markdown("---")
    view_col1, view_col2 = st.columns(2)
    
    with view_col1:
        st.subheader("📊 엑셀 데이터 미리보기")
        df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
        # 데이터 시작점(No.) 찾기
        header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
        df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
        # 엑셀 시트처럼 보이기 위해 스타일 적용 후 출력
        st.dataframe(df_display, height=300, use_container_width=True)

    with view_col2:
        st.subheader("🖼️ PDF 라벨 이미지")
        img, ocr_data = get_image_and_data(pdf_file)
        # 이미지 크기를 엑셀 표 높이와 비슷하게 조절
        st.image(img, width=450)

    # 분석 버튼
    if st.button("🚀 위 데이터를 바탕으로 정밀 분석 시작", use_container_width=True):
        # [줄바꿈 대응 매칭 로직]
        standard_list = df_display[lang_choice].dropna().astype(str).tolist()
        
        words = [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip()]
        full_blob = "".join(words)
        
        # Ingredients 이후 텍스트 압축 매칭
        match_start = re.search(r'ingredient', full_blob, re.IGNORECASE)
        search_blob = full_blob[match_start.start():] if match_start else full_blob
        
        comparison = []
        curr_pos = 0
        for i, std in enumerate(standard_list):
            clean_std = re.sub(r'[^a-zA-Z0-9가-힣]', '', std)
            match = re.search(re.escape(clean_std), search_blob[curr_pos:], re.IGNORECASE)
            
            if match:
                res, status = std, "✅ 일치"
                curr_pos += match.end()
            else:
                res, status = "미검출", "❌ 오류"
            comparison.append({"No": i+1, "Excel 기준": std, "인식 결과": res, "상태": status})

        # 결과 리포트 출력
        st.markdown("---")
        st.subheader("📋 최종 검증 결과")
        res_df = pd.DataFrame(comparison)
        st.table(res_df.style.applymap(lambda x: f'background-color: {"#d4edda" if x == "✅ 일치" else "#f8d7da"}', subset=['상태']))