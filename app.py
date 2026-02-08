import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="라벨 체크 테스트 용훈", layout="wide")

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

def get_image_and_data(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 인식률을 높이기 위한 이미지 전처리 (흑백화)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ocr_data = pytesseract.image_to_data(gray, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.radio("분석 유형", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=16)

# --- 메인 로직 ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 스마트 매칭 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        # 이미지를 즉시 보여줌으로써 시각적 피드백 제공
        img, ocr_data = get_image_and_data(pdf_file)
        st.subheader("🖼️ 업로드된 이미지 확인")
        st.image(img, caption="검토 중인 라벨 이미지", use_container_width=True)
        
        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            try:
                # 1. 엑셀 파싱
                df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
                header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
                df_clean = pd.read_excel(excel_file, skiprows=header_idx + 1)
                standard_list = df_clean[lang_choice].dropna().astype(str).tolist()[:int(compare_limit)]

                # 2. 텍스트 추출 (Ingredients 이후)
                all_words = [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 20]
                full_blob = " ".join(all_words)
                
                # "Ingredients" 위치 찾기 (없으면 전체 텍스트 사용)
                ingredients_match = re.search(r'ingredient', full_blob, re.IGNORECASE)
                search_area = full_blob[ingredients_match.start():] if ingredients_match else full_blob

                # 3. 매칭 로직 (1,2- 콤마 유연 대응)
                comparison = []
                for i, std_name in enumerate(standard_list):
                    # 글자 사이의 특수문자/공백/줄바꿈을 무시하는 강력한 정규식
                    pattern = "".join([re.escape(c) if c.isalnum() else r'[\s\W]*' for c in std_name])
                    match = re.search(pattern, search_area, re.IGNORECASE)
                    
                    if match:
                        detected, status = match.group(), "✅ 일치"
                        search_area = search_area[match.end():] # 찾은 이후부터 검색
                    else:
                        detected, status = "미검출 (확인 필요)", "❌ 오류"
                    
                    comparison.append({"No": i+1, "Excel 표준": std_name, "PDF 인식": detected, "상태": status})

                st.subheader("📋 분석 리포트")
                res_df = pd.DataFrame(comparison)
                st.dataframe(res_df.style.applymap(lambda x: f'background-color: {"#d4edda" if x == "✅ 일치" else "#f8d7da"}', subset=['상태']), use_container_width=True)

            except Exception as e:
                st.error(f"오류: {e}")

elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 시각적 차이 분석")
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.file_uploader("원본", type=['pdf', 'jpg', 'png'], key="f1")
    with col2:
        f2 = st.file_uploader("수정본", type=['pdf', 'jpg', 'png'], key="f2")

    if f1 and f2:
        i1 = get_image_and_data(f1)[0]
        i2 = get_image_and_data(f2)[0]
        
        if st.button("🔍 차이점 분석"):
            # 사이즈 맞춤
            i1_res = cv2.resize(i1, (i2.shape[1], i2.shape[0]))
            diff = cv2.absdiff(cv2.cvtColor(i1_res, cv2.COLOR_RGB2GRAY), cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY))
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            res_img = i2.copy()
            for c in contours:
                if cv2.contourArea(c) > 50:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(res_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            c1, c2 = st.columns(2)
            c1.image(i1_res, caption="원본", use_container_width=True)
            c2.image(res_img, caption="변화 감지", use_container_width=True)