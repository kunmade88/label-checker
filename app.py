import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

# 페이지 설정
st.set_page_config(page_title="문안확인용 테스트 용훈", layout="wide")

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
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

# --- 사이드바 모드 설정 ---
with st.sidebar:
    st.header("⚙️ 작업 설정")
    mode = st.radio("분석 유형 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=16)
        st.info("💡 1,2-Hexanediol 등 성분명 내부 콤마를 자동으로 인식하여 순서 꼬임을 방지합니다.")
    else:
        st.info("✨ 두 파일의 디자인 변경이나 오타 위치를 시각적으로 비교합니다.")

# --- 메인 로직 ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 스마트 매칭 (순서 & 오타 검증)")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            try:
                # 1. 엑셀 데이터 파싱
                df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
                # 'No.' 컬럼이 있는 행을 찾아 헤더로 설정
                header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
                df_clean = pd.read_excel(excel_file, skiprows=header_idx + 1)
                standard_list = df_clean[lang_choice].dropna().astype(str).tolist()[:int(compare_limit)]

                # 2. 이미지 OCR 수행
                img = get_image_from_upload(pdf_file)
                ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
                all_words = [t.strip() for i, t in enumerate(ocr_data['text']) if t.strip() and int(ocr_data['conf'][i]) >= 30]
                
                # Ingredients 문구 이후부터 텍스트 바다 생성
                start_idx = 0
                for i, word in enumerate(all_words):
                    if "ingredient" in word.lower():
                        start_idx = i + 1
                        break
                full_blob = " ".join(all_words[start_idx:])

                # 3. 지능형 매칭 (성분명 덩어리 찾기)
                comparison = []
                search_area = full_blob
                
                for i, std_name in enumerate(standard_list):
                    # 특수문자 무시 정규식 패턴 생성
                    pattern = "".join([re.escape(c) if c.isalnum() else r'[^a-zA-Z0-9가-힣]*' for c in std_name])
                    match = re.search(pattern, search_area, re.IGNORECASE)
                    
                    if match:
                        detected, status = match.group(), "✅ 일치"
                        search_area = search_area[match.end():] # 찾은 이후 지점부터 다시 검색
                    else:
                        detected, status = "미검출 (오타/누락 확인)", "❌ 오류"
                    
                    comparison.append({"No": i+1, "표준 성분명 (Excel)": std_name, "인식 결과 (PDF)": detected, "상태": status})

                # 4. 결과 출력
                st.subheader("📋 분석 리포트")
                res_df = pd.DataFrame(comparison)
                
                def style_status(val):
                    return f'background-color: {"#d4edda" if val == "✅ 일치" else "#f8d7da"}'

                st.dataframe(res_df.style.applymap(style_status, subset=['상태']), use_container_width=True, height=500)

            except Exception as e:
                st.error(f"⚠️ 오류 발생: {e}")

elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ PDF/이미지 시각적 차이 분석")
    st.markdown("두 이미지 사이의 픽셀 변화를 감지하여 **바뀐 부분에 빨간색 박스**를 표시합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        file_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if file_old and file_new:
        if st.button("🔍 차이점 분석 실행", use_container_width=True):
            with st.spinner("이미지 대조 중..."):
                img_old = get_image_from_upload(file_old)
                img_new = get_image_from_upload(file_new)

                # 사이즈 일치화 (수정본 기준)
                h, w, _ = img_new.shape
                img_old_res = cv2.resize(img_old, (w, h))

                # 차이 계산 로직
                gray_old = cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY)
                gray_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
                
                diff = cv2.absdiff(gray_old, gray_new)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                output = img_new.copy()
                for cnt in contours:
                    if cv2.contourArea(cnt) > 50:
                        x, y, w_b, h_b = cv2.boundingRect(cnt)
                        cv2.rectangle(output, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)

                # 결과 레이아웃
                res_col1, res_col2 = st.columns(2)
                res_col1.image(img_old_res, caption="원본 (Base)", use_container_width=True)
                res_col2.image(output, caption="차이점 감지 (빨간 박스)", use_container_width=True)