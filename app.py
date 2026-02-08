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
        # DPI 300으로 설정하여 고해상도 OCR 소스 확보
        pages = convert_from_bytes(file_bytes, dpi=300)
        return np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clean_for_match(text):
    """알파벳, 숫자, 한글만 남기고 모두 제거 (매칭용 알맹이)"""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text)).lower().strip()

def get_similarity(a, b):
    """두 문자열 사이의 유사도 측정 (0.0 ~ 1.0)"""
    return SequenceMatcher(None, a, b).ratio()

def apply_row_style(val):
    """상태값에 따른 배경색 지정"""
    if val == "✅ 일치":
        return "background-color: #d4edda" # 연초록
    elif val == "🟡 유사(확인필요)":
        return "background-color: #fff3cd" # 연노랑
    else:
        return "background-color: #f8d7da" # 연빨강

# --- 사이드바 모드 설정 ---
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")
    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26)
        st.info("💡 100% 일치는 초록색, 90% 이상 유사도는 노란색으로 표시합니다.")
    else:
        st.info("🖼️ 원본과 수정본 이미지를 대조하여 차이점을 빨간 박스로 표시합니다.")

# --- 모드 1: Excel vs PDF (성분 검증) ---
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 문안확인용 전성분 검토 테스트 용훈")
    
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=['xlsx', 'csv'])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=['pdf', 'jpg', 'png'])

    if excel_file and pdf_file:
        st.markdown("---")
        view_c1, view_c2 = st.columns(2)
        
        with view_c1:
            st.subheader("📊 엑셀 데이터 미리보기")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)
            df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            st.dataframe(df_display, height=600, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 검토 대상 이미지")
            img = get_image(pdf_file)
            st.image(img, use_container_width=True)

        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            try:
                # OCR 수행 및 '글자 바다' 생성
                ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
                compact_ocr = clean_for_match(ocr_text)

                standard_list = df_display[lang_choice].dropna().astype(str).tolist()
                comparison = []
                search_area = compact_ocr # 순차 검색용 영역

                for i, std_name in enumerate(standard_list):
                    clean_std = clean_for_match(std_name)
                    found_status = "❌ 미검출"
                    
                    if not clean_std: continue

                    # 1. 완전 일치 (100%)
                    if clean_std in search_area:
                        found_status = "✅ 일치"
                        pos = search_area.find(clean_std)
                        search_area = search_area[pos + len(clean_std):]
                    
                    # 2. 유사도 체크 (90% 이상)
                    else:
                        std_len = len(clean_std)
                        best_sim = 0
                        best_pos = -1
                        
                        # 슬라이딩 윈도우 방식으로 주변 텍스트와 비교
                        # (검색 효율을 위해 주변 1000자 내외에서 비교하는 것이 좋으나 전체에서 탐색)
                        for j in range(len(search_area) - std_len + 1):
                            segment = search_area[j : j + std_len]
                            sim = get_similarity(clean_std, segment)
                            if sim > best_sim:
                                best_sim = sim
                                best_pos = j
                        
                        if best_sim >= 0.90: # 유사도 90% 임계점
                            found_status = "🟡 유사(확인필요)"
                            search_area = search_area[best_pos + std_len:]
                    
                    comparison.append({"No": i+1, "Excel 기준": std_name, "상태": found_status})

                st.markdown("---")
                st.subheader("📋 최종 분석 리포트")
                res_df = pd.DataFrame(comparison)
                # 스타일 적용하여 테이블 출력
                st.table(res_df.style.applymap(apply_row_style, subset=['상태']))
                
            except Exception as e:
                st.error(f"오류 발생: {e}")

# --- 모드 2: PDF vs PDF (시각적 차이 분석) ---
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ 전성분 수정전/후 비교 테스트 용훈")
    st.markdown("원본과 수정본의 이미지를 겹쳐서 **픽셀 차이**를 찾아냅니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        file_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if file_old and file_new:
        if st.button("🔍 시각적 차이점 찾기", use_container_width=True):
            with st.spinner("이미지 비교 분석 중..."):
                img_old = get_image(file_old)
                img_new = get_image(file_new)

                # 수정본 이미지 크기에 맞춰 원본 리사이즈
                h, w, _ = img_new.shape
                img_old_res = cv2.resize(img_old, (w, h))

                # 그레이스케일 변환 및 차이 추출
                gray_old = cv2.cvtColor(img_old_res, cv2.COLOR_RGB2GRAY)
                gray_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
                
                # 두 이미지의 절대 차이 계산
                diff = cv2.absdiff(gray_old, gray_new)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                
                # 차이가 있는 부분에 윤곽선(Box) 그리기
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                output = img_new.copy()
                diff_count = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) > 50: # 미세 노이즈 무시
                        x, y, w_b, h_b = cv2.boundingRect(cnt)
                        cv2.rectangle(output, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
                        diff_count += 1

                st.success(f"분석 완료! 총 {diff_count}곳의 차이점이 발견되었습니다.")
                
                res_col1, res_col2 = st.columns(2)
                res_col1.image(img_old_res, caption="원본 (Base)", use_container_width=True)
                res_col2.image(output, caption="차이 발생 구역 (빨간 박스)", use_container_width=True)