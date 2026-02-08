import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re
from difflib import SequenceMatcher

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="라벨 체크 AI 통합 시스템", layout="wide")

# =========================
# 유틸 함수
# =========================
def get_image_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        # ✅ DPI 올려 OCR 정확도 향상
        pages = convert_from_bytes(file_bytes, dpi=300)
        return np.array(pages[0].convert("RGB"))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize_ocr_confusions(s: str) -> str:
    """OCR에서 자주 나오는 문자 혼동을 완화"""
    if s is None:
        return ""
    s = str(s)
    # 대시류 통일
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return s

def clean_text(text):
    """기호/공백 제거 후 비교용 텍스트 생성"""
    text = normalize_ocr_confusions(text)
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", str(text)).lower().strip()

def preprocess_for_ocr(img_rgb):
    """OCR 정확도 향상을 위한 전처리 (확대 + 이진화)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    return thr

def fuzzy_find_in_blob(blob: str, target: str, min_ratio: float = 0.90):
    """
    긴 blob 문자열에서 target과 유사한 구간을 탐색.
    반환: (found_bool, best_ratio, best_substring)
    """
    if not target:
        return False, 0.0, ""

    tlen = len(target)
    if tlen < 4:
        ok = target in blob
        return ok, 1.0 if ok else 0.0, target if ok else ""

    min_w = max(4, int(tlen * 0.8))
    max_w = int(tlen * 1.2)

    best_ratio = 0.0
    best_sub = ""

    step = max(1, tlen // 10)  # 속도/정확도 절충

    for w in range(min_w, max_w + 1):
        for i in range(0, max(1, len(blob) - w + 1), step):
            sub = blob[i : i + w]
            ratio = SequenceMatcher(None, target, sub).ratio()
            if ratio > best_ratio:
                best_ratio, best_sub = ratio, sub
                if ratio >= min_ratio:
                    return True, best_ratio, best_sub

    return False, best_ratio, best_sub

def status_color(s):
    if s == "✅ 일치":
        return "background-color: #d4edda"
    if s == "🟡 유사":
        return "background-color: #fff3cd"
    return "background-color: #f8d7da"

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.header("🛠️ 작업 모드")
    mode = st.radio("분석 유형 선택", ["Excel vs PDF (성분 검증)", "PDF vs PDF (시각적 차이)"])
    st.markdown("---")

    if mode == "Excel vs PDF (성분 검증)":
        lang_choice = st.radio("검증 언어", ["영문명", "한글명"])
        compare_limit = st.number_input("비교 성분 개수", value=26, min_value=1)
        st.info("💡 OCR 전처리 + DPI 향상 + 유사매칭(퍼지)로 미검출을 크게 줄입니다.")
        min_ratio = st.slider("유사매칭 기준(유사도)", 0.70, 0.99, 0.90, 0.01)
        show_debug = st.checkbox("디버그(OCR 원문 일부 보기)", value=False)
    else:
        st.info("🖼️ 원본과 수정본 PDF/이미지를 대조하여 바뀐 부분을 표시합니다.")

# =========================
# 모드 1: Excel vs PDF
# =========================
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 스마트 매칭 (정밀 + 유사매칭)")

    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=["xlsx", "csv"])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=["pdf", "jpg", "png"])

    if excel_file and pdf_file:
        st.markdown("---")
        view_c1, view_c2 = st.columns(2)

        with view_c1:
            st.subheader("📊 엑셀 데이터 미리보기 (확대)")
            if excel_file.name.endswith(".xlsx"):
                df_raw = pd.read_excel(excel_file)
            else:
                df_raw = pd.read_csv(excel_file)

            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)

            # 엑셀 파일 포인터가 이미 읽혀있을 수 있어 재로딩
            excel_file.seek(0)
            if excel_file.name.endswith(".xlsx"):
                df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            else:
                df_display = pd.read_csv(excel_file, skiprows=header_idx + 1).head(int(compare_limit))

            st.dataframe(df_display, height=650, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 검토 대상 이미지")
            img = get_image_from_upload(pdf_file)
            st.image(img, use_container_width=True)

        if st.button("🚀 정밀 분석 시작", use_container_width=True):
            try:
                # OCR
                pre = preprocess_for_ocr(img)
                ocr_text = pytesseract.image_to_string(
                    pre,
                    lang="kor+eng",
                    config="--oem 3 --psm 6"
                )
                compact_blob = clean_text(ocr_text)

                if show_debug:
                    st.markdown("---")
                    st.subheader("🧪 디버그: OCR 원문(일부)")
                    st.text(ocr_text[:3000])

                # 비교 리스트
                if lang_choice not in df_display.columns:
                    st.error(f"엑셀에 '{lang_choice}' 컬럼이 없습니다. 현재 컬럼: {list(df_display.columns)}")
                    st.stop()

                standard_list = df_display[lang_choice].dropna().astype(str).tolist()

                comparison = []
                for i, std_name in enumerate(standard_list):
                    clean_std = clean_text(std_name)

                    # 1) 정확 매칭
                    if clean_std and (clean_std in compact_blob):
                        detected = std_name
                        status = "✅ 일치"
                        score = 1.00
                    else:
                        # 2) 유사(퍼지) 매칭
                        found, ratio, _sub = fuzzy_find_in_blob(compact_blob, clean_std, min_ratio=float(min_ratio))
                        if found:
                            detected = f"(유사매칭)"
                            status = "🟡 유사"
                            score = ratio
                        else:
                            detected = "미검출"
                            status = "❌ 오류"
                            score = ratio

                    comparison.append({
                        "No": i + 1,
                        "Excel 기준": std_name,
                        "인식 결과": detected,
                        "유사도": f"{score:.2f}",
                        "상태": status
                    })

                st.markdown("---")
                st.subheader("📋 최종 분석 리포트")
                res_df = pd.DataFrame(comparison)

                st.dataframe(
                    res_df.style.applymap(status_color, subset=["상태"]),
                    use_container_width=True,
                    height=520
                )

            except Exception as e:
                st.error(f"오류 발생: {e}")

# =========================
# 모드 2: PDF vs PDF
# =========================
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ PDF/이미지 시각적 차이 분석")
    st.markdown("원본과 수정본을 업로드하면 **바뀐 부분만 빨간색 박스**로 표시합니다.")

    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("📄 원본(Base) 업로드", type=["pdf", "jpg", "png"], key="old")
    with col2:
        file_new = st.file_uploader("📄 수정본(New) 업로드", type=["pdf", "jpg", "png"], key="new")

    if file_old and file_new:
        if st.button("🔍 시각적 차이점 찾기", use_container_width=True):
            with st.spinner("이미지 정렬 및 차이 분석 중..."):
                img_old = get_image_from_upload(file_old)
                img_new = get_image_from_upload(file_new)

                # 사이즈 일치화 (수정본 기준)
                h, w, _ = img_new.shape
                img_old_res = cv2.resize(img_old, (w, h))

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

                res_col1, res_col2 = st.columns(2)
                res_col1.image(img_old_res, caption="원본 (Base)", use_container_width=True)
                res_col2.image(output, caption="차이 발생 구역 (빨간 박스)", use_container_width=True)
