import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="라벨 체크 AI 통합 시스템", layout="wide")

# =========================
# 유틸리티 함수
# =========================
def get_image_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        # ✅ PDF는 DPI 올려 OCR 정확도 향상
        pages = convert_from_bytes(file_bytes, dpi=300)
        return np.array(pages[0].convert("RGB"))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_for_ocr(img_rgb):
    """
    OCR 정확도 향상용 전처리:
    - 그레이스케일
    - 2배 확대
    - 블러
    - adaptive threshold
    """
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

def norm(s: str) -> str:
    """OCR/텍스트 비교 전 기본 정규화"""
    if s is None:
        return ""
    s = str(s)
    # 다양한 대시를 통일
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # 중간점/불릿류를 콤마로 통일(라벨에서 자주 나옴)
    s = s.replace("•", ",").replace("·", ",").replace(";", ",")
    return s

def build_flexible_pattern(name: str) -> str:
    """
    성분명을 토큰으로 쪼개고, 토큰 사이에 어떤 구분기호(공백/줄바꿈/콤마/하이픈 등)도 허용하는 정규식 생성.
    - 오검출을 줄이기 위해 '토큰 자체'는 그대로 일치해야 함.
    예) 1,2-Hexanediol -> 1\W*2\W*hexanediol
    예) C12-15 Alkyl Benzoate -> c12\W*15\W*alkyl\W*benzoate
    """
    s = norm(name).lower()
    tokens = re.findall(r"[a-z0-9가-힣]+", s)
    if not tokens:
        return ""
    return r"\W*".join(map(re.escape, tokens))

def status_color(v):
    if v == "✅ 일치":
        return "background-color:#d4edda"
    if v == "🟡 순서/문단 차이":
        return "background-color:#fff3cd"
    return "background-color:#f8d7da"

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

        st.markdown("### 🔧 정확도 튜닝")
        psm = st.selectbox("Tesseract PSM", [4, 6], index=0)  # 표/정렬이면 4가 유리한 경우 많음
        window_back = st.slider("커서 되돌림(문단 내려감 대응)", 200, 3000, 1200, 100)
        window_ahead = st.slider("탐색 범위(오검출 방지)", 1000, 20000, 8000, 500)

        show_debug = st.checkbox("디버그: OCR 원문 보기(일부)", value=False)
        st.info("💡 콤마/줄바꿈/하이픈 차이를 정규식으로 흡수하면서, 순차 탐색(커서)로 오검출을 줄입니다.")
    else:
        st.info("🖼️ 원본과 수정본 PDF/이미지를 대조하여 바뀐 부분을 표시합니다.")

# =========================
# 모드 1: Excel vs PDF (성분 검증)
# =========================
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 문안확인 테스트 용훈")

    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("📂 기준 엑셀 업로드", type=["xlsx", "csv"])
    with col2:
        pdf_file = st.file_uploader("📄 검토 PDF/이미지 업로드", type=["pdf", "jpg", "png"])

    if excel_file and pdf_file:
        st.markdown("---")
        view_c1, view_c2 = st.columns(2)

        with view_c1:
            st.subheader("📊 엑셀 데이터 미리보기")
            df_raw = pd.read_excel(excel_file) if excel_file.name.endswith(".xlsx") else pd.read_csv(excel_file)
            header_idx = next((i for i, row in df_raw.iterrows() if "No." in row.values), 0)

            # 파일 포인터 리셋 후 재로딩 (중요)
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

        if st.button("🚀 정밀 비교 시작", use_container_width=True):
            try:
                # 1) OCR
                pre = preprocess_for_ocr(img)
                ocr_text = pytesseract.image_to_string(
                    pre,
                    lang="kor+eng",
                    config=f"--oem 3 --psm {psm}"
                )
                ocr_norm = norm(ocr_text).lower()

                if show_debug:
                    st.markdown("---")
                    st.subheader("🧪 디버그: OCR 원문(일부)")
                    st.text(ocr_text[:4000])

                # 2) Excel 리스트 준비
                if lang_choice not in df_display.columns:
                    st.error(f"엑셀에 '{lang_choice}' 컬럼이 없습니다. 현재 컬럼: {list(df_display.columns)}")
                    st.stop()

                standard_list = df_display[lang_choice].dropna().astype(str).tolist()

                # 3) 순차 매칭(커서 기반) + 문단 내려감 대응(커서 되돌림)
                comparison = []
                cursor = 0

                for i, std_name in enumerate(standard_list):
                    pat = build_flexible_pattern(std_name)

                    if not pat:
                        comparison.append({
                            "No": i + 1,
                            "Excel 기준": std_name,
                            "매칭 상태": "❌ 오류",
                            "비고": "패턴 생성 실패"
                        })
                        continue

                    start = max(0, cursor - int(window_back))
                    end = min(len(ocr_norm), cursor + int(window_ahead))
                    region = ocr_norm[start:end]

                    m = re.search(pat, region, flags=re.IGNORECASE)

                    if m:
                        status = "✅ 일치"
                        # cursor 갱신 (원문 전체 기준)
                        cursor = start + m.end()
                        note = ""
                    else:
                        # 커서 근처에서 못 찾으면 '순서/문단 차이' 가능성을 체크:
                        # 바로 앞/뒤 구간만 조금 넓혀 1번 더 (전체 검색은 오검출↑라 안 함)
                        start2 = max(0, cursor - int(window_back * 2))
                        end2 = min(len(ocr_norm), cursor + int(window_ahead * 2))
                        region2 = ocr_norm[start2:end2]
                        m2 = re.search(pat, region2, flags=re.IGNORECASE)

                        if m2:
                            status = "🟡 순서/문단 차이"
                            cursor = start2 + m2.end()
                            note = "문단/줄바꿈/순서 영향 가능"
                        else:
                            status = "❌ 미검출"
                            note = "라벨 누락/오타/OCR 오류 가능"

                    comparison.append({
                        "No": i + 1,
                        "Excel 기준": std_name,
                        "매칭 상태": status,
                        "비고": note
                    })

                st.markdown("---")
                st.subheader("📋 최종 비교 리포트")
                res_df = pd.DataFrame(comparison)
                st.dataframe(
                    res_df.style.applymap(status_color, subset=["매칭 상태"]),
                    use_container_width=True,
                    height=520
                )

                # 추가: 요약
                ok = (res_df["매칭 상태"] == "✅ 일치").sum()
                warn = (res_df["매칭 상태"] == "🟡 순서/문단 차이").sum()
                bad = (res_df["매칭 상태"] == "❌ 미검출").sum()
                st.markdown("---")
                st.write(f"✅ 일치: {ok}  |  🟡 주의: {warn}  |  ❌ 미검출: {bad}")

            except Exception as e:
                st.error(f"오류 발생: {e}")

# =========================
# 모드 2: PDF vs PDF (시각적 차이)
# =========================
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ PDF/이미지 시각적 차이 분석 테스트 용훈")
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
