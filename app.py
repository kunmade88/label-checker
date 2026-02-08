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
# 유틸리티
# =========================
def get_image_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes, dpi=300)  # DPI 올림
        return np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_for_ocr(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return thr

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # 대시 통일
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # OCR이 자주 섞는 구분기호 통일
    s = s.replace("•", ",").replace("·", ",").replace(";", ",")
    return s

def clean_key(text: str) -> str:
    """비교용 키: 기호/공백 제거"""
    text = normalize_text(text).lower()
    return re.sub(r"[^a-z0-9가-힣]", "", text).strip()

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# =========================
# OCR 텍스트를 "성분 항목 리스트"로 파싱
# =========================
def extract_ingredient_items(ocr_text: str):
    """
    OCR 텍스트에서 성분 항목을 최대한 리스트 형태로 뽑아냄.
    핵심:
    - 기본적으로 콤마/줄바꿈 기준으로 쪼개되,
    - '1,2-Hexanediol' 같은 "숫자,숫자-" 패턴의 콤마는 분리하지 않도록 보호
    """
    t = normalize_text(ocr_text)

    # 1) "1,2-" 같은 케이스 보호: "1,2-" -> "1§2-" 로 임시 치환
    t = re.sub(r"(\d)\s*,\s*(\d)\s*-", r"\1§\2-", t)

    # 2) 콤마 / 줄바꿈 기준으로 분리
    parts = re.split(r"[,|\n]+", t)

    items = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # 보호문자 복구
        p = p.replace("§", ",")

        # 너무 짧은 조각 제거(노이즈)
        # 단, "C12-15" 같이 짧지만 의미 있는 건 살려야 하므로 숫자/문자 조합이면 허용
        if len(p) < 3 and not re.search(r"[a-zA-Z0-9가-힣]", p):
            continue

        # OCR이 중간에 공백을 이상하게 넣은 경우 정리
        p = re.sub(r"\s{2,}", " ", p).strip()

        items.append(p)

    # 3) 중복/유사 중복 정리(간단히 key 기준 unique)
    seen = set()
    uniq = []
    for it in items:
        k = clean_key(it)
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)

    return uniq

# =========================
# 매칭: Excel 항목 vs OCR 항목
# =========================
def match_lists(excel_list, ocr_items, exact_first=True):
    """
    excel_list: 엑셀 기준 리스트(정답)
    ocr_items: OCR에서 뽑은 항목 리스트
    결과:
      - 매칭/유사/미검출
      - OCR에만 있는 추가 항목도 탐지
      - 순서 차이(인덱스 차이)도 확인 가능
    """
    ocr_keys = [clean_key(x) for x in ocr_items]
    used = set()

    rows = []
    for idx, ex in enumerate(excel_list, start=1):
        ex_key = clean_key(ex)

        # 1) exact key 매칭
        hit_j = None
        if exact_first and ex_key:
            for j, ok in enumerate(ocr_keys):
                if j in used:
                    continue
                if ex_key == ok:
                    hit_j = j
                    score = 1.0
                    break
        else:
            score = 0.0

        # 2) 유사도 매칭(가장 높은 것 선택)
        if hit_j is None:
            best_j = None
            best_score = 0.0
            for j, ok in enumerate(ocr_keys):
                if j in used:
                    continue
                if not ex_key or not ok:
                    continue
                s = similarity(ex_key, ok)
                if s > best_score:
                    best_score = s
                    best_j = j

            hit_j = best_j
            score = best_score

        if hit_j is None:
            rows.append({
                "No": idx,
                "Excel 기준": ex,
                "OCR 후보": "",
                "유사도": 0.00,
                "판정": "❌ 미검출",
                "비고": ""
            })
            continue

        used.add(hit_j)
        ocr_val = ocr_items[hit_j]

        # 판정 기준(너가 목적이 '오타/다름' 탐지이므로 3단계 추천)
        # - 1.00: 완전일치
        # - 0.92~0.999: 오타 가능(유사)
        # - <0.92: 다른 항목일 가능성(주의)
        if score >= 0.999:
            판정 = "✅ 일치"
            비고 = ""
        elif score >= 0.92:
            판정 = "🟡 오타/표기차이 가능"
            비고 = "Excel과 OCR 표기가 다를 수 있음"
        else:
            판정 = "🟠 매칭불안(다른 성분 가능)"
            비고 = "OCR이 잘못 읽었거나 다른 항목일 수 있음"

        rows.append({
            "No": idx,
            "Excel 기준": ex,
            "OCR 후보": ocr_val,
            "유사도": round(score, 2),
            "판정": 판정,
            "비고": 비고
        })

    # OCR에만 남은 항목(추가/불필요 항목 후보)
    extras = []
    for j, it in enumerate(ocr_items):
        if j not in used:
            extras.append(it)

    return pd.DataFrame(rows), extras

def style_result(df):
    def color(v):
        if v == "✅ 일치":
            return "background-color:#d4edda"
        if v == "🟡 오타/표기차이 가능":
            return "background-color:#fff3cd"
        if v == "🟠 매칭불안(다른 성분 가능)":
            return "background-color:#ffe5cc"
        if v == "❌ 미검출":
            return "background-color:#f8d7da"
        return ""
    return df.style.applymap(color, subset=["판정"])


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
        sim_threshold = st.slider("오타 판정 기준(유사도)", 0.85, 0.99, 0.92, 0.01)
        show_ocr_debug = st.checkbox("OCR 파싱 결과(성분 리스트) 보기", value=False)
    else:
        st.info("🖼️ 원본과 수정본 PDF/이미지를 대조하여 바뀐 부분을 표시합니다.")

# =========================
# 모드 1: Excel vs PDF
# =========================
if mode == "Excel vs PDF (성분 검증)":
    st.title("🔍 전성분 문안확인용 테스트 용훈")

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

            excel_file.seek(0)
            if excel_file.name.endswith('.xlsx'):
                df_display = pd.read_excel(excel_file, skiprows=header_idx + 1).head(int(compare_limit))
            else:
                df_display = pd.read_csv(excel_file, skiprows=header_idx + 1).head(int(compare_limit))

            st.dataframe(df_display, height=650, use_container_width=True)

        with view_c2:
            st.subheader("🖼️ 검토 대상 이미지")
            img = get_image_from_upload(pdf_file)
            st.image(img, use_container_width=True)

        if st.button("🚀 비교 분석 시작", use_container_width=True):
            try:
                pre = preprocess_for_ocr(img)
                ocr_text = pytesseract.image_to_string(pre, lang='kor+eng', config='--oem 3 --psm 6')

                # ✅ OCR 텍스트 -> 성분 항목 리스트 파싱
                ocr_items = extract_ingredient_items(ocr_text)

                if show_ocr_debug:
                    st.subheader("🧪 OCR 파싱 성분 리스트")
                    st.write(ocr_items)

                if lang_choice not in df_display.columns:
                    st.error(f"엑셀에 '{lang_choice}' 컬럼이 없습니다. 현재 컬럼: {list(df_display.columns)}")
                    st.stop()

                excel_list = df_display[lang_choice].dropna().astype(str).tolist()

                # ✅ 매칭 수행
                res_df, extras = match_lists(excel_list, ocr_items)

                # 사용자 슬라이더 기준으로 판정 업데이트(오타 기준 커스터마이즈)
                # (위 match_lists 기본 기준도 있지만, 여기서 너 기준(sim_threshold)으로 한번 더 정리)
                for i in range(len(res_df)):
                    if res_df.loc[i, "판정"] == "✅ 일치":
                        continue
                    score = float(res_df.loc[i, "유사도"])
                    if res_df.loc[i, "판정"] == "❌ 미검출":
                        continue
                    if score >= sim_threshold:
                        res_df.loc[i, "판정"] = "🟡 오타/표기차이 가능"
                    else:
                        res_df.loc[i, "판정"] = "🟠 매칭불안(다른 성분 가능)"

                st.markdown("---")
                st.subheader("📋 비교 리포트 (오타/차이 탐지)")
                st.dataframe(style_result(res_df), use_container_width=True, height=520)

                # ✅ OCR에만 있는 항목 표시(엑셀에 없는데 라벨에 있는 것)
                st.markdown("---")
                st.subheader("➕ OCR에만 존재하는 항목(추가/불필요 성분 후보)")
                if extras:
                    st.write(extras)
                else:
                    st.write("없음")

            except Exception as e:
                st.error(f"오류 발생: {e}")

# =========================
# 모드 2: PDF vs PDF (원본 유지)
# =========================
elif mode == "PDF vs PDF (시각적 차이)":
    st.title("🖼️ PDF/이미지 시각적 차이 분석")
    st.markdown("원본과 수정본을 업로드하면 **바뀐 부분만 빨간색 박스**로 표시합니다.")

    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("📄 원본(Base) 업로드", type=['pdf', 'jpg', 'png'], key="old")
    with col2:
        file_new = st.file_uploader("📄 수정본(New) 업로드", type=['pdf', 'jpg', 'png'], key="new")

    if file_old and file_new:
        if st.button("🔍 시각적 차이점 찾기", use_container_width=True):
            with st.spinner("이미지 정렬 및 차이 분석 중..."):
                img_old = get_image_from_upload(file_old)
                img_new = get_image_from_upload(file_new)

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
