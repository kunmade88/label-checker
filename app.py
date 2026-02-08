import streamlit as st
import cv2
import numpy as np
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🧪 전성분 및 문구 변경 내역 정밀 분석 test 용훈")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0])
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # OCR 추출 (문장 단위를 위해 명확하게 추출)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    return img_bgr, text

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 및 리포트 정돈 시작"):
        with st.spinner('노이즈를 제거하며 리포트를 생성 중입니다...'):
            try:
                img1, text1 = get_data_from_upload(uploaded_files[0])
                img2, text2 = get_data_from_upload(uploaded_files[1])

                # 1. 시각적 하이라이트 처리 (투명도 적용)
                h, w, _ = img2.shape
                img1_res = cv2.resize(img1, (w, h))
                diff = cv2.absdiff(img1_res, img2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY) # 감도 약간 조절
                
                kernel = np.ones((15,15), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=1)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500: # 의미 있는 크기의 차이만 하이라이트
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        roi = overlay[y:y+bh, x:x+bw]
                        rect = np.full(roi.shape, (0, 0, 255), dtype=np.uint8)
                        res = cv2.addWeighted(roi, 0.7, rect, 0.3, 0)
                        overlay[y:y+bh, x:x+bw] = res

                # 이미지 출력
                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="수정 전 원본", use_container_width=True)
                with col2: st.image(overlay, caption="수정 후 투명 하이라이트", use_container_width=True)

                # 2. 리포트 생성 (유사도 검사 추가)
                lines1 = [l.strip() for l in text1.splitlines() if len(l.strip()) > 2]
                lines2 = [l.strip() for l in text2.splitlines() if len(l.strip()) > 2]
                
                d = difflib.Differ()
                diff_result = list(d.compare(lines1, lines2))
                
                changes = []
                idx = 0
                while idx < len(diff_result):
                    if idx + 1 < len(diff_result) and diff_result[idx].startswith('- ') and diff_result[idx+1].startswith('+ '):
                        old_txt = diff_result[idx][2:]
                        new_txt = diff_result[idx+1][2:]
                        
                        # 두 문장의 유사도 계산 (0.0 ~ 1.0)
                        similarity = difflib.SequenceMatcher(None, old_txt, new_txt).ratio()
                        
                        # 유사도가 너무 높으면(90% 이상) 단순 OCR 오타일 가능성이 크므로 '내용 수정'으로 묶음
                        if similarity > 0.4: # 문장 구조가 어느정도 비슷할 때만 수정으로 표시
                            changes.append({"구분": "📝 문장 수정", "기존 내용": old_txt, "변경 내용": new_txt})
                        idx += 2
                    elif diff_result[idx].startswith('- '):
                        changes.append({"구분": "❌ 문장 삭제", "기존 내용": diff_result[idx][2:], "변경 내용": "-"})
                        idx += 1
                    elif diff_result[idx].startswith('+ '):
                        changes.append({"구분": "✅ 문장 추가", "기존 내용": "-", "변경 내용": diff_result[idx][2:]})
                        idx += 1
                    else:
                        idx += 1

                st.subheader("📋 정돈된 상세 변경 리포트")
                if changes:
                    # 너무 짧거나 의미 없는 특수문자 위주 데이터는 한 번 더 필터링
                    filtered_changes = [c for c in changes if len(str(c.get('기존 내용')) + str(c.get('변경 내용'))) > 5]
                    st.table(pd.DataFrame(filtered_changes))
                    st.balloons()
                else:
                    st.success("의미 있는 변경 사항이 발견되지 않았습니다.")

            except Exception as e:
                st.error(f"오류: {e}")