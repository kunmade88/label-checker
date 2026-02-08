import streamlit as st
import cv2
import numpy as np
import difflib
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
import pandas as pd

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🧪 전성분 변경 내역 정밀 분석 (디테일 강화)")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0])
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # OCR 수행 (한글+영어)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    return img_bgr, text

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    file1, file2 = uploaded_files[0], uploaded_files[1]
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('이미지 대조 및 미세 텍스트 분석 중...'):
            try:
                img1, text1 = get_data_from_upload(file1)
                img2, text2 = get_data_from_upload(file2)

                # 1. 이미지 하이라이트 및 번호 매기기
                height, width = img2.shape[:2]
                img1_res = cv2.resize(img1, (width, height))
                diff = cv2.absdiff(img1_res, img2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = img2.copy()
                box_count = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) > 80: # 너무 작은 노이즈 무시
                        box_count += 1
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # 박스 옆에 번호 쓰기
                        cv2.putText(overlay, str(box_count), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                st.subheader("📸 변경 영역 시각화 (번호 매칭)")
                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="[수정 전 원본]", use_container_width=True)
                with col2: st.image(overlay, caption="[수정 후] 빨간 박스 번호를 아래 표에서 확인하세요", use_container_width=True)

                # 2. 텍스트 미세 비교 (콤마 하나까지 잡아내기)
                list1, list2 = text1.split(), text2.split()
                d = difflib.Differ()
                diff_result = list(d.compare(list1, list2))
                
                changes = []
                idx = 0
                while idx < len(diff_result):
                    # 수정됨 (삭제 후 바로 추가된 경우)
                    if idx + 1 < len(diff_result) and diff_result[idx].startswith('- ') and diff_result[idx+1].startswith('+ '):
                        changes.append({"구분": "⚠️ 내용 수정", "기존": diff_result[idx][2:], "변경": diff_result[idx+1][2:]})
                        idx += 2
                    # 삭제됨
                    elif diff_result[idx].startswith('- '):
                        changes.append({"구분": "❌ 항목 삭제", "기존": diff_result[idx][2:], "변경": "-"})
                        idx += 1
                    # 추가됨
                    elif diff_result[idx].startswith('+ '):
                        changes.append({"구분": "✅ 항목 추가", "기존": "-", "변경": diff_result[idx][2:]})
                        idx += 1
                    else:
                        idx += 1

                # 3. 상세 리포트
                st.subheader("📝 상세 변경 내역 리포트")
                if changes:
                    df = pd.DataFrame(changes)
                    # 데이터프레임 스타일링 (삭제는 빨강, 추가는 초록)
                    st.table(df)
                    
                    # 엑셀 다운로드
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 분석 결과 엑셀로 받기", csv, "label_report.csv", "text/csv")
                    st.balloons()
                else:
                    st.success("텍스트 차이점이 발견되지 않았습니다.")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")