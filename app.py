import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문구 변경 정밀 분석 리포트 test 용훈")

# 2. 이미지 처리 함수
def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

# 3. 파일 업로더 (여기가 누락되어서 에러가 났던 겁니다!)
uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('전성분 텍스트 내용 정밀 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                overlay = img2.copy()
                changes = []

                # 이미지 1의 텍스트를 집합으로 저장 (공백 제거)
                list1_content = set(t.strip().replace(" ", "") for t in data1['text'] if t.strip())

                for i in range(len(data2['text'])):
                    txt2_raw = data2['text'][i].strip()
                    if not txt2_raw or int(data2['conf'][i]) < 45: 
                        continue
                    
                    txt2_clean = txt2_raw.replace(" ", "")
                    
                    # 이미 있는 단어면 패스, 없는 단어면 하이라이트
                    if txt2_clean in list1_content:
                        continue 
                    
                    tx, ty, tw, th = data2['left'][i], data2['top'][i], data2['width'][i], data2['height'][i]
                    roi = overlay[ty:ty+th, tx:tx+tw]
                    red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                    overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)
                    
                    changes.append({"상태": "🔄 문구 변경/추가", "내용": txt2_raw, "비고": "원본에 없는 텍스트"})

                # 결과 출력
                col1, col2 = st.columns(2)
                with col1: st.image(img1, caption="수정 전 (원본)", use_container_width=True)
                with col2: st.image(overlay, caption="변경 문구 분석 (빨간색 확인)", use_container_width=True)

                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("모든 문구가 일치합니다.")

            except Exception as e:
                st.error(f"분석 오류: {e}")