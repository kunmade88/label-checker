import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문구 변경 정밀 분석 리포트 테스트중 용훈")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OCR 데이터 추출
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('문구 내용 정밀 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])
                overlay = img2.copy()
                changes = []

                # [의도 반영 핵심 로직]
                # 원본 이미지의 모든 단어를 '공백과 특수문자를 제거'하고 저장합니다.
                source_words = set(t.strip().replace(" ", "").replace(",", "") for t in data1['text'] if t.strip())

                for i in range(len(data2['text'])):
                    txt2_raw = data2['text'][i].strip()
                    # 신뢰도가 너무 낮은 데이터는 제외
                    if not txt2_raw or int(data2['conf'][i]) < 40: continue
                    
                    # 비교용 텍스트 가공 (공백, 쉼표 제거)
                    txt2_clean = txt2_raw.replace(" ", "").replace(",", "")
                    
                    # [비교] 가공된 텍스트가 원본 단어 주머니에 있다면? -> 이미 있는 성분이므로 패스!
                    if txt2_clean in source_words:
                        continue 
                    
                    # 여기에 걸리는 것들만 진짜 '추가/수정'된 단어입니다.
                    tx, ty, tw, th = data2['left'][i], data2['top'][i], data2['width'][i], data2['height'][i]
                    roi = overlay[ty:ty+th, tx:tx+tw]
                    red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                    overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)
                    changes.append({"내용": txt2_raw, "상태": "🔄 변경/추가됨"})

                col1, col2 = st.columns(2)
                with col1: st.image(img1, caption="원본(수정 전)", use_container_width=True)
                with col2: st.image(overlay, caption="변경 영역 하이라이트", use_container_width=True)
                
                if changes: 
                    st.subheader("📋 실질적 변경 내용")
                    st.table(pd.DataFrame(changes))
                else: 
                    st.success("내용상 변경된 문구가 없습니다.")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")