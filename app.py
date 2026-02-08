import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문안확인용_test 용훈")

# 텍스트에서 특수문자와 공백을 제거하는 함수
def clean_text(text):
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text)

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

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('순서 및 내용 정밀 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])
                overlay = img2.copy()
                changes = []

                # 원본(이미지1)의 텍스트 리스트 생성
                list1 = [clean_text(t) for t in data1['text'] if t.strip()]

                # 비교 대상(이미지2)의 단어들을 하나씩 검사
                for i in range(len(data2['text'])):
                    txt2_raw = data2['text'][i].strip()
                    
                    # 신뢰도가 낮은 데이터 제외
                    if not txt2_raw or int(data2['conf'][i]) < 40:
                        continue
                    
                    txt2_clean = clean_text(txt2_raw)
                    
                    # [의도 반영 핵심 로직] 
                    # 현재 단어가 원본의 '비슷한 위치(인덱스)'에 있는지 확인
                    is_changed = True
                    # 주변 5단어 정도의 범위를 탐색하여 순서 밀림 허용
                    search_range = range(max(0, i-3), min(len(list1), i+4))
                    for idx in search_range:
                        if list1[idx] == txt2_clean:
                            is_changed = False
                            break
                    
                    # 만약 주변 순서에 이 단어가 없다면 (내용이 바뀌었거나 순서가 심하게 밀림)
                    if is_changed:
                        tx, ty, tw, th = data2['left'][i], data2['top'][i], data2['width'][i], data2['height'][i]
                        
                        # 빨간색 하이라이트 표시
                        roi = overlay[ty:ty+th, tx:tx+tw]
                        red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                        overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)
                        
                        changes.append({"내용": txt2_raw, "상태": "🔄 위치/내용 변경"})

                # 결과 레이아웃 출력
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1, caption="원본(수정 전)", use_container_width=True)
                with col2:
                    st.image(overlay, caption="변경 영역 하이라이트", use_container_width=True)
                
                if changes: 
                    st.subheader("📋 실질적 변경 내용 (순서 불일치 포함)")
                    st.table(pd.DataFrame(changes).drop_duplicates('내용'))
                else: 
                    st.success("내용 및 순서상 변경된 문구가 없습니다.")
                    
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")