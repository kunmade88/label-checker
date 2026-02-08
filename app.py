import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pdf2image import convert_from_bytes
import re

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문안 확인 테스트중 용훈")

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
    # OCR 데이터 추출 (좌표 정보 포함)
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name) # 파일명 순 정렬
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('문구 순서 및 내용 1:1 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])
                
                overlay = img2.copy()
                changes = []

                # 1. 이미지 1(전)의 순수 텍스트 리스트 생성
                list1_clean = [clean_text(t) for t in data1['text'] if t.strip()]

                # 2. 이미지 2(후)의 유효한 텍스트 인덱스만 추출
                # (OCR 데이터 중 실제 글자가 있는 인덱스만 골라냄)
                valid_indices2 = [i for i, t in enumerate(data2['text']) if t.strip() and int(data2['conf'][i]) >= 40]
                
                # 3. 이미지 2의 단어들을 순서대로(j번째) 이미지 1과 비교
                for j, i in enumerate(valid_indices2):
                    txt2_raw = data2['text'][i].strip()
                    txt2_clean = clean_text(txt2_raw)
                    
                    is_changed = False
                    remark = ""

                    # 상황 A: 이미지 1의 해당 순서에 단어가 없거나 (리스트가 짧음)
                    # 상황 B: 해당 순서의 단어가 서로 다를 때 (순서 바뀜 또는 오타)
                    if j >= len(list1_clean):
                        is_changed = True
                        remark = "항목 추가됨"
                    elif list1_clean[j] != txt2_clean:
                        is_changed = True
                        remark = f"불일치 (원본: {list1_clean[j]})"

                    if is_changed:
                        # 좌표 정보 추출 및 음영 표시
                        tx, ty, tw, th = data2['left'][i], data2['top'][i], data2['width'][i], data2['height'][i]
                        
                        roi = overlay[ty:ty+th, tx:tx+tw]
                        red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                        # 투명도 30%의 빨간색 음영 적용
                        overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)
                        
                        changes.append({
                            "순서": j + 1,
                            "상태": remark,
                            "검출 단어": txt2_raw
                        })

                # 결과 화면 출력
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1, caption="[전] 이미지", use_container_width=True)
                with col2:
                    st.image(overlay, caption="[후] 변경사항 음영 표시", use_container_width=True)
                
                if changes:
                    st.subheader("📋 상세 변경 리포트")
                    st.table(pd.DataFrame(changes))
                else:
                    st.success("전/후 문구 순서와 내용이 완벽하게 일치합니다.")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")