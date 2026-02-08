import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd
from difflib import SequenceMatcher

st.set_page_config(page_title="라벨 체크 AI 리포트", layout="wide")
st.title("🔍 전성분 문안확인 테스트 용훈")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0].convert('RGB'))
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # OCR 데이터 및 좌표 추출
    ocr_data = pytesseract.image_to_data(img, lang='kor+eng', output_type=pytesseract.Output.DICT)
    return img, ocr_data

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('이미지 배경을 무시하고 텍스트 내용만 정밀 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                # 이미지 크기 맞춤 (시각화용)
                h2, w2, _ = img2.shape
                img1_res = cv2.resize(img1, (w2, h2))
                
                overlay = img2.copy()
                changes = []

                # 이미지 1의 텍스트 리스트 생성 (공백 제거 버전)
                # 정밀한 비교를 위해 텍스트가 있는 인덱스만 추출
                list1_clean = [t.strip().replace(" ", "") for t in data1['text'] if t.strip()]
                list1_raw = [t.strip() for t in data1['text'] if t.strip()]

                # 이미지 2의 텍스트를 순회하며 이미지 1과 대조
                for i in range(len(data2['text'])):
                    txt2_raw = data2['text'][i].strip()
                    if not txt2_raw or int(data2['conf'][i]) < 45: continue
                    
                    txt2_clean = txt2_raw.replace(" ", "")
                    
                    # 1. 완벽 일치 여부 확인 (공백 포함)
                    found_exact = False
                    found_content_only = False
                    
                    for j, txt1_raw in enumerate(list1_raw):
                        if txt1_raw == txt2_raw:
                            found_exact = True
                            break
                        # 2. 공백 제거 후 내용만 일치하는지 확인 (자간 차이)
                        elif txt1_raw.replace(" ", "") == txt2_clean:
                            found_content_only = True
                            break

                    # 하이라이트 및 리포트 로직
                    tx, ty, tw, th = data2['left'][i], data2['top'][i], data2['width'][i], data2['height'][i]
                    
                    if not found_exact and not found_content_only:
                        # 아예 새로운 내용이거나 수정된 경우
                        roi = overlay[ty:ty+th, tx:tx+tw]
                        red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                        overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)
                        changes.append({"상태": "🔄 내용 수정/추가", "내용": txt2_raw, "비고": "이미지1에 없는 텍스트"})
                    
                    elif found_content_only:
                        # 내용은 같으나 자간/공백이 다른 경우 (노란색 하이라이트 - 선택 사항)
                        # 여기서는 빨간색 대신 노란색으로 스타일 차이를 표시할 수도 있습니다.
                        roi = overlay[ty:ty+th, tx:tx+tw]
                        yellow_rect = np.full(roi.shape, (255, 255, 0), dtype=np.uint8)
                        overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.8, yellow_rect, 0.2, 0)
                        changes.append({"상태": "⚠️ 스타일 변경", "내용": txt2_raw, "비고": "텍스트 동일 / 자간 및 공백 차이"})

                # 결과 출력
                col1, col2 = st.columns(2)
                with col1: st.image(img1_res, caption="수정 전 (원본)", use_container_width=True)
                with col2: st.image(overlay, caption="변경 영역 하이라이트 (노랑:스타일 / 빨강:내용)", use_container_width=True)

                st.subheader("📋 실질적 변경 내용 정밀 분석 결과")
                if changes:
                    st.table(pd.DataFrame(changes))
                    st.balloons()
                else:
                    st.success("텍스트 내용 및 스타일이 완벽히 일치합니다.")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")