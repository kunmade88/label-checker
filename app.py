import streamlit as st
import cv2
import numpy as np
import difflib
import pytesseract
from pdf2image import convert_from_bytes
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="라벨 체크 AI 스마트 리포트", layout="wide")
st.title("🎯 스마트 인터랙티브 분석 리포트")
st.write("이미지 위 하이라이트에 마우스를 올려 상세 변경 내용을 확인하세요.")

def get_data_from_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith('.pdf'):
        pages = convert_from_bytes(file_bytes)
        img = np.array(pages[0])
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    return img, text

uploaded_files = st.file_uploader("비교할 파일 2개를 선택하세요", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    if st.button("🚀 스마트 분석 시작"):
        try:
            img1, text1 = get_data_from_upload(uploaded_files[0])
            img2, text2 = get_data_from_upload(uploaded_files[1])

            # 1. 차이점 영역 계산
            h, w, _ = img2.shape
            img1_res = cv2.resize(img1, (w, h))
            diff = cv2.absdiff(img1_res, img2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((20,20), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 2. 텍스트 비교 (표 데이터 생성)
            d = difflib.Differ()
            diff_res = list(d.compare(text1.splitlines(), text2.splitlines()))
            changes = [line[2:].strip() for line in diff_res if line.startswith('+ ') or line.startswith('- ')]
            
            # 3. Plotly를 이용한 인터랙티브 이미지 생성
            fig = px.imshow(img2) # 수정 후 이미지를 배경으로 설정
            
            box_idx = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 600:
                    box_idx += 1
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    
                    # 마우스를 올렸을 때 보여줄 텍스트 (최대 3개 문장만 예시로 매칭)
                    hover_text = f"<b>영역 #{box_idx}</b><br>변경 내용 확인 필요"
                    if len(changes) >= box_idx:
                        hover_text = f"<b>영역 #{box_idx}</b><br>내용: {changes[box_idx-1][:30]}..."

                    # 이미지 위에 투명한 사각형 레이어 추가
                    fig.add_shape(
                        type="rect", x0=x, y0=y, x1=x+bw, y1=y+bh,
                        line=dict(color="Red", width=2),
                        fillcolor="Red", opacity=0.2 # 투명도 조절
                    )
                    # 툴팁(Hover) 데이터 추가
                    fig.add_trace(go.Scatter(
                        x=[x + bw/2], y=[y + bh/2],
                        text=[hover_text],
                        mode="markers",
                        marker=dict(opacity=0), # 점은 안 보이게
                        hoverinfo="text",
                        showlegend=False
                    ))

            fig.update_layout(dragmode="pan", width=1000, height=800)
            st.plotly_chart(fig, use_container_width=True)

            # 4. 하단 상세 표
            st.subheader("📝 전체 변경 목록")
            diff_df = []
            for line in diff_res:
                if line.startswith('- '): diff_df.append({"상태": "기존", "내용": line[2:]})
                elif line.startswith('+ '): diff_df.append({"상태": "변경", "내용": line[2:]})
            st.table(pd.DataFrame(diff_df))

        except Exception as e:
            st.error(f"오류: {e}")