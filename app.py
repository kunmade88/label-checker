# ... (상단 import 및 get_data_from_upload 함수는 동일) ...

if len(uploaded_files) >= 2:
    uploaded_files.sort(key=lambda x: x.name)
    
    if st.button("🚀 정밀 분석 시작"):
        with st.spinner('전성분 텍스트 내용 정밀 대조 중...'):
            try:
                img1, data1 = get_data_from_upload(uploaded_files[0])
                img2, data2 = get_data_from_upload(uploaded_files[1])

                # 시각화용 배경 이미지 (이미지 2 기준)
                overlay = img2.copy()
                changes = []

                # [STEP 1] 이미지 1의 단어들을 집합(Set)으로 저장 (중복 제거 및 검색 최적화)
                # 띄어쓰기 차이로 인한 오탐지를 막기 위해 공백을 제거하고 저장합니다.
                list1_content = set(t.strip().replace(" ", "") for t in data1['text'] if t.strip())

                # [STEP 2] 이미지 2의 단어들을 하나씩 검사
                for i in range(len(data2['text'])):
                    txt2_raw = data2['text'][i].strip()
                    
                    # 빈 칸이거나 OCR 신뢰도가 낮은 단어는 건너뜀
                    if not txt2_raw or int(data2['conf'][i]) < 45: 
                        continue
                    
                    # 비교를 위해 이미지 2의 단어도 공백 제거
                    txt2_clean = txt2_raw.replace(" ", "")
                    
                    # [STEP 3] 핵심 대조 로직
                    # 이미지 1의 전성분 목록에 이미 존재하는 단어라면 하이라이트 하지 않음!
                    if txt2_clean in list1_content:
                        continue 
                    
                    # 여기에 걸린다면 "내용이 바뀌었거나 새로 추가된" 단어임
                    tx, ty, tw, th = data2['left'][i], data2['top'][i], data2['width'][i], data2['height'][i]
                    
                    # 해당 단어 위치에만 빨간색 음영 처리
                    roi = overlay[ty:ty+th, tx:tx+tw]
                    red_rect = np.full(roi.shape, (255, 0, 0), dtype=np.uint8)
                    overlay[ty:ty+th, tx:tx+tw] = cv2.addWeighted(roi, 0.7, red_rect, 0.3, 0)
                    
                    changes.append({
                        "상태": "🔄 문구 변경/추가", 
                        "내용": txt2_raw, 
                        "비고": "원본에 없는 텍스트"
                    })

                # [STEP 4] 결과 출력
                col1, col2 = st.columns(2)
                with col1: st.image(img1, caption="수정 전 (원본)", use_container_width=True)
                with col2: st.image(overlay, caption="변경 문구 타겟 분석 (빨간색만 확인하세요)", use_container_width=True)

                st.subheader("📋 변경 내용 리포트")
                if changes:
                    st.table(pd.DataFrame(changes))
                else:
                    st.success("모든 전성분 문구가 일치합니다.")

            except Exception as e:
                st.error(f"분석 오류: {e}")