import base64
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

# 모델 로딩 시 불필요한 인터넷 연결 경고 방지
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 🌟 반드시 파일 최상단에 위치해야 하는 설정
st.set_page_config(page_title='당근막캐', page_icon='🥕', layout='centered')

# ---------------------------------------------------------
# [1. 기본 설정 및 모바일 라이트모드 CSS]
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* 전체 배경 라이트모드 및 모바일 앱처럼 좁게 설정 */
    .stApp { background-color: #FFFFFF; color: #212529; }
    .block-container { max-width: 500px !important; padding: 0 !important; padding-bottom: 80px !important; margin: 0 auto; }
    
    /* 숨길 기본 Streamlit 요소들 */
    header {visibility: hidden;}
    
    /* 상단 헤더 (석촌동, 아이콘) */
    .app-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px 20px; position: sticky; top: 0; background-color: #FFFFFF; z-index: 100;
    }
    .header-title { font-size: 18px; font-weight: bold; color: #212529; display: flex; align-items: center; gap: 4px; }
    .header-icons span { font-size: 22px; margin-left: 15px; color: #212529; cursor: pointer; }
    
    /* 가로 스크롤 필터 버튼 */
    .filter-scroll {
        display: flex; gap: 8px; padding: 0 20px 15px 20px; overflow-x: auto; scrollbar-width: none;
        border-bottom: 1px solid #f1f3f5;
    }
    .filter-scroll::-webkit-scrollbar { display: none; }
    .filter-btn {
        background-color: #f1f3f5; border-radius: 20px; padding: 7px 14px;
        font-size: 13px; white-space: nowrap; color: #495057; border: 1px solid #f1f3f5;
    }
    .filter-btn.active { background-color: #212529; color: #FFFFFF; font-weight: bold; border: 1px solid #212529; }
    
    /* 피드 아이템 리스트 */
    .daangn-feed-container {
        display: flex; padding: 15px 20px; border-bottom: 1px solid #f1f3f5; position: relative;
    }
    .daangn-feed-img {
        width: 108px; height: 108px; border-radius: 8px; object-fit: cover; flex-shrink: 0; border: 1px solid #f8f9fa;
    }
    .daangn-feed-info {
        padding-left: 15px; display: flex; flex-direction: column; width: 100%; justify-content: flex-start;
    }
    .daangn-feed-title {
        font-size: 15px; color: #212529; margin: 0 0 4px 0; line-height: 1.4;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; padding-right: 15px;
    }
    .daangn-feed-meta { font-size: 12px; color: #868e96; margin-bottom: 2px; }
    .daangn-feed-price { font-size: 16px; font-weight: bold; color: #212529; margin: 0;}
    
    /* 상태 뱃지 및 하단 아이콘 */
    .badge-status {
        background-color: #009e73; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;
    }
    .feed-actions { position: absolute; right: 20px; bottom: 15px; color: #868e96; font-size: 13px; display: flex; gap: 5px; align-items: center;}
    .kebab-menu { position: absolute; right: 20px; top: 15px; color: #adb5bd; font-size: 18px; line-height: 1; }
    
    /* 하단 네비게이션 바 */
    .bottom-nav {
        position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); width: 100%; max-width: 500px;
        background-color: #FFFFFF; border-top: 1px solid #f1f3f5;
        display: flex; justify-content: space-around; padding: 10px 0 20px 0; z-index: 100;
    }
    .nav-item { display: flex; flex-direction: column; align-items: center; font-size: 10px; color: #495057; position: relative;}
    .nav-item.active { color: #212529; font-weight: bold; }
    .nav-icon { font-size: 22px; margin-bottom: 3px; }
    .nav-badge { background: #FF7E36; color: white; border-radius: 10px; padding: 1px 5px; font-size: 9px; font-weight: bold; position: absolute; top: -2px; right: -5px; }
    
    /* '+ 글쓰기' 플로팅 버튼 */
    button[kind="primary"] {
        position: fixed !important; bottom: 90px !important; left: 50% !important; transform: translateX(120px) !important;
        background-color: #FF7E36 !important; color: white !important;
        border-radius: 30px !important; padding: 12px 20px !important; font-size: 16px !important; font-weight: bold !important;
        border: none !important; box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important; z-index: 999 !important;
    }
    
    /* 스플래시 화면 (글자 없이 로고만! 매끄럽게 사라짐) */
    .splash-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: #FFFFFF; z-index: 9999;
        display: flex; justify-content: center; align-items: center;
        /* 2초 동안 보여준 후 0.5초 동안 투명해지며 사라짐 */
        animation: fadeOutSplash 0.5s ease-in-out 2.0s forwards;
    }
    @keyframes fadeOutSplash {
        0% { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- 백엔드 모듈 임포트 -------------------
from src.predict_pipeline import predict_sell_probability
from src.siglip_predictor import SiglipSinglePredictor

if 'page' not in st.session_state:
    st.session_state.page = 'buyer'


# ---------------------------------------------------------
# [2. 모델 및 데이터 로드 함수]
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    buy_model = CatBoostClassifier().load_model('data/models/daangn_buy_predictor.cbm')
    sell_model = CatBoostClassifier().load_model(
        'data/models/daangn_sell_predictor_7839.cbm'
    )
    siglip_predictor = SiglipSinglePredictor()
    return buy_model, sell_model, siglip_predictor


buy_model, sell_model, siglip_predictor = load_all_models()


def get_image_base64(image_path):
    """로컬 이미지를 HTML에 바로 뿌리기 위해 Base64로 인코딩"""
    try:
        with open(image_path, 'rb') as f:
            return f'data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}'
    except:
        return 'https://via.placeholder.com/100?text=Error'


@st.cache_data
def load_real_feed_data():
    # 1. 하위 폴더를 포함하여 data/images/ 내의 모든 사진 파일 긁어오기
    image_dir = Path('data/images')
    image_paths_dict = {}

    if image_dir.exists():
        # jpg, png, jpeg 등 모든 확장자 검색
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            for file_path in image_dir.rglob(ext):
                file_id = file_path.stem  # 파일명에서 확장자를 제외한 이름 추출 (id)
                image_paths_dict[str(file_id)] = str(file_path)

    if not image_paths_dict:
        st.warning('data/images/ 폴더 내에 이미지가 하나도 없습니다.')
        return pd.DataFrame()

    # 2. 지정해주신 CSV 로드
    csv_path = 'data/team_csv/merged_dedup_siglip2_labeled.csv'

    # 만약 저 경로에 없다면 혹시 몰라 data/csv/ 쪽도 체크
    if not os.path.exists(csv_path):
        csv_path = 'data/csv/merged_dedup_siglip2_labeled.csv'
        if not os.path.exists(csv_path):
            st.error(f'CSV 파일을 찾을 수 없습니다: {csv_path}')
            return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        df['id_str'] = df['id'].astype(str)

        # 3. 내 컴퓨터에 진짜로 존재하는 이미지의 ID만 남기기 (핵심!)
        df = df[df['id_str'].isin(image_paths_dict.keys())].copy()

        if df.empty:
            return pd.DataFrame()

        # 10개만 자르기
        df = df.head(10)

        # 실제 매핑된 이미지 경로를 df에 넣기
        df['img_path'] = df['id_str'].map(image_paths_dict)
        df['img_base64'] = df['img_path'].apply(get_image_base64)

        # 모델 예측용 피처 조립
        test_df = pd.DataFrame()
        test_df['price'] = df.get('price', 0).fillna(0).astype(float)
        test_df['sellerTemperature'] = (
            df.get('sellerTemperature', 36.5).fillna(36.5).astype(float)
        )
        test_df['viewCount'] = df.get('viewCount', 0).fillna(0).astype(float)
        test_df['favoriteCount'] = df.get('favoriteCount', 0).fillna(0).astype(float)
        test_df['chatCount'] = df.get('chatCount', 0).fillna(0).astype(float)
        test_df['title'] = df.get('title', '').fillna('').astype(str)
        test_df['content'] = df.get('content', '').fillna('').astype(str)
        test_df['region_name'] = (
            df.get('region_name', '알 수 없음').fillna('알 수 없음').astype(str)
        )
        test_df['title_len'] = test_df['title'].apply(len)
        test_df['has_keyword_new'] = test_df['title'].apply(
            lambda x: 1 if re.search(r'새상품|미개봉|새제품', x) else 0
        )
        test_df['price_ratio_to_brand'] = 1.0
        test_df['favorite_per_view'] = test_df['favoriteCount'] / (
            test_df['viewCount'] + 1
        )
        test_df['chat_per_view'] = test_df['chatCount'] / (test_df['viewCount'] + 1)
        test_df['is_boosted'] = 0

        # 한 번에 예측
        probs = buy_model.predict_proba(test_df)[:, 1] * 100
        df['prob'] = [round(p, 1) for p in probs]
        return df
    except Exception as e:
        st.error(f'Data Load Error: {e}')
        return pd.DataFrame()


# ---------------------------------------------------------
# [3. 페이지 라우팅]
# ---------------------------------------------------------

# --- 1. BUYER FEED ---
if st.session_state.page == 'buyer':
    # 🌟 스플래시 오버레이 (당근 로고만 화면 중앙에 표시)
    if 'splash_shown' not in st.session_state:
        # github에 올려진 글자 없는 당근마켓 심볼 로고 사용
        st.markdown(
            """
            <div class="splash-overlay">
                <img src="https://avatars.githubusercontent.com/u/60900933?s=200&v=4" width="85" style="border-radius:18px;">
            </div>
        """,
            unsafe_allow_html=True,
        )
        st.session_state.splash_shown = True

    feed_data = load_real_feed_data()

    # 상단 헤더 및 스크롤 필터
    st.markdown(
        """
        <div class="app-header">
            <div class="header-title">석촌동 <span style="font-size: 14px; margin-top:2px;">⌄</span></div>
            <div class="header-icons">
                <span style="font-size: 18px;">🔍</span><span style="font-size: 18px;">🔔</span><span style="font-size: 20px;">≡</span>
            </div>
        </div>
        <div class="filter-scroll">
            <div class="filter-btn active">전체</div>
            <div class="filter-btn">중고거래</div>
            <div class="filter-btn">중고차 직거래</div>
            <div class="filter-btn">방금 전</div>
            <div class="filter-btn">가까운 동네</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # 피드 리스트
    if feed_data.empty:
        st.markdown(
            "<p style='text-align:center; padding: 50px; color:#868e96;'>조건에 맞는 게시글(이미지 포함)이 없습니다.</p>",
            unsafe_allow_html=True,
        )
    else:
        feed_data = feed_data.sort_values(by='prob', ascending=False)
        html_feed = ''
        for _, row in feed_data.iterrows():
            badge = (
                "<span class='badge-status'>예약중</span> " if row['prob'] >= 80 else ''
            )

            chat_cnt = int(row.get('chatCount', 0))
            fav_cnt = int(row.get('favoriteCount', 0))
            actions_html = "<div class='feed-actions'>"
            if chat_cnt > 0:
                actions_html += f'<span>💬 {chat_cnt}</span>'
            if fav_cnt > 0:
                actions_html += f'<span>🤍 {fav_cnt}</span>'
            actions_html += '</div>' if (chat_cnt > 0 or fav_cnt > 0) else ''

            # 가격 밑에 실시간 판매 확률 추가!
            prob_html = f"<div style='color: #FF7E36; font-size: 12px; font-weight: bold; margin-top: 4px;'>⚡ 판매 확률 {row['prob']}%</div>"

            html_feed += f"""
            <div class="daangn-feed-container">
                <img class="daangn-feed-img" src="{row['img_base64']}">
                <div class="daangn-feed-info">
                    <div class="kebab-menu">⋮</div>
                    <p class="daangn-feed-title">{row['title']}</p>
                    <p class="daangn-feed-meta">{row['region_name']} · 방금 전</p>
                    <p class="daangn-feed-price">{badge}{int(row['price']):,}원</p>
                    {prob_html}
                    {actions_html}
                </div>
            </div>
            """
        st.markdown(html_feed, unsafe_allow_html=True)

    # 하단 네비게이션
    st.markdown(
        """
        <div class="bottom-nav">
            <div class="nav-item active"><span class="nav-icon">🏠</span>홈</div>
            <div class="nav-item"><span class="nav-icon">👥</span>커뮤니티</div>
            <div class="nav-item"><span class="nav-icon">📍</span>동네지도</div>
            <div class="nav-item"><span class="nav-icon">💬</span>채팅 <span class="nav-badge">11</span></div>
            <div class="nav-item"><span class="nav-icon">👤</span>나의 당근</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # 플로팅 버튼 (+ 글쓰기)
    if st.button('+ 글쓰기', type='primary'):
        st.session_state.page = 'seller'
        st.experimental_rerun()

# --- 2. SELLER PAGE ---
elif st.session_state.page == 'seller':
    if st.button('⬅️ 뒤로가기'):
        st.session_state.page = 'buyer'
        st.experimental_rerun()

    st.header('📤 내 물건 올리기')
    img_file = st.file_uploader('사진 등록', type=['png', 'jpg', 'jpeg', 'webp'])
    title = st.text_input('제목', placeholder='제품명을 입력하세요')
    price = st.number_input('가격 (원)', min_value=0, step=1000)
    content = st.text_area('내용', height=100)

    if title or content or price > 0 or img_file:
        with st.spinner('AI 분석 중...'):
            p, b, l = predict_sell_probability(
                sell_model, siglip_predictor, img_file, title, content, price
            )
        st.markdown('---')
        st.metric('실시간 판매 확률', f'{p:.1f}%')
        st.progress(p / 100)
        st.write(f'💡 분석 결과: **{b}** 브랜드의 **{l}** 항목으로 보이네요!')

    if st.button('등록 완료', type='primary', use_container_width=True):
        st.balloons()
        time.sleep(1)
        st.session_state.page = 'buyer'
        st.experimental_rerun()
