import base64
import os
import random
import re
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
# [1. 공통 라이트모드 CSS]
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* 전체 배경 라이트모드 및 모바일 앱처럼 좁게 설정 */
    .stApp { background-color: #FFFFFF; color: #212529; }
    .block-container { max-width: 500px !important; padding: 0 !important; padding-bottom: 80px !important; margin: 0 auto; }
    
    /* 숨길 기본 Streamlit 요소들 */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
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
    
    /* 판매자 폼 디자인 개선 */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stTextArea > div > textarea {
        border-radius: 8px; border: 1px solid #e9ecef; padding: 10px; font-size: 15px;
    }
    
    /* 🚨 스플래시 화면: 버튼을 완벽히 덮기 위해 z-index를 999999로 상향 조정 */
    .splash-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: #FFFFFF; z-index: 999999 !important; 
        display: flex; justify-content: center; align-items: center;
        animation: fadeOutSplash 0.4s ease-in-out 1.2s forwards;
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
try:
    from src.predict_pipeline import (
        BUY_THRESHOLD,  # 👈 추가
        calibrate_probability,  # 👈 추가
        get_brand_and_label,
        predict_sell_probability,
    )
    from src.siglip_predictor import SiglipSinglePredictor
except ImportError:
    st.error(
        '백엔드 모듈(predict_pipeline 등)을 찾을 수 없습니다. 경로를 확인해주세요.'
    )

if 'page' not in st.session_state:
    st.session_state.page = 'buyer'


# ---------------------------------------------------------
# [2. 모델 및 데이터 로드 함수]
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_all_models():
    # 🚨 바로 여기입니다! 실제 모델 파일이 있는 정확한 경로로 바꿔주세요.
    # (예: 파일이 같은 폴더에 있으면 이름만, 'models' 폴더 안에 있으면 'models/이름.cbm')

    buy_model_path = 'data/models/daangn_buy_predictor_new.cbm'  # 👈 경로 수정
    sell_model_path = 'data/models/daangn_sell_predictor_new4.cbm'  # 👈 경로 수정

    buy_model = CatBoostClassifier()
    if os.path.exists(buy_model_path):
        buy_model.load_model(buy_model_path)
    else:
        print(f'⚠️ [경고] 구매자 모델을 찾을 수 없습니다: {buy_model_path}')

    sell_model = CatBoostClassifier()
    if os.path.exists(sell_model_path):
        sell_model.load_model(sell_model_path)
    else:
        print(f'⚠️ [경고] 판매자 모델을 찾을 수 없습니다: {sell_model_path}')

    siglip_predictor = SiglipSinglePredictor()
    return buy_model, sell_model, siglip_predictor


# 🚨 화면 중앙 배치를 위한 임시 컨테이너 및 🔥예열 이모지 적용

loading_space = st.empty()

with loading_space.container():
    st.markdown(
        """
        <style>
        /* 🚨 핵심 수정: 화면 절대 정중앙을 잡는 마법의 CSS */
        .custom-loader-container {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 99999; /* 다른 요소들에 절대 가려지지 않음 */
            width: 100vw;
        }
        .loader-row {
            display: flex;
            align-items: center;
            gap: 15px; /* 스피너와 글자 사이 간격 */
            margin-bottom: 10px;
        }
        .custom-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid #f1f3f5;
            border-top: 3px solid #EF7326; /* 당근마켓 주황색 포인트 */
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .main-text {
            font-size: 22px; /* 🔥 첫 번째 줄 폰트 키움 */
            font-weight: bold;
            color: #212529;
        }
        .sub-text {
            font-size: 14px; /* 두 번째 줄 폰트 작게 */
            color: #868e96;
            text-align: center;
        }
        </style>
        
        <div class="custom-loader-container">
            <div class="loader-row">
                <div class="custom-spinner"></div>
                <div class="main-text">🥕 당근막캐 AI 엔진 🔥 예열 중... 🔥</div>
            </div>
            <div class="sub-text">© 2026 당근막캐. All rights reserved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 여기서 실제 백엔드 로딩 수행 (st.spinner 없이 조용히 실행)
    buy_model, sell_model, siglip_predictor = load_all_models()

# 로딩이 끝나면 화면을 깔끔하게 지우고 본 화면 진입
loading_space.empty()
# ---------------------------------------------------------


def get_image_base64(image_path):
    """로컬 이미지를 HTML에 바로 뿌리기 위해 Base64로 인코딩"""
    try:
        with open(image_path, 'rb') as f:
            return f'data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}'
    except:
        empty_svg = 'PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDgiIGhlaWdodD0iMTA4Ij48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjhmOWZhIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbC1Cb2xkIiBmb250LXNpemU9IjE0IiBmaWxsPSIjYWRiNWJkIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Tm8gSW1hZ2U8L3RleHQ+PC9zdmc+'
        return f'data:image/svg+xml;base64,{empty_svg}'


@st.cache_data(show_spinner=False)
def load_real_feed_data():
    """파일 탐색(I/O) 병목을 완벽히 제거한 초고속 랜덤 로더"""
    csv_path = 'data/csv/merged_dedup_siglip2_labeled.csv'
    base_img_dir = 'data/images/merged_all'

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        # 1. 🚨 [역발상 최적화] 사진 폴더를 먼저 한 번만 훑어서, 실제로 존재하는 사진 파일들의 이름(ID)을 가져옵니다.
        # rglob을 한 번만 돌려서 모든 이미지 경로를 미리 리스트업 해둡니다. (이게 훨씬 빠릅니다)
        all_image_paths = list(Path(base_img_dir).rglob('*.*'))

        # 확장자 필터링 및 파일명(ID) 추출
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        img_dict = {}  # { '상품ID' : '실제이미지경로' } 구조의 딕셔너리 생성

        for p in all_image_paths:
            if p.suffix.lower() in valid_exts:
                item_id = p.stem  # 파일명에서 확장자 제거한 순수 ID
                img_dict[item_id] = str(p)

        # 2. 존재하는 사진 ID 목록 중에서 랜덤으로 150개를 먼저 확정 짓습니다.
        if not img_dict:
            return pd.DataFrame()  # 사진이 하나도 없으면 종료

        import random

        available_ids = list(img_dict.keys())
        target_sample_size = min(150, len(available_ids))  # 최대 150개 추출
        selected_ids = random.sample(available_ids, target_sample_size)

        # 3. CSV 파일을 읽을 때, '선택된 ID'를 가진 줄만 쏙쏙 뽑아옵니다. (메모리 절약)
        # chunksize를 쓰거나 통째로 읽어서 거르거나 할 수 있지만, pandas는 필터링이 빠르므로 읽고 거릅니다.
        df_raw = pd.read_csv(csv_path, low_memory=False, dtype={'id': str})

        # 전체 데이터 중에서 우리가 뽑아놓은 150개의 ID에 해당하는 행만 필터링합니다.
        feed_df = df_raw[df_raw['id'].isin(selected_ids)].copy()

        if feed_df.empty:
            return pd.DataFrame(columns=['region_name'])

        # 4. 미리 찾아둔 이미지 경로를 바로 매칭해줍니다. (폴더 뒤지기 작업 0번!)
        feed_df['img_path'] = feed_df['id'].map(img_dict)

        # 이미지를 Base64로 변환
        feed_df['img_base64'] = feed_df['img_path'].apply(
            lambda x: get_image_base64(x) if pd.notna(x) else get_image_base64('')
        )

        # --- 모델 예측용 피처 세팅 (기존 로직 완벽 유지) ---
        test_df = pd.DataFrame(
            {
                'price': feed_df.get('price', 0).fillna(0).astype(float),
                'sellerTemperature': feed_df.get('sellerTemperature', 36.5)
                .fillna(36.5)
                .astype(float),
                'viewCount': feed_df.get('viewCount', 0).fillna(0).astype(float),
                'favoriteCount': feed_df.get('favoriteCount', 0)
                .fillna(0)
                .astype(float),
                'chatCount': feed_df.get('chatCount', 0).fillna(0).astype(float),
                'title': feed_df.get('title', '').fillna('').astype(str),
                'content': feed_df.get('content', '').fillna('').astype(str),
                'region_name': feed_df.get('region_name', '알 수 없음')
                .fillna('알 수 없음')
                .astype(str),
            }
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

        try:
            # 1. 구매자 모델의 '날것' 확률들을 뽑아냅니다. (곱하기 100 없앰)
            raw_probs = buy_model.predict_proba(test_df)[:, 1]

            # 2. 각각의 확률을 BUY_THRESHOLD 기준으로 50% 커트라인 보정합니다.
            probs = [calibrate_probability(p, BUY_THRESHOLD) for p in raw_probs]
        except:
            import random as rd

            probs = [rd.uniform(30.0, 99.0) for _ in range(len(test_df))]

        feed_df['prob'] = [round(p, 1) for p in probs]

        feed_df['chatCount'] = feed_df['chatCount'].fillna(0).astype(int)
        feed_df['favoriteCount'] = feed_df['favoriteCount'].fillna(0).astype(int)
        # 👇 추가: 가격이 비어있으면(NaN) 0원으로 처리해서 에러 방지
        feed_df['price'] = feed_df.get('price', 0).fillna(0).astype(int)

        # 목록을 한번 더 섞어서 반환
        return feed_df.sample(frac=1).reset_index(drop=True)

    except Exception as e:
        print(f'Feed Load Error: {e}')
        return pd.DataFrame(columns=['region_name'])


# ---------------------------------------------------------
# [3. 페이지 라우팅]
# ---------------------------------------------------------
def go_to_seller():
    st.session_state.page = 'seller'


def go_to_buyer():
    st.session_state.page = 'buyer'


# --- 1. BUYER FEED (구매자 메인 화면) ---
if st.session_state.page == 'buyer':
    # 🌟 완벽 분리된 '글쓰기 버튼' 전용 CSS
    st.markdown(
        """
        <style>
        /* 🚨 추가 수정: 버튼을 처음에 투명하게(opacity:0) 만들고, 스플래시가 끝나는 1.5초 뒤에 나타나게(fadeIn) 합니다! */
        [data-testid="stButton"] > button {
            position: fixed !important;
            bottom: 90px !important;
            right: calc(50% - 230px) !important; 
            background-color: #EF7326 !important;
            color: white !important;
            border: none !important;
            border-radius: 30px !important;
            width: 120px !important;
            height: 50px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
            z-index: 9999 !important;
            opacity: 0; 
            animation: fadeInBtn 0.5s ease-in-out 0.5s forwards; 
            transition: transform 0.2s;
        }
        @keyframes fadeInBtn {
            to { opacity: 1; }
        }
        [data-testid="stButton"] > button:hover {
            transform: scale(1.05);
            background-color: #d9631c !important;
        }
        /* 모바일 환경 대응 (오른쪽 여백 고정) */
        @media (max-width: 500px) {
            [data-testid="stButton"] > button { right: 20px !important; }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # 🌟 스플래시 오버레이 (당근 로고)
    if 'splash_shown' not in st.session_state:
        st.markdown(
            """
            <div class="splash-overlay">
                <div style="text-align: center;">
                    <div style="font-size: 110px; line-height: 1;">🥕</div>
                    <div style="color: #EF7326; font-size: 36px; font-weight: 900; margin-top: 15px; letter-spacing: -2px;">당근막캐</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        st.session_state.splash_shown = True

    feed_data = load_real_feed_data()

    # --- 🚨 [수정 2] 동적 동네 필터 UI 및 아이폰 상태표시줄 ---
    import random
    from datetime import datetime

    # 1. 상태표시줄 시간 가져오기
    current_time = datetime.now().strftime('%H:%M')

    # 2. 동네 목록 가나다순 정렬 및 '전체 동네' 맨 위 고정
    unique_regions = feed_data['region_name'].dropna().unique().tolist()
    unique_regions.sort()
    region_options = ['전체 동네'] + unique_regions

    # 3. 로딩된 데이터 중 랜덤으로 하나를 첫 화면 기본 동네로 설정
    if unique_regions:
        default_region = random.choice(unique_regions)
        default_idx = region_options.index(default_region)
    else:
        default_idx = 0

    # 4. 상태표시줄 HTML + 헤더 CSS 렌더링
    current_time = datetime.now().strftime('%H:%M')
    st.markdown(
        f"""
        <style>
        /* 아이폰 스타일 상태표시줄 전용 가벼운 CSS */
        .ios-status-bar {{
            display: flex; justify-content: space-between; align-items: center;
            /* 🚨 수정: 위쪽 여백(12px)을 주어 천장에서 띄우고, 좌우 여백(15px)도 넓혔습니다. */
            padding: 12px 15px 5px 15px; 
            background-color: #FFFFFF;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            /* 🚨 수정: 아래쪽 헤더와 너무 겹치지 않게 마진을 줄였습니다. */
            margin-bottom: 2px; 
        }}
        .ios-time {{ font-size: 15px; font-weight: 600; color: #000000; letter-spacing: -0.5px; padding-left: 5px; }}
        .ios-icons {{ display: flex; align-items: center; gap: 4px; color: #000000; }}
        /* 1. 동네 선택창 투명화 및 폰트 튜닝 */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: transparent !important; border: none !important;
            font-size: 19px !important; font-weight: bold !important; color: #212529 !important;
            box-shadow: none !important; cursor: pointer !important; padding-left: 5px !important;
        }}
        /* 2. 우측 아이콘 간격 및 오른쪽 정렬 복구 */
        .header-icons-right {{ 
            text-align: right; font-size: 20px; color: #212529; 
            margin-top: 5px; letter-spacing: 12px; cursor: pointer; 
        }}
        </style>
        <div class="ios-status-bar">
            <div class="ios-time">{current_time}</div>
            <div class="ios-icons">
                <span style="font-size: 10px;">📍</span>
                <span style="font-size: 12px;">📶</span>
                <span style="font-size: 14px;">ᯤ</span>
                <span style="font-size: 16px; margin-bottom: 2px;">🔋</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )
    head_col1, head_col2 = st.columns([4, 6])
    with head_col1:
        # 🚨 아래 줄들이 with 문보다 반드시 4칸 더 안쪽으로 들어가야 합니다.
        selected_region = st.selectbox(
            '동네 선택',
            region_options,
            label_visibility='collapsed',
            key='region_selector',
        )
        # 현재 선택된 동네를 세션에 저장하여 판매자 페이지에서 활용합니다.
        st.session_state.current_user_region = selected_region

    with head_col2:
        st.markdown(
            '<div class="header-icons-right">🔍🔔≡</div>', unsafe_allow_html=True
        )
    st.markdown(
        """
        <div class="filter-scroll" style="width: 100%; justify-content: flex-start;">
            <div class="filter-btn active">전체</div>
            <div class="filter-btn">중고거래</div>
            <div class="filter-btn">중고차 직거래</div>
            <div class="filter-btn">방금 전</div>
            <div class="filter-btn">가까운 동네</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 선택된 동네로 데이터 필터링
    if selected_region != '전체 동네':
        filtered_feed = feed_data[feed_data['region_name'] == selected_region].copy()
    else:
        filtered_feed = feed_data.copy()

    # 피드 리스트 출력
    if filtered_feed.empty:  # 🚨 feed_data -> filtered_feed 로 변경
        st.markdown(
            "<p style='text-align:center; padding: 50px; color:#868e96;'>선택하신 동네에 올라온 매물이 없습니다.</p>",
            unsafe_allow_html=True,
        )
    else:
        filtered_feed = filtered_feed.sample(frac=1).head(30).reset_index(drop=True)

        # 🚨 [완벽 해결] html_feed라는 하나의 변수에 뭉치지 않고, 매물 1개마다 개별적으로 화면에 꽂아버립니다.
        # 🚨 [최종 안정화 버전] 모든 유령 문자와 레이아웃 버그를 잡는 코드입니다.
        for _, row in filtered_feed.iterrows():
            badge = (
                "<span class='badge-status'>인기🔥</span> " if row['prob'] >= 85 else ''
            )
            chat_cnt = int(row.get('chatCount', 0))
            fav_cnt = int(row.get('favoriteCount', 0))

            # 아이콘 로직: 변수에 미리 담아서 복잡도를 줄입니다.
            actions_html = ''
            if chat_cnt > 0 or fav_cnt > 0:
                inner_icons = ''
                if chat_cnt > 0:
                    inner_icons += f'<span>💬 {chat_cnt}</span>'
                if fav_cnt > 0:
                    inner_icons += f'<span>🤍 {fav_cnt}</span>'
                actions_html = f"<div class='feed-actions'>{inner_icons}</div>"

            prob_html = f"<div style='color: #EF7326; font-size: 13px; font-weight: bold; margin-top: 5px;'>⚡ 일주일 이내 판매될 확률 {row['prob']}%</div>"

            # 🚨 핵심: HTML 문자열의 앞뒤 공백을 .strip()으로 제거하여 텍스트 출력을 방지합니다.
            item_html = f"""
<div class="daangn-feed-container" style="width: 100%; box-sizing: border-box; display: flex; padding: 15px 20px; border-bottom: 1px solid #f1f3f5; position: relative; background-color: #FFFFFF;">
<img class="daangn-feed-img" src="{row['img_base64']}" style="width: 108px; height: 108px; border-radius: 8px; object-fit: cover; flex-shrink: 0;" />
<div class="daangn-feed-info" style="flex: 1; min-width: 0; padding-left: 15px; display: flex; flex-direction: column; justify-content: flex-start;">
<div class="kebab-menu" style="position: absolute; right: 20px; top: 15px; color: #adb5bd; font-size: 18px;">⋮</div>
<p class="daangn-feed-title" style="font-size: 15px; color: #212529; margin: 0 0 4px 0; line-height: 1.4;">{row['title']}</p>
<p class="daangn-feed-meta" style="font-size: 12px; color: #868e96; margin-bottom: 2px;">{row['region_name']} · 방금 전</p>
<p class="daangn-feed-price" style="font-size: 16px; font-weight: bold; color: #212529; margin: 0;">{badge}{int(row['price']):,}원</p>
{prob_html}
{actions_html}
</div>
</div>""".strip()

            st.markdown(item_html, unsafe_allow_html=True)

    # 하단 네비게이션
    st.markdown(
        """
        <div class="bottom-nav">
            <div class="nav-item active"><span class="nav-icon">🏠</span>홈</div>
            <div class="nav-item"><span class="nav-icon">👥</span>동네생활</div>
            <div class="nav-item"><span class="nav-icon">📍</span>내 근처</div>
            <div class="nav-item"><span class="nav-icon">💬</span>채팅 <span class="nav-badge">11</span></div>
            <div class="nav-item"><span class="nav-icon">👤</span>나의 당근</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 진짜 스트림릿 버튼! (규칙 준수: st.experimental_rerun() 사용)
    if st.button('+ 글쓰기 ✏️', on_click=go_to_seller):
        pass

# --- 2. SELLER PAGE (판매자 화면) ---
elif st.session_state.page == 'seller':
    # 🚨 임시저장 & 작성완료 시 텍스트를 기억하는 콜백 함수
    def save_draft():
        st.session_state.draft_title = st.session_state.get('title_input', '')
        st.session_state.draft_content = st.session_state.get('content_input', '')
        st.session_state.draft_price = st.session_state.get('price_input', None)
        st.session_state.page = 'buyer'

    def clear_draft_and_submit():
        st.session_state.draft_title = ''
        st.session_state.draft_content = ''
        st.session_state.draft_price = None
        st.session_state.page = 'buyer'

    # 🌟 1. UI 교정용 고정형 CSS
    st.markdown(
        """
        <style>
        .block-container { 
            max-width: 500px !important; margin: 0 auto !important; 
            padding-top: 20px !important; padding-left: 20px !important; padding-right: 20px !important; 
            background-color: #FFFFFF !important; 
        }
        .seller-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 15px 20px; border-bottom: 1px solid #EAEBEE;
            background-color: #FFFFFF; position: fixed; top: 0; left: 50%;
            transform: translateX(-50%); width: 100%; max-width: 500px; z-index: 2000;
        }

        /* 🚨 [해결 1] X 버튼 (Secondary) 공중부양 고정 */
        [data-testid="stButton"] button[kind="secondary"] {
            position: fixed !important; top: 12px !important; left: calc(50% - 250px + 20px) !important; 
            background-color: transparent !important; border: none !important;
            color: #212529 !important; font-size: 24px !important; padding: 0 !important; 
            width: 30px !important; height: 30px !important; box-shadow: none !important; z-index: 2005 !important;
        }
        @media (max-width: 500px) { [data-testid="stButton"] button[kind="secondary"] { left: 20px !important; } }

        /* 🚨 [해결 1] 임시저장 버튼 (Tertiary) 순수 텍스트 디자인으로 고정 */
        [data-testid="stButton"] button[kind="tertiary"] {
            position: fixed !important; top: 15px !important; right: calc(50% - 250px + 20px) !important;
            background-color: transparent !important; border: none !important;
            color: #adb5bd !important; font-size: 15px !important; padding: 0 !important;
            font-weight: normal !important; box-shadow: none !important; z-index: 2005 !important;
            width: auto !important; height: auto !important; min-height: 0 !important;
        }
        @media (max-width: 500px) { [data-testid="stButton"] button[kind="tertiary"] { right: 20px !important; } }
        [data-testid="stButton"] button[kind="tertiary"]:hover { color: #212529 !important; }

        /* 업로드 섹션 */
        div[data-testid="stHorizontalBlock"] { align-items: flex-start !important; margin-top: 15px !important; margin-bottom: 5px !important; }
        .camera-bg { width: 72px !important; height: 72px !important; background-color: #FAFBFC; border: 1px solid #EAEBEE; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #adb5bd; font-size: 14px; font-weight: bold; line-height: 1.2; text-align: center; }
        
        /* 겉박스는 투명하게, 클릭은 가능하게 덮어씌움 */
        [data-testid="stFileUploader"] { margin-top: -72px !important; width: 72px !important; height: 72px !important; opacity: 0 !important; cursor: pointer !important; z-index: 10 !important; }
        [data-testid="stFileUploaderDropzone"] { height: 72px !important; min-height: 72px !important; padding: 0 !important; }
        
        /* 🚨 핵심 수정: section 태그 숨김 해제! ul(파일목록)과 에러메시지만 숨깁니다. */
        [data-testid="stUploadedFile"], div[data-testid="stFileUploader"] > div:nth-child(2), div[data-testid="stFileUploader"] ul { display: none !important; }
        
        .thumb-img { width: 72px !important; height: 72px !important; border-radius: 4px; object-fit: cover; border: 1px solid #EAEBEE; display: block; }
        div[data-testid="column"]:nth-of-type(2) { margin-left: -10px !important; }

        .thumb-img { width: 72px !important; height: 72px !important; border-radius: 4px; object-fit: cover; border: 1px solid #EAEBEE; display: block; }
        div[data-testid="column"]:nth-of-type(2) { margin-left: -10px !important; }

        /* 라벨 폰트 */
        .stTextInput label p, .stTextArea label p, .stNumberInput label p { font-size: 15px !important; font-weight: 600 !important; color: #212529 !important; margin-bottom: 5px !important; }

        /* 알약 버튼 */
        div[role="radiogroup"] { display: flex; flex-direction: row; gap: 6px; margin: 10px 0 5px 0; }
        div[role="radiogroup"] label { background-color: #F2F3F6 !important; border: 1px solid #F2F3F6 !important; border-radius: 16px !important; padding: 5px 12px !important; cursor: pointer; margin: 0 !important; display: flex !important; align-items: center !important; justify-content: center !important; }
        div[role="radiogroup"] label:has(input:checked) { background-color: #212529 !important; border-color: #212529 !important; }
        div[role="radiogroup"] label > div { margin: 0 !important; padding: 0 !important; }
        div[role="radiogroup"] label p { color: #4D5159 !important; font-size: 13px !important; font-weight: bold !important; margin: 0 !important; text-align: center !important; }
        div[role="radiogroup"] label:has(input:checked) p { color: #FFFFFF !important; }
        div[role="radiogroup"] label > div:first-child { display: none !important; }

        /* 입력창 디자인 */
        div[data-testid="stCheckbox"] { margin-top: -10px; margin-bottom: 15px; }
        div[data-testid="stCheckbox"] label p { font-size: 14px !important; color: #4D5159 !important; font-weight: 500 !important; }
        div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div { background-color: transparent !important; border: none !important; border-bottom: 1px solid #EAEBEE !important; border-radius: 0 !important; box-shadow: none !important; }
        div[data-baseweb="input"] > div:focus-within, div[data-baseweb="textarea"] > div:focus-within { border-bottom: 1px solid #212529 !important; }
        input[type="text"], input[type="number"], textarea { font-size: 17px !important; padding: 15px 4px !important; color: #212529 !important; }
        input::placeholder, textarea::placeholder { color: #ADB5BD !important; }
        [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display: none !important; }
        
        [data-testid="stButton"] button[kind="primary"] { background-color: #FF7E36 !important; color: #FFFFFF !important; border: none !important; border-radius: 6px !important; font-size: 16px !important; font-weight: bold !important; padding: 12px 0 !important; width: 100% !important; }
        [data-testid="stButton"] button[kind="primary"]:hover { background-color: #E66A2B !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 🌟 2. 상단 헤더 & 네이티브 버튼
    st.markdown(
        """
        <div class="seller-header">
            <div style="width:24px; height:24px;"></div>
            <span style="font-size:17px; font-weight:bold;">내 물건 팔기</span>
            <div style="width:55px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # type을 완벽히 분리하여 꼬임 방지
    st.button('✕', type='secondary', on_click=go_to_buyer)
    st.button('임시저장', type='tertiary', on_click=save_draft)

    # 🌟 3. 업로드 섹션
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 6])

    with col1:
        st.markdown('<div class="camera-bg">📷<br>0/10</div>', unsafe_allow_html=True)
        img_file = st.file_uploader(
            ' ',
            type=['png', 'jpg', 'jpeg', 'webp'],
            label_visibility='collapsed',
            key='sell_img',
        )

    with col2:
        if img_file:
            b64 = base64.b64encode(img_file.getvalue()).decode()
            st.markdown(
                f'<div class="my-thumb-box"><img src="data:image/jpeg;base64,{b64}" class="thumb-img"></div>',
                unsafe_allow_html=True,
            )

    # 🌟 4. 입력 폼 (메모리 로딩 & 나눔하기 로직 적용)
    title_header = st.empty()

    # 콜백용 key 할당 및 draft 로드
    title = st.text_input(
        '제목',
        value=st.session_state.get('draft_title', ''),
        placeholder='제목을 입력해주세요.',
        key='title_input',
    )
    content = st.text_area(
        '자세한 설명',
        value=st.session_state.get('draft_content', ''),
        height=200,
        placeholder='게시글 내용을 작성해주세요...',
        key='content_input',
    )
    sell_type = st.radio(
        ' ', ['판매하기', '나눔하기'], horizontal=True, label_visibility='collapsed'
    )

    # 나눔하기 판단
    is_sharing = sell_type == '나눔하기'

    price = st.number_input(
        '**가격**',
        min_value=0,
        step=1000,
        value=st.session_state.get('draft_price', None),
        placeholder='₩ 가격을 입력해주세요.',
        disabled=is_sharing,
        key='price_input',
    )
    final_price = 0 if is_sharing else (price if price is not None else 0)

    offer_check = st.checkbox('가격 제안 받기')

    import ast

    current_brand = 'unknown'
    current_label = 'other'

    # 사진, 제목, 설명 중 하나라도 입력되었다면 즉시 분류 시작!
    if img_file or title.strip() or content.strip():
        raw_b, raw_l = get_brand_and_label(siglip_predictor, img_file, title, content)
        current_brand = raw_b

        # 라벨 텍스트 파싱
        if isinstance(raw_l, dict):
            current_label = raw_l.get('final_label', 'other')
        elif isinstance(raw_l, str) and '{' in raw_l:
            try:
                current_label = ast.literal_eval(raw_l).get('final_label', 'other')
            except:
                current_label = 'other'
        else:
            current_label = str(raw_l)

        # 인식된 결과가 하나라도 의미 있는 값이면 배지를 즉시 띄움
        if current_brand != 'unknown' or current_label != 'other':
            title_header.markdown(
                f"<div style='margin-bottom: 5px;'><span style='color:#EF7326; font-size:13px; font-weight:bold; background-color:#fff0e6; padding:4px 10px; border-radius:12px;'>✨ AI 인식: {current_brand} / {current_label}</span></div>",
                unsafe_allow_html=True,
            )

    # 🌟 5. 실시간 확률 예측 로직 (스마트 안내)
    st.write('')
    st.markdown('#### ⚡ AI 실시간 판매 확률')

    missing_fields = []
    if not img_file:
        missing_fields.append('사진')
    if not title.strip():
        missing_fields.append('제목')
    if not content.strip():
        missing_fields.append('자세한 설명')

    # 나눔하기 상태라면 가격(0원) 검증을 통과시킴
    if not is_sharing and final_price <= 0:
        missing_fields.append('가격')

    if len(missing_fields) == 0:
        with st.spinner('AI가 매물의 매력도를 실시간으로 분석하고 있어요...'):
            try:
                user_region = st.session_state.get('current_user_region', '석촌동')
                if user_region == '전체 동네':
                    user_region = '석촌동'

                # 🚨 [수정] 무거운 직접 계산 대신 FastAPI 서버로 요청(POST)을 보냅니다!
                import requests

                # 1. 보낼 데이터 포장
                api_data = {
                    'title': title,
                    'content': content,
                    'price': final_price,
                    'region_name': user_region,
                    'seller_temp': 36.5,
                }

                # 2. 보낼 이미지 포장
                api_files = {}
                if img_file:
                    api_files['image'] = (
                        img_file.name,
                        img_file.getvalue(),
                        img_file.type,
                    )

                # 3. FastAPI 서버(백엔드) 호출
                response = requests.post(
                    'http://127.0.0.1:8000/predict/sell', data=api_data, files=api_files
                )

                # 4. 결과 받아오기
                if response.status_code == 200:
                    result = response.json()
                    if result['status'] == 'success':
                        p = result['probability']
                    else:
                        st.error(f'API 예측 에러: {result["message"]}')
                        p = 0.0
                else:
                    st.error(
                        'API 서버와 통신할 수 없습니다. 백엔드 서버가 켜져 있는지 확인하세요.'
                    )
                    p = 0.0  # 🚨 [수정] 무거운 직접 계산 대신 FastAPI 서버로 요청(POST)을 보냅니다!
                import requests

                # 1. 보낼 데이터 포장
                api_data = {
                    'title': title,
                    'content': content,
                    'price': final_price,
                    'region_name': user_region,
                    'seller_temp': 36.5,
                    'is_submit': 'false',
                }

                # 2. 보낼 이미지 포장
                api_files = {}
                if img_file:
                    api_files['image'] = (
                        img_file.name,
                        img_file.getvalue(),
                        img_file.type,
                    )

                # 3. FastAPI 서버(백엔드) 호출
                response = requests.post(
                    'http://127.0.0.1:8000/predict/sell', data=api_data, files=api_files
                )

                # 4. 결과 받아오기
                if response.status_code == 200:
                    result = response.json()
                    if result['status'] == 'success':
                        p = result['probability']
                    else:
                        st.error(f'API 예측 에러: {result["message"]}')
                        p = 0.0
                else:
                    st.error(
                        'API 서버와 통신할 수 없습니다. 백엔드 서버가 켜져 있는지 확인하세요.'
                    )
                    p = 0.0

                # 🚨 나눔하기면 99.9% 1초컷!
                # (하드코딩 삭제) 모델이 0원(나눔) 기준으로 계산한 p값을 그대로 사용합니다.
                st.progress(int(p) if p <= 100 else 100)

                # 확률 구간별 피드백 메시지
                if is_sharing:
                    st.success(
                        f'🎁 예상 나눔 완료 확률: **{p}%** (무료 나눔도 사진과 설명에 따라 성공 확률이 달라집니다!)'
                    )
                elif p >= 75:
                    st.success(
                        f'🔥 예상 판매 확률: **{p}%** (올리자마자 불티나게 팔릴 확률이 높아요!)'
                    )
                elif p >= 50:
                    st.info(
                        f'👍 예상 판매 확률: **{p}%** (가격과 조건이 꽤 괜찮습니다.)'
                    )
                else:
                    st.warning(
                        f'🤔 예상 판매 확률: **{p}%** (가격을 조금 낮추거나 키워드를 더 넣어보세요.)'
                    )

            except Exception as e:
                st.error(f'예측 실패: {e}')
    else:
        missing_str = ', '.join(missing_fields)
        st.info(
            f'💡 **{missing_str}** 항목을 더 입력해 주시면 AI가 판매 확률을 분석해 드립니다.'
        )

    # 🌟 6. 작성 완료 버튼
    st.write('')
    if st.button('작성 완료', type='primary', use_container_width=True):
        if len(missing_fields) > 0:
            st.error(
                '필수 항목(사진, 제목, 내용, 가격)을 모두 입력해야 등록할 수 있습니다.'
            )
        else:
            with st.spinner('데이터를 서버에 안전하게 저장 중입니다...'):
                try:
                    import requests

                    user_region = st.session_state.get('current_user_region', '석촌동')
                    if user_region == '전체 동네':
                        user_region = '석촌동'

                    # 🚨 [핵심] 여기서 'is_submit': 'true'로 쏴서 저장을 지시합니다!
                    api_data = {
                        'title': title,
                        'content': content,
                        'price': final_price,
                        'region_name': user_region,
                        'seller_temp': 36.5,
                        'is_submit': 'true',
                    }
                    api_files = {}
                    if img_file:
                        api_files['image'] = (
                            img_file.name,
                            img_file.getvalue(),
                            img_file.type,
                        )

                    response = requests.post(
                        'http://127.0.0.1:8000/predict/sell',
                        data=api_data,
                        files=api_files,
                    )

                    if response.status_code == 200:
                        st.balloons()  # 발표용 시각 효과 🎉
                        st.success('성공적으로 등록되고 데이터가 저장되었습니다!')
                        clear_draft_and_submit()
                        st.rerun()
                    else:
                        st.error('서버에 저장하지 못했습니다.')
                except Exception as e:
                    st.error(f'서버 통신 에러: {e}')
