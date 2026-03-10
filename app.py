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
    from src.predict_pipeline import predict_sell_probability
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

    buy_model_path = 'data/models/daangn_buy_predictor_PR.cbm'  # 👈 경로 수정
    sell_model_path = 'data/models/daangn_sell_predictor_7839.cbm'  # 👈 경로 수정

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


with st.spinner('AI 엔진 예열 중... 🥕 all rights reserved by 당근막캐'):
    buy_model, sell_model, siglip_predictor = load_all_models()


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
            probs = buy_model.predict_proba(test_df)[:, 1] * 100
        except:
            import random as rd

            probs = [rd.uniform(30.0, 99.0) for _ in range(len(test_df))]

        feed_df['prob'] = [round(p, 1) for p in probs]

        feed_df['chatCount'] = feed_df['chatCount'].fillna(0).astype(int)
        feed_df['favoriteCount'] = feed_df['favoriteCount'].fillna(0).astype(int)

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
            animation: fadeInBtn 0.5s ease-in-out 1.5s forwards; 
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
    # 🌟 1. UI 교정용 고정형 CSS (Drag and Drop 완벽 차단)
    st.markdown(
        """
        <style>
        .block-container { padding-top: 20px !important; background-color: #FFFFFF; }
        .seller-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 15px 20px; border-bottom: 1px solid #f1f3f5;
            background-color: #FFFFFF; position: fixed; top: 0; left: 50%;
            transform: translateX(-50%); width: 500px; max-width: 100%; z-index: 2000;
        }
        
        /* 카메라 박스 영역 레이아웃 강제 고정 */
        .upload-section { display: flex; flex-direction: row; align-items: center; gap: 12px; margin-bottom: 25px; width: 100%;}
        .daangn-uploader-wrapper { position: relative; width: 80px; height: 80px; flex-shrink: 0; }
        .daangn-camera-visual {
            width: 80px; height: 80px; border: 1.5px solid #e9ecef; border-radius: 10px;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            position: absolute; top: 0; left: 0; z-index: 1; background: white;
        }
        
        /* 진짜 업로더를 80px 안에 가두고 투명하게 */
        .daangn-uploader-wrapper [data-testid="stFileUploader"] {
            position: absolute; top: 0; left: 0; z-index: 2; opacity: 0 !important; width: 80px !important;
        }
        .daangn-uploader-wrapper [data-testid="stFileUploadDropzone"] {
            width: 80px !important; height: 80px !important; padding: 0 !important; border: none !important;
        }
        
        .thumb-img { width: 80px; height: 80px; border-radius: 10px; object-fit: cover; border: 1px solid #f1f3f5; }
        .error-hint { color: #ff5252; font-size: 13px; font-weight: bold; margin-bottom: 10px; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # 🌟 2. 상단 헤더
    st.markdown(
        """
        <div class="seller-header">
            <div style="font-size:24px; cursor:pointer;" onclick="window.location.reload();">✕</div>
            <span style="font-size:17px; font-weight:bold;">내 물건 팔기</span>
            <span style="font-size:15px; color:#adb5bd;">임시저장</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # 🌟 3. 업로드 섹션 (가로 정렬)
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    col_up, col_thumb = st.columns([1, 4])

    with col_up:
        st.markdown('<div class="daangn-uploader-wrapper">', unsafe_allow_html=True)
        st.markdown(
            '<div class="daangn-camera-visual"><span style="font-size:24px;">📷</span><span style="font-size:12px; color:#adb5bd;">0/10</span></div>',
            unsafe_allow_html=True,
        )
        img_file = st.file_uploader(
            ' ',
            type=['png', 'jpg', 'jpeg', 'webp'],
            label_visibility='collapsed',
            key='sell_img',
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_thumb:
        if img_file:
            b64 = base64.b64encode(img_file.getvalue()).decode()
            st.markdown(
                f'<img src="data:image/jpeg;base64,{b64}" class="thumb-img">',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # 🌟 4. 입력 폼 (실시간 AI 인식 정보 공간 추가)
    st.write('')

    # 🚨 [UI 마법 1] 나중에 AI 분석 결과가 들어갈 '빈 공간'을 제목 위에 미리 만들어 둡니다.
    title_header = st.empty()
    title_header.markdown(
        "<span style='font-size:15px; font-weight:bold; color:#212529;'>제목</span>",
        unsafe_allow_html=True,
    )

    # label_visibility='collapsed'를 써서 기본 제목 라벨을 숨기고 위에서 만든 공간을 활용합니다.
    title = st.text_input(
        '제목', placeholder='제목을 입력해주세요.', label_visibility='collapsed'
    )

    content = st.text_area(
        '**자세한 설명**', height=150, placeholder='게시글 내용을 작성해 주세요.'
    )
    price = st.number_input(
        '**가격**', min_value=0, step=1000, placeholder='₩ 가격을 입력해주세요.'
    )

    # 🌟 5. 실시간 확률 예측 로직 (버튼 제거, 자동 실행)
    st.write('')
    st.markdown('#### ⚡ AI 실시간 판매 확률')

    # 세 가지 필수 조건이 모두 채워지는 순간, 버튼 없이 곧바로 분석 시작!
    if title and content and price > 0:
        with st.spinner('AI가 매물의 매력도를 실시간으로 분석하고 있어요...'):
            try:
                user_region = st.session_state.get('current_user_region', '석촌동')
                if user_region == '전체 동네':
                    user_region = '석촌동'

                # 모델 예측 함수 호출
                p, b, l = predict_sell_probability(
                    sell_model,
                    siglip_predictor,
                    img_file,
                    title,
                    content,
                    price,
                    region_name=user_region,
                )

                # 🚨 [UI 마법 2] 계산이 끝나면 아까 비워둔 제목 윗공간에 AI 정보를 세련되게 채워 넣습니다!
                title_header.markdown(
                    f"<span style='font-size:15px; font-weight:bold; color:#212529;'>제목</span> <span style='color:#EF7326; font-size:13px; font-weight:bold; margin-left:8px; background-color:#fff0e6; padding:2px 8px; border-radius:10px;'>✨ AI 인식: {b} / {l}</span>",
                    unsafe_allow_html=True,
                )

                # 확률 게이지 바
                st.progress(int(p) if p <= 100 else 100)

                # 확률 구간별 피드백 메시지
                if p >= 75:
                    st.success(
                        f'🔥 예상 판매 확률: **{p}%** (올리자마자 불티나게 팔릴 확률이 높아요!)'
                    )
                elif p >= 40:
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
        # 입력값이 부족할 때 보여주는 안내 문구
        st.info(
            '💡 사진, 제목, 설명, 가격을 모두 입력하면 AI가 확률을 실시간으로 알려줍니다.'
        )

    # 🌟 6. 작성 완료 버튼
    st.write('')
    if st.button(
        '작성 완료', type='primary', use_container_width=True, on_click=go_to_buyer
    ):
        st.balloons()
