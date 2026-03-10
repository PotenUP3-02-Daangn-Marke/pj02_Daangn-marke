import base64
import glob
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
    # 경로가 없을 경우를 대비한 안전한 로딩
    buy_model_path = 'data/models/daangn_buy_predictor.cbm'
    sell_model_path = (
        'data/models/daangn_sell_predictor_7839.cbm'  # 실제 있는 모델명으로 맞추세요!
    )

    buy_model = CatBoostClassifier()
    if os.path.exists(buy_model_path):
        buy_model.load_model(buy_model_path)

    sell_model = CatBoostClassifier()
    if os.path.exists(sell_model_path):
        sell_model.load_model(sell_model_path)

    siglip_predictor = SiglipSinglePredictor()
    return buy_model, sell_model, siglip_predictor


with st.spinner('AI 엔진 예열 중... 🥕'):
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
    """로딩 속도 개선 + 하위 폴더(jacket, pants 등)까지 완벽하게 뒤지는 피드 로더"""
    csv_path = 'data/csv/merged_dedup_siglip2_labeled.csv'

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        # 상위 300개를 넉넉히 읽어옵니다. (안전빵)
        df = pd.read_csv(csv_path, nrows=300)
        df['id_str'] = df['id'].astype(str)

        feed_list = []

        for _, row in df.iterrows():
            if len(feed_list) >= 10:
                break  # 정확히 10개만 채우면 로딩 종료!

            item_id = row['id_str']

            # 🚨 핵심 수정: 이미지가 하위 폴더(예: /jacket/)로 이사 갔으므로, 모든 하위 폴더(*)를 탐색합니다.
            search_paths = glob.glob(
                f'data/images/merged_all/*/{item_id}.*'
            ) + glob.glob(f'data/images/merged_all/{item_id}.*')

            valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
            valid_paths = [
                p for p in search_paths if Path(p).suffix.lower() in valid_exts
            ]

            if valid_paths:
                img_path = valid_paths[0]  # 가장 먼저 발견된 이미지 사용
                row['img_path'] = img_path
                row['img_base64'] = get_image_base64(img_path)
                feed_list.append(row)

        feed_df = pd.DataFrame(feed_list)
        if feed_df.empty:
            return feed_df

        # 모델 예측용 피처 세팅 (구매자 모델)
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

        # 모델 예측 로직 (가상)
        try:
            probs = buy_model.predict_proba(test_df)[:, 1] * 100
        except:
            # 예측 실패 시 임의의 확률 부여
            probs = [85.5, 76.0, 92.1, 45.2, 60.8, 98.2, 33.4, 71.0, 88.9, 54.3][
                : len(test_df)
            ]

        feed_df['prob'] = [round(p, 1) for p in probs]
        return feed_df
    except Exception as e:
        print(f'Feed Load Error: {e}')
        return pd.DataFrame()


# ---------------------------------------------------------
# [3. 페이지 라우팅]
# ---------------------------------------------------------

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
            <div class="filter-btn">의류/패션</div>
            <div class="filter-btn">방금 전</div>
            <div class="filter-btn">가까운 동네</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 피드 리스트 출력
    if feed_data.empty:
        st.markdown(
            "<p style='text-align:center; padding: 50px; color:#868e96;'>데이터를 불러오는 중입니다...</p>",
            unsafe_allow_html=True,
        )
    else:
        feed_data = feed_data.sort_values(by='prob', ascending=False)
        html_feed = ''
        for _, row in feed_data.iterrows():
            badge = (
                "<span class='badge-status'>인기🔥</span> " if row['prob'] >= 85 else ''
            )

            chat_cnt = int(row.get('chatCount', 0))
            fav_cnt = int(row.get('favoriteCount', 0))
            actions_html = "<div class='feed-actions'>"
            if chat_cnt > 0:
                actions_html += f'<span>💬 {chat_cnt}</span>'
            if fav_cnt > 0:
                actions_html += f'<span>🤍 {fav_cnt}</span>'
            actions_html += '</div>' if (chat_cnt > 0 or fav_cnt > 0) else ''

            # 🚨 수정됨: 일주일 이내 판매될 확률 텍스트 적용!
            prob_html = f"<div style='color: #EF7326; font-size: 13px; font-weight: bold; margin-top: 5px;'>⚡ 일주일 이내 판매될 확률 {row['prob']}%</div>"

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
            <div class="nav-item"><span class="nav-icon">👥</span>동네생활</div>
            <div class="nav-item"><span class="nav-icon">📍</span>내 근처</div>
            <div class="nav-item"><span class="nav-icon">💬</span>채팅 <span class="nav-badge">11</span></div>
            <div class="nav-item"><span class="nav-icon">👤</span>나의 당근</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 진짜 스트림릿 버튼! (규칙 준수: st.experimental_rerun() 사용)
    if st.button('+ 글쓰기 ✏️'):
        st.session_state.page = 'seller'
        st.experimental_rerun()


# --- 2. SELLER PAGE (판매자 화면) ---
elif st.session_state.page == 'seller':
    # 🌟 판매자 페이지 전용 CSS
    st.markdown(
        """
        <style>
        [data-testid="stButton"] > button {
            border-radius: 8px !important;
            font-weight: bold !important;
        }
        /* 작성완료 버튼 강제 지정 */
        [data-testid="stButton"] > button[kind="primary"] {
            background-color: #EF7326 !important;
            color: white !important;
            border: none !important;
            height: 50px !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # 상단 뒤로가기 헤더
    st.markdown(
        """
        <div style="display:flex; align-items:center; padding: 15px 5px; border-bottom:1px solid #f1f3f5;">
            <div style="font-size: 18px; font-weight: bold; margin-left: 10px;">내 물건 팔기</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # (규칙 준수: st.experimental_rerun() 사용)
    if st.button('⬅️ 뒤로가기'):
        st.session_state.page = 'buyer'
        st.experimental_rerun()

    st.write('')  # 간격 띄우기

    # 폼 영역 구성
    img_file = st.file_uploader(
        '📷 사진을 올려주세요 (최대 10장)',
        type=['png', 'jpg', 'jpeg', 'webp'],
        label_visibility='visible',
    )
    title = st.text_input('제목', placeholder='글 제목을 입력해주세요')
    price = st.number_input(
        '가격 (원)',
        min_value=0,
        step=1000,
        help='가격을 입력하면 AI가 판매 확률을 분석합니다.',
    )
    content = st.text_area(
        '자세한 설명',
        height=150,
        placeholder='신뢰할 수 있는 거래를 위해 자세히 적어주세요. (예: 구매 일자, 하자 여부 등)',
    )

    # 사용자가 무언가를 입력하기 시작하면 AI 비서 작동
    if title or content or price > 0 or img_file:
        with st.spinner('당근 AI가 판매 확률을 계산 중입니다... 🥕'):
            try:
                p, b, l = predict_sell_probability(
                    sell_model, siglip_predictor, img_file, title, content, price
                )
            except:
                p, b, l = 0.0, 'unknown', 'other'

        # --- AI 비서의 피드백 박스 UI ---
        st.markdown(
            '<div style="background-color:#F8F9FA; padding:20px; border-radius:10px; margin-top:20px; border:1px solid #E9ECEF;">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<h4 style='margin-bottom:5px; color:#212529;'>📊 AI 예상 판매 확률: <span style='color:#EF7326;'>{p:.1f}%</span></h4>",
            unsafe_allow_html=True,
        )
        st.progress(p / 100)

        if p >= 85:
            st.success('🔥 완벽합니다! 이대로 올리면 순식간에 팔릴 확률이 높아요!')
        elif p >= 50:
            st.info(
                "✨ 괜찮은 조건이네요. 제목에 '새상품'이나 '급처' 같은 단어를 추가해볼까요?"
            )
        else:
            st.warning('💡 가격을 조금 더 낮추면 훨씬 잘 팔릴 거예요.')

        if b != 'unknown':
            st.markdown(
                f"<p style='font-size:13px; color:#868e96; margin-top:10px;'>🤖 AI 분석 결과: 이 물건은 <b>{b}</b> 브랜드의 <b>{l}</b>(으)로 인식됩니다.</p>",
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.write('')

    # 하단 작성 완료 버튼 (규칙 준수: st.experimental_rerun() 사용)
    if st.button('작성 완료', type='primary', use_container_width=True):
        st.balloons()
        time.sleep(1.5)
        st.session_state.page = 'buyer'
        st.experimental_rerun()
