import ast
import importlib
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. 브랜드 추출 모듈 로드
try:
    brand_module = importlib.import_module('03_brandName_from_title_content')
except Exception:
    brand_module = None


# ---------------------------------------------------------
# 🚨 [근본 해결] 공유받은 clean_df2.csv 기반 사실 통계 로드
# ---------------------------------------------------------
def load_fact_statistics():
    # 파일 경로를 새로 공유받은 clean_df2.csv로 설정합니다.
    csv_path = 'data/team_csv/clean_df2.csv'
    if not os.path.exists(csv_path):
        print(f'⚠️ {csv_path} 파일을 찾을 수 없습니다. 기본값(3.5만)을 사용합니다.')
        return {}, {}, 35000.0

    try:
        # 데이터 로드 (불필요한 공백 제거)
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # 가격 데이터 정제
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df = df[df['price'] > 0]

        # 브랜드별/라벨별 평균 가격 계산 (사실 기반의 기준점)
        brand_means = df.groupby('brandName')['price'].mean().to_dict()
        label_means = df.groupby('label')['price'].mean().to_dict()
        global_mean = df['price'].mean()

        print(f'✅ [사실 기반] {len(brand_means)}개 브랜드 통계 로드 완료!')
        return brand_means, label_means, global_mean
    except Exception as e:
        print(f'❌ 데이터 분석 중 에러: {e}')
        return {}, {}, 35000.0


# 앱 시작 시 통계치를 딱 한 번 로드합니다.
BRAND_MEAN_DICT, LABEL_MEAN_DICT, GLOBAL_MEAN = load_fact_statistics()

# ---------------------------------------------------------


def get_brand_and_label(siglip_predictor, image_file, title, content):
    label = 'other'
    brandName = 'unknown'
    if (title or content) and brand_module:
        try:
            t, c = str(title or ''), str(content or '')
            brandName, _, _ = brand_module.extract_brand_name_fast(t, c, f'{t}\n{c}')
        except:
            brandName = 'unknown'
    if image_file and siglip_predictor:
        try:
            label = siglip_predictor.predict(image_file, title, content, brandName)
        except:
            label = 'other'
    return brandName, label


def predict_sell_probability(
    catboost_model,
    siglip_predictor,
    image_file,
    title,
    content,
    price,
    region_name='unknown',
    seller_temp=36.5,
):
    if not title and not content and price == 0:
        return 0.0, 'unknown', 'other'

    # 1. 브랜드/라벨 추출 및 파싱
    brandName, label_raw = get_brand_and_label(
        siglip_predictor, image_file, title, content
    )

    # 딕셔너리 형태의 라벨에서 알맹이만 추출
    if isinstance(label_raw, dict):
        label_str = label_raw.get('final_label', 'other')
    elif isinstance(label_raw, str) and '{' in label_raw:
        try:
            label_str = ast.literal_eval(label_raw).get('final_label', 'other')
        except:
            label_str = 'other'
    else:
        label_str = str(label_raw)

    # 2. 🚨 [중요] 사실 기반 비교 로직
    # CSV에서 가져온 '진짜 평균가'와 내 가격을 비교합니다.
    b_mean = BRAND_MEAN_DICT.get(brandName, GLOBAL_MEAN)
    l_mean = LABEL_MEAN_DICT.get(label_str, GLOBAL_MEAN)

    # 가격이 바뀔 때마다 이 비율이 변하며 모델의 예측값을 흔듭니다.
    price_ratio_to_brand = float(price / b_mean)
    price_ratio_to_label = float(price / l_mean)

    # 3. 기타 피처 계산
    title_len = len(str(title))
    has_keyword_new = 1 if re.search(r'새상품|미개봉|새제품|택포', str(title)) else 0

    # 4. 모델 입력 데이터 생성 (마스터키 적용)
    feature_dict = {
        'price': float(price),
        'price_ratio_to_brand': float(price_ratio_to_brand),
        'sellerTemperature': float(seller_temp),
        'price_ratio_to_label': float(price_ratio_to_label),
        'title_len': float(title_len),
        'has_keyword_new': str(int(has_keyword_new)),
        'title': str(title) if title else '',
        'content': str(content) if content else '',
        'region_name': str(region_name),
        'brandName': str(brandName),
        'label': str(label_str),
    }

    try:
        df_f = pd.DataFrame([feature_dict])
        df_f = df_f[catboost_model.feature_names_]  # 모델 순서대로 정렬
        prob = catboost_model.predict_proba(df_f)[0][1] * 100
        return round(prob, 1), brandName, label_str
    except Exception as e:
        print(f'Prediction Error: {e}')
        return 0.0, brandName, label_str
