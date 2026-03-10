"""
=============================================================================
[파일 3] src/predict_pipeline.py
- 역할: UI(스트림릿)와 AI 모델(브랜드 추출, 이미지 분류, 판매 확률 예측)을 연결합니다.
=============================================================================
"""

import importlib
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    brand_module = importlib.import_module('03_brandName_from_title_content')
except Exception as e:
    print(f'Warning: 03번 브랜드 모듈 로드 실패 ({e})')
    brand_module = None

BRAND_MEAN_DICT = {}
LABEL_MEAN_DICT = {}


def get_brand_and_label(siglip_predictor, image_file, title, content):
    label = 'other'
    brandName = 'unknown'

    if (title or content) and brand_module:
        try:
            title_norm = brand_module.normalize_text_for_brand(title if title else '')
            content_norm = brand_module.normalize_text_for_brand(
                content if content else ''
            )
            full_text = f'{title_norm}\n{content_norm}'
            extracted_brand, _, _ = brand_module.extract_brand_name_fast(
                title_norm, content_norm, full_text
            )
            brandName = extracted_brand
        except Exception as e:
            print(f'Brand Extraction Error: {e}')
            brandName = 'unknown'

    if image_file is not None and siglip_predictor is not None:
        try:
            label = siglip_predictor.predict(image_file, title, content, brandName)
        except Exception as e:
            print(f'Image Labeling Error: {e}')
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

    brandName, label = get_brand_and_label(siglip_predictor, image_file, title, content)

    # 🚨 [핵심 수정] 딕셔너리로 나온 라벨을 텍스트로 안전하게 변환
    if isinstance(label, dict):
        label_str = label.get('fine', str(label))
    else:
        label_str = str(label)

    # label 대신 label_str을 사용하여 에러 원천 차단!
    b_mean = BRAND_MEAN_DICT.get(brandName, price)
    l_mean = LABEL_MEAN_DICT.get(label_str, price)

    price_ratio_to_brand = price / (b_mean + 1)
    price_ratio_to_label = price / (l_mean + 1)

    title_len = len(str(title)) if title else 0
    has_keyword_new = 1 if re.search(r'새상품|미개봉|새제품|택포', str(title)) else 0

    df_features = pd.DataFrame(
        [
            {
                'price': price,
                'sellerTemperature': seller_temp,
                'price_ratio_to_brand': price_ratio_to_brand,
                'price_ratio_to_label': price_ratio_to_label,
                'title_len': title_len,
                'has_keyword_new': has_keyword_new,
                'title': title if title else '',
                'content': content if content else '',
                'region_name': region_name,
            }
        ]
    )

    try:
        probability = catboost_model.predict_proba(df_features)[0][1] * 100
        return round(probability, 1), brandName, label_str
    except Exception as e:
        print(f'Sell Prediction Error: {e}')
        return 0.0, brandName, label_str
