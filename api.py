import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import csv
import datetime
import io
import os
import uuid
from contextlib import asynccontextmanager

from catboost import CatBoostClassifier
from fastapi import FastAPI, File, Form, UploadFile

# 🚨 기존에 만들어둔 파이프라인 모듈들 임포트
from src.predict_pipeline import predict_sell_probability
from src.siglip_predictor import SiglipSinglePredictor

# --- [설정] 데이터 저장 경로 ---
DATA_SAVE_DIR = 'collected_data'
IMAGE_SAVE_DIR = os.path.join(DATA_SAVE_DIR, 'images')
CSV_FILE_PATH = os.path.join(DATA_SAVE_DIR, 'metadata.csv')

# 폴더가 없으면 생성합니다.
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# CSV 파일이 없으면 헤더(컬럼명)를 먼저 만듭니다.
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'timestamp',
                'image_path',
                'title',
                'price',
                'region',
                'temp',
                'probability',
                'content',
            ]
        )

# 전역 변수로 모델 보관
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('🚀 AI 엔진 예열 및 모델 로딩 중...')
    sell_model = CatBoostClassifier()
    sell_model.load_model('data/models/daangn_sell_predictor_new4.cbm')
    siglip_predictor = SiglipSinglePredictor()

    models['sell_model'] = sell_model
    models['siglip_predictor'] = siglip_predictor
    print('✅ 모든 모델 로딩 완료! 데이터 수집 준비 완료.')
    yield
    models.clear()


app = FastAPI(title='당근막캐 AI API', lifespan=lifespan)


@app.get('/')
def read_root():
    return {'message': '당근막캐 API 서버 작동 중 🥕'}


@app.post('/predict/sell')
async def api_predict_sell(
    title: str = Form(...),
    content: str = Form(...),
    price: int = Form(...),
    region_name: str = Form('알 수 없음'),
    seller_temp: float = Form(36.5),
    image: UploadFile = File(None),
    is_submit: str = Form('false'),  # 🚨 [핵심 추가] 저장 여부를 묻는 스위치!
):
    try:
        # 1. 이미지 처리 (바이트 -> 파일 객체)
        img_bytes = await image.read() if image else None
        image_file_obj = io.BytesIO(img_bytes) if img_bytes else None

        # 2. 예측 파이프라인 실행 (저장과 상관없이 예측은 항상 수행)
        p, b, l = predict_sell_probability(
            catboost_model=models['sell_model'],
            siglip_predictor=models['siglip_predictor'],
            image_file=image_file_obj,
            title=title,
            content=content,
            price=price,
            region_name=region_name,
            seller_temp=seller_temp,
        )

        # ---------------------------------------------------------
        # 🚨 [신규] is_submit이 'true'일 때만(작성 완료를 눌렀을 때만) 저장!
        # ---------------------------------------------------------
        if is_submit.lower() == 'true':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            image_save_path = 'None'

            # (1) 이미지 파일 저장
            if img_bytes:
                image_filename = f'{timestamp}_{unique_id}.jpg'
                image_save_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
                with open(image_save_path, 'wb') as f:
                    f.write(img_bytes)

            # (2) CSV 파일에 텍스트 정보 추가
            with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                safe_content = content.replace('\n', ' ')
                writer.writerow(
                    [
                        timestamp,
                        image_save_path,
                        title,
                        price,
                        region_name,
                        seller_temp,
                        p,
                        safe_content,
                    ]
                )
        # ---------------------------------------------------------

        return {'status': 'success', 'probability': p, 'brand': b, 'label': l}

    except Exception as e:
        print(f'🚨 API Error: {e}')
        return {'status': 'error', 'message': str(e)}
