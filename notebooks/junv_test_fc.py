import os
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List

FIRECRAWL_API_KEY = "fc-fe305164e19f4d1095e908d0330976c8"
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

class DaangnItem(BaseModel):
    article_id: str = Field(description="상품 상세 링크의 숫자 ID 번호")
    title: str = Field(description="상품명 (예: 폴로 랄프로렌 셔츠)")
    price: str = Field(description="상품 가격 숫자만 (예: 35000)")
    status: str = Field(description="현재 상태 (판매중, 예약중, 판매완료 중 하나)")
    upload_time_str: str = Field(description="표시된 업로드 시간 (예: 10분 전, 끌올 1일전)")
    image_url: str = Field(description="첫 번째 상품 이미지 절대 경로 URL")

class DaangnExtractSchema(BaseModel):
    items: List[DaangnItem]

url = "https://www.daangn.com/search/%ED%8F%B4%EB%A1%9C?page=1"

try:
    print("Testing app.scrape with json format...")
    res = app.scrape(
        url,
        formats=[{
            'type': 'json',
            'schema': DaangnExtractSchema.model_json_schema(),
            'prompt': "당근마켓 페이지의 검색결과 목록에서 '폴로' 상품들의 ID, 이름, 가격, 판매상태, 등록시간, 사진URL을 모두 추출해줘."
        }]
    )
    if isinstance(res, dict):
        print("Keys:", res.keys())
        print("JSON Output:", res.get('json', 'N/A'))
    else:
        print("Fields:", dir(res))
        print("JSON Output:", getattr(res, 'json', 'N/A'))
except Exception as e:
    print("Error with app.scrape (json):", type(e), e)
