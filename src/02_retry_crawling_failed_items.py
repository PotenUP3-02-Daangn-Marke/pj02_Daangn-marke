import json
import random
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import pytz
import requests
from tqdm import tqdm

# =========================
# Config & Helpers
# =========================
BASE = 'https://www.daangn.com'
UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/122.0.0.0 Safari/537.36'
)

HEADERS = {
    'User-Agent': UA,
    'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
    'Referer': BASE,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

REMIX_RE = re.compile(r'window\.__remixContext\s*=\s*(\{.*?\})\s*;\s*</script>', re.S)


def extract_remix_context(html: str) -> dict:
    m = REMIX_RE.search(html)
    if not m:
        return {}
    return json.loads(m.group(1))


def dig(obj: object, path: list[str]) -> object | None:
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def parse_detail(html: str) -> dict:
    remix = extract_remix_context(html)
    if not remix:
        return {}

    loader = dig(remix, ['state', 'loaderData'])
    if not isinstance(loader, dict):
        return {}

    route_key = next((k for k in loader if 'buy_sell_id' in k), None)
    if route_key is None:
        return {}

    product = dig(loader, [route_key, 'product'])
    if not isinstance(product, dict):
        return {}

    user = product.get('user', {}) if isinstance(product.get('user', {}), dict) else {}

    images = product.get('images')
    image_count = len(images) if isinstance(images, list) else None

    image_urls: list[str] = []
    if isinstance(images, list):
        for img in images:
            if isinstance(img, str) and img:
                image_urls.append(img)
            elif isinstance(img, dict):
                u = img.get('url') or img.get('imageUrl') or img.get('src')
                if isinstance(u, str) and u:
                    image_urls.append(u)

    return {
        'favoriteCount': product.get('favoriteCount'),
        'chatCount': product.get('chatCount'),
        'viewCount': product.get('viewCount'),
        'sellerTemperature': user.get('score'),
        'imageCount': image_count,
        'imageUrls': image_urls if image_urls else None,
    }


# =========================
# Main Execution
# =========================
def retry_failed_items(
    input_csv='failed_status_detail.csv', output_csv='retried_status_detail.csv'
):
    # 실패한 파일 불러오기
    df = pd.read_csv(input_csv)

    # 시간 업데이트를 위해 한국 시간(KST) 설정
    kst = pytz.timezone('Asia/Seoul')

    session = requests.Session()
    session.headers.update(HEADERS)

    # 진행 상황을 보기 위해 tqdm 사용
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Retrying failed items'):
        url = row['href']
        if not url.startswith('http'):
            url = urljoin(BASE, url)

        try:
            # 429 Too Many Requests 에러 방지를 위한 딜레이 (1초 ~ 2.5초 사이)
            # 상태에 따라 더 늘리셔도 좋습니다.
            time.sleep(random.uniform(1.0, 2.5))

            r = session.get(url, timeout=15)
            r.raise_for_status()

            parsed_data = parse_detail(r.text)

            # 파싱된 데이터로 지정된 컬럼 업데이트
            if parsed_data:
                df.at[idx, 'favoriteCount'] = parsed_data.get('favoriteCount')
                df.at[idx, 'chatCount'] = parsed_data.get('chatCount')
                df.at[idx, 'viewCount'] = parsed_data.get('viewCount')
                df.at[idx, 'sellerTemperature'] = parsed_data.get('sellerTemperature')
                df.at[idx, 'imageCount'] = parsed_data.get('imageCount')

                # 리스트 형태의 이미지 URL을 JSON 문자열로 저장
                image_urls = parsed_data.get('imageUrls')
                df.at[idx, 'imageUrls'] = (
                    json.dumps(image_urls, ensure_ascii=False) if image_urls else None
                )

                # 성공 상태 및 에러 메시지 초기화
                df.at[idx, 'status_detail'] = 'ok'
                df.at[idx, 'error'] = None

        except Exception as e:
            # 실패 시 에러 메시지 갱신
            df.at[idx, 'error'] = repr(e)

        # 크롤링 시간 업데이트 (KST 기준)
        df.at[idx, 'crawledAt'] = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S%z')

    # 결과 저장
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'\n완료! 결과가 {output_csv}에 저장되었습니다.')

    # 성공한 항목 수 확인
    success_count = (df['status_detail'] == 'ok').sum()
    print(f'총 {len(df)}개 중 {success_count}개 복구 성공')


if __name__ == '__main__':
    retry_failed_items(
        './data/csv/failed_status_detail.csv', './data/csv/retried_status_detail.csv'
    )
