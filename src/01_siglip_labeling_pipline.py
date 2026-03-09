# 01_siglip_labeling_pipeline_merged_final.py
# Python 3.12+
#
# Requirements
#   uv add torch torchvision pillow pandas tqdm transformers accelerate safetensors
#
# Optional
#   uv add bitsandbytes
#
# Features
# - SigLIP2 NaFlex
# - 2-stage coarse -> fine labeling
# - prompt ensemble
# - keyword text fusion
# - keyword prior
# - brand extraction -> brandName column
# - unknown when no brand found
# - brand prior for fine labels
# - confidence / margin thresholding
# - top-k 저장
# - review queue CSV 생성
# - image feature cache
# - move/copy to label directories
# - robust HF output handling (Tensor / BaseModelOutputWithPooling)
#
# ✅ merged_dedup.csv 기준
#   - title/content 사용
#   - id 기준으로 단일 이미지 폴더에서 찾음
#
# ✅ unknown 줄이기 반영
#   - softmax confidence 사용
#   - threshold 완화
#   - very_low_conf일 때만 unknown
#   - 대부분 애매한 샘플은 fine_pred 유지 + review 로 보냄
#   - keyword prior를 CLIP 코드 수준으로 강화

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# =========================
# Config
# =========================
DATASET_NAME = 'merged_dedup'

# ✅ 실제 경로로 수정
IMG_DIR = Path('./data/images/merged_all')
CSV_PATH = Path('./data/csv/merged_dedup.csv')

OUT_CSV_PATH = Path(f'./data/csv/{DATASET_NAME}_siglip2_labeled.csv')
REVIEW_CSV_PATH = Path(f'./data/csv/{DATASET_NAME}_siglip2_review.csv')
FEATURE_CACHE_PATH = Path(f'./data/cache/{DATASET_NAME}_siglip2_image_features.pt')

MODEL_NAME = 'google/siglip2-so400m-patch16-naflex'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

BATCH_SIZE = 8
TEXT_BATCH_SIZE = 128
SAVE_IMAGE_FEATURES = False  # 한번만 True한뒤 나중에 False
LOAD_IMAGE_FEATURES_IF_EXISTS = True  # 처음엔 False 이후에 True
USE_TORCH_COMPILE = False

ALPHA_IMG = 0.88
BETA_TXT = 1.0 - ALPHA_IMG

TOPK = 3

MOVE_TO_LABEL_DIR = True
COPY_INSTEAD_OF_MOVE = False
DRY_RUN_MOVE = False
MOVE_UNKNOWN_TOO = True
UNKNOWN_DIR_NAME = '_unknown'
MOVE_BY = 'coarse'  # "coarse" | "fine"

# ✅ unknown 줄이기용 완화 threshold
COARSE_MARGIN_THRESH = 0.06
COARSE_CONF_THRESH = 0.22

FINE_MARGIN_THRESH = 0.03
FINE_CONF_THRESH = 0.18

TOP1_TOP2_RATIO_THRESH = 1.02
REVIEW_IF_IMAGE_TEXT_DISAGREE = True
IMAGE_TEXT_DISAGREE_MARGIN = 0.20

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.jfif'}

PRINT_MISSING_LIMIT = 30
PRINT_BAD_IMAGE_LIMIT = 30

MAX_TEXT_LENGTH = 64
TEXT_PREFIX_TEMPLATE = 'This is a photo of {}.'


# =========================
# 2-stage taxonomy
# =========================
COARSE_LABELS = [
    'top',
    'bottom',
    'outer',
    'dress',
    'skirt',
    'bags',
    'shoes',
    'other',
]

FINE_LABELS: dict[str, list[str]] = {
    'top': [
        't-shirt',
        'polo shirt',
        'shirt',
        'hoodie',
        'sweatshirt',
        'knit sweater',
        'cardigan',
        'sleeveless',
    ],
    'bottom': [
        'jeans',
        'pants',
        'shorts',
    ],
    'outer': [
        'jacket',
        'denim jacket',
        'coat',
        'padding jacket',
        'blazer',
    ],
    'dress': [
        'dress',
    ],
    'skirt': [
        'skirt',
    ],
    'bags': [
        'backpack',
        'shoulder bag',
        'crossbody bag',
        'tote bag',
        'handbag',
    ],
    'shoes': [
        'sneakers',
        'boots',
        'sandals',
        'heels',
        'loafers',
    ],
    'other': [
        'other',
    ],
}


# =========================
# Keyword extraction
# =========================
FASHION_KEYWORDS = [
    '티셔츠',
    '반팔',
    '긴팔',
    '셔츠',
    '와이셔츠',
    '카라티',
    '폴로',
    '후드',
    '후드티',
    '맨투맨',
    '니트',
    '스웨터',
    '가디건',
    '나시',
    '민소매',
    '바지',
    '청바지',
    '데님',
    '슬랙스',
    '반바지',
    '쇼츠',
    '자켓',
    '재킷',
    '코트',
    '롱코트',
    '패딩',
    '다운',
    '점퍼',
    '블레이저',
    '원피스',
    '드레스',
    '치마',
    '스커트',
    '가방',
    '백팩',
    '숄더백',
    '크로스백',
    '토트백',
    '핸드백',
    '신발',
    '운동화',
    '스니커즈',
    '부츠',
    '샌들',
    '로퍼',
    '구두',
]

KW_NORMALIZE = {
    '폴로': '카라티',
    '재킷': '자켓',
    '쇼츠': '반바지',
    '데님': '청바지',
    '다운': '패딩',
    '스니커즈': '운동화',
}


def extract_fashion_keywords(title: str, content: str) -> str:
    text = f'{title}\n{content}'.lower()
    found: list[str] = []
    for kw in FASHION_KEYWORDS:
        if kw.lower() in text:
            found.append(kw)

    norm = [KW_NORMALIZE.get(kw, kw) for kw in found]
    norm = list(dict.fromkeys(norm))
    return ' '.join(norm).strip()


def build_keyword_sentence(keywords: str) -> str:
    if not keywords:
        return ''
    return f'의류 카테고리 키워드: {keywords}'


def coarse_prior_from_keywords(keywords: str, coarse_labels: list[str]) -> torch.Tensor:
    """
    keyword 기반으로 coarse score bias를 준다.
    CLIP 코드 수준으로 약간 더 강하게 반영.
    """
    prior_map = {
        'top': {
            '티셔츠',
            '반팔',
            '긴팔',
            '셔츠',
            '와이셔츠',
            '카라티',
            '후드',
            '후드티',
            '맨투맨',
            '니트',
            '스웨터',
            '가디건',
            '나시',
            '민소매',
        },
        'bottom': {'바지', '청바지', '슬랙스', '반바지'},
        'outer': {'자켓', '코트', '롱코트', '패딩', '점퍼', '블레이저'},
        'dress': {'원피스', '드레스'},
        'skirt': {'치마', '스커트'},
        'bags': {'가방', '백팩', '숄더백', '크로스백', '토트백', '핸드백'},
        'shoes': {'신발', '운동화', '부츠', '샌들', '로퍼', '구두'},
        'other': set(),
    }

    kw_set = set(keywords.split()) if keywords else set()
    bias = torch.zeros(len(coarse_labels), dtype=torch.float32)

    for i, label in enumerate(coarse_labels):
        overlap = len(kw_set & prior_map.get(label, set()))
        if overlap > 0:
            bias[i] = 1.5 + 0.5 * overlap
    return bias


# =========================
# Brand extraction + brand prior
# =========================
BRAND_ALIASES: dict[str, list[str]] = {
    'polo ralph lauren': [
        '폴로',
        '폴로랄프로렌',
        '랄프로렌',
        '랄프 로렌',
        'polo',
        'polo ralph lauren',
        'ralph lauren',
        'polo by ralph lauren',
    ],
    'lacoste': ['라코스테', 'lacoste'],
    'beams': ['빔즈', 'beams'],
    'brooks brothers': ['브룩스브라더스', '브룩스 브라더스', 'brooks brothers'],
    'the north face': ['노스페이스', '노페', 'north face', 'the north face', 'tnf'],
    'nike': ['나이키', 'nike', 'nk'],
    'adidas': ['아디다스', '아디', 'adidas', 'adidas originals'],
    'new balance': ['뉴발란스', '뉴발', 'new balance', 'nb'],
    'puma': ['푸마', 'puma'],
    'reebok': ['리복', 'reebok'],
    'asics': ['아식스', 'asics'],
    'under armour': ['언더아머', '언더 아머', 'under armour', 'under armor', 'ua'],
    'fila': ['휠라', 'fila'],
    'kappa': ['카파', 'kappa'],
    'descente': ['데상트', 'descente'],
    'mizuno': ['미즈노', 'mizuno'],
    'salomon': ['살로몬', 'salomon'],
    "arc'teryx": ['아크테릭스', '아크', "arc'teryx", 'arcteryx'],
    'patagonia': ['파타고니아', 'patagonia'],
    'columbia': ['컬럼비아', 'columbia'],
    'montbell': ['몽벨', 'mont bell', 'montbell'],
    'millet': ['밀레', 'millet'],
    'blackyak': ['블랙야크', 'blackyak', 'black yak'],
    'k2': ['k2', '케이투'],
    'nepa': ['네파', 'nepa'],
    'discovery expedition': [
        '디스커버리',
        '디스커버리 익스페디션',
        'discovery',
        'discovery expedition',
    ],
    'national geographic': [
        '내셔널지오그래픽',
        '내셔널 지오그래픽',
        'national geographic',
    ],
    'snow peak': ['스노우피크', 'snow peak'],
    'uniqlo': ['유니클로', 'uniqlo', '유니클로u', 'uniqlo u'],
    'gu': ['지유', 'gu'],
    'zara': ['자라', 'zara', 'zaraman', 'zara man'],
    'h&m': ['h&m', 'hm', '에이치앤엠', '에이치 앤 엠'],
    'cos': ['cos', '코스'],
    'massimo dutti': ['마시모두띠', '마시모 두띠', 'massimo dutti'],
    'weekday': ['weekday', '위크데이'],
    'arket': ['arket', '아르켓'],
    'muji': ['무인양품', '무지', 'muji'],
    'gap': ['갭', 'gap'],
    'banana republic': ['바나나리퍼블릭', '바나나 리퍼블릭', 'banana republic'],
    'abercrombie & fitch': [
        '아베크롬비',
        '아베크롬비앤피치',
        'abercrombie',
        'abercrombie & fitch',
        'a&f',
    ],
    'hollister': ['홀리스터', 'hollister'],
    'american eagle': ['아메리칸이글', '아메리칸 이글', 'american eagle'],
    'carhartt': ['칼하트', 'carhartt', 'carhartt wip'],
    'dickies': ['디키즈', 'dickies'],
    'ben davis': ['벤데이비스', '벤 데이비스', 'ben davis'],
    'stussy': ['스투시', 'stussy'],
    'supreme': ['슈프림', 'supreme'],
    'thisisneverthat': ['디스이즈네버댓', 'thisisneverthat', 'tnn'],
    'musinsa standard': ['무신사스탠다드', '무신사 스탠다드', 'musinsa standard'],
    '8seconds': ['에잇세컨즈', '8seconds', '8 seconds'],
    'spao': ['스파오', 'spao'],
    'topten': ['탑텐', 'topten', 'topten10', 'topten 10'],
    'giordano': ['지오다노', 'giordano'],
    'codes combine': ['코데즈컴바인', 'codes combine'],
    'tngt': ['tngt', '티엔지티'],
    'time homme': ['타임옴므', '타임 옴므', 'time homme'],
    'system homme': ['시스템옴므', '시스템 옴므', 'system homme'],
    "levi's": ['리바이스', "levi's", 'levis'],
    'lee': ['lee', '리'],
    'edwin': ['에드윈', 'edwin'],
    'nudie jeans': ['누디진', '누디진스', 'nudie jeans', 'nudie'],
    'diesel': ['디젤', 'diesel'],
    'burberry': ['버버리', 'burberry'],
    'gucci': ['구찌', 'gucci'],
    'chanel': ['샤넬', 'chanel'],
    'louis vuitton': ['루이비통', '루이 비통', 'louis vuitton', 'lv'],
    'hermes': ['에르메스', 'hermes'],
    'prada': ['프라다', 'prada'],
    'miu miu': ['미우미우', '미우 미우', 'miu miu'],
    'celine': ['셀린느', '셀린', 'celine'],
    'dior': ['디올', 'christian dior', 'dior'],
    'saint laurent': ['생로랑', 'saint laurent', 'ysl'],
    'balenciaga': ['발렌시아가', 'balenciaga'],
    'givenchy': ['지방시', 'givenchy'],
    'valentino': ['발렌티노', 'valentino'],
    'fendi': ['펜디', 'fendi'],
    'bottega veneta': ['보테가베네타', '보테가 베네타', 'bottega veneta'],
    'loewe': ['로에베', 'loewe'],
    'marni': ['마르니', 'marni'],
    'maison margiela': [
        '메종마르지엘라',
        '메종 마르지엘라',
        'maison margiela',
        'margiela',
    ],
    'jil sander': ['질샌더', '질 샌더', 'jil sander'],
    'thom browne': ['톰브라운', '톰 브라운', 'thom browne'],
    'moncler': ['몽클레어', 'moncler'],
    'stone island': ['스톤아일랜드', '스톤 아일랜드', 'stone island'],
    'converse': ['컨버스', 'converse'],
    'vans': ['반스', 'vans'],
    'dr. martens': ['닥터마틴', '닥마', 'dr martens', 'dr. martens'],
    'timberland': ['팀버랜드', 'timberland'],
    'birkenstock': ['버켄스탁', 'birkenstock'],
    'crocs': ['크록스', 'crocs'],
    'ugg': ['어그', 'ugg'],
    'golden goose': ['골든구스', 'golden goose'],
    'porter': ['포터', 'porter', '요시다포터', '요시다 포터'],
    'tumi': ['투미', 'tumi'],
    'samsonite': ['샘소나이트', 'samsonite'],
    'longchamp': ['롱샴', 'longchamp'],
    'coach': ['코치', 'coach'],
    'michael kors': ['마이클코어스', '마이클 코어스', 'michael kors'],
    'tory burch': ['토리버치', '토리 버치', 'tory burch'],
    'andersson bell': ['앤더슨벨', '앤더슨 벨', 'andersson bell'],
    'ader error': ['아더에러', '아더 에러', 'ader error'],
    'wooyoungmi': ['우영미', 'wooyoungmi'],
    'juun.j': ['준지', 'juun j', 'juun.j'],
    'low classic': ['로우클래식', '로우 클래식', 'low classic'],
    'matin kim': ['마뗑킴', '마틴킴', 'matin kim'],
    'emis': ['이미스', 'emis'],
    'covernat': ['커버낫', 'covernat'],
    'marithe francois girbaud': [
        '마리떼',
        '마리떼프랑소와저버',
        'marithe',
        'marithe francois girbaud',
    ],
    'kirsh': ['키르시', 'kirsh'],
    'mahagrid': ['마하그리드', 'mahagrid'],
    'mlb': ['mlb', '엠엘비'],
}

BRAND_FINE_PRIOR: dict[str, dict[str, float]] = {
    'polo ralph lauren': {
        'polo shirt': 1.4,
        'shirt': 0.8,
        'knit sweater': 0.6,
        'cardigan': 0.4,
    },
    'the north face': {
        'padding jacket': 1.3,
        'jacket': 0.9,
        'coat': 0.5,
    },
    'nike': {
        'sneakers': 1.2,
        'hoodie': 0.7,
        't-shirt': 0.6,
        'pants': 0.3,
    },
    'adidas': {
        'sneakers': 1.2,
        'hoodie': 0.7,
        't-shirt': 0.6,
        'pants': 0.3,
    },
    'new balance': {
        'sneakers': 1.3,
    },
    'lacoste': {
        'polo shirt': 1.2,
        'shirt': 0.6,
        'knit sweater': 0.4,
    },
}


def normalize_text_for_brand(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9가-힣\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_brand_name(title: str, content: str) -> str:
    text = normalize_text_for_brand(f'{title}\n{content}')
    matched: list[tuple[str, str]] = []

    for canonical_brand, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_text_for_brand(alias)
            if not alias_norm:
                continue

            if (
                re.search(rf'(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])', text)
                or alias_norm in text
            ):
                matched.append((canonical_brand, alias_norm))

    if not matched:
        return 'unknown'

    matched.sort(key=lambda x: len(x[1]), reverse=True)
    return matched[0][0]


def fine_prior_from_brand(brand_name: str, fine_labels: list[str]) -> torch.Tensor:
    bias = torch.zeros(len(fine_labels), dtype=torch.float32)

    if brand_name == 'unknown':
        return bias

    prior_map = BRAND_FINE_PRIOR.get(brand_name, {})
    for i, label in enumerate(fine_labels):
        if label in prior_map:
            bias[i] = prior_map[label]

    return bias


# =========================
# Prompt engineering
# =========================
def build_prompts_for_label(label: str) -> list[str]:
    base = [
        f'a product photo of {label}',
        f'a marketplace photo of {label}',
        f'a secondhand marketplace listing photo of {label}',
        f'a close-up product photo of {label}',
        f'a full-body or worn photo of {label}',
        TEXT_PREFIX_TEMPLATE.format(label),
    ]

    ko: list[str] = []

    if label == 'top':
        ko = [
            '상의 사진',
            '티셔츠 셔츠 카라티 니트 맨투맨 후드티 같은 상의',
            '상체에 입는 옷',
        ]
    elif label == 'bottom':
        ko = [
            '하의 사진',
            '바지 청바지 슬랙스 반바지 같은 하의',
            '하체에 입는 옷',
        ]
    elif label == 'outer':
        ko = [
            '아우터 사진',
            '자켓 코트 패딩 점퍼 블레이저 같은 겉옷',
            '겉에 걸치는 옷',
        ]
    elif label == 'dress':
        ko = [
            '원피스 사진',
            '드레스 사진',
            '상의와 하의가 한 벌로 이어진 옷',
        ]
    elif label == 'skirt':
        ko = [
            '스커트 사진',
            '치마 사진',
            '허리 아래로 내려오는 하의',
        ]
    elif label == 'bags':
        ko = [
            '가방 사진',
            '백팩 숄더백 크로스백 토트백 핸드백 같은 패션 잡화',
            '패션 가방 제품 사진',
        ]
    elif label == 'shoes':
        ko = [
            '신발 사진',
            '운동화 부츠 샌들 로퍼 구두 같은 신발',
            '신발 제품 사진',
        ]
    elif label == 't-shirt':
        ko = [
            '반팔 티셔츠 사진',
            '라운드넥 티셔츠',
            '기본 티셔츠',
        ]
    elif label == 'polo shirt':
        ko = [
            '카라티 사진',
            '폴로 셔츠 사진',
            '카라와 단추가 있는 반팔 상의',
        ]
    elif label == 'shirt':
        ko = [
            '셔츠 사진',
            '와이셔츠 사진',
            '카라와 단추가 있는 긴팔 상의',
        ]
    elif label == 'hoodie':
        ko = [
            '후드티 사진',
            '모자가 달린 상의',
            '후드가 있는 스웨트 상의',
        ]
    elif label == 'sweatshirt':
        ko = [
            '맨투맨 사진',
            '후드 없는 스웨트셔츠',
            '라운드넥 맨투맨',
        ]
    elif label == 'knit sweater':
        ko = [
            '니트 사진',
            '스웨터 사진',
            '뜨개 느낌 상의',
        ]
    elif label == 'cardigan':
        ko = [
            '가디건 사진',
            '앞이 열리는 니트',
            '단추가 있는 니트 아우터',
        ]
    elif label == 'sleeveless':
        ko = [
            '민소매 사진',
            '나시 사진',
            '슬리브리스 상의',
        ]
    elif label == 'jeans':
        ko = [
            '청바지 사진',
            '데님 팬츠 사진',
            '데님 소재 바지',
        ]
    elif label == 'pants':
        ko = [
            '긴 바지 사진',
            '슬랙스 사진',
            '면바지 사진',
        ]
    elif label == 'shorts':
        ko = [
            '반바지 사진',
            '쇼츠 사진',
            '무릎 위 길이 바지',
        ]
    elif label == 'jacket':
        ko = [
            '자켓 사진',
            '재킷 사진',
            '가벼운 아우터',
        ]
    elif label == 'denim jacket':
        ko = [
            '청자켓 사진',
            '데님 재킷 사진',
            '데님 소재 아우터',
        ]
    elif label == 'coat':
        ko = [
            '코트 사진',
            '롱코트 사진',
            '긴 겨울 아우터',
        ]
    elif label == 'padding jacket':
        ko = [
            '패딩 사진',
            '다운 자켓 사진',
            '두꺼운 겨울 점퍼',
        ]
    elif label == 'blazer':
        ko = [
            '블레이저 사진',
            '정장 자켓 사진',
            '테일러드 자켓',
        ]
    elif label == 'backpack':
        ko = [
            '백팩 사진',
            '배낭 스타일 가방',
        ]
    elif label == 'shoulder bag':
        ko = [
            '숄더백 사진',
            '어깨에 메는 가방',
        ]
    elif label == 'crossbody bag':
        ko = [
            '크로스백 사진',
            '몸에 가로로 메는 가방',
        ]
    elif label == 'tote bag':
        ko = [
            '토트백 사진',
            '손잡이가 있는 가방',
        ]
    elif label == 'handbag':
        ko = [
            '핸드백 사진',
            '작은 여성 가방',
        ]
    elif label == 'sneakers':
        ko = [
            '운동화 사진',
            '스니커즈 사진',
        ]
    elif label == 'boots':
        ko = [
            '부츠 사진',
            '발목 또는 종아리까지 올라오는 신발',
        ]
    elif label == 'sandals':
        ko = [
            '샌들 사진',
            '여름 신발',
        ]
    elif label == 'heels':
        ko = [
            '구두 사진',
            '하이힐 사진',
        ]
    elif label == 'loafers':
        ko = [
            '로퍼 사진',
            '끈 없는 구두',
        ]
    elif label == 'other':
        ko = [
            '의류 외 기타 제품',
            '분류 불가 이미지',
            '기타',
        ]

    prompts = base + ko
    prompts = list(dict.fromkeys(prompts))
    return prompts[:10]


def build_prompt_ensemble(labels: list[str]) -> dict[str, list[str]]:
    return {lbl: build_prompts_for_label(lbl) for lbl in labels}


COARSE_PROMPTS = build_prompt_ensemble(COARSE_LABELS)
FINE_PROMPTS = {g: build_prompt_ensemble(lbls) for g, lbls in FINE_LABELS.items()}


# =========================
# File helpers
# =========================
def sanitize_for_filename(label: str) -> str:
    label = label.strip().lower()
    label = re.sub(r'\s+', '-', label)
    label = re.sub(r'[^a-z0-9가-힣\-_.]+', '', label)
    return label[:60] if label else 'unknown'


def find_image_by_id(img_dir: Path, item_id: str) -> Path | None:
    item_id = item_id.strip()
    for ext in EXTS:
        p = img_dir / f'{item_id}{ext}'
        if p.exists():
            return p
        p2 = img_dir / f'{item_id}{ext.upper()}'
        if p2.exists():
            return p2
    return None


def move_image_to_label_dir(img_path: Path, label: str, base_dir: Path) -> Path:
    safe_label = sanitize_for_filename(label)
    label_dir = base_dir / safe_label
    label_dir.mkdir(parents=True, exist_ok=True)

    dest = label_dir / img_path.name

    if img_path.parent == label_dir:
        return dest

    if dest.exists():
        stem = img_path.stem
        ext = img_path.suffix
        k = 1
        while True:
            cand = label_dir / f'{stem}_{k}{ext}'
            if not cand.exists():
                dest = cand
                break
            k += 1

    if DRY_RUN_MOVE:
        print(
            f'[DRY] {"copy" if COPY_INSTEAD_OF_MOVE else "move"} {img_path.name} -> {dest.relative_to(base_dir)}'
        )
        return dest

    if COPY_INSTEAD_OF_MOVE:
        shutil.copy2(img_path, dest)
    else:
        img_path.replace(dest)
    return dest


def audit_leftovers(img_dir: Path) -> None:
    leftovers = [
        p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS
    ]
    print(f'\nLeftovers in root: {len(leftovers)}')
    print('Sample leftovers:', leftovers[:20])

    ext_counts = Counter([Path(x).suffix.lower() for x in leftovers])
    if leftovers:
        print('Root extension counts:', dict(ext_counts))


# =========================
# HF output handling
# =========================
def extract_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    for attr in ('text_embeds', 'image_embeds', 'pooler_output', 'last_hidden_state'):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if value is not None:
                if attr == 'last_hidden_state':
                    return value[:, 0]
                return value

    if isinstance(output, dict):
        for key in (
            'text_embeds',
            'image_embeds',
            'pooler_output',
            'last_hidden_state',
        ):
            if key in output and output[key] is not None:
                value = output[key]
                if key == 'last_hidden_state':
                    return value[:, 0]
                return value

    raise TypeError(f'Unsupported output type for feature extraction: {type(output)}')


# =========================
# SigLIP2 utilities
# =========================
def load_siglip2_model_and_processor() -> tuple[Any, Any]:
    model = (
        AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        .to(DEVICE)
        .eval()
    )

    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        model = torch.compile(model)  # type: ignore[assignment]

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def safe_open_rgb(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        return None


@torch.no_grad()
def get_text_features(
    model,
    processor,
    texts: list[str],
    batch_size: int = TEXT_BATCH_SIZE,
) -> torch.Tensor:
    feats: list[torch.Tensor] = []

    for start in tqdm(range(0, len(texts), batch_size), desc='Encoding text prompts'):
        batch = texts[start : start + batch_size]
        inputs = processor(
            text=batch,
            padding='max_length',
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        output = model.get_text_features(**inputs)
        out = extract_feature_tensor(output)
        out = F.normalize(out, dim=-1)
        feats.append(out.detach().cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def get_image_features_batch(
    model,
    processor,
    images: list[Image.Image],
) -> torch.Tensor:
    inputs = processor(
        images=images,
        return_tensors='pt',
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    output = model.get_image_features(**inputs)
    out = extract_feature_tensor(output)
    out = F.normalize(out, dim=-1)
    return out


def mean_pool_prompt_embeddings(
    label_to_prompt_features: dict[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    labels = list(label_to_prompt_features.keys())
    pooled = []

    for lbl in labels:
        feat = label_to_prompt_features[lbl].mean(dim=0, keepdim=True)
        feat = F.normalize(feat, dim=-1)
        pooled.append(feat)

    return labels, torch.cat(pooled, dim=0)


def build_label_text_embeddings(
    model,
    processor,
    prompt_dict: dict[str, list[str]],
) -> tuple[list[str], torch.Tensor, dict[str, torch.Tensor]]:
    labels = list(prompt_dict.keys())

    all_texts: list[str] = []
    index: dict[str, tuple[int, int]] = {}
    cursor = 0

    for lbl in labels:
        prompts = prompt_dict[lbl]
        all_texts.extend(prompts)
        index[lbl] = (cursor, cursor + len(prompts))
        cursor += len(prompts)

    all_feats = get_text_features(
        model, processor, all_texts, batch_size=TEXT_BATCH_SIZE
    )

    by_label: dict[str, torch.Tensor] = {}
    for lbl in labels:
        s, e = index[lbl]
        by_label[lbl] = all_feats[s:e]

    pooled_labels, pooled_embs = mean_pool_prompt_embeddings(by_label)
    return pooled_labels, pooled_embs, by_label


def cosine_scores(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    return (vec @ mat.T).squeeze(0)


def sigmoid_probs(scores: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(scores)


def softmax_confidence(scores: torch.Tensor) -> float:
    return float(torch.softmax(scores, dim=0).max().item())


def topk(
    scores: torch.Tensor, labels: list[str], k: int = 3
) -> list[tuple[str, float]]:
    k = min(k, len(labels))
    vals, idx = torch.topk(scores, k=k)
    return [(labels[int(i)], float(v.item())) for v, i in zip(vals, idx, strict=True)]


def top1_margin(scores: torch.Tensor) -> float:
    k = min(2, int(scores.numel()))
    vals, _ = torch.topk(scores, k=k)
    if vals.numel() < 2:
        return float('inf')
    return float((vals[0] - vals[1]).item())


def top1_top2_ratio(scores: torch.Tensor) -> float:
    probs = sigmoid_probs(scores)
    k = min(2, int(probs.numel()))
    vals, _ = torch.topk(probs, k=k)
    if vals.numel() < 2:
        return float('inf')
    return float((vals[0] / (vals[1] + 1e-8)).item())


def soft_vote_scores(
    image_scores: torch.Tensor,
    text_scores: torch.Tensor | None,
    alpha_img: float = ALPHA_IMG,
) -> torch.Tensor:
    if text_scores is None:
        return image_scores
    return alpha_img * image_scores + (1.0 - alpha_img) * text_scores


# =========================
# Keyword text encoder cache
# =========================
class KeywordTextEncoder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.cache: dict[str, torch.Tensor] = {}

    def encode(self, text: str) -> torch.Tensor | None:
        if not text:
            return None
        if text in self.cache:
            return self.cache[text]

        inputs = self.processor(
            text=[text],
            padding='max_length',
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.get_text_features(**inputs)
            feat = extract_feature_tensor(output)
            feat = F.normalize(feat, dim=-1).detach().cpu()

        self.cache[text] = feat
        return feat


# =========================
# Classification
# =========================
def classify_2stage(
    img_feat_cpu: torch.Tensor,
    kw_feat_cpu: torch.Tensor | None,
    keywords: str,
    brand_name: str,
    coarse_labels: list[str],
    coarse_embs_cpu: torch.Tensor,
    fine_embs_map_cpu: dict[str, tuple[list[str], torch.Tensor]],
) -> dict[str, Any]:
    img_feat = img_feat_cpu
    kw_feat = kw_feat_cpu
    coarse_embs = coarse_embs_cpu

    # ---- stage 1: coarse
    coarse_img_scores = cosine_scores(img_feat, coarse_embs)

    coarse_txt_scores = None
    if kw_feat is not None:
        coarse_txt_scores = cosine_scores(kw_feat, coarse_embs)

    coarse_scores = soft_vote_scores(coarse_img_scores, coarse_txt_scores)
    coarse_scores = coarse_scores + coarse_prior_from_keywords(keywords, coarse_labels)

    coarse_top = topk(coarse_scores, coarse_labels, k=TOPK)
    coarse_pred = coarse_top[0][0]
    coarse_conf = softmax_confidence(coarse_scores)
    coarse_margin = top1_margin(coarse_scores)
    coarse_ratio = top1_top2_ratio(coarse_scores)

    coarse_low = (coarse_conf < COARSE_CONF_THRESH) or (
        coarse_margin < COARSE_MARGIN_THRESH
    )

    # ---- stage 2: fine
    fine_labels, fine_embs = fine_embs_map_cpu[coarse_pred]

    fine_img_scores = cosine_scores(img_feat, fine_embs)

    fine_txt_scores = None
    if kw_feat is not None:
        fine_txt_scores = cosine_scores(kw_feat, fine_embs)

    fine_scores = soft_vote_scores(fine_img_scores, fine_txt_scores)
    fine_scores = fine_scores + fine_prior_from_brand(brand_name, fine_labels)

    fine_top = topk(fine_scores, fine_labels, k=TOPK)
    fine_pred = fine_top[0][0]
    fine_conf = softmax_confidence(fine_scores)
    fine_margin = top1_margin(fine_scores)
    fine_ratio = top1_top2_ratio(fine_scores)

    fine_low = (
        (fine_conf < FINE_CONF_THRESH)
        or (fine_margin < FINE_MARGIN_THRESH)
        or (fine_ratio < TOP1_TOP2_RATIO_THRESH)
    )

    image_text_disagree = False
    if REVIEW_IF_IMAGE_TEXT_DISAGREE and fine_txt_scores is not None:
        img_best = fine_labels[int(torch.argmax(fine_img_scores).item())]
        txt_best = fine_labels[int(torch.argmax(fine_txt_scores).item())]

        img_best_score = float(torch.max(sigmoid_probs(fine_img_scores)).item())
        txt_best_score = float(torch.max(sigmoid_probs(fine_txt_scores)).item())

        if (
            img_best != txt_best
            and abs(img_best_score - txt_best_score) <= IMAGE_TEXT_DISAGREE_MARGIN
        ):
            image_text_disagree = True

    # ✅ truly ambiguous일 때만 unknown
    very_low = fine_conf < 0.12 and fine_margin < 0.015 and coarse_margin < 0.03

    final_label = fine_pred
    triage = 'auto_accept'
    review_reason = ''

    if coarse_low and fine_low and very_low:
        final_label = 'unknown'
        triage = 'review'
        review_reason = 'very_low_conf_unknown'
    elif fine_low:
        final_label = fine_pred
        triage = 'review'
        review_reason = 'fine_low_conf'
    elif coarse_low:
        final_label = fine_pred
        triage = 'review'
        review_reason = 'coarse_low_conf'
    elif image_text_disagree:
        final_label = fine_pred
        triage = 'review'
        review_reason = 'image_text_disagree'

    return {
        'coarse_pred': coarse_pred,
        'coarse_conf': coarse_conf,
        'coarse_margin': coarse_margin,
        'coarse_ratio': coarse_ratio,
        'coarse_top3': coarse_top,
        'fine_group': coarse_pred,
        'fine_pred': fine_pred,
        'fine_conf': fine_conf,
        'fine_margin': fine_margin,
        'fine_ratio': fine_ratio,
        'fine_top3': fine_top,
        'final_label': final_label,
        'final_conf': fine_conf,
        'triage': triage,
        'review_reason': review_reason,
    }


# =========================
# Feature cache
# =========================
@torch.no_grad()
def build_or_load_image_feature_cache(
    model,
    processor,
    paths_by_id: dict[str, Path],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    feature_cache: dict[str, torch.Tensor] = {}
    bad_ids: set[str] = set()

    if LOAD_IMAGE_FEATURES_IF_EXISTS and FEATURE_CACHE_PATH.exists():
        print(f'Loading cached image features: {FEATURE_CACHE_PATH.resolve()}')
        obj = torch.load(FEATURE_CACHE_PATH, map_location='cpu')
        feature_cache = obj.get('features', {})
        bad_ids = set(obj.get('bad_ids', []))

        missing = [
            k for k in paths_by_id if k not in feature_cache and k not in bad_ids
        ]
        if not missing:
            return feature_cache, bad_ids
        print(f'Cache partial. Need to encode {len(missing)} more images.')

    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    todo_ids = [k for k in paths_by_id if k not in feature_cache and k not in bad_ids]
    images: list[Image.Image] = []
    ids_in_batch: list[str] = []

    def flush_batch() -> None:
        nonlocal images, ids_in_batch, feature_cache
        if not images:
            return
        feats = get_image_features_batch(model, processor, images).detach().cpu()
        for item_id, feat in zip(ids_in_batch, feats, strict=True):
            feature_cache[item_id] = feat.unsqueeze(0)
        images = []
        ids_in_batch = []

    for item_id in tqdm(todo_ids, desc='Encoding image features'):
        img = safe_open_rgb(paths_by_id[item_id])
        if img is None:
            bad_ids.add(item_id)
            continue

        images.append(img)
        ids_in_batch.append(item_id)

        if len(images) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    if SAVE_IMAGE_FEATURES:
        torch.save(
            {
                'features': feature_cache,
                'bad_ids': sorted(bad_ids),
                'model_name': MODEL_NAME,
            },
            FEATURE_CACHE_PATH,
        )
        print(f'Saved image feature cache: {FEATURE_CACHE_PATH.resolve()}')

    return feature_cache, bad_ids


# =========================
# Main
# =========================
def main() -> None:
    if not IMG_DIR.exists():
        raise FileNotFoundError(f'IMG_DIR not found: {IMG_DIR.resolve()}')
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'CSV_PATH not found: {CSV_PATH.resolve()}')

    df = pd.read_csv(CSV_PATH)

    if 'id' not in df.columns:
        raise ValueError("CSV must contain an 'id' column.")

    if 'title' not in df.columns:
        df['title'] = ''
    if 'content' not in df.columns:
        df['content'] = ''

    df['id'] = df['id'].astype(str).str.strip()

    print('CSV_PATH:', CSV_PATH.resolve())
    print('IMG_DIR :', IMG_DIR.resolve())
    print('DEVICE  :', DEVICE)

    root_image_count = sum(
        1 for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS
    )
    print('df rows :', len(df))
    print('root image files:', root_image_count)

    df_ids = set(df['id'])
    img_ids = {
        p.stem.strip()
        for p in IMG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    }

    print('PRECHECK matched:', len(df_ids & img_ids))
    print('PRECHECK df-only :', len(df_ids - img_ids))
    print('PRECHECK img-only:', len(img_ids - df_ids))

    paths_by_id: dict[str, Path] = {}
    for item_id in df['id']:
        p = find_image_by_id(IMG_DIR, item_id)
        if p is not None:
            paths_by_id[item_id] = p

    print(f'Loading model: {MODEL_NAME}')
    model, processor = load_siglip2_model_and_processor()

    coarse_labels, coarse_embs_cpu, _ = build_label_text_embeddings(
        model, processor, COARSE_PROMPTS
    )

    fine_embs_map_cpu: dict[str, tuple[list[str], torch.Tensor]] = {}
    for g in COARSE_LABELS:
        lbls, embs, _ = build_label_text_embeddings(model, processor, FINE_PROMPTS[g])
        fine_embs_map_cpu[g] = (lbls, embs)

    kw_encoder = KeywordTextEncoder(model, processor)

    image_feature_cache, bad_ids = build_or_load_image_feature_cache(
        model, processor, paths_by_id
    )

    # output columns
    df['fashion_keywords'] = ''
    df['brandName'] = 'unknown'
    df['label'] = ''
    df['label_conf'] = 0.0
    df['coarse_label'] = ''
    df['coarse_conf'] = 0.0
    df['coarse_margin'] = 0.0
    df['coarse_ratio'] = 0.0
    df['fine_conf'] = 0.0
    df['fine_margin'] = 0.0
    df['fine_ratio'] = 0.0
    df['top3_labels'] = ''
    df['top3_scores'] = ''
    df['triage'] = ''
    df['review_reason'] = ''
    df['img_path'] = ''

    missing_path = 0
    bad_image = 0
    moved = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc='SigLIP2 2-stage labeling'):
        item_id = str(row['id']).strip()

        img_path = paths_by_id.get(item_id)
        if img_path is None:
            missing_path += 1
            df.at[i, 'label'] = 'missing_image'
            df.at[i, 'triage'] = 'missing_image'
            df.at[i, 'review_reason'] = 'missing_path'
            df.at[i, 'label_conf'] = 0.0
            df.at[i, 'brandName'] = 'unknown'
            if missing_path <= PRINT_MISSING_LIMIT:
                print(f'[MISSING_PATH] id={item_id!r} expected under {IMG_DIR}')
            continue

        df.at[i, 'img_path'] = str(img_path)

        if item_id in bad_ids:
            bad_image += 1
            df.at[i, 'label'] = 'bad_image'
            df.at[i, 'triage'] = 'bad_image'
            df.at[i, 'review_reason'] = 'pil_open_failed'
            df.at[i, 'label_conf'] = 0.0
            df.at[i, 'brandName'] = 'unknown'
            if bad_image <= PRINT_BAD_IMAGE_LIMIT:
                print(f'[BAD_IMAGE] path={str(img_path)}')
            continue

        img_feat = image_feature_cache.get(item_id)
        if img_feat is None:
            bad_image += 1
            df.at[i, 'label'] = 'bad_image'
            df.at[i, 'triage'] = 'bad_image'
            df.at[i, 'review_reason'] = 'feature_missing'
            df.at[i, 'label_conf'] = 0.0
            df.at[i, 'brandName'] = 'unknown'
            continue

        title = str(row.get('title', '') or '')
        content = str(row.get('content', '') or '')

        brand_name = extract_brand_name(title, content)
        df.at[i, 'brandName'] = brand_name

        kws = extract_fashion_keywords(title, content)
        df.at[i, 'fashion_keywords'] = kws

        kw_sentence = build_keyword_sentence(kws)
        kw_feat = kw_encoder.encode(kw_sentence)

        out = classify_2stage(
            img_feat_cpu=img_feat,
            kw_feat_cpu=kw_feat,
            keywords=kws,
            brand_name=brand_name,
            coarse_labels=coarse_labels,
            coarse_embs_cpu=coarse_embs_cpu,
            fine_embs_map_cpu=fine_embs_map_cpu,
        )

        final_label = str(out['final_label'])
        coarse_label = str(out['coarse_pred'])

        df.at[i, 'label'] = final_label
        df.at[i, 'label_conf'] = float(out['final_conf'])
        df.at[i, 'coarse_label'] = coarse_label
        df.at[i, 'coarse_conf'] = float(out['coarse_conf'])
        df.at[i, 'coarse_margin'] = float(out['coarse_margin'])
        df.at[i, 'coarse_ratio'] = float(out['coarse_ratio'])
        df.at[i, 'fine_conf'] = float(out['fine_conf'])
        df.at[i, 'fine_margin'] = float(out['fine_margin'])
        df.at[i, 'fine_ratio'] = float(out['fine_ratio'])
        df.at[i, 'triage'] = str(out['triage'])
        df.at[i, 'review_reason'] = str(out['review_reason'])

        fine_top3 = out['fine_top3']
        df.at[i, 'top3_labels'] = ','.join([x[0] for x in fine_top3])
        df.at[i, 'top3_scores'] = ','.join([f'{x[1]:.4f}' for x in fine_top3])

        if MOVE_TO_LABEL_DIR:
            if final_label in {'missing_image', 'bad_image'}:
                pass
            elif final_label == 'unknown':
                if MOVE_UNKNOWN_TOO:
                    move_image_to_label_dir(img_path, UNKNOWN_DIR_NAME, IMG_DIR)
                    moved += 1
            else:
                target_label = coarse_label if MOVE_BY == 'coarse' else final_label
                move_image_to_label_dir(img_path, target_label, IMG_DIR)
                moved += 1

    df.to_csv(OUT_CSV_PATH, index=False)
    print(f'\nSaved: {OUT_CSV_PATH.resolve()}')

    review_df = df[df['triage'] == 'review'].copy()
    review_df.to_csv(REVIEW_CSV_PATH, index=False)
    print(f'Saved review queue: {REVIEW_CSV_PATH.resolve()}')

    print(f'Moved/copied: {moved}')
    print(f'Missing path: {missing_path}')
    print(f'Bad images  : {bad_image}')

    print('\nLabel distribution (top 20):')
    print(df['label'].value_counts(dropna=False).head(20).to_string())

    print('\nBrand distribution (top 20):')
    print(df['brandName'].value_counts(dropna=False).head(20).to_string())

    print('\nTriage distribution:')
    print(df['triage'].value_counts(dropna=False).to_string())

    audit_leftovers(IMG_DIR)


if __name__ == '__main__':
    main()
