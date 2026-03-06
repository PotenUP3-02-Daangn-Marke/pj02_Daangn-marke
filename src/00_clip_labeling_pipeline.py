# clip_2stage_labeling_move_to_label_dirs.py
# Python 3.12+ (modern typing), OpenAI CLIP, 2-stage, prompt-ensemble,
# keyword-only text fusion, top-3 저장, confidence threshold,
# df에 label 컬럼 추가 + (옵션) label별 subdir로 "이동/복사" (파일명 유지)
#
# ✅ Robust diagnostics:
#   - PRECHECK: df id vs root image stems (should match your audit)
#   - missing_path logs (first N)
#   - bad_image logs (PIL read failures)
#   - leftovers audit in root after processing

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

import clip  # pip install git+https://github.com/openai/CLIP.git
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# =========================
# Config
# =========================
KEYWORD = '폴로'
IMG_DIR = Path(f'./data/images/{KEYWORD}')
CSV_PATH = Path(f'./data/csv/daangn_{KEYWORD}.csv')
OUT_CSV_PATH = Path(f'./data/csv/daangn_{KEYWORD}_labeled.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLIP_MODEL_NAME = 'ViT-B/32'

# Image vs keyword-text fusion
ALPHA_IMG = 0.85
BETA_TXT = 1.0 - ALPHA_IMG

# 운영 파라미터
TOPK = 3

# ✅ Move to label subdir (keep filename)
MOVE_TO_LABEL_DIR = True
COPY_INSTEAD_OF_MOVE = False  # True면 원본 유지 + 라벨 폴더에 복사본 생성
DRY_RUN_MOVE = False  # True면 실제 이동/복사 안 하고 계획만 출력
MOVE_UNKNOWN_TOO = True
UNKNOWN_DIR_NAME = '_unknown'

# Confidence policy
COARSE_MARGIN_THRESH = 0.12
FINE_MARGIN_THRESH = 0.08
FINE_CONF_THRESH = 0.18

# Supported image extensions
EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.jfif'}

# Debug limits
PRINT_MISSING_LIMIT = 30
PRINT_BAD_IMAGE_LIMIT = 30


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
# Keyword extraction (title/content -> keywords only)
# =========================
FASHION_KEYWORDS = [
    # tops
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
    # bottoms
    '바지',
    '청바지',
    '데님',
    '슬랙스',
    '반바지',
    '쇼츠',
    # outers
    '자켓',
    '재킷',
    '코트',
    '롱코트',
    '패딩',
    '다운',
    '점퍼',
    '블레이저',
    # dress/skirt
    '원피스',
    '드레스',
    '치마',
    '스커트',
    # bags
    '가방',
    '백팩',
    '숄더백',
    '크로스백',
    '토트백',
    '핸드백',
    # shoes
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


def coarse_prior_from_keywords(keywords: str, coarse_labels: list[str]) -> torch.Tensor:
    """
    keyword 기반으로 coarse score bias를 준다.
    반환 shape: (len(coarse_labels),)
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
            bias[i] = 1.5 + 0.5 * overlap  # 적당한 가중치
    return bias


def extract_fashion_keywords(title: str, content: str) -> str:
    text = f'{title}\n{content}'.lower()
    found: list[str] = []
    for kw in FASHION_KEYWORDS:
        if kw.lower() in text:
            found.append(kw)
    norm: list[str] = []
    for kw in found:
        norm.append(KW_NORMALIZE.get(kw, kw))
    norm = list(dict.fromkeys(norm))  # unique preserve order
    return ' '.join(norm).strip()


def build_keyword_sentence(keywords: str) -> str:
    if not keywords:
        return ''
    return f'의류 카테고리 키워드: {keywords}'


# =========================
# Prompt engineering (5~10 prompts per label)
# =========================
def build_prompts_for_label(label: str) -> list[str]:
    base = [
        f'a photo of {label}',
        f'a marketplace photo of {label}',
        f'a close-up product photo of {label}',
        f'a photo of {label} laid flat on the floor',
        f'a photo of someone wearing {label}',
        f'a photo of {label} clothing item',
    ]

    ko: list[str] = []
    if label == 'top':
        ko = [
            '상의: 티셔츠, 셔츠, 카라티, 니트, 맨투맨, 후드티 같은 상체에 입는 옷',
            '상체에 입는 옷 사진, 상의',
        ]
    elif label == 'bottom':
        ko = [
            '하의: 바지, 청바지, 슬랙스, 반바지 같은 하체에 입는 옷',
            '하체에 입는 옷 사진, 하의',
        ]
    elif label == 'outer':
        ko = [
            '아우터: 자켓, 코트, 패딩, 점퍼, 블레이저 같은 겉옷',
            '겉에 걸치는 옷 사진, 아우터',
        ]
    elif label == 'dress':
        ko = ['원피스, 드레스: 상의와 하의가 한 벌로 이어진 옷', '원피스 사진']
    elif label == 'skirt':
        ko = ['치마, 스커트: 허리부터 아래로 내려오는 하의', '스커트 사진']
    elif label == 'polo shirt':
        ko = [
            '카라가 있는 반팔 상의, 폴로 셔츠, 카라티',
            '카라와 단추가 있는 티셔츠 스타일 상의',
        ]
    elif label == 't-shirt':
        ko = ['반팔 티셔츠, 기본 티, 라운드넥 티', '프린팅이 있을 수 있는 티셔츠']
    elif label == 'shirt':
        ko = ['셔츠, 와이셔츠, 단추가 있는 긴팔 상의', '카라가 있고 단추로 여미는 셔츠']
    elif label == 'hoodie':
        ko = ['후드티, 모자가 달린 상의', '후드가 있는 스웨트셔츠']
    elif label == 'sweatshirt':
        ko = ['맨투맨, 후드 없는 스웨트셔츠', '라운드넥 맨투맨 상의']
    elif label == 'knit sweater':
        ko = ['니트, 스웨터, 뜨개 느낌의 상의', '두께감 있는 니트 상의']
    elif label == 'cardigan':
        ko = ['가디건, 앞이 열리는 니트 아우터', '단추가 있는 가디건']
    elif label == 'sleeveless':
        ko = ['민소매, 나시, 슬리브리스 상의', '어깨가 드러나는 상의']
    elif label == 'jeans':
        ko = ['청바지, 데님 팬츠', '데님 소재의 바지']
    elif label == 'pants':
        ko = ['바지, 슬랙스, 면바지', '긴 바지']
    elif label == 'shorts':
        ko = ['반바지, 쇼츠', '무릎 위 길이의 바지']
    elif label == 'jacket':
        ko = ['자켓, 재킷, 가벼운 아우터', '지퍼/단추로 여미는 아우터']
    elif label == 'denim jacket':
        ko = ['청자켓, 데님 재킷', '데님 소재의 아우터']
    elif label == 'coat':
        ko = ['코트, 롱코트, 겨울 아우터', '무릎 아래로 내려오는 코트']
    elif label == 'padding jacket':
        ko = ['패딩, 다운 자켓, 퀼팅 아우터', '두꺼운 겨울 점퍼']
    elif label == 'blazer':
        ko = ['블레이저, 정장 자켓', '테일러드 자켓']
    elif label == 'bags':
        ko = [
            '가방: 백팩, 크로스백, 숄더백, 토트백 같은 패션 잡화',
            '가방 제품 사진',
        ]
    elif label == 'shoes':
        ko = [
            '신발: 운동화, 부츠, 샌들, 로퍼 같은 신발',
            '신발 제품 사진',
        ]
    elif label == 'backpack':
        ko = ['백팩, 배낭 스타일 가방']
    elif label == 'shoulder bag':
        ko = ['숄더백, 어깨에 메는 가방']
    elif label == 'crossbody bag':
        ko = ['크로스백, 몸에 가로로 메는 가방']
    elif label == 'tote bag':
        ko = ['토트백, 손잡이가 있는 가방']
    elif label == 'handbag':
        ko = ['핸드백, 작은 여성 가방']
    elif label == 'sneakers':
        ko = ['운동화, 스니커즈']
    elif label == 'boots':
        ko = ['부츠, 발목 또는 종아리까지 올라오는 신발']
    elif label == 'sandals':
        ko = ['샌들, 여름 신발']
    elif label == 'heels':
        ko = ['하이힐, 굽 있는 여성 신발']
    elif label == 'loafers':
        ko = ['로퍼, 끈 없는 구두']
    elif label == 'other':
        ko = ['의류 외 기타, 분류 불가', '기타']

    prompts = base[:6] + ko[:3]
    return prompts[:10]


def build_prompt_ensemble(labels: list[str]) -> dict[str, list[str]]:
    return {lbl: build_prompts_for_label(lbl) for lbl in labels}


COARSE_PROMPTS = build_prompt_ensemble(COARSE_LABELS)
FINE_PROMPTS = {g: build_prompt_ensemble(lbls) for g, lbls in FINE_LABELS.items()}


# =========================
# CLIP encoding utilities
# =========================
@torch.no_grad()
def encode_image(model, preprocess, image_path: Path) -> torch.Tensor | None:
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception:
        # caller will handle
        return None
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


@torch.no_grad()
def encode_text(model, texts: list[str]) -> torch.Tensor:
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def mean_pool_text_feats(text_feats: torch.Tensor) -> torch.Tensor:
    feat = text_feats.mean(dim=0, keepdim=True)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def build_label_text_embeddings(
    model, prompt_dict: dict[str, list[str]]
) -> tuple[list[str], torch.Tensor]:
    labels = list(prompt_dict.keys())
    embs: list[torch.Tensor] = []
    for lbl in labels:
        tf = encode_text(model, prompt_dict[lbl])
        embs.append(mean_pool_text_feats(tf))
    return labels, torch.cat(embs, dim=0)


def scores_with_scale(
    vec: torch.Tensor, mat: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    return (logit_scale * (vec @ mat.T)).squeeze(0)


def softmax_confidence(scores: torch.Tensor) -> float:
    return float(torch.softmax(scores, dim=0).max().item())


def top1_margin(scores: torch.Tensor) -> float:
    k = min(2, int(scores.numel()))
    v, _ = torch.topk(scores, k=k)
    if v.numel() < 2:
        return float('inf')
    return float((v[0] - v[1]).item())


def topk(
    scores: torch.Tensor, labels: list[str], k: int = 3
) -> list[tuple[str, float]]:
    k = min(k, len(labels))
    vals, idx = torch.topk(scores, k=k)
    return [(labels[int(i)], float(v.item())) for v, i in zip(vals, idx, strict=True)]


# =========================
# 2-stage classification
# =========================
def classify_2stage(
    model,
    img_feat: torch.Tensor,
    kw_feat: torch.Tensor | None,
    keywords: str,
    coarse_labels: list[str],
    coarse_embs: torch.Tensor,
    fine_embs_map: dict[str, tuple[list[str], torch.Tensor]],
) -> dict[str, object]:
    logit_scale = model.logit_scale.exp()

    # ---- Stage 1: coarse
    s_img = scores_with_scale(img_feat, coarse_embs, logit_scale)

    if kw_feat is not None:
        s_txt = scores_with_scale(kw_feat, coarse_embs, logit_scale)
        s = ALPHA_IMG * s_img + BETA_TXT * s_txt
    else:
        s = s_img

    # keyword prior
    kw_prior = coarse_prior_from_keywords(keywords, coarse_labels).to(s.device)
    s = s + kw_prior

    coarse_top = topk(s, coarse_labels, k=TOPK)
    coarse_pred = coarse_top[0][0]
    coarse_conf = softmax_confidence(s)
    coarse_margin = top1_margin(s)
    coarse_low = coarse_margin < COARSE_MARGIN_THRESH

    # ---- Stage 2: fine
    fine_labels, fine_embs = fine_embs_map[coarse_pred]
    s2_img = scores_with_scale(img_feat, fine_embs, logit_scale)

    if kw_feat is not None:
        s2_txt = scores_with_scale(kw_feat, fine_embs, logit_scale)
        s2 = ALPHA_IMG * s2_img + BETA_TXT * s2_txt
    else:
        s2 = s2_img

    fine_top = topk(s2, fine_labels, k=TOPK)
    fine_pred = fine_top[0][0]
    fine_conf = softmax_confidence(s2)
    fine_margin = top1_margin(s2)

    fine_low = (fine_margin < FINE_MARGIN_THRESH) or (fine_conf < FINE_CONF_THRESH)

    final_label = fine_pred
    if coarse_low and fine_low:
        final_label = 'unknown'

    return {
        'coarse_pred': coarse_pred,
        'coarse_conf': coarse_conf,
        'coarse_margin': coarse_margin,
        'coarse_top3': coarse_top,
        'fine_group': coarse_pred,
        'fine_pred': fine_pred,
        'fine_conf': fine_conf,
        'fine_margin': fine_margin,
        'fine_top3': fine_top,
        'final_label': final_label,
        'final_conf': fine_conf,
    }


# =========================
# File helpers (move to label dir)
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

    dest = label_dir / img_path.name  # keep original filename

    # already in right folder
    if img_path.parent == label_dir:
        return dest

    # collision avoidance
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

    # title/content optional
    if 'title' not in df.columns:
        df['title'] = ''
    if 'content' not in df.columns:
        df['content'] = ''

    # ✅ sanity prints (path mismatch detector)
    print('CSV_PATH:', CSV_PATH.resolve())
    print('IMG_DIR :', IMG_DIR.resolve())
    root_image_count = sum(
        1 for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS
    )
    print('df rows :', len(df))
    print('root image files:', root_image_count)

    # ✅ PRECHECK (should match your audit numbers)
    df_ids = set(df['id'].astype(str).str.strip())
    img_ids = {
        p.stem.strip()
        for p in IMG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    }

    print('PRECHECK matched:', len(df_ids & img_ids))
    print('PRECHECK df-only :', len(df_ids - img_ids))
    print('PRECHECK img-only:', len(img_ids - df_ids))

    model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    model.eval()

    # Precompute prompt-ensemble embeddings
    coarse_labels, coarse_embs = build_label_text_embeddings(model, COARSE_PROMPTS)

    fine_embs_map: dict[str, tuple[list[str], torch.Tensor]] = {}
    for g in COARSE_LABELS:
        lbls, embs = build_label_text_embeddings(model, FINE_PROMPTS[g])
        fine_embs_map[g] = (lbls, embs)

    # Output columns
    df['fashion_keywords'] = ''
    df['label'] = ''
    df['label_conf'] = 0.0
    df['coarse_label'] = ''
    df['coarse_conf'] = 0.0
    df['coarse_margin'] = 0.0
    df['fine_conf'] = 0.0
    df['fine_margin'] = 0.0
    df['top3_labels'] = ''
    df['top3_scores'] = ''

    missing_path = 0
    bad_image = 0
    moved = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc='CLIP 2-stage labeling'):
        item_id = str(row['id']).strip()

        img_path = find_image_by_id(IMG_DIR, item_id)
        if img_path is None:
            missing_path += 1
            df.at[i, 'label'] = 'missing_image'
            df.at[i, 'label_conf'] = 0.0
            if missing_path <= PRINT_MISSING_LIMIT:
                print(f'[MISSING_PATH] id={item_id!r} expected under {IMG_DIR}')
            continue

        title = str(row.get('title', '') or '')
        content = str(row.get('content', '') or '')

        kws = extract_fashion_keywords(title, content)
        df.at[i, 'fashion_keywords'] = kws
        kw_sentence = build_keyword_sentence(kws)

        img_feat = encode_image(model, preprocess, img_path)
        if img_feat is None:
            bad_image += 1
            df.at[i, 'label'] = 'bad_image'
            df.at[i, 'label_conf'] = 0.0
            if bad_image <= PRINT_BAD_IMAGE_LIMIT:
                print(f'[BAD_IMAGE] path={str(img_path)}')
            continue

        kw_feat = None
        if kw_sentence:
            kw_feat = encode_text(model, [kw_sentence])

        out = classify_2stage(
            model=model,
            img_feat=img_feat,
            kw_feat=kw_feat,
            keywords=kws,
            coarse_labels=coarse_labels,
            coarse_embs=coarse_embs,
            fine_embs_map=fine_embs_map,
        )

        final_label = str(out['final_label'])
        df.at[i, 'label'] = final_label
        df.at[i, 'label_conf'] = float(out['final_conf'])
        df.at[i, 'coarse_label'] = str(out['coarse_pred'])
        df.at[i, 'coarse_conf'] = float(out['coarse_conf'])
        df.at[i, 'coarse_margin'] = float(out['coarse_margin'])
        df.at[i, 'fine_conf'] = float(out['fine_conf'])
        df.at[i, 'fine_margin'] = float(out['fine_margin'])

        fine_top3 = out['fine_top3']  # list[tuple[str, float]]
        df.at[i, 'top3_labels'] = ','.join([x[0] for x in fine_top3])
        df.at[i, 'top3_scores'] = ','.join([f'{x[1]:.4f}' for x in fine_top3])

        # Move/copy to label dir (keep filename)
        coarse_label = str(out['coarse_pred'])

        if MOVE_TO_LABEL_DIR:
            if final_label == 'missing_image':
                pass
            elif final_label == 'unknown':
                if MOVE_UNKNOWN_TOO:
                    move_image_to_label_dir(img_path, UNKNOWN_DIR_NAME, IMG_DIR)
                    moved += 1
            else:
                move_image_to_label_dir(img_path, coarse_label, IMG_DIR)
                moved += 1

    df.to_csv(OUT_CSV_PATH, index=False)
    print(f'\nSaved: {OUT_CSV_PATH.resolve()}')
    print(f'Moved/copied: {moved}')
    print(f'Missing path: {missing_path}')
    print(f'Bad images  : {bad_image}')

    # label distribution
    print('\nLabel distribution (top 20):')
    print(df['label'].value_counts(dropna=False).head(20).to_string())

    audit_leftovers(IMG_DIR)


if __name__ == '__main__':
    main()
