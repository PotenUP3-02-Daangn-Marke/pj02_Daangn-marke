# clip_2stage_labeling.py
# Python 3.12+ (modern typing), OpenAI CLIP, 2-stage, prompt-ensemble,
# keyword-only text fusion, top-3 저장, confidence threshold, df에 label 컬럼 추가,

from __future__ import annotations

import re
import shutil
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

# image vs keyword-text fusion
ALPHA_IMG = 0.85
BETA_TXT = 1.0 - ALPHA_IMG

# 운영 파라미터
TOPK = 3

MOVE_TO_LABEL_DIR = True  # 라벨 폴더로 이동할지
COPY_INSTEAD_OF_MOVE = False  # True면 원본 유지 + 라벨폴더에 복사본 생성
DRY_RUN_MOVE = False  # True면 이동 계획만 출력

# confidence 정책 (softmax peak는 낮게 나올 수 있어서 margin 기반도 같이 씀)
COARSE_MARGIN_THRESH = 0.25  # coarse top1-top2 margin
FINE_MARGIN_THRESH = 0.20  # fine top1-top2 margin

FINE_CONF_THRESH = 0.23  # softmax peak 보조(너무 낮으면 unknown)


# =========================
# 2-stage taxonomy
# =========================
COARSE_LABELS = ['top', 'bottom', 'outer', 'dress', 'skirt', 'other']

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
    'bottom': ['jeans', 'pants', 'shorts'],
    'outer': ['jacket', 'denim jacket', 'coat', 'padding jacket', 'blazer'],
    'dress': ['dress'],
    'skirt': ['skirt'],
    'other': ['other'],
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
]

KW_NORMALIZE = {
    '폴로': '카라티',
    '재킷': '자켓',
    '쇼츠': '반바지',
    '데님': '청바지',
    '다운': '패딩',
}


def extract_fashion_keywords(title: str, content: str) -> str:
    text = f'{title}\n{content}'.lower()
    found: list[str] = []
    for kw in FASHION_KEYWORDS:
        if kw.lower() in text:
            found.append(kw)
    norm: list[str] = []
    for kw in found:
        norm.append(KW_NORMALIZE.get(kw, kw))
    # unique, preserve order
    norm = list(dict.fromkeys(norm))
    return ' '.join(norm).strip()


def build_keyword_sentence(keywords: str) -> str:
    if not keywords:
        return ''
    return f'의류 카테고리 키워드: {keywords}'


# =========================
# Prompt engineering (5~10 prompts per label) + improved coarse prompts
# =========================
def build_prompts_for_label(label: str) -> list[str]:
    # sentence-style prompts (CLIP likes these)
    base = [
        f'a photo of {label}',
        f'a marketplace photo of {label}',
        f'a close-up product photo of {label}',
        f'a photo of {label} laid flat on the floor',
        f'a photo of someone wearing {label}',
        f'a photo of {label} clothing item',
    ]

    # label-specific Korean cues (+ examples for coarse groups)
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
    elif label == 'other':
        ko = ['의류 외 기타, 분류 불가', '기타']
    else:
        ko = [label]

    prompts = base[:6] + ko[:3]  # 6~9개
    return prompts[:10]


def build_prompt_ensemble(labels: list[str]) -> dict[str, list[str]]:
    return {lbl: build_prompts_for_label(lbl) for lbl in labels}


COARSE_PROMPTS = build_prompt_ensemble(COARSE_LABELS)
FINE_PROMPTS = {g: build_prompt_ensemble(lbls) for g, lbls in FINE_LABELS.items()}


# =========================
# CLIP encoding utilities
# =========================
@torch.no_grad()
def encode_image(model, preprocess, image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # (1, D)


@torch.no_grad()
def encode_text(model, texts: list[str]) -> torch.Tensor:
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # (N, D)


def mean_pool_text_feats(text_feats: torch.Tensor) -> torch.Tensor:
    feat = text_feats.mean(dim=0, keepdim=True)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # (1, D)


def build_label_text_embeddings(
    model,
    prompt_dict: dict[str, list[str]],
) -> tuple[list[str], torch.Tensor]:
    labels = list(prompt_dict.keys())
    embs: list[torch.Tensor] = []
    for lbl in labels:
        prompts = prompt_dict[lbl]
        tf = encode_text(model, prompts)  # (P, D)
        emb = mean_pool_text_feats(tf)  # (1, D)
        embs.append(emb)
    return labels, torch.cat(embs, dim=0)  # (L, D)


def scores_with_scale(
    vec: torch.Tensor, mat: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    # vec: (1,D), mat: (L,D) => (L,)
    return (logit_scale * (vec @ mat.T)).squeeze(0)


def softmax_confidence(scores: torch.Tensor) -> float:
    probs = torch.softmax(scores, dim=0)
    return float(probs.max().item())


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
    coarse_labels: list[str],
    coarse_embs: torch.Tensor,
    fine_embs_map: dict[str, tuple[list[str], torch.Tensor]],
) -> dict[str, object]:
    """
    Returns dict with:
      coarse_pred, coarse_conf, coarse_margin, coarse_top3
      fine_group, fine_pred, fine_conf, fine_margin, fine_top3
      final_label, final_conf
    """
    logit_scale = model.logit_scale.exp()

    # ---- Stage 1 (coarse): DO NOT hard-fallback to "other" here
    s_img = scores_with_scale(img_feat, coarse_embs, logit_scale)
    if kw_feat is not None:
        s_txt = scores_with_scale(kw_feat, coarse_embs, logit_scale)
        s = ALPHA_IMG * s_img + BETA_TXT * s_txt
    else:
        s = s_img

    coarse_top = topk(s, coarse_labels, k=TOPK)
    coarse_pred = coarse_top[0][0]
    coarse_conf = softmax_confidence(s)
    coarse_margin = top1_margin(s)
    coarse_low = coarse_margin < COARSE_MARGIN_THRESH

    # ---- Stage 2 (fine within predicted coarse group)
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
    fine_low = fine_margin < FINE_MARGIN_THRESH or fine_conf < FINE_CONF_THRESH

    final_label = fine_pred
    final_conf = fine_conf

    # Final reject policy (unknown)
    if coarse_low or fine_low:
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
        'final_conf': final_conf,
    }


# =========================
# File helpers
# =========================
def sanitize_for_filename(label: str) -> str:
    label = label.strip().lower()
    label = re.sub(r'\s+', '-', label)
    label = re.sub(r'[^a-z0-9가-힣\-_.]+', '', label)
    return label[:60] if label else 'unknown'


def find_image_by_id(img_dir: Path, item_id: str) -> Path | None:
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        p = img_dir / f'{item_id}{ext}'
        if p.exists():
            return p
        p2 = img_dir / f'{item_id}{ext.upper()}'
        if p2.exists():
            return p2
    return None


def move_image_to_label_dir(img_path: Path, label: str, base_dir: Path) -> Path:
    """
    Move/copy img_path into base_dir/label/ keeping original filename.
    """
    safe_label = sanitize_for_filename(label)
    label_dir = base_dir / safe_label
    label_dir.mkdir(parents=True, exist_ok=True)

    dest = label_dir / img_path.name  # keep original filename

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


# =========================
# Main
# =========================
def main():
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

    missing = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc='CLIP 2-stage labeling'):
        item_id = str(row['id']).strip()

        img_path = find_image_by_id(IMG_DIR, item_id)
        if img_path is None:
            missing += 1
            df.at[i, 'label'] = 'missing_image'
            df.at[i, 'label_conf'] = 0.0
            continue

        title = str(row.get('title', '') or '')
        content = str(row.get('content', '') or '')

        # keyword-only text
        kws = extract_fashion_keywords(title, content)
        df.at[i, 'fashion_keywords'] = kws
        kw_sentence = build_keyword_sentence(kws)

        with torch.no_grad():
            img_feat = encode_image(model, preprocess, img_path)

            kw_feat = None
            if kw_sentence:
                kw_feat = encode_text(model, [kw_sentence])

            out = classify_2stage(
                model=model,
                img_feat=img_feat,
                kw_feat=kw_feat,
                coarse_labels=coarse_labels,
                coarse_embs=coarse_embs,
                fine_embs_map=fine_embs_map,
            )

        final_label = str(out['final_label'])
        final_conf = float(out['final_conf'])
        coarse_pred = str(out['coarse_pred'])

        df.at[i, 'label'] = final_label
        df.at[i, 'label_conf'] = final_conf
        df.at[i, 'coarse_label'] = coarse_pred
        df.at[i, 'coarse_conf'] = float(out['coarse_conf'])
        df.at[i, 'coarse_margin'] = float(out['coarse_margin'])
        df.at[i, 'fine_conf'] = float(out['fine_conf'])
        df.at[i, 'fine_margin'] = float(out['fine_margin'])

        fine_top3 = out['fine_top3']  # list[tuple[str, float]]
        df.at[i, 'top3_labels'] = ','.join([x[0] for x in fine_top3])
        df.at[i, 'top3_scores'] = ','.join([f'{x[1]:.4f}' for x in fine_top3])

        # Debug first few rows (uncomment if needed)
        # if i < 20:
        #     print(item_id, out)

        if MOVE_TO_LABEL_DIR and final_label not in {'unknown', 'missing_image'}:
            move_image_to_label_dir(img_path, final_label, IMG_DIR)

    df.to_csv(OUT_CSV_PATH, index=False)
    print(f'Saved: {OUT_CSV_PATH.resolve()}')
    if missing:
        print(f"Missing images: {missing} rows (label='missing_image').")


if __name__ == '__main__':
    main()
