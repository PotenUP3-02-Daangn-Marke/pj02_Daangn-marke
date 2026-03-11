"""
=============================================================================
[파일 2] src/siglip_predictor.py
- 역할: 무거운 SigLIP2 모델을 캐싱하고, 단일 이미지에 대한 실시간 추론을 담당합니다.
=============================================================================
"""

import importlib
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    siglip_module = importlib.import_module('01_siglip_labeling_pipeline')
except Exception as e:
    raise ImportError(
        f'01_siglip_labeling_pipeline.py 모듈을 불러오는 데 실패했습니다: {e}'
    )


class SiglipSinglePredictor:
    def __init__(self):
        print('🚀 SigLIP2 모델 및 텍스트 임베딩 로드 중... (초기 1회 약 30초~1분 소요)')
        self.model, self.processor = siglip_module.load_siglip2_model_and_processor()

        self.coarse_labels, self.coarse_embs_cpu, _ = (
            siglip_module.build_label_text_embeddings(
                self.model, self.processor, siglip_module.COARSE_PROMPTS
            )
        )

        self.fine_embs_map_cpu = {}
        for g in siglip_module.COARSE_LABELS:
            lbls, embs, _ = siglip_module.build_label_text_embeddings(
                self.model, self.processor, siglip_module.FINE_PROMPTS[g]
            )
            self.fine_embs_map_cpu[g] = (lbls, embs)

        self.kw_encoder = siglip_module.KeywordTextEncoder(self.model, self.processor)
        print('✅ SigLIP2 실시간 예측기 세팅 완료!')

    @torch.no_grad()
    def predict(self, image_file, title, content, brand_name='unknown'):
        if image_file is None:
            return 'other'

        try:
            if isinstance(image_file, Image.Image):
                img = image_file.convert('RGB')
            else:
                img = Image.open(image_file).convert('RGB')
        except Exception as e:
            print(f'이미지 변환 에러: {e}')
            return 'other'

        img_feat = (
            siglip_module.get_image_features_batch(self.model, self.processor, [img])
            .detach()
            .cpu()
        )

        title_str = str(title) if title else ''
        content_str = str(content) if content else ''
        kws = siglip_module.extract_fashion_keywords(title_str, content_str)
        kw_sentence = siglip_module.build_keyword_sentence(kws)
        kw_feat = self.kw_encoder.encode(kw_sentence)

        out = siglip_module.classify_2stage(
            img_feat_cpu=img_feat,
            kw_feat_cpu=kw_feat,
            keywords=kws,
            brand_name=brand_name,
            coarse_labels=self.coarse_labels,
            coarse_embs_cpu=self.coarse_embs_cpu,
            fine_embs_map_cpu=self.fine_embs_map_cpu,
        )
        return out
