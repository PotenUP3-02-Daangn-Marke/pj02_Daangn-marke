import pandas as pd
from sentence_transformers import SentenceTransformer

# 한국어 문장 임베딩에 성능이 좋은 모델 로드 (GPU가 있다면 자동으로 할당됩니다)
# 처음 실행 시 모델을 다운로드하므로 시간이 조금 걸릴 수 있습니다.
model_name = 'jhgan/ko-sroberta-multitask'
embedder = SentenceTransformer(model_name)


def create_text_embeddings(df, text_columns):
    """
    지정된 텍스트 컬럼들을 임베딩 벡터로 변환하여 기존 데이터프레임에 병합합니다.
    """
    df_embedded = df.copy()

    for col in text_columns:
        print(f'[{col}] 임베딩 추출 중...')
        # 텍스트 결측치 처리 (빈 문자열로)
        texts = df_embedded[col].fillna('').tolist()

        # 임베딩 생성 (768차원 벡터로 변환됨)
        embeddings = embedder.encode(texts, show_progress_bar=True)

        # 임베딩 결과를 데이터프레임 컬럼으로 변환
        emb_df = pd.DataFrame(
            embeddings,
            columns=[f'{col}_emb_{i}' for i in range(embeddings.shape[1])],
            index=df_embedded.index,
        )

        # 기존 텍스트 컬럼 삭제 및 임베딩 컬럼 병합
        df_embedded = df_embedded.drop(columns=[col])
        df_embedded = pd.concat([df_embedded, emb_df], axis=1)

    return df_embedded


# 사용 예시 (df는 원본 데이터프레임)
df = pd.read_csv('./data/csv/train_df.csv')

text_cols = ['title', 'content']
df_vectorized = create_text_embeddings(df, text_cols)

df_vectorized.to_csv('./data/csv/vectorized_df.csv', index=False)
