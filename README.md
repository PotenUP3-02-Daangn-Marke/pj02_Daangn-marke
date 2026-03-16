# 🥕 당근막캐 (Daangn Market Sales Prediction) MVP
## 🇰🇷 [FEAT] 발표용 MVP 최종 스트림릿 업데이트 및 예측 파이프라인 연동

### 💡 개요
발표용으로 사용할 당근마켓 의류 판매 예측 모델의 MVP Streamlit 화면(UI/UX) 최종 업데이트 및 백엔드 예측 파이프라인 연동을 완료했습니다. 팀원들이 바로 테스트해 볼 수 있도록 샘플 데이터도 함께 추가했습니다.

### 🛠️ 주요 변경 사항
**프론트엔드 & API 연동**
* `app.py`: Streamlit 기반 최종 UI 업데이트 및 디자인 레이아웃 수정
* `api.py` (신규): FastAPI/백엔드 서버와 통신하기 위한 로직 추가

**예측 파이프라인 업데이트**
* `src/predict_pipeline.py`: 모델 예측 파이프라인과 프론트엔드 연결 로직 수정

**패키지 의존성 동기화**
* `pyproject.toml`, `uv.lock`: 실행 환경 구성을 위한 패키지 버전 업데이트

**테스트용 데이터 추가 (Git LFS)**
* `collected_data/`: 앱 구동 테스트를 위한 샘플 이미지 폴더 및 `metadata.csv` 파일 추가

### 🏃‍♂️ 테스트 방법 (Reviewer's Guide)
1. 최신 코드를 pull 받습니다. (`building_streamlit` 브랜치)
2. 환경 의존성을 동기화합니다:
   ```bash
   uv sync
---
3. 백엔드 서버와 Streamlit 앱을 실행하여 UI가 정상적으로 렌더링되고 예측 결과가 잘 나오는지 확인해 주세요:
   ```bash
   uvicorn api:app --reload --port 8000
   uv run streamlit run app.py
### 📌 참고사항
* 생성되었던 다수의 백업 복사본 파일(app copy_...py 등)은 히스토리를 깔끔하게 유지하기 위해 커밋에서 제외했습니다.

## 🇺🇸 [FEAT] Final Streamlit MVP Update & Prediction Pipeline Integration

### 💡 Overview
Completed the final Streamlit UI/UX updates for the Daangn Market clothing sales prediction MVP and successfully integrated it with the backend prediction pipeline. Sample data has also been included so the team can easily test the application locally.

### 🛠️ Key Changes
**Frontend & API Integration**
* `app.py`: Finalized Streamlit UI updates and refined the overall design layout.
* `api.py` (New): Added logic to handle communication with the FastAPI backend server.

**Prediction Pipeline**
* `src/predict_pipeline.py`: Updated logic to seamlessly connect the machine learning prediction pipeline with the frontend.

**Dependency Synchronization**
* `pyproject.toml`, `uv.lock`: Synchronized package versions to ensure consistent execution environments across the team.

**Sample Data Added (via Git LFS)**
* `collected_data/`: Added a sample image folder and `metadata.csv` to facilitate local app testing for reviewers.

### 🏃‍♂️ Reviewer's Guide (How to Test)
1. Pull the latest code from this branch (`building_streamlit`).
2. Sync the environment dependencies:
   ```bash
   uv sync
3. Run the Streamlit app and verify that the UI renders correctly and predictions work as expected using the provided sample images:
   ```bash
   uvicorn api:app --reload --port 8000
   uv run streamlit run app.py
### 📌 Notes
* Numerous backup files generated during the development process (e.g., app copy_...py) were intentionally excluded from this commit to maintain a clean repository history.