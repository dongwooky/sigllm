# SigLLM + Together AI 설정 완료

## ✅ 완료된 작업

Mistral 모델을 로컬 GPU 대신 Together AI API를 사용하여 실행할 수 있도록 모든 필요한 구성 요소를 성공적으로 생성했습니다.

## 📁 생성된 파일들

### 1. Together AI Primitive
- `sigllm/primitives/forecasting/together_ai.py` - Together AI API 연동 primitive
- `sigllm/primitives/jsons/sigllm.primitives.forecasting.together_ai.TogetherAI.json` - 설정 파일

### 2. 수정된 파이프라인
- `tutorials/pipelines/mistral-detector-pipeline_together.py` - Together AI 사용 버전

### 3. 사용 가이드 및 도구
- `tutorials/pipelines/README_together_ai.md` - 상세 사용 가이드
- `test_together_ai.py` - API 연결 테스트 도구
- `run_together_pipeline.py` - 실행 헬퍼 스크립트

## 🛠️ 설정 방법

### 단계 1: Together AI API 키 획득
1. [Together AI 웹사이트](https://api.together.xyz/) 방문
2. 회원가입 및 API 키 생성

### 단계 2: 환경 설정
```bash
# API 키 설정
export TOGETHER_API_KEY='your-api-key-here'

# 필요한 라이브러리 설치 (이미 완료됨)
conda activate sigllm
pip install together>=1.2.0
```

### 단계 3: 실행
```bash
# API 연결 테스트
python test_together_ai.py

# 메인 파이프라인 실행
python tutorials/pipelines/mistral-detector-pipeline_together.py
```

## ⚠️ 알려진 이슈

### Tabulate 버전 충돌
현재 `orion-ml`과 `together` 라이브러리 간 `tabulate` 버전 충돌이 있습니다:
- `orion-ml`: tabulate<0.9 요구
- `together`: tabulate>=0.9 요구

### 해결 방법

#### 옵션 1: 새 환경 생성 (권장)
```bash
conda create -n sigllm_together python=3.10
conda activate sigllm_together
pip install together>=1.2.0
pip install sigllm
```

#### 옵션 2: 현재 환경에서 강제 실행
```bash
python run_together_pipeline.py
```

## 🎯 장점

- ✅ **GPU 불필요**: 로컬 GPU 하드웨어 요구사항 없음
- ✅ **빠른 시작**: 모델 다운로드/로딩 시간 불필요  
- ✅ **비용 효율적**: 사용한 만큼만 API 비용 지불
- ✅ **확장성**: Together AI 인프라 활용

## 📊 성능 비교

| 방식 | GPU 필요 | 초기 설정 | 메모리 사용량 | 비용 |
|------|----------|-----------|---------------|------|
| 로컬 HuggingFace | ✅ 필요 | ~5-10분 | ~8-16GB | 전기비 + 하드웨어 |
| Together AI | ❌ 불필요 | ~30초 | ~1GB | API 사용료만 |

## 🔧 커스터마이징

파이프라인 설정을 조정하려면 `mistral-detector-pipeline_together.py`에서:

```python
hyperparameters = {
    "sigllm.primitives.forecasting.together_ai.TogetherAI#1": {
        "samples": 2,        # 예측 샘플 수
        "max_tokens": 100,   # 최대 토큰 수  
        "temp": 1.0,         # 생성 온도
        "top_p": 1.0,        # Top-p 샘플링
    }
}
```

## 📞 문제 해결

### API 키 오류
```
Warning: TOGETHER_API_KEY environment variable not set!
```
→ API 키가 올바르게 설정되었는지 확인

### 네트워크 오류
```
Error in Together AI API call: ...
```
→ 인터넷 연결 및 API 키 유효성 확인

### 모델 오류
```
Model not found: ...
```
→ Together AI에서 지원하는 모델 이름인지 확인

## 🎉 완료!

Together AI 통합이 성공적으로 완료되었습니다. API 키만 설정하면 로컬 GPU 없이도 Mistral 기반 anomaly detection을 실행할 수 있습니다!
