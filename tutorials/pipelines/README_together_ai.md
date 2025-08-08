# SigLLM with Together AI

이 튜토리얼은 로컬 GPU 대신 Together AI API를 사용하여 Mistral 모델을 실행하는 방법을 설명합니다.

## 🎯 주요 장점

- ✅ **GPU 불필요**: 로컬 GPU 하드웨어 요구사항 없음
- ✅ **빠른 시작**: 모델 다운로드나 로딩 시간 불필요
- ✅ **비용 효율적**: 사용한 만큼만 API 비용 지불
- ✅ **확장성**: Together AI의 인프라 활용

## 📋 사전 요구사항

1. **Together AI API 키**
   - [Together AI 웹사이트](https://api.together.xyz/)에서 회원가입
   - API 키 생성

2. **Python 패키지**
   ```bash
   conda activate sigllm
   pip install together>=1.2.0
   ```

## 🚀 사용 방법

### 1. API 키 설정

```bash
export TOGETHER_API_KEY='your-api-key-here'
```

또는 Python 스크립트에서:
```python
import os
os.environ['TOGETHER_API_KEY'] = 'your-api-key-here'
```

### 2. 스크립트 실행

```bash
conda activate sigllm
cd /home/dongwook/github/sigllm
python tutorials/pipelines/mistral-detector-pipeline_together.py
```

## 📊 예상 출력

```
=== SigLLM Detector Pipeline (Together AI) ===
This script uses Together AI API - no local GPU required!
Make sure TOGETHER_API_KEY environment variable is set.

1. Loading data...
Data shape: (1624, 2)

2. Setting up pipeline...
Pipeline primitives: 12
Pipeline primitives: ['mlstars.custom.timeseries_preprocessing.time_segments_aggregate', ...]

3. Step-by-step pipeline execution...
--- Step 0: Time segments aggregate ---
--- Step 1: SimpleImputer ---
--- Step 2: Float2Scalar ---
--- Step 3: Rolling window sequences ---
--- Step 4: Format as string ---
--- Step 5: TogetherAI model ---
Note: This uses Together AI API instead of local GPU
--- Step 6: Format as integer ---
--- Step 7: Scalar2Float ---
--- Step 8: Aggregate rolling window ---
--- Step 9: Reshape ---
--- Step 10: Regression errors ---
--- Step 11: Find anomalies ---

4. Results...
🎯 Detected Anomalies (by Mistral via Together AI):
   start         end     score
0  1.310945e+09  1.310946e+09  0.8542

=== Final Mistral Detector (Together AI) Accuracy Summary ===
F1-Score: 0.7834
Precision: 0.8921
Recall: 0.6952
Accuracy: 0.9245

🌐 API Usage Note: This run used Together AI API for Mistral model inference
```

## 🔧 설정 옵션

`mistral-detector-pipeline_together.py` 파일에서 다음 설정을 조정할 수 있습니다:

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

## 📈 성능 비교

| 방식 | GPU 필요 | 초기 설정 시간 | 메모리 사용량 | 비용 |
|------|----------|----------------|---------------|------|
| 로컬 HuggingFace | ✅ 필요 | ~5-10분 | ~8-16GB | 전기비 + 하드웨어 |
| Together AI | ❌ 불필요 | ~30초 | ~1GB | API 사용료만 |

## 🛠️ 커스터마이징

### 다른 모델 사용

Together AI에서 지원하는 다른 모델을 사용하려면:

```python
hyperparameters = {
    "sigllm.primitives.forecasting.together_ai.TogetherAI#1": {
        "name": "meta-llama/Llama-2-7b-chat-hf",  # 다른 모델로 변경
        "samples": 2,
        "max_tokens": 100,
    }
}
```

### 더 작은 데이터셋으로 테스트

빠른 테스트를 위해 데이터 크기를 줄이려면:

```python
# 파일에서 이 부분의 주석을 해제하세요
start = 900
end = start + 200
data = data.iloc[start: end]
print(f"Using subset: {data.shape}")
```

## 🔍 문제 해결

### API 키 오류
```
Warning: TOGETHER_API_KEY environment variable not set!
```
→ API 키를 올바르게 설정했는지 확인하세요.

### 네트워크 오류
```
Error in Together AI API call: ...
```
→ 인터넷 연결과 API 키 유효성을 확인하세요.

### 모델 오류
```
Model not found: ...
```
→ 지원되는 모델 이름인지 Together AI 문서를 확인하세요.

## 📚 추가 리소스

- [Together AI 문서](https://docs.together.ai/)
- [SigLLM 원본 논문](https://arxiv.org/pdf/2310.07820.pdf)
- [Mistral 모델 정보](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
