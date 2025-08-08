# 🚀 SigLLM Together AI Performance Optimization

## 성능 최적화 개요

기존 Together AI 파이프라인의 속도 문제를 해결하기 위해 다음과 같은 최적화를 구현했습니다:

### 🎯 주요 성능 개선사항

| 최적화 기법 | 예상 성능 향상 | 설명 |
|------------|-------------|------|
| **비동기/병렬 처리** | 3-10x 빠름 | API 호출을 병렬로 처리 |
| **배치 처리** | 20-50% 향상 | 요청을 배치 단위로 그룹화 |
| **샘플 수 최적화** | 2x 빠름 | samples=2 → 1로 변경 |
| **토큰 수 최적화** | 10-20% 향상 | 더 효율적인 토큰 계산 |
| **연결 풀링** | 5-15% 향상 | HTTP 연결 재사용 |

## 📁 새로 생성된 파일들

### 1. 최적화된 Together AI 프리미티브
- `sigllm/primitives/forecasting/together_ai_optimized.py`
- 기존 `TogetherAI` 클래스의 고성능 버전
- 비동기 처리 및 배치 처리 지원

### 2. 최적화된 파이프라인 스크립트
- `tutorials/pipelines/mistral-detector-pipeline_together_optimized.py`
- 기존 스크립트의 최적화 버전
- 성능 모니터링 및 시간 측정 포함

### 3. 의존성 설치 스크립트
- `install_optimized_deps.py`
- 최적화에 필요한 추가 패키지 설치

## 🚀 사용 방법

### 1. 의존성 설치
```bash
cd /home/dongwook/github/sigllm
python install_optimized_deps.py
```

### 2. 최적화된 파이프라인 실행
```bash
# Together AI API 키 설정 (필요한 경우)
export TOGETHER_API_KEY='your-api-key'

# 최적화된 버전 실행
python tutorials/pipelines/mistral-detector-pipeline_together_optimized.py
```

## ⚙️ 최적화 파라미터

최적화된 버전에서 조정 가능한 주요 파라미터:

```python
hyperparameters = {
    "sigllm.primitives.forecasting.together_ai.TogetherAI#1": {
        "samples": 1,              # 샘플 수 (기존: 2)
        "max_tokens": 50,          # 최대 토큰 수
        "temp": 0.7,               # 온도 (기존: 1.0)
        "top_p": 0.9,              # top_p (기존: 1.0)
        "max_concurrent": 10,      # 🆕 동시 요청 수
        "batch_size": 20,          # 🆕 배치 크기
        "use_async": True,         # 🆕 비동기 처리 활성화
    }
}
```

## 📊 성능 비교

### 기존 버전 vs 최적화 버전

| 메트릭 | 기존 버전 | 최적화 버전 | 개선율 |
|-------|----------|------------|-------|
| **API 호출 방식** | 순차적 | 병렬/비동기 | 3-10x |
| **샘플 수** | 2 | 1 | 2x |
| **배치 처리** | ❌ | ✅ | 20-50% |
| **연결 재사용** | ❌ | ✅ | 5-15% |
| **메모리 사용량** | 보통 | 약간 증가 | -10% |

### 예상 실행 시간 (1000 시퀀스 기준)

```
기존 버전:    ~20-30분
최적화 버전:  ~3-5분
속도 향상:    5-8x 빠름
```

## 🔧 고급 튜닝

### 1. 동시 요청 수 조정
```python
"max_concurrent": 10,  # 기본값
# - 5-15: 안정적, 느림
# - 10-20: 균형잡힌 설정 (권장)
# - 20+: 빠르지만 API 제한 가능
```

### 2. 배치 크기 조정
```python
"batch_size": 20,  # 기본값
# - 10-20: 작은 데이터셋
# - 20-50: 중간 데이터셋 (권장)
# - 50+: 큰 데이터셋
```

### 3. 메모리 사용량 최적화
```python
"use_async": False,  # 메모리 부족시 False로 설정
# True: 빠르지만 더 많은 메모리 사용
# False: 느리지만 메모리 효율적
```

## 🐛 문제 해결

### 1. 메모리 부족 에러
```python
# 해결방법: 배치 크기와 동시 요청 수 줄이기
"max_concurrent": 5,
"batch_size": 10,
```

### 2. API 제한 에러
```python
# 해결방법: 동시 요청 수 줄이기
"max_concurrent": 5,
```

### 3. 타임아웃 에러
```python
# 해결방법: 비동기 모드 비활성화
"use_async": False,
```

## 📈 성능 모니터링

최적화된 버전은 다음과 같은 성능 메트릭을 실시간으로 표시합니다:

- 각 단계별 실행 시간
- API 호출 통계
- 평균 시퀀스당 처리 시간
- 예상 속도 향상 배수

## 🔮 향후 개선 방안

1. **GPU 캐싱**: 반복적인 패턴에 대한 로컬 캐싱
2. **압축**: 입력 데이터 압축으로 네트워크 대역폭 절약
3. **스마트 배치**: 유사한 길이의 시퀀스끼리 배치 그룹화
4. **적응형 파라미터**: 데이터 특성에 따른 자동 파라미터 조정

## 📞 문의사항

최적화 관련 문제나 추가 개선사항이 있으시면 이슈를 등록해주세요.
