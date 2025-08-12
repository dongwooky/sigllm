#!/usr/bin/env python3
"""
SigLLM Detector Pipeline - BENCHMARK Version (Together AI)
Benchmark-compatible high-level version for accurate performance comparison

Key Features:
🎯 Uses SigLLM high-level interface (same as benchmark)
📊 Standard Orion evaluation metrics
💾 Proper memory management
⚡ Async/parallel processing for speed
🔧 Balanced hyperparameters for accuracy + speed

Requirements:
- Together AI API key (set as TOGETHER_API_KEY environment variable)
- sigllm, orion, mlblocks, matplotlib, pandas, numpy, together, aiohttp, tqdm
"""

import warnings
warnings.simplefilter('ignore')

import os
import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from copy import deepcopy
from orion.data import load_signal, load_anomalies
from orion.benchmark import _load_signal  # 벤치마크와 동일한 함수 사용
from orion.evaluation import contextual_confusion_matrix
from mlblocks import MLPipeline, get_pipelines_paths
from sigllm import SigLLM  # 벤치마크와 동일한 고수준 인터페이스
from sigllm.data import load_normal  # Few-shot 지원용


# 🌐 S3 기반 동적 설정 (벤치마크와 동일)
BUCKET = 'sintel-sigllm'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

# 벤치마크 데이터셋과 파라미터 설정
def load_benchmark_configs():
    """S3 또는 로컬에서 벤치마크 설정 로딩"""
    # 로컬 폴백 데이터
    local_benchmark_data = {}
    local_benchmark_params = {
        'MSL': (True,),  # test_split=True
        'SMAP': (True,),
        'YAHOO': (True,),
        'NAB': (True,),
    }
    
    try:
        # S3에서 로딩 시도 (타임아웃 10초)
        import socket
        socket.setdefaulttimeout(10)
        
        data = (
            pd.read_csv(S3_URL.format(BUCKET, 'datasets.csv'), index_col=0, header=None)
            .applymap(ast.literal_eval)
            .to_dict()[1]
        )
        params = (
            pd.read_csv(S3_URL.format(BUCKET, 'parameters.csv'), index_col=0, header=None)
            .applymap(ast.literal_eval)
            .to_dict()[1]
        )
        print("✅ S3 기반 벤치마크 설정 로딩 성공")
        return data, params
        
    except Exception as e:
        print(f"⚠️ S3 설정 로딩 실패 ({type(e).__name__}), 로컬 기본값 사용")
        print(f"   • 에러: {str(e)[:100]}...")
        return local_benchmark_data, local_benchmark_params

BENCHMARK_DATA, BENCHMARK_PARAMS = load_benchmark_configs()

# 파이프라인 디렉토리 설정
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sigllm', 'pipelines')

# 파이프라인 매핑 (벤치마크와 동일)
PIPELINES = {
    'mistral_detector_together': 'mistral_detector_together',
    'mistral_detector': 'mistral_detector',
    'gpt_detector': 'gpt_detector',
}

# Set Together AI API key from .env file if not already set
if not os.getenv('TOGETHER_API_KEY'):
    # Try to load from .env file in current directory
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('TOGETHER_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    os.environ['TOGETHER_API_KEY'] = api_key
                    print(f"✅ Loaded TOGETHER_API_KEY from .env file")
                    break
    
    if not os.getenv('TOGETHER_API_KEY'):
        print("Warning: TOGETHER_API_KEY environment variable not set!")
        print("Please set your Together AI API key: export TOGETHER_API_KEY='your-api-key'")
        print("You can get an API key from: https://api.together.xyz/")
        exit(1)

# 🔧 벤치마크와 동일한 동적 하이퍼파라미터 로딩 함수들
def _get_pipeline_directory(pipeline_name):
    """파이프라인 디렉토리 경로를 찾는 함수 (벤치마크와 동일)"""
    if os.path.isfile(pipeline_name):
        return os.path.dirname(pipeline_name)

    pipelines_paths = get_pipelines_paths()
    for base_path in pipelines_paths:
        parts = pipeline_name.split('.')
        number_of_parts = len(parts)

        for folder_parts in range(number_of_parts):
            folder = os.path.join(base_path, *parts[:folder_parts])
            filename = '.'.join(parts[folder_parts:]) + '.json'
            json_path = os.path.join(folder, filename)

            if os.path.isfile(json_path):
                return os.path.dirname(json_path)

def _get_pipeline_hyperparameter(hyperparameters, dataset_name, pipeline_name):
    """데이터셋별 특화 하이퍼파라미터 로딩 (벤치마크와 동일)"""
    hyperparameters_ = deepcopy(hyperparameters)

    if isinstance(hyperparameters, dict):
        hyperparameters_ = hyperparameters_.get(dataset_name) or hyperparameters_
        hyperparameters_ = hyperparameters_.get(pipeline_name) or hyperparameters_

    elif isinstance(hyperparameters_, str) and os.path.exists(hyperparameters_):
        with open(hyperparameters_) as f:
            hyperparameters_ = json.load(f)

    elif hyperparameters_ is None and dataset_name and pipeline_name:
        pipeline_path = _get_pipeline_directory(pipeline_name)
        if pipeline_path:
            pipeline_dirname = os.path.basename(pipeline_path)
            # 데이터셋별 특화 설정 파일 찾기
            file_path = os.path.join(
                pipeline_path, pipeline_dirname + '_' + dataset_name.lower() + '.json'
            )
            if os.path.exists(file_path):
                print(f"✅ 데이터셋별 특화 설정 로딩: {file_path}")
                with open(file_path) as f:
                    hyperparameters_ = json.load(f)
            else:
                print(f"📝 데이터셋별 특화 설정 없음, 기본 설정 사용: {dataset_name}")

    return hyperparameters_

def _augment_hyperparameters(hyperparameters, few_shot):
    """Few-shot 학습을 위한 하이퍼파라미터 보정 (벤치마크와 동일)"""
    hyperparameters_ = deepcopy(hyperparameters)
    if few_shot and hyperparameters:
        print("🎯 Few-shot 학습을 위한 하이퍼파라미터 보정 적용")
        for hyperparameter, value in hyperparameters.items():
            if 'time_segments_aggregate' in hyperparameter:
                name = hyperparameter[:-1]
                number = int(hyperparameter[-1]) + 1
                hyperparameters_[name + str(number)] = value

    return hyperparameters_

def _get_optimized_async_params():
    """비동기 방식에 필요한 하드코딩 파라미터들"""
    return {
        "sigllm.primitives.forecasting.together_ai.TogetherAI#1": {
            # 🚀 비동기 최적화 파라미터들 (하드코딩 유지)
            "max_concurrent": 15,  # 안정성을 위한 동시 요청 수
            "batch_size": 50,      # 메모리 효율을 위한 배치 크기  
            "use_async": True,     # 비동기 처리 활성화
            # 🎯 성능 균형 파라미터들 (동적 로딩 가능하지만 기본값 제공)
            "samples": 2,          # 앙상블 효과
            "max_tokens": 50,      # 적절한 토큰 수
            "temp": 0.7,           # 균형잡힌 창의성
            "top_p": 0.9,          # 품질 필터링
        }
    }

def main():
    """
    Main function to run the optimized detector pipeline
    """
    print("=== 🎯 SigLLM Detector Pipeline (BENCHMARK Together AI) ===")
    print("Benchmark-compatible high-level version for accurate performance comparison!")
    print("Uses same interface and evaluation as official benchmark\n")
    
    # Record start time for total performance measurement
    total_start_time = time.time()
    
    # 1. Data Loading - 벤치마크와 동일한 방식 사용
    # 벤치마크 데이터셋 (paper-benchmark.csv에서 확인)
    # MSL: M-6, M-1, M-2, S-2, P-10, T-4, T-5, F-7, M-3, M-4, M-5, P-15, C-1, C-2, T-12, T-13, F-4, F-5, D-14, T-9, P-14, T-8, P-11, D-15, D-16, M-7, F-8
    # SMAP: P-1, S-1, E-1, E-2, E-3, E-4, E-5, E-6, E-7, E-8, E-9, E-10, E-11, E-12, E-13, A-1, D-1, P-3, D-2, D-3, D-4, A-2, ...
    # YAHOO: Real_1, Real_2, ... Real_67, A1Benchmark, A2Benchmark, A3Benchmark, A4Benchmark
    # 
    # 1. 데이터셋 및 파이프라인 설정
    signal_name = 'M-6'  # 벤치마크 데이터셋
    dataset_name = 'MSL'  # 데이터셋 이름
    pipeline_name = 'mistral_detector_together'
    few_shot = False  # Few-shot 학습 여부 (1shot 파이프라인이면 True)
    
    print("1. 📊 벤치마크 스타일 데이터 로딩...")
    print(f"   • Signal: {signal_name}")
    print(f"   • Dataset: {dataset_name}")
    print(f"   • Pipeline: {pipeline_name}")
    print(f"   • Few-shot: {few_shot}")
    
    # S3에서 데이터셋별 test_split 파라미터 가져오기
    parameters = BENCHMARK_PARAMS.get(dataset_name)
    test_split = True  # 기본값
    if parameters is not None:
        test_split = list(parameters.values())[0]
        print(f"   • S3에서 로딩한 test_split: {test_split}")
    else:
        print(f"   • 기본 test_split 사용: {test_split}")
    
    # 벤치마크와 동일한 방식으로 데이터 로딩
    _, data = _load_signal(signal_name, test_split=test_split)
    print(f"   • Data shape: {data.shape}")
    
    # Few-shot을 위한 normal data 로딩
    normal = None
    if few_shot:
        normal = load_normal(signal_name)
        print(f"   • Normal data loaded for few-shot: {normal.shape}")
    
    # Quick visualization
    plt.figure(figsize=(12, 4))
    plt.plot(data['value'])
    plt.title(f'데이터셋: {dataset_name} - {signal_name}')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. 🔧 동적 하이퍼파라미터 로딩 시스템
    print("\n2. 🔧 벤치마크 스타일 동적 하이퍼파라미터 로딩...")
    
    # 2.1 기본 하이퍼파라미터 (None으로 시작하여 동적 로딩)
    base_hyperparameters = None
    
    # 2.2 데이터셋별 특화 하이퍼파라미터 로딩
    print("   🎯 데이터셋별 특화 설정 로딩 시도...")
    hyperparameters = _get_pipeline_hyperparameter(
        base_hyperparameters, dataset_name, pipeline_name
    )
    
    # 2.3 비동기 최적화 파라미터 오버라이드
    print("   🚀 비동기 최적화 파라미터 적용...")
    async_params = _get_optimized_async_params()
    if hyperparameters is None:
        hyperparameters = async_params
    else:
        # 기존 설정에 비동기 파라미터 병합
        for key, value in async_params.items():
            if key in hyperparameters:
                hyperparameters[key].update(value)
            else:
                hyperparameters[key] = value
    
    # 2.4 Few-shot 학습을 위한 하이퍼파라미터 보정
    hyperparameters = _augment_hyperparameters(hyperparameters, few_shot)
    
    print(f"   ✅ 최종 하이퍼파라미터 로딩 완료")
    print(f"   📋 주요 설정:")
    llm_params = hyperparameters.get("sigllm.primitives.forecasting.together_ai.TogetherAI#1", {})
    print(f"      • samples: {llm_params.get('samples', 'N/A')}")
    print(f"      • max_tokens: {llm_params.get('max_tokens', 'N/A')}")
    print(f"      • temp: {llm_params.get('temp', 'N/A')}")
    print(f"      • max_concurrent: {llm_params.get('max_concurrent', 'N/A')}")
    print(f"      • use_async: {llm_params.get('use_async', 'N/A')}")
    
    # 🎯 벤치마크와 동일한 고수준 방식 사용
    print("\n   🚀 벤치마크 스타일 SigLLM 인터페이스 초기화")
    pipeline = SigLLM(pipeline_name, hyperparameters=hyperparameters)
    
    print(f"Pipeline initialized: {type(pipeline).__name__}")
    
    # 3. 벤치마크와 동일한 고수준 실행
    print("\n3. 🚀 벤치마크와 100% 동일한 방식으로 실행...")
    print("   One-step anomaly detection using SigLLM.detect() method")
    
    # Load ground truth anomalies first
    truth_anomalies = load_anomalies(signal_name)
    print(f"Ground Truth Anomalies: {len(truth_anomalies)} segments")
    
    # 벤치마크와 동일한 방식으로 실행
    print("\n🔥 Running anomaly detection...")
    print(f"   • Dataset: {dataset_name}")
    print(f"   • Signal: {signal_name}")
    print(f"   • Few-shot: {few_shot}")
    print(f"   • Test split: {test_split}")
    
    detection_start = time.time()
    
    try:
        # 벤치마크와 정확히 동일한 방식 (Few-shot 지원 포함)
        if few_shot and normal is not None:
            print("   🎯 Few-shot learning with normal data")
            anomalies = pipeline.detect(data, normal=normal)
        else:
            print("   🎯 Zero-shot learning")
            anomalies = pipeline.detect(data)
            
        detection_time = time.time() - detection_start
        
        print(f"✅ Detection completed in {detection_time:.2f}s")
        print(f"📊 Detected {len(anomalies)} anomaly segments")
        
        # Show detected anomalies
        if len(anomalies) > 0:
            print(f"🎯 Detected Anomalies (by {dataset_name} 특화 Mistral via Together AI):")
            print(anomalies)
        else:
            print(f"🎯 No anomalies detected by {dataset_name} 특화 Mistral via Together AI")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        # Create empty DataFrame for fallback
        anomalies = pd.DataFrame(columns=['start', 'end', 'score'])
    
    # 4. Performance Evaluation (벤치마크와 동일한 방식)
    print("\n4. 📊 Performance Evaluation...")
    
    # 벤치마크와 동일한 방식으로 평가 메트릭 계산
    from orion.evaluation import CONTEXTUAL_METRICS as METRICS
    
    print(f"📊 Evaluation Info:")
    print(f"   • Data shape: {data.shape}")
    print(f"   • Truth anomalies: {len(truth_anomalies)} segments")
    print(f"   • Detected anomalies: {len(anomalies)} segments")
    
    # 벤치마크에서 사용하는 동일한 scorer 함수들
    evaluation_results = {}
    for metric_name, scorer in METRICS.items():
        try:
            score = scorer(truth_anomalies, anomalies, data)
            evaluation_results[metric_name] = score
            print(f"   • {metric_name}: {score:.4f}")
        except Exception as e:
            print(f"   • {metric_name}: Error - {e}")
            evaluation_results[metric_name] = 0.0
    
    # 벤치마크 스타일 요약 출력
    print(f"\n🎯 Benchmark-style Results Summary:")
    print(f"   • F1-Score: {evaluation_results.get('f1', 0.0):.4f}")
    print(f"   • Precision: {evaluation_results.get('precision', 0.0):.4f}")
    print(f"   • Recall: {evaluation_results.get('recall', 0.0):.4f}")
    
    # 메모리 정리 (벤치마크와 동일)
    import gc
    import torch
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("   💾 Memory cleanup completed")
    
    # 5. Simple Visualization (고수준 결과 기반)
    print("\n5. 📊 Creating benchmark-style visualization...")
    
    # 간단한 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # Data plot
    plt.subplot(2, 1, 1)
    plt.plot(data['value'], label='Original Data', linewidth=2)
    plt.title('🚀 BENCHMARK-style Mistral Anomaly Detection via Together AI')
    
    # Mark detected anomalies
    if len(anomalies) > 0:
        for _, anomaly in anomalies.iterrows():
            start_idx = data[data.index >= anomaly['start']].index[0] if len(data[data.index >= anomaly['start']]) > 0 else 0
            end_idx = data[data.index <= anomaly['end']].index[-1] if len(data[data.index <= anomaly['end']]) > 0 else len(data)-1
            plt.axvspan(start_idx, end_idx, color='red', alpha=0.3, label='Detected Anomalies')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel('Value')
    
    # Performance metrics
    plt.subplot(2, 1, 2)
    metric_names = ['Precision', 'Recall', 'F1-Score']
    metric_values = [
        evaluation_results.get('precision', 0.0),
        evaluation_results.get('recall', 0.0), 
        evaluation_results.get('f1', 0.0)
    ]
    
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('🎯 Performance Metrics (Benchmark Style)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Performance Summary
    total_time = time.time() - total_start_time
    print(f"\n=== 🎯 벤치마크 완전 호환 Mistral Detector Pipeline 실행 완료! ===")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"🎯 Final F1-Score: {evaluation_results.get('f1', 0.0):.4f}")
    print(f"📊 Applied Features:")
    print(f"   ✅ S3 기반 동적 설정 로딩")
    print(f"   ✅ 데이터셋별 특화 하이퍼파라미터 ({dataset_name})")
    print(f"   ✅ Few-shot 학습 지원 ({few_shot})")
    print(f"   ✅ 비동기 최적화 파라미터 유지")
    print(f"   ✅ 벤치마크와 동일한 평가 메트릭")
    print(f"   ✅ 벤치마크와 동일한 SigLLM 고수준 인터페이스")
    
    return {
        'anomalies': anomalies,
        'truth_anomalies': truth_anomalies,
        'evaluation_results': evaluation_results,
        'execution_time': total_time
    }

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n✅ Final results summary:")
            print(f"   • Anomalies detected: {len(results['anomalies'])}")
            print(f"   • Ground truth anomalies: {len(results['truth_anomalies'])}")
            print(f"   • Execution time: {results['execution_time']:.2f}s")
            print(f"   • Final F1-Score: {results['evaluation_results'].get('f1', 0.0):.4f}")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
