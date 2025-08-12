#!/usr/bin/env python3
"""
MSL 데이터셋 Multivariate 분석 가능성 탐색
- 시간 동기화 및 리샘플링 방법
- 공통 시간 구간 찾기
- Multivariate time series 변환 가능성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orion.benchmark import _load_signal
import warnings
warnings.simplefilter('ignore')

def analyze_multivariate_potential():
    """MSL 데이터셋의 multivariate 분석 가능성 탐색"""
    
    print("=== 🔬 MSL Multivariate 분석 가능성 탐색 ===\n")
    
    # 대표 센서들 선택 (서로 다른 타입)
    target_sensors = {
        'M-1': 'Main_Sensor_1',
        'P-10': 'Power_System', 
        'T-4': 'Temperature',
        'F-5': 'Function',
        'C-1': 'Control',
        'D-14': 'Diagnostics'
    }
    
    sensor_data = {}
    
    # 1. 각 센서 데이터 로딩
    print("1. 📊 센서 데이터 로딩...")
    for sensor_id, sensor_name in target_sensors.items():
        try:
            _, data = _load_signal(sensor_id, test_split=False)
            sensor_data[sensor_id] = {
                'name': sensor_name,
                'data': data,
                'length': len(data),
                'start': data.index[0],
                'end': data.index[-1],
                'freq': estimate_frequency(data)
            }
            print(f"   ✅ {sensor_id} ({sensor_name}): {len(data):,} points, 주기: {sensor_data[sensor_id]['freq']:.2f}")
        except Exception as e:
            print(f"   ❌ {sensor_id}: {e}")
    
    return sensor_data

def estimate_frequency(data):
    """데이터의 샘플링 주기 추정"""
    if len(data) < 2:
        return 0
    
    # 시간 인덱스 차이들의 평균
    time_diffs = np.diff(data.index[:100])  # 처음 100개 포인트로 추정
    return np.mean(time_diffs) if len(time_diffs) > 0 else 1.0

def find_common_timerange(sensor_data):
    """공통 시간 구간 찾기"""
    print("\n2. ⏰ 공통 시간 구간 분석...")
    
    # 모든 센서의 시작/끝 시간
    all_starts = [info['start'] for info in sensor_data.values()]
    all_ends = [info['end'] for info in sensor_data.values()]
    
    # 공통 구간 = 가장 늦은 시작 ~ 가장 이른 끝
    common_start = max(all_starts)
    common_end = min(all_ends)
    common_length = common_end - common_start + 1
    
    print(f"   📅 공통 시간 구간:")
    print(f"      시작: {common_start}")
    print(f"      끝: {common_end}")
    print(f"      길이: {common_length:,} time units")
    
    # 각 센서의 공통 구간 커버리지
    print(f"\n   📊 센서별 공통 구간 커버리지:")
    for sensor_id, info in sensor_data.items():
        coverage = common_length / info['length'] * 100
        available = common_end <= info['end'] and common_start >= info['start']
        status = "✅" if available else "❌"
        print(f"      {status} {sensor_id}: {coverage:.1f}% ({info['length']:,} -> {common_length:,})")
    
    return common_start, common_end, common_length

def create_multivariate_dataset(sensor_data, common_start, common_end):
    """Multivariate 데이터셋 생성"""
    print("\n3. 🔧 Multivariate 데이터셋 생성...")
    
    # 공통 시간 인덱스 생성
    time_index = range(common_start, common_end + 1)
    
    # 각 센서별로 공통 구간 데이터 추출 및 리샘플링
    multivariate_data = pd.DataFrame(index=time_index)
    
    successful_sensors = []
    
    for sensor_id, info in sensor_data.items():
        try:
            # 공통 구간에 해당하는 데이터 추출
            sensor_df = info['data']
            
            # 공통 구간 필터링
            common_data = sensor_df[
                (sensor_df.index >= common_start) & 
                (sensor_df.index <= common_end)
            ]
            
            if len(common_data) > 0:
                # 리샘플링/보간으로 공통 인덱스에 맞춤
                resampled = common_data.reindex(time_index, method='nearest')
                
                # NaN 값 처리 (선형 보간)
                resampled = resampled.interpolate(method='linear')
                
                # 여전히 NaN이 있으면 forward/backward fill
                resampled = resampled.fillna(method='ffill').fillna(method='bfill')
                
                # multivariate 데이터에 추가
                column_name = f"{sensor_id}_{info['name']}"
                multivariate_data[column_name] = resampled['value']
                successful_sensors.append(sensor_id)
                
                print(f"   ✅ {sensor_id}: {len(common_data):,} -> {len(resampled):,} points")
            else:
                print(f"   ❌ {sensor_id}: 공통 구간에 데이터 없음")
                
        except Exception as e:
            print(f"   ❌ {sensor_id}: {e}")
    
    print(f"\n   🎯 성공한 센서 수: {len(successful_sensors)}/{len(sensor_data)}")
    print(f"   📊 최종 multivariate 데이터: {multivariate_data.shape}")
    
    return multivariate_data, successful_sensors

def analyze_correlations(multivariate_data):
    """센서 간 상관관계 분석"""
    print("\n4. 🔗 센서 간 상관관계 분석...")
    
    # 상관관계 행렬 계산
    correlation_matrix = multivariate_data.corr()
    
    print("   📊 센서 간 상관계수 (상위 10개 쌍):")
    
    # 상관계수 추출 (대각선 제외)
    correlations = []
    for i, col1 in enumerate(correlation_matrix.columns):
        for j, col2 in enumerate(correlation_matrix.columns):
            if i < j:  # 중복 제거
                corr_val = correlation_matrix.loc[col1, col2]
                if not np.isnan(corr_val):
                    correlations.append((col1, col2, abs(corr_val), corr_val))
    
    # 절댓값 기준으로 정렬
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    for i, (sensor1, sensor2, abs_corr, corr) in enumerate(correlations[:10]):
        direction = "+" if corr > 0 else "-"
        print(f"      {i+1:2d}. {sensor1.split('_')[0]} ↔ {sensor2.split('_')[0]}: {direction}{abs_corr:.3f}")
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 상관관계 히트맵
    plt.subplot(2, 2, 1)
    try:
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    except ImportError:
        # seaborn 없이 matplotlib으로 히트맵
        im = plt.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation')
        plt.xticks(range(len(correlation_matrix.columns)), 
                   [col.split('_')[0] for col in correlation_matrix.columns], rotation=45)
        plt.yticks(range(len(correlation_matrix.index)), 
                   [col.split('_')[0] for col in correlation_matrix.index])
    plt.title('센서 간 상관관계 행렬')
    
    # 시계열 플롯
    plt.subplot(2, 2, 2)
    sample_length = min(1000, len(multivariate_data))
    for col in multivariate_data.columns:
        plt.plot(multivariate_data[col][:sample_length], alpha=0.7, label=col.split('_')[0])
    plt.title(f'Multivariate 시계열 (첫 {sample_length} points)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 정규화된 시계열
    plt.subplot(2, 2, 3)
    normalized_data = (multivariate_data - multivariate_data.mean()) / multivariate_data.std()
    for col in normalized_data.columns:
        plt.plot(normalized_data[col][:sample_length], alpha=0.7, label=col.split('_')[0])
    plt.title(f'정규화된 Multivariate 시계열')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 분산 기여도
    plt.subplot(2, 2, 4)
    variances = multivariate_data.var().sort_values(ascending=True)
    plt.barh(range(len(variances)), variances.values)
    plt.yticks(range(len(variances)), [name.split('_')[0] for name in variances.index])
    plt.title('센서별 분산 크기')
    plt.xlabel('Variance')
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def multivariate_anomaly_potential(multivariate_data, successful_sensors):
    """Multivariate 이상 탐지 활용 가능성"""
    print("\n5. 🎯 Multivariate 이상 탐지 활용 방안...")
    
    print(f"   📊 활용 가능한 방법들:")
    print(f"      1. 🔗 상관관계 기반 이상 탐지")
    print(f"         - 센서 간 비정상적 상관관계 변화 감지")
    print(f"         - 정상 상관 패턴에서 벗어나는 구간 탐지")
    
    print(f"      2. 📈 주성분 분석 (PCA) 기반")
    print(f"         - 다차원 센서 데이터의 주요 패턴 추출")
    print(f"         - 주성분 공간에서의 이상치 탐지")
    
    print(f"      3. 🧠 오토인코더 (Autoencoder)")
    print(f"         - 정상 패턴 학습 후 재구성 오차로 이상 탐지")
    print(f"         - 센서 간 복합적 패턴 모델링")
    
    print(f"      4. 🔄 그랜저 인과관계 (Granger Causality)")
    print(f"         - 센서 간 시간적 영향관계 모델링")
    print(f"         - 비정상적 인과관계 변화 감지")
    
    print(f"      5. 🌐 Dynamic Time Warping (DTW)")
    print(f"         - 센서 간 시간적 동기화 및 패턴 비교")
    print(f"         - 시간 지연을 고려한 이상 탐지")
    
    # 실제 활용 예시 (간단한 PCA)
    print(f"\n   🔬 간단한 PCA 기반 이상 탐지 예시:")
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # 데이터 정규화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(multivariate_data.fillna(0))
        
        # PCA 적용
        pca = PCA(n_components=min(3, scaled_data.shape[1]))
        pca_result = pca.fit_transform(scaled_data)
        
        print(f"      📊 PCA 결과:")
        print(f"         - 주성분 수: {pca.n_components_}")
        print(f"         - 설명 분산비: {pca.explained_variance_ratio_}")
        print(f"         - 누적 설명력: {np.cumsum(pca.explained_variance_ratio_)}")
        
        # 재구성 오차 계산
        reconstructed = pca.inverse_transform(pca_result)
        reconstruction_error = np.mean((scaled_data - reconstructed) ** 2, axis=1)
        
        # 이상치 임계값 (95번째 백분위수)
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold
        
        print(f"         - 탐지된 이상치: {np.sum(anomalies)} / {len(anomalies)} ({np.mean(anomalies)*100:.1f}%)")
        
    except ImportError:
        print(f"      ⚠️ scikit-learn이 필요합니다 (pip install scikit-learn)")
    except Exception as e:
        print(f"      ❌ PCA 분석 실패: {e}")

def resampling_strategies():
    """리샘플링 전략들"""
    print("\n6. 🔄 리샘플링 전략 및 해결책...")
    
    strategies = {
        "1. 🕐 시간 동기화": [
            "공통 시간 구간 추출",
            "선형/스플라인 보간으로 동일 간격 맞춤",
            "가장 느린 센서에 맞춰 다운샘플링"
        ],
        "2. 📊 통계적 정렬": [
            "슬라이딩 윈도우로 통계값 계산 (평균, 분산 등)",
            "시간 구간별 집계 (1분, 5분 단위 등)",
            "적응적 샘플링 (중요 이벤트 중심)"
        ],
        "3. 🤖 머신러닝 기반": [
            "센서별 임베딩 후 융합",
            "변분 오토인코더로 시간 동기화",
            "어텐션 메커니즘으로 중요 시점 동기화"
        ],
        "4. 🔗 센서 관계 모델링": [
            "센서 간 지연시간 학습",
            "물리적 관계 기반 모델링",
            "계층적 클러스터링으로 센서 그룹화"
        ]
    }
    
    for strategy, methods in strategies.items():
        print(f"   {strategy}:")
        for method in methods:
            print(f"      • {method}")
        print()

if __name__ == "__main__":
    print("🚀 MSL Multivariate 분석 시작...\n")
    
    try:
        # 1. 센서 데이터 로딩
        sensor_data = analyze_multivariate_potential()
        
        if len(sensor_data) < 2:
            print("❌ 충분한 센서 데이터가 없습니다.")
            exit(1)
        
        # 2. 공통 시간 구간 찾기
        common_start, common_end, common_length = find_common_timerange(sensor_data)
        
        if common_length <= 0:
            print("❌ 공통 시간 구간이 없습니다.")
            exit(1)
        
        # 3. Multivariate 데이터셋 생성
        multivariate_data, successful_sensors = create_multivariate_dataset(
            sensor_data, common_start, common_end
        )
        
        if len(successful_sensors) < 2:
            print("❌ Multivariate 분석에 충분한 센서가 없습니다.")
            exit(1)
        
        # 4. 상관관계 분석
        correlation_matrix = analyze_correlations(multivariate_data)
        
        # 5. Multivariate 이상 탐지 활용 방안
        multivariate_anomaly_potential(multivariate_data, successful_sensors)
        
        # 6. 리샘플링 전략
        resampling_strategies()
        
        print("\n✅ MSL Multivariate 분석 완료!")
        print(f"🎯 결론: MSL 데이터는 적절한 전처리를 통해 multivariate 분석 가능!")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
