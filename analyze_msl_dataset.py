#!/usr/bin/env python3
"""
MSL 데이터셋 분석 스크립트
- 서로 다른 MSL 신호들의 시간대 및 길이 비교
- 데이터 일관성 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orion.data import load_signal, load_anomalies
from orion.benchmark import _load_signal
import warnings
warnings.simplefilter('ignore')

def analyze_msl_signals():
    """MSL 데이터셋의 여러 신호들을 분석"""
    
    # MSL 데이터셋의 신호들 (paper-benchmark.csv에서 확인된 MSL 신호들)
    msl_signals = [
        'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7',
        'S-2', 'P-10', 'P-11', 'P-14', 'P-15', 
        'T-4', 'T-5', 'T-8', 'T-9', 'T-12', 'T-13',
        'F-4', 'F-5', 'F-7', 'F-8',
        'C-1', 'C-2', 'D-14', 'D-15', 'D-16'
    ]
    
    print("=== 🔍 MSL 데이터셋 신호 분석 ===\n")
    
    results = []
    
    # 각 신호별로 분석
    for signal_name in msl_signals:
        try:
            print(f"📊 분석 중: {signal_name}")
            
            # test_split=False로 전체 데이터 로딩
            _, data_full = _load_signal(signal_name, test_split=False)
            # test_split=True로 테스트 데이터만 로딩  
            _, data_test = _load_signal(signal_name, test_split=True)
            
            # 기본 정보
            full_length = len(data_full)
            test_length = len(data_test)
            
            # 시간 정보 (인덱스 확인)
            full_start_time = data_full.index[0] if not data_full.empty else None
            full_end_time = data_full.index[-1] if not data_full.empty else None
            test_start_time = data_test.index[0] if not data_test.empty else None
            test_end_time = data_test.index[-1] if not data_test.empty else None
            
            # 값 범위
            full_min = data_full['value'].min() if not data_full.empty else None
            full_max = data_full['value'].max() if not data_full.empty else None
            test_min = data_test['value'].min() if not data_test.empty else None
            test_max = data_test['value'].max() if not data_test.empty else None
            
            # 이상치 정보
            try:
                anomalies = load_anomalies(signal_name)
                num_anomalies = len(anomalies)
            except:
                num_anomalies = 0
            
            result = {
                'signal': signal_name,
                'full_length': full_length,
                'test_length': test_length,
                'test_ratio': test_length / full_length if full_length > 0 else 0,
                'full_start': full_start_time,
                'full_end': full_end_time,
                'test_start': test_start_time,
                'test_end': test_end_time,
                'full_range': (full_min, full_max),
                'test_range': (test_min, test_max),
                'num_anomalies': num_anomalies
            }
            
            results.append(result)
            
            print(f"   • 전체 길이: {full_length:,}")
            print(f"   • 테스트 길이: {test_length:,}")
            print(f"   • 테스트 비율: {result['test_ratio']:.2%}")
            print(f"   • 전체 시간: {full_start_time} ~ {full_end_time}")
            print(f"   • 테스트 시간: {test_start_time} ~ {test_end_time}")
            print(f"   • 이상치 수: {num_anomalies}")
            print()
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            continue
    
    return results

def compare_time_periods(results):
    """시간대 비교 분석"""
    print("\n=== ⏰ 시간대 비교 분석 ===")
    
    # 시간 정보가 있는 결과만 필터링
    time_results = [r for r in results if r['full_start'] is not None]
    
    if not time_results:
        print("⚠️ 시간 정보를 가진 신호가 없습니다.")
        return
    
    # 전체 데이터의 시간 범위
    all_starts = [r['full_start'] for r in time_results]
    all_ends = [r['full_end'] for r in time_results]
    
    earliest_start = min(all_starts)
    latest_end = max(all_ends)
    
    print(f"📅 MSL 데이터셋 전체 시간 범위:")
    print(f"   시작: {earliest_start}")
    print(f"   종료: {latest_end}")
    print(f"   총 기간: {latest_end - earliest_start}")
    print()
    
    # 시간대별 그룹화
    print("📊 신호별 시간 범위:")
    for result in time_results:
        duration = result['full_end'] - result['full_start']
        print(f"   {result['signal']:>4}: {result['full_start']} ~ {result['full_end']} (기간: {duration})")
    
    # 겹치는 시간대 확인
    print(f"\n🔍 시간대 겹침 분석:")
    overlapping_pairs = []
    for i, r1 in enumerate(time_results):
        for j, r2 in enumerate(time_results[i+1:], i+1):
            # 겹치는 시간 확인
            overlap_start = max(r1['full_start'], r2['full_start'])
            overlap_end = min(r1['full_end'], r2['full_end'])
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                overlapping_pairs.append((r1['signal'], r2['signal'], overlap_duration))
    
    if overlapping_pairs:
        print(f"   겹치는 신호 쌍: {len(overlapping_pairs)}개")
        for pair in overlapping_pairs[:10]:  # 상위 10개만 표시
            print(f"     {pair[0]} ↔ {pair[1]}: {pair[2]}")
    else:
        print("   겹치는 시간대 없음 - 모든 신호가 서로 다른 시간대")

def analyze_length_distribution(results):
    """길이 분포 분석"""
    print("\n=== 📏 길이 분포 분석 ===")
    
    lengths = [r['full_length'] for r in results if r['full_length'] > 0]
    test_lengths = [r['test_length'] for r in results if r['test_length'] > 0]
    test_ratios = [r['test_ratio'] for r in results if r['test_ratio'] > 0]
    
    print(f"📊 전체 데이터 길이 통계:")
    print(f"   평균: {np.mean(lengths):,.0f}")
    print(f"   중앙값: {np.median(lengths):,.0f}")
    print(f"   최소: {np.min(lengths):,}")
    print(f"   최대: {np.max(lengths):,}")
    print(f"   표준편차: {np.std(lengths):,.0f}")
    
    print(f"\n📊 테스트 데이터 길이 통계:")
    print(f"   평균: {np.mean(test_lengths):,.0f}")
    print(f"   중앙값: {np.median(test_lengths):,.0f}")
    print(f"   최소: {np.min(test_lengths):,}")
    print(f"   최대: {np.max(test_lengths):,}")
    
    print(f"\n📊 테스트 비율 통계:")
    print(f"   평균: {np.mean(test_ratios):.2%}")
    print(f"   중앙값: {np.median(test_ratios):.2%}")
    print(f"   최소: {np.min(test_ratios):.2%}")
    print(f"   최대: {np.max(test_ratios):.2%}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 전체 길이 분포
    axes[0,0].hist(lengths, bins=20, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('전체 데이터 길이 분포')
    axes[0,0].set_xlabel('길이')
    axes[0,0].set_ylabel('빈도')
    
    # 테스트 길이 분포
    axes[0,1].hist(test_lengths, bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].set_title('테스트 데이터 길이 분포')
    axes[0,1].set_xlabel('길이')
    axes[0,1].set_ylabel('빈도')
    
    # 테스트 비율 분포
    axes[1,0].hist(test_ratios, bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('테스트 비율 분포')
    axes[1,0].set_xlabel('비율')
    axes[1,0].set_ylabel('빈도')
    
    # 전체 vs 테스트 길이 산점도
    axes[1,1].scatter(lengths, test_lengths, alpha=0.7)
    axes[1,1].set_title('전체 길이 vs 테스트 길이')
    axes[1,1].set_xlabel('전체 길이')
    axes[1,1].set_ylabel('테스트 길이')
    
    # 대각선 추가 (1:1 비율)
    max_val = max(max(lengths), max(test_lengths))
    axes[1,1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 비율')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def sample_signals_comparison():
    """몇 개 신호의 상세 비교"""
    print("\n=== 🔍 샘플 신호 상세 비교 ===")
    
    sample_signals = ['M-1', 'M-2', 'M-6', 'S-2']  # 대표 신호들
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, signal_name in enumerate(sample_signals):
        try:
            _, data_full = _load_signal(signal_name, test_split=False)
            _, data_test = _load_signal(signal_name, test_split=True)
            
            # 전체 데이터 플롯
            axes[i].plot(data_full.index, data_full['value'], alpha=0.7, label='전체 데이터')
            # 테스트 데이터 강조
            axes[i].plot(data_test.index, data_test['value'], alpha=0.9, label='테스트 데이터', linewidth=2)
            
            axes[i].set_title(f'{signal_name} (전체: {len(data_full):,}, 테스트: {len(data_test):,})')
            axes[i].set_xlabel('시간')
            axes[i].set_ylabel('값')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'에러: {e}', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{signal_name} - 로딩 실패')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 MSL 데이터셋 분석 시작...\n")
    
    # 1. 신호별 분석
    results = analyze_msl_signals()
    
    # 2. 시간대 비교
    compare_time_periods(results)
    
    # 3. 길이 분포 분석
    analyze_length_distribution(results)
    
    # 4. 샘플 신호 비교
    sample_signals_comparison()
    
    print("\n✅ MSL 데이터셋 분석 완료!")
