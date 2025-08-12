#!/usr/bin/env python3
"""
MSL 데이터셋 출처 식별 스크립트
- NAB 포맷의 단일 채널 vs NASA MSL 멀티변량 데이터 확인
- 데이터 구조, 메타데이터, 컬럼 수 분석
"""

import pandas as pd
import numpy as np
from orion.benchmark import _load_signal
from orion.data import load_anomalies
import warnings
warnings.simplefilter('ignore')

def analyze_data_structure():
    """데이터 구조 및 출처 분석"""
    
    print("=== 🔍 MSL 데이터셋 출처 식별 ===\n")
    
    # 샘플 신호들
    sample_signals = ['M-1', 'P-10', 'T-4', 'S-2']
    
    for signal_name in sample_signals:
        print(f"📊 분석 중: {signal_name}")
        
        try:
            # 전체 데이터 로딩
            _, data = _load_signal(signal_name, test_split=False)
            
            print(f"   📋 기본 정보:")
            print(f"      • 데이터 타입: {type(data)}")
            print(f"      • 데이터 형태: {data.shape}")
            print(f"      • 컬럼 수: {len(data.columns)}")
            print(f"      • 컬럼 이름: {list(data.columns)}")
            print(f"      • 인덱스 타입: {type(data.index[0])}")
            print(f"      • 인덱스 범위: {data.index[0]} ~ {data.index[-1]}")
            
            # 데이터 샘플 확인
            print(f"   📈 데이터 샘플 (첫 5행):")
            print(data.head())
            
            # 값 범위 및 통계
            if 'value' in data.columns:
                print(f"   📊 값 통계:")
                print(f"      • 최솟값: {data['value'].min():.6f}")
                print(f"      • 최댓값: {data['value'].max():.6f}")
                print(f"      • 평균: {data['value'].mean():.6f}")
                print(f"      • 표준편차: {data['value'].std():.6f}")
                print(f"      • 결측값: {data['value'].isnull().sum()}")
            
            # 다른 컬럼이 있는지 확인
            other_cols = [col for col in data.columns if col != 'value']
            if other_cols:
                print(f"   🔍 추가 컬럼 발견: {other_cols}")
                for col in other_cols:
                    print(f"      • {col}: {data[col].dtype}, 샘플: {data[col].head(3).tolist()}")
            
            # 이상치 정보
            try:
                anomalies = load_anomalies(signal_name)
                print(f"   🚨 이상치 정보:")
                print(f"      • 이상치 수: {len(anomalies)}")
                if len(anomalies) > 0:
                    print(f"      • 이상치 구조: {anomalies.columns.tolist()}")
                    print(f"      • 첫 번째 이상치:")
                    print(anomalies.head(1))
            except:
                print(f"   ⚠️ 이상치 정보 없음")
            
            print(f"   {'='*50}")
            print()
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            print()

def check_multivariate_possibility():
    """실제 멀티변량 데이터 가능성 확인"""
    
    print("\n=== 🔬 멀티변량 데이터 가능성 검증 ===")
    
    # 모든 MSL 신호 확인
    all_signals = [
        'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7',
        'S-1', 'S-2', 
        'P-10', 'P-11', 'P-14', 'P-15', 
        'T-4', 'T-5', 'T-8', 'T-9', 'T-12', 'T-13',
        'F-4', 'F-5', 'F-7', 'F-8',
        'C-1', 'C-2', 'D-14', 'D-15', 'D-16'
    ]
    
    working_signals = []
    signal_info = {}
    
    print("📊 모든 MSL 신호 스캔...")
    
    for signal in all_signals:
        try:
            _, data = _load_signal(signal, test_split=False)
            working_signals.append(signal)
            
            signal_info[signal] = {
                'length': len(data),
                'columns': len(data.columns),
                'start': data.index[0],
                'end': data.index[-1],
                'value_range': (data['value'].min(), data['value'].max()),
                'has_other_cols': len(data.columns) > 1
            }
            
        except:
            pass
    
    print(f"✅ 사용 가능한 신호: {len(working_signals)}개")
    print(f"   신호 목록: {working_signals}")
    
    # 멀티변량 특성 분석
    print(f"\n🔍 멀티변량 특성 분석:")
    
    # 1. 컬럼 수 확인
    multi_column_signals = [s for s, info in signal_info.items() if info['has_other_cols']]
    print(f"   • 다중 컬럼 신호: {len(multi_column_signals)}개")
    if multi_column_signals:
        print(f"     {multi_column_signals}")
    
    # 2. 신호 이름 패턴 분석
    prefixes = {}
    for signal in working_signals:
        prefix = signal.split('-')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(signal)
    
    print(f"   • 신호 그룹 (접두사별):")
    for prefix, signals in prefixes.items():
        print(f"     {prefix}: {len(signals)}개 - {signals}")
    
    # 3. 시간 동기화 가능성
    time_overlaps = 0
    total_pairs = 0
    
    for i, s1 in enumerate(working_signals):
        for s2 in working_signals[i+1:]:
            total_pairs += 1
            info1, info2 = signal_info[s1], signal_info[s2]
            
            # 시간 겹침 확인
            overlap_start = max(info1['start'], info2['start'])
            overlap_end = min(info1['end'], info2['end'])
            
            if overlap_start <= overlap_end:
                time_overlaps += 1
    
    overlap_ratio = time_overlaps / total_pairs if total_pairs > 0 else 0
    print(f"   • 시간 겹침 비율: {overlap_ratio:.1%} ({time_overlaps}/{total_pairs})")
    
    return working_signals, signal_info, prefixes

def determine_data_source(working_signals, signal_info, prefixes):
    """데이터 출처 결정"""
    
    print(f"\n=== 🎯 데이터 출처 결정 ===")
    
    # 증거 수집
    evidence = {
        'nab_like': [],
        'nasa_like': [],
        'unknown': []
    }
    
    # 1. 데이터 구조 분석
    single_column_count = sum(1 for info in signal_info.values() if not info['has_other_cols'])
    if single_column_count == len(signal_info):
        evidence['nab_like'].append("모든 신호가 단일 컬럼 ('value'만)")
    else:
        evidence['nasa_like'].append(f"다중 컬럼 신호 {len(signal_info) - single_column_count}개 발견")
    
    # 2. 신호 명명 규칙
    if all(len(signal.split('-')) == 2 for signal in working_signals):
        evidence['nasa_like'].append("NASA 스타일 명명 규칙 (접두사-번호)")
    
    # 3. 신호 그룹 분석
    if len(prefixes) > 1:
        evidence['nasa_like'].append(f"다양한 센서 타입 {len(prefixes)}개 (M, P, T, F, C, D, S)")
    
    # 4. 시간 인덱스 분석
    integer_indices = sum(1 for info in signal_info.values() 
                         if isinstance(info['start'], (int, np.integer)))
    if integer_indices == len(signal_info):
        evidence['nab_like'].append("모든 신호가 정수 시간 인덱스 사용")
    
    # 5. 데이터 길이 패턴
    lengths = [info['length'] for info in signal_info.values()]
    length_variety = len(set(lengths)) / len(lengths) if lengths else 0
    
    if length_variety > 0.5:  # 50% 이상이 서로 다른 길이
        evidence['nasa_like'].append(f"다양한 데이터 길이 ({len(set(lengths))}종류)")
    else:
        evidence['nab_like'].append("유사한 데이터 길이 패턴")
    
    # 결론 도출
    print("📋 증거 분석:")
    
    print("   🔵 NAB 포맷 증거:")
    for item in evidence['nab_like']:
        print(f"      • {item}")
    
    print("   🟢 NASA MSL 증거:")
    for item in evidence['nasa_like']:
        print(f"      • {item}")
    
    if evidence['unknown']:
        print("   ⚪ 불확실한 증거:")
        for item in evidence['unknown']:
            print(f"      • {item}")
    
    # 최종 판단
    nab_score = len(evidence['nab_like'])
    nasa_score = len(evidence['nasa_like'])
    
    print(f"\n🎯 최종 판단:")
    print(f"   NAB 스코어: {nab_score}")
    print(f"   NASA 스코어: {nasa_score}")
    
    if nasa_score > nab_score:
        conclusion = "NASA MSL 기반 (단일 채널로 추출된 부분집합)"
        details = [
            "각 신호가 NASA MSL의 개별 센서 채널",
            "원본은 멀티변량이지만 단일 채널로 분리",
            "벤치마크용으로 전처리된 데이터"
        ]
    elif nab_score > nasa_score:
        conclusion = "NAB 포맷 유사 데이터"
        details = [
            "Numenta Anomaly Benchmark 스타일",
            "단일 채널 시계열 데이터",
            "이상 탐지 벤치마크용 데이터"
        ]
    else:
        conclusion = "혼합 또는 불확실"
        details = [
            "NASA와 NAB 특성 모두 보유",
            "추가 메타데이터 필요"
        ]
    
    print(f"   결론: {conclusion}")
    print(f"   특징:")
    for detail in details:
        print(f"      • {detail}")

if __name__ == "__main__":
    print("🚀 MSL 데이터셋 출처 식별 시작...\n")
    
    # 1. 데이터 구조 분석
    analyze_data_structure()
    
    # 2. 멀티변량 가능성 확인
    working_signals, signal_info, prefixes = check_multivariate_possibility()
    
    # 3. 데이터 출처 결정
    determine_data_source(working_signals, signal_info, prefixes)
    
    print("\n✅ 출처 식별 완료!")
