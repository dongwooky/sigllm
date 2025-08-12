#!/usr/bin/env python3
"""
MSL 데이터셋 이상치 시간대 분석
- 각 센서별 이상치 발생 시간 비교
- 동시 발생 vs 독립 발생 패턴 분석
- Multivariate 이상 탐지 관점에서 해석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orion.data import load_anomalies
from orion.benchmark import _load_signal
from datetime import datetime
import warnings
warnings.simplefilter('ignore')

def analyze_anomaly_timing():
    """각 센서별 이상치 발생 시간 분석"""
    
    print("=== 🚨 MSL 센서별 이상치 시간대 분석 ===\n")
    
    # 모든 MSL 센서들
    all_sensors = [
        'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7',
        'S-1', 'S-2', 
        'P-10', 'P-11', 'P-14', 'P-15', 
        'T-4', 'T-5', 'T-8', 'T-9', 'T-12', 'T-13',
        'F-4', 'F-5', 'F-7', 'F-8',
        'C-1', 'C-2', 'D-14', 'D-15', 'D-16'
    ]
    
    anomaly_data = {}
    
    # 각 센서별 이상치 정보 수집
    print("📊 센서별 이상치 정보 수집...")
    for sensor in all_sensors:
        try:
            anomalies = load_anomalies(sensor)
            if len(anomalies) > 0:
                # 실제 timestamp 데이터 로딩
                _, data = _load_signal(sensor, test_split=False)
                
                anomaly_info = []
                for _, anomaly in anomalies.iterrows():
                    start_ts = anomaly['start']
                    end_ts = anomaly['end']
                    
                    # Unix timestamp를 datetime으로 변환
                    start_dt = datetime.fromtimestamp(start_ts)
                    end_dt = datetime.fromtimestamp(end_ts)
                    duration = end_ts - start_ts
                    
                    anomaly_info.append({
                        'start_timestamp': start_ts,
                        'end_timestamp': end_ts,
                        'start_datetime': start_dt,
                        'end_datetime': end_dt,
                        'duration_seconds': duration,
                        'duration_days': duration / (24 * 3600)
                    })
                
                anomaly_data[sensor] = {
                    'count': len(anomalies),
                    'anomalies': anomaly_info,
                    'data_length': len(data)
                }
                
                print(f"   ✅ {sensor}: {len(anomalies)}개 이상치")
                for i, anom in enumerate(anomaly_info):
                    print(f"      {i+1}. {anom['start_datetime']} ~ {anom['end_datetime']} ({anom['duration_days']:.1f}일)")
            else:
                print(f"   ⚪ {sensor}: 이상치 없음")
                
        except Exception as e:
            print(f"   ❌ {sensor}: {e}")
    
    return anomaly_data

def analyze_temporal_patterns(anomaly_data):
    """시간적 패턴 분석"""
    
    print(f"\n=== ⏰ 시간적 패턴 분석 ===")
    
    # 모든 이상치 이벤트를 시간순으로 정렬
    all_events = []
    
    for sensor, info in anomaly_data.items():
        for i, anomaly in enumerate(info['anomalies']):
            all_events.append({
                'sensor': sensor,
                'anomaly_id': i,
                'start': anomaly['start_timestamp'],
                'end': anomaly['end_timestamp'],
                'start_dt': anomaly['start_datetime'],
                'end_dt': anomaly['end_datetime'],
                'duration': anomaly['duration_seconds']
            })
    
    # 시작 시간 기준으로 정렬
    all_events.sort(key=lambda x: x['start'])
    
    print(f"📋 전체 이상치 이벤트: {len(all_events)}개")
    print(f"   시간순 정렬:")
    
    for i, event in enumerate(all_events):
        print(f"      {i+1:2d}. {event['sensor']:>4} | {event['start_dt']} ~ {event['end_dt']} ({event['duration']/3600:.1f}h)")
    
    # 시간 겹침 분석
    print(f"\n🔍 시간 겹침 분석:")
    
    overlapping_pairs = []
    simultaneous_groups = []
    
    for i, event1 in enumerate(all_events):
        overlaps_with = []
        
        for j, event2 in enumerate(all_events):
            if i != j:
                # 시간 겹침 확인
                overlap_start = max(event1['start'], event2['start'])
                overlap_end = min(event1['end'], event2['end'])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlap_ratio1 = overlap_duration / event1['duration']
                    overlap_ratio2 = overlap_duration / event2['duration']
                    
                    overlaps_with.append({
                        'sensor': event2['sensor'],
                        'overlap_duration': overlap_duration,
                        'overlap_ratio1': overlap_ratio1,
                        'overlap_ratio2': overlap_ratio2
                    })
        
        if overlaps_with:
            overlapping_pairs.append({
                'sensor1': event1['sensor'],
                'start1': event1['start_dt'],
                'overlaps': overlaps_with
            })
    
    if overlapping_pairs:
        print(f"   ✅ 겹치는 이상치 발견:")
        for pair in overlapping_pairs:
            print(f"      📍 {pair['sensor1']} ({pair['start1']}):")
            for overlap in pair['overlaps']:
                print(f"         ↔ {overlap['sensor']:>4}: {overlap['overlap_duration']/3600:.1f}h 겹침 "
                      f"({overlap['overlap_ratio1']:.1%}/{overlap['overlap_ratio2']:.1%})")
    else:
        print(f"   ❌ 겹치는 이상치 없음 - 모든 이상치가 독립적으로 발생")
    
    return all_events, overlapping_pairs

def analyze_sensor_groups(anomaly_data, all_events):
    """센서 그룹별 이상치 패턴 분석"""
    
    print(f"\n=== 🔧 센서 그룹별 이상치 패턴 ===")
    
    # 센서 그룹 정의
    sensor_groups = {
        'M': 'Main_Sensors',
        'P': 'Power_System',
        'T': 'Temperature',
        'F': 'Function',
        'C': 'Control',
        'D': 'Diagnostics',
        'S': 'System'
    }
    
    group_anomalies = {}
    
    for group_prefix, group_name in sensor_groups.items():
        group_events = [event for event in all_events if event['sensor'].startswith(group_prefix)]
        
        if group_events:
            group_anomalies[group_prefix] = {
                'name': group_name,
                'count': len(group_events),
                'sensors': list(set([event['sensor'] for event in group_events])),
                'events': group_events
            }
            
            print(f"   {group_prefix} ({group_name}):")
            print(f"      • 이상치 수: {len(group_events)}개")
            print(f"      • 영향받은 센서: {group_anomalies[group_prefix]['sensors']}")
            
            # 그룹 내 시간 분포
            if len(group_events) > 1:
                times = [event['start'] for event in group_events]
                time_span = max(times) - min(times)
                print(f"      • 시간 범위: {time_span / (24*3600):.1f}일")
    
    return group_anomalies

def visualize_anomaly_timeline(all_events, anomaly_data):
    """이상치 타임라인 시각화"""
    
    print(f"\n📊 이상치 타임라인 시각화...")
    
    if not all_events:
        print("   ⚠️ 시각화할 이상치가 없습니다.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. 전체 타임라인
    sensors = list(anomaly_data.keys())
    sensor_y_pos = {sensor: i for i, sensor in enumerate(sensors)}
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sensors)))
    
    for i, event in enumerate(all_events):
        sensor = event['sensor']
        y_pos = sensor_y_pos[sensor]
        
        # 이상치 구간을 바 형태로 표시
        start_dt = event['start_dt']
        end_dt = event['end_dt']
        duration_hours = (event['end'] - event['start']) / 3600
        
        ax1.barh(y_pos, duration_hours, left=event['start'], 
                height=0.6, alpha=0.7, color=colors[y_pos % len(colors)],
                label=f"{sensor}" if sensor not in [e['sensor'] for e in all_events[:i]] else "")
        
        # 이상치 번호 표시
        ax1.text(event['start'] + duration_hours/2, y_pos, f"{i+1}", 
                ha='center', va='center', fontweight='bold', fontsize=8)
    
    ax1.set_yticks(range(len(sensors)))
    ax1.set_yticklabels(sensors)
    ax1.set_xlabel('Unix Timestamp')
    ax1.set_title('MSL 센서별 이상치 발생 타임라인')
    ax1.grid(True, alpha=0.3)
    
    # X축을 날짜로 변환
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    
    timestamps = [event['start'] for event in all_events] + [event['end'] for event in all_events]
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    ax1.set_xlim(min(timestamps) - 86400*30, max(timestamps) + 86400*30)  # 앞뒤 30일 여유
    
    # 2. 센서 그룹별 집계
    group_counts = {}
    for event in all_events:
        group = event['sensor'][0]  # 첫 글자 (M, P, T, F, C, D, S)
        group_counts[group] = group_counts.get(group, 0) + 1
    
    groups = list(group_counts.keys())
    counts = list(group_counts.values())
    
    ax2.bar(groups, counts, alpha=0.7, color=colors[:len(groups)])
    ax2.set_xlabel('센서 그룹')
    ax2.set_ylabel('이상치 개수')
    ax2.set_title('센서 그룹별 이상치 분포')
    ax2.grid(True, alpha=0.3)
    
    for i, count in enumerate(counts):
        ax2.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def determine_multivariate_implications(all_events, overlapping_pairs, group_anomalies):
    """Multivariate 분석 관점에서의 함의"""
    
    print(f"\n=== 🎯 Multivariate 분석 관점에서의 함의 ===")
    
    total_events = len(all_events)
    total_overlaps = len(overlapping_pairs)
    
    print(f"📊 이상치 발생 패턴:")
    print(f"   • 전체 이상치 이벤트: {total_events}개")
    print(f"   • 시간 겹침 이벤트: {total_overlaps}개")
    print(f"   • 독립 발생 비율: {(total_events - total_overlaps) / total_events * 100:.1f}%")
    
    if total_overlaps > 0:
        print(f"\n🔗 동시 발생 이상치 의미:")
        print(f"   • 시스템 레벨 장애 또는 연쇄 반응")
        print(f"   • 물리적 연결된 센서들의 동시 영향")
        print(f"   • Multivariate 모델의 필요성 높음")
        
        print(f"\n💡 Multivariate 접근법 권장:")
        print(f"   1. 🔗 Cross-sensor correlation 모니터링")
        print(f"   2. 🌊 Cascade failure detection")
        print(f"   3. 🎯 Root cause analysis through sensor dependency")
        print(f"   4. ⚡ Early warning through leading indicators")
        
    else:
        print(f"\n🔍 독립 발생 이상치 의미:")
        print(f"   • 각 센서별 독립적 이상 발생")
        print(f"   • 센서별 개별 임계값 기반 탐지 가능")
        print(f"   • Univariate 접근법도 효과적")
        
        print(f"\n💡 분석 접근법 권장:")
        print(f"   1. 🎯 센서별 개별 모니터링 (Univariate)")
        print(f"   2. 📊 센서 그룹별 패턴 분석")
        print(f"   3. 🔍 시간 지연 상관관계 탐지")
        print(f"   4. 📈 Long-term trend analysis")
    
    # 센서 그룹별 패턴
    print(f"\n🔧 센서 그룹별 이상치 특성:")
    for group, info in group_anomalies.items():
        vulnerability = info['count'] / len(info['sensors'])
        print(f"   {group} ({info['name']}): {info['count']}개 이상치, 취약도 {vulnerability:.1f}")

if __name__ == "__main__":
    print("🚀 MSL 센서별 이상치 시간대 분석 시작...\n")
    
    # 1. 센서별 이상치 정보 수집
    anomaly_data = analyze_anomaly_timing()
    
    if not anomaly_data:
        print("❌ 분석할 이상치 데이터가 없습니다.")
        exit(1)
    
    # 2. 시간적 패턴 분석
    all_events, overlapping_pairs = analyze_temporal_patterns(anomaly_data)
    
    # 3. 센서 그룹별 분석
    group_anomalies = analyze_sensor_groups(anomaly_data, all_events)
    
    # 4. 타임라인 시각화
    visualize_anomaly_timeline(all_events, anomaly_data)
    
    # 5. Multivariate 분석 함의
    determine_multivariate_implications(all_events, overlapping_pairs, group_anomalies)
    
    print("\n✅ 이상치 시간대 분석 완료!")
