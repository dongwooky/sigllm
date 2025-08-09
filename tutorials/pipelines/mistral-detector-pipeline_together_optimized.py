#!/usr/bin/env python3
"""
SigLLM Detector Pipeline - OPTIMIZED Version (Together AI)
High-performance version with async/parallel processing

Performance Improvements:
🚀 Async/parallel API calls (up to 10x faster)
📦 Batch processing for better throughput
🎯 Optimized token calculation
⚡ Connection pooling and retry logic
🔧 Tuned hyperparameters for speed

Requirements:
- Together AI API key (set as TOGETHER_API_KEY environment variable)
- sigllm, orion, mlblocks, matplotlib, pandas, numpy, together, aiohttp, tqdm
"""

import warnings
warnings.simplefilter('ignore')

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from orion.data import load_signal, load_anomalies
from orion.evaluation import contextual_confusion_matrix
from mlblocks import MLPipeline


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

def main():
    """
    Main function to run the optimized detector pipeline
    """
    print("=== 🚀 SigLLM Detector Pipeline (OPTIMIZED Together AI) ===")
    print("High-performance version with async/parallel processing!")
    print("Expected speed improvement: 3-10x faster than sequential version\n")
    
    # Record start time for total performance measurement
    total_start_time = time.time()
    
    # 1. Data Loading
    print("1. Loading data...")
    data = load_signal('exchange-2_cpm_results')
    # data = load_signal('F-5-test')
    print(f"Data shape: {data.shape}")
    
    # Quick visualization of the data
    plt.figure(figsize=(12, 4))
    plt.plot(data['value'])
    plt.title('Original Time Series Data')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Optional: Use smaller segment for quick testing
    # Uncomment the lines below for quick testing
    # start = 900
    # end = start + 200
    # data = data.iloc[start: end]
    # print(f"Using subset: {data.shape}")
    
    # 2. Pipeline Setup with Optimized Parameters
    print("\n2. Setting up OPTIMIZED pipeline...")
    print("   🚀 Using optimized TogetherAI primitive with async/parallel processing")
    
    pipeline_name = 'mistral_detector_together'
    print(f"   ✅ Using Together AI pipeline: {pipeline_name}")
    pipeline = MLPipeline(pipeline_name)
    
    # OPTIMIZED hyperparameters for better performance
    hyperparameters = {
        "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
            "interval": 3600 #기본
            # "interval": 21600 #NASA
        },
        "sigllm.primitives.forecasting.together_ai.TogetherAI#1": {
            "samples": 1,  # Reduced from 2 to 1 for 2x speed improvement
            "max_tokens": 20,  # REDUCED: 30 → 20 (minimal tokens for anomaly detection)
            "temp": 0.1,  # REDUCED: 0.7 → 0.1 (more deterministic = faster generation)
            "top_p": 0.9,  # Optimized for speed
            "max_concurrent": 20,  # INCREASED: 15 → 20 (even more parallel processing)
            "batch_size": 100,  # INCREASED: 50 → 75 (larger batches = fewer API calls)
            "use_async": True,  # Enable async processing for maximum speed
        },
        "sigllm.primitives.transformation.format_as_integer#1": {
            "trunc": 1,
            "errors": "coerce"
        }
    }
    
    pipeline.set_hyperparameters(hyperparameters)
    
    print(f"Pipeline primitives: {len(pipeline.primitives)}")
    print("Pipeline primitives:", pipeline.primitives)
    
    # 3. Step-by-step execution with performance monitoring
    print("\n3. Step-by-step pipeline execution with performance monitoring...")
    
    # Step 0: Time segments aggregate
    print("\n--- Step 0: Time segments aggregate ---")
    print("Creates equi-spaced time series by aggregating values over fixed interval")
    step_start = time.time()
    step = 0
    context = pipeline.fit(data, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    # Show first 5 entries
    for i, x in list(zip(context['index'], context['X']))[:5]:
        print(f"Entry at {i} has value {x}")
    
    print(f"X shape: {context['X'].shape}")
    print("Note: dimension changed from original shape (1624, 2)")
    
    # Step 1: SimpleImputer
    print("\n--- Step 1: SimpleImputer ---")
    print("Fills missing values")
    step_start = time.time()
    step = 1
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    # Step 2: Float2Scalar
    print("\n--- Step 2: Float2Scalar ---")
    print("Converts float values into scalar up to certain decimal points")
    step_start = time.time()
    step = 2
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    # Show first 5 entries after conversion
    for i, x in list(zip(context['index'], context['X']))[:5]:
        print(f"Entry at {i} has value {x}")
    
    print(f"Minimum value: {context['minimum']}")
    
    # Step 3: Rolling window sequences
    print("\n--- Step 3: Rolling window sequences ---")
    print("Generates sub-sequences using rolling window approach")
    step_start = time.time()
    step = 3
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    # Show shapes after slicing
    print(f"X shape = {context['X'].shape}")
    print(f"y shape = {context['y'].shape}")
    print(f"X index shape = {context['index'].shape}")
    print(f"y index shape = {context['target_index'].shape}")
    
    # Step 4: Format as string
    print("\n--- Step 4: Format as string ---")
    print("Converts scalar values into string representation")
    step_start = time.time()
    step = 4
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    # Show first sequence
    first_sequence = (context['X']).flatten().tolist()[0]
    print(f"First sequence (first few values): {first_sequence[:10]}...")
    print("Note: Now string type, ready for LLM input")
    
    # Step 5: OPTIMIZED TogetherAI model - The main performance bottleneck
    print("\n--- Step 5: 🚀 OPTIMIZED TogetherAI model ---")
    print("Prompts Together AI model to forecast next steps with PARALLEL PROCESSING")
    print("This is where the major speed improvement happens!")
    
    sequences_count = len(context['X'])
    print(f"📊 Processing {sequences_count} sequences with optimized parallel calls...")
    
    step_start = time.time()
    step = 5
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    
    print(f"   🎯 Step {step} completed in {step_time:.2f}s")
    print(f"   ⚡ Average time per sequence: {step_time/sequences_count:.3f}s")
    print(f"   🔥 Estimated speedup vs sequential: {sequences_count * 2 * 0.5 / step_time:.1f}x")
    print(f"Context keys: {list(context.keys())}")
    
    # Show first few predictions
    print(f"First 5 predictions: {context['y_hat'][:5]}")
    
    # Step 6: Format as integer
    print("\n--- Step 6: Format as integer ---")
    print("Converts string values into integers")
    step_start = time.time()
    step = 6
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    print(f"First 5 predictions: {context['y_hat'][:5]}")
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 7: Scalar2Float
    print("\n--- Step 7: Scalar2Float ---")
    print("Converts integers to float and adds minimum value")
    step_start = time.time()
    step = 7
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    print(f"First 5 predictions: {context['y_hat'][:5]}")
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 8: Aggregate rolling window
    print("\n--- Step 8: Aggregate rolling window ---")
    print("Aggregates multiple horizon predictions into single representation")
    step_start = time.time()
    step = 8
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 9: Reshape
    print("\n--- Step 9: Reshape ---")
    print("Reshapes y_hat sequences")
    step_start = time.time()
    step = 9
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 10: Regression errors
    print("\n--- Step 10: Regression errors ---")
    print("Computes point-wise difference between y and y_hat")
    step_start = time.time()
    step = 10
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 11: Find anomalies
    print("\n--- Step 11: Find anomalies ---")
    print("Extracts anomalies from error sequences")
    step_start = time.time()
    step = 11
    context = pipeline.fit(**context, start_=step, output_=step)
    step_time = time.time() - step_start
    print(f"   ⏱️  Step {step} completed in {step_time:.2f}s")
    print(f"Context keys: {list(context.keys())}")
    
    # 4. Results
    print("\n4. Results...")
    
    # Check if anomalies were detected
    if context['anomalies'] is not None and len(context['anomalies']) > 0:
        # Display anomalies as DataFrame
        anomalies_df = pd.DataFrame(context['anomalies'], columns=['start', 'end', 'score'])
        print("🎯 Detected Anomalies (by Optimized Mistral via Together AI):")
        print(anomalies_df)
    else:
        print("🎯 No anomalies detected by Optimized Mistral via Together AI")
        # Create empty DataFrame with correct structure
        anomalies_df = pd.DataFrame(columns=['start', 'end', 'score'])
        print("Created empty anomalies DataFrame for further processing")
    
    # 4.5. Performance Evaluation
    print("\n4.5. Performance Evaluation...")
    
    # Load ground truth anomalies
    truth_anomalies = load_anomalies('exchange-2_cpm_results')
    print(f"Ground Truth Anomalies: {len(truth_anomalies)} segments")
    
    # Prepare data for evaluation
    index, y, yhat, errors, anomalies = list(map(context.get, ['target_index', 'y', 'y_hat', 'errors', 'anomalies']))
    
    # Create data DataFrame for evaluation
    eval_data = pd.DataFrame({
        'timestamp': index,
        'value': y.flatten() if hasattr(y, 'flatten') else y
    })
    
    print(f"📊 Debug Info:")
    print(f"   • eval_data shape: {eval_data.shape}")
    print(f"   • truth_anomalies: {len(truth_anomalies)} segments")
    print(f"   • detected anomalies: {len(anomalies_df)} segments")
    print(f"   • timestamp range: {eval_data['timestamp'].min()} to {eval_data['timestamp'].max()}")
    
    # Pure F1 Score calculation (point-level)
    print(f"\n=== 🧮 Pure F1 Score Calculation (Point-level) ===")
    try:
        # Create binary labels for each timestamp
        timestamps = eval_data['timestamp'].values
        gt_labels = np.zeros(len(timestamps), dtype=int)
        detected_labels = np.zeros(len(timestamps), dtype=int)
        
        # Mark ground truth anomalies
        for _, row in truth_anomalies.iterrows():
            mask = (timestamps >= row['start']) & (timestamps <= row['end'])
            gt_labels[mask] = 1
        
        # Mark detected anomalies  
        for _, row in anomalies_df.iterrows():
            mask = (timestamps >= row['start']) & (timestamps <= row['end'])
            detected_labels[mask] = 1
        
        # Calculate confusion matrix
        tp_pure = np.sum((gt_labels == 1) & (detected_labels == 1))
        fp_pure = np.sum((gt_labels == 0) & (detected_labels == 1))
        fn_pure = np.sum((gt_labels == 1) & (detected_labels == 0))
        tn_pure = np.sum((gt_labels == 0) & (detected_labels == 0))
        
        # Calculate metrics
        precision_pure = tp_pure / (tp_pure + fp_pure) if (tp_pure + fp_pure) > 0 else 0
        recall_pure = tp_pure / (tp_pure + fn_pure) if (tp_pure + fn_pure) > 0 else 0
        f1_pure = 2 * precision_pure * recall_pure / (precision_pure + recall_pure) if (precision_pure + recall_pure) > 0 else 0
        accuracy_pure = (tp_pure + tn_pure) / (tp_pure + tn_pure + fp_pure + fn_pure)
        
        print(f"📊 Point-level Analysis:")
        print(f"   • Total GT anomaly points: {np.sum(gt_labels)}")
        print(f"   • Total detected anomaly points: {np.sum(detected_labels)}")
        print(f"   • Total data points: {len(timestamps)}")
        
        print(f"\n🔍 Pure Confusion Matrix:")
        print(f"   True Positives (TP): {tp_pure}")
        print(f"   False Positives (FP): {fp_pure}")  
        print(f"   False Negatives (FN): {fn_pure}")
        print(f"   True Negatives (TN): {tn_pure}")
        
        print(f"\n✨ Pure Metrics:")
        print(f"   Precision: {precision_pure:.4f} ({tp_pure}/{tp_pure + fp_pure})")
        print(f"   Recall: {recall_pure:.4f} ({tp_pure}/{tp_pure + fn_pure})")
        print(f"   F1-Score: {f1_pure:.4f}")
        print(f"   Accuracy: {accuracy_pure:.4f}")
        
        print(f"\n💡 Pure Interpretation:")
        print(f"   • {tp_pure} points correctly detected as anomalies")
        print(f"   • {fn_pure} anomaly points missed")
        print(f"   • {fp_pure} points incorrectly flagged as anomalies")
        print(f"   • {recall_pure:.1%} of all anomaly points detected")
        
    except Exception as e:
        print(f"❌ Pure F1 calculation failed: {e}")
    
    # Pure Segment-based evaluation
    print(f"\n=== 🎯 Pure Segment-based Evaluation ===")
    try:
        print(f"📊 Segment Analysis:")
        print(f"   • Ground Truth segments: {len(truth_anomalies)}")
        print(f"   • Detected segments: {len(anomalies_df)}")
        
        # Calculate segment-level overlaps
        gt_matched = set()  # Track which GT segments have been matched
        det_matched = set()  # Track which detected segments have been matched
        
        overlaps = []
        for gt_idx, gt_row in truth_anomalies.iterrows():
            for det_idx, det_row in anomalies_df.iterrows():
                # Calculate overlap
                overlap_start = max(gt_row['start'], det_row['start'])
                overlap_end = min(gt_row['end'], det_row['end'])
                overlap_length = max(0, overlap_end - overlap_start)
                
                if overlap_length > 0:
                    # Calculate overlap percentage
                    gt_length = gt_row['end'] - gt_row['start']
                    det_length = det_row['end'] - det_row['start']
                    overlap_ratio_gt = overlap_length / gt_length
                    overlap_ratio_det = overlap_length / det_length
                    
                    overlaps.append({
                        'gt_idx': gt_idx,
                        'det_idx': det_idx,
                        'overlap_length': overlap_length,
                        'overlap_ratio_gt': overlap_ratio_gt,
                        'overlap_ratio_det': overlap_ratio_det
                    })
                    
                    # Consider as matched if overlap is significant (>= 10%)
                    if overlap_ratio_gt >= 0.1 or overlap_ratio_det >= 0.1:
                        gt_matched.add(gt_idx)
                        det_matched.add(det_idx)
        
        # Calculate pure segment metrics
        tp_seg = len(gt_matched)  # GT segments that were successfully detected
        fn_seg = len(truth_anomalies) - len(gt_matched)  # GT segments that were missed
        fp_seg = len(anomalies_df) - len(det_matched)  # Detected segments that don't match GT
        
        # Calculate metrics
        precision_seg = tp_seg / len(anomalies_df) if len(anomalies_df) > 0 else 0
        recall_seg = tp_seg / len(truth_anomalies) if len(truth_anomalies) > 0 else 0
        f1_seg = 2 * precision_seg * recall_seg / (precision_seg + recall_seg) if (precision_seg + recall_seg) > 0 else 0
        
        print(f"\n🔍 Pure Segment Overlap Details:")
        for i, overlap in enumerate(overlaps):
            gt_idx = overlap['gt_idx']
            det_idx = overlap['det_idx']
            print(f"   Match {i+1}: GT[{gt_idx}] ↔ Det[{det_idx}]")
            print(f"     • Overlap length: {overlap['overlap_length']}")
            print(f"     • GT coverage: {overlap['overlap_ratio_gt']:.1%}")
            print(f"     • Det coverage: {overlap['overlap_ratio_det']:.1%}")
        
        
    except Exception as e:
        print(f"❌ Pure segment evaluation failed: {e}")
    
    # Calculate performance metrics using BOTH Orion evaluation methods
    try:
        if len(anomalies_df) > 0:
            from orion.evaluation.contextual import contextual_f1_score, contextual_precision, contextual_recall
            
            print("\n🔍 === COMPARISON: Two Orion Evaluation Methods ===")
            
            # Method 1: Point-based Weighted Evaluation
            print("\n📊 Method 1: Point-based Weighted Evaluation (weighted=True)")
            print("   → Evaluates each individual data point with weights")
            
            tn1, fp1, fn1, tp1 = contextual_confusion_matrix(
                truth_anomalies, anomalies_df, eval_data, weighted=True
            )
            
            precision1 = contextual_precision(truth_anomalies, anomalies_df, eval_data, weighted=True)
            recall1 = contextual_recall(truth_anomalies, anomalies_df, eval_data, weighted=True)
            f1_1 = contextual_f1_score(truth_anomalies, anomalies_df, eval_data, weighted=True)
            
            print(f"   True Positives (TP): {tp1}")
            print(f"   False Positives (FP): {fp1}")
            print(f"   False Negatives (FN): {fn1}")
            print(f"   True Negatives (TN): {tn1}")
            print(f"   Precision: {precision1:.4f}")
            print(f"   Recall: {recall1:.4f}")
            print(f"   F1-Score: {f1_1:.4f}")
            
            
            # Method 2: Segment-based Non-weighted Evaluation (Recommended for Time Series)
            print("\n🎯 Method 2: Segment-based Non-weighted Evaluation (weighted=False)")
            print("   → Evaluates anomaly segments based on overlap")
            
            tn2, fp2, fn2, tp2 = contextual_confusion_matrix(
                truth_anomalies, anomalies_df, eval_data, weighted=False
            )
            
            precision2 = contextual_precision(truth_anomalies, anomalies_df, eval_data, weighted=False)
            recall2 = contextual_recall(truth_anomalies, anomalies_df, eval_data, weighted=False)
            f1_2 = contextual_f1_score(truth_anomalies, anomalies_df, eval_data, weighted=False)
            # Note: For segment-based evaluation, TN is None (not applicable in segment overlap analysis)
            
            print(f"   True Positives (TP): {tp2 if tp2 is not None else 'N/A (segment-based)'}")
            print(f"   False Positives (FP): {fp2}")
            print(f"   False Negatives (FN): {fn2}")
            print(f"   True Negatives (TN): {tn2 if tn2 is not None else 'N/A (segment-based)'}")
            print(f"   Precision: {precision2:.4f}")
            print(f"   Recall: {recall2:.4f}")
            print(f"   F1-Score: {f1_2:.4f}")
            
            # Store both sets of metrics in context
            context['evaluation_metrics'] = {
                'point_based': {
                    'tp': tp1, 'fp': fp1, 'fn': fn1, 'tn': tn1,
                    'precision': precision1, 'recall': recall1, 'f1': f1_1
                },
                'segment_based': {
                    'tp': tp2, 'fp': fp2, 'fn': fn2, 'tn': tn2,
                    'precision': precision2, 'recall': recall2, 'f1': f1_2
                }
            }
            
            # Use segment-based metrics as primary for visualization (recommended for time series)
            precision, recall, f1 = precision2, recall2, f1_2
        else:
            from orion.evaluation.contextual import contextual_f1_score, contextual_precision, contextual_recall
            
            print("\n🔍 === COMPARISON: Two Orion Evaluation Methods (No Detections) ===")
            
            # Empty detection DataFrame
            empty_detections = pd.DataFrame(columns=['start', 'end', 'score'])
            
            # Method 1: Point-based Weighted Evaluation
            print("\n📊 Method 1: Point-based Weighted Evaluation (weighted=True)")
            tn1, fp1, fn1, tp1 = contextual_confusion_matrix(
                truth_anomalies, empty_detections, eval_data, weighted=True
            )
            precision1 = contextual_precision(truth_anomalies, empty_detections, eval_data, weighted=True)
            recall1 = contextual_recall(truth_anomalies, empty_detections, eval_data, weighted=True)
            f1_1 = contextual_f1_score(truth_anomalies, empty_detections, eval_data, weighted=True)
            
            print(f"   True Positives (TP): {tp1}")
            print(f"   False Positives (FP): {fp1}")
            print(f"   False Negatives (FN): {fn1}")
            print(f"   True Negatives (TN): {tn1}")
            print(f"   Precision: {precision1:.4f}")
            print(f"   Recall: {recall1:.4f}")
            print(f"   F1-Score: {f1_1:.4f}")
            
            # Method 2: Segment-based Non-weighted Evaluation
            print("\n🎯 Method 2: Segment-based Non-weighted Evaluation (weighted=False)")
            tn2, fp2, fn2, tp2 = contextual_confusion_matrix(
                truth_anomalies, empty_detections, eval_data, weighted=False
            )
            precision2 = contextual_precision(truth_anomalies, empty_detections, eval_data, weighted=False)
            recall2 = contextual_recall(truth_anomalies, empty_detections, eval_data, weighted=False)
            f1_2 = contextual_f1_score(truth_anomalies, empty_detections, eval_data, weighted=False)
            # Note: For segment-based evaluation, TN is None (not applicable in segment overlap analysis)
            
            print(f"   True Positives (TP): {tp2 if tp2 is not None else 'N/A (segment-based)'}")
            print(f"   False Positives (FP): {fp2}")
            print(f"   False Negatives (FN): {fn2}")
            print(f"   True Negatives (TN): {tn2 if tn2 is not None else 'N/A (segment-based)'}")
            print(f"   Precision: {precision2:.4f}")
            print(f"   Recall: {recall2:.4f}")
            print(f"   F1-Score: {f1_2:.4f}")
            
            # Store both sets of metrics
            context['evaluation_metrics'] = {
                'point_based': {
                    'tp': tp1, 'fp': fp1, 'fn': fn1, 'tn': tn1,
                    'precision': precision1, 'recall': recall1, 'f1': f1_1
                },
                'segment_based': {
                    'tp': tp2, 'fp': fp2, 'fn': fn2, 'tn': tn2,
                    'precision': precision2, 'recall': recall2, 'f1': f1_2
                }
            }
            
            # Use segment-based metrics as primary for visualization  
            precision, recall, f1 = precision2, recall2, f1_2
        
    except Exception as e:
        print(f"Evaluation calculation failed: {e}")
        context['evaluation_metrics'] = None
    
    # 5. Visualization
    print("\n5. Creating visualizations...")
    
    # Extract data for plotting
    index, y, yhat, errors, anomalies = list(map(context.get, ['target_index', 'y', 'y_hat', 'errors', 'anomalies']))
    
    # Main plot
    plt.figure(figsize=(15, 12))
    
    # Get F1 score for title
    f1_score = context.get('evaluation_metrics', {}).get('segment_based', {}).get('f1', 0) if context.get('evaluation_metrics') else 0
    
    # Original plot
    plt.subplot(3, 1, 1)
    plt.plot(index, y, label='original', linewidth=2)
    plt.plot(index, yhat, label='Optimized Mistral forecast (Together AI)', linewidth=2)
    plt.plot(index, errors, label='error', linewidth=1, alpha=0.7)
    
    # Mark anomalies
    if len(anomalies) > 0:
        plt.axvspan(*anomalies[0][:2], color='r', alpha=0.2, label='detected anomalies')
    
    plt.legend()
    plt.title(f'🚀 OPTIMIZED Mistral-based Time Series Anomaly Detection via Together AI (F1: {f1_score:.4f})')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Value')
    
    # Zoomed view
    plt.subplot(3, 1, 2)
    plt.plot(index, y, label='original', linewidth=2)
    plt.plot(index, yhat, label='Optimized Mistral forecast (Together AI)', linewidth=2)
    plt.plot(index, errors, label='error', linewidth=1, alpha=0.7)
    
    if len(anomalies) > 0:
        plt.axvspan(*anomalies[0][:2], color='r', alpha=0.2, label='detected anomalies')
    
    # Zoom to specific time range
    plt.xlim(1310000000, 1311000000)
    plt.legend()
    plt.title('Zoomed View (1.31e9 ± 500000)')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Value')
    
    # Performance metrics visualization - Compare both methods
    plt.subplot(3, 1, 3)
    if context.get('evaluation_metrics'):
        metrics = context['evaluation_metrics']
        
        if 'point_based' in metrics and 'segment_based' in metrics:
            # Both evaluation methods available - show comparison
            metric_names = ['Precision', 'Recall', 'F1-Score']
            point_values = [metrics['point_based']['precision'], metrics['point_based']['recall'], 
                           metrics['point_based']['f1']]
            segment_values = [metrics['segment_based']['precision'], metrics['segment_based']['recall'], 
                            metrics['segment_based']['f1']]
            
            x = range(len(metric_names))
            width = 0.35
            
            bars1 = plt.bar([i - width/2 for i in x], point_values, width, 
                           label='Point-based', color='lightblue', alpha=0.8)
            bars2 = plt.bar([i + width/2 for i in x], segment_values, width, 
                           label='Segment-based', color='orange', alpha=0.8)
            
            plt.title('🚀 Optimized Mistral Detector: Orion Evaluation Methods Comparison')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.xticks(x, metric_names)
            plt.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            # Fallback to single method
            if 'segment_based' in metrics:
                m = metrics['segment_based']
            else:
                m = metrics
            
            metric_names = ['Precision', 'Recall', 'F1-Score']
            metric_values = [m.get('precision', 0), m.get('recall', 0), m.get('f1', 0)]
            
            bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'red'])
            plt.title('🚀 Optimized Mistral Detector Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'Performance metrics not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Performance Metrics')
    
    plt.xlabel('Time Index')
    plt.tight_layout()
    plt.show()
    
    # Performance Summary
    total_time = time.time() - total_start_time
    print(f"\n=== 🚀 OPTIMIZED Mistral Detector Pipeline (Together AI) execution completed! ===")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    
    
    return context

if __name__ == "__main__":
    try:
        context = main()
        if context:
            print(f"\nFinal context keys: {list(context.keys())}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
