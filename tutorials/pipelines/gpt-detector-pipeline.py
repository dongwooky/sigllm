#!/usr/bin/env python3
"""
SigLLM GPT Detector Pipeline - Python Script Version

This script demonstrates the step-by-step execution of the gpt_detector pipeline
for time series anomaly detection using OpenAI GPT models.

Requirements:
- OpenAI API key (set as OPENAI_API_KEY environment variable or in .env file)
- sigllm, orion, mlblocks, matplotlib, pandas, numpy, python-dotenv
"""

import warnings
warnings.simplefilter('ignore')

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from orion.data import load_signal, load_anomalies
from orion.evaluation import contextual_confusion_matrix
from mlblocks import MLPipeline

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env 파일에서 환경변수를 로드했습니다.")
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다. 시스템 환경변수를 사용합니다.")
    print("   설치하려면: pip install python-dotenv")

def check_openai_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-openai-api-key-here':
        print("❌ ERROR: OpenAI API key가 설정되지 않았습니다!")
        print()
        print("다음 중 하나의 방법으로 API key를 설정하세요:")
        print("1. .env 파일에서 OPENAI_API_KEY 수정")
        print("2. 터미널에서: export OPENAI_API_KEY='your-api-key-here'")
        print()
        print("OpenAI API key는 https://platform.openai.com/api-keys 에서 생성할 수 있습니다.")
        return False
    else:
        print("✅ OpenAI API key가 설정되었습니다")
        print(f"   Key 시작 부분: {api_key[:20]}...")
        return True

def main():
    """
    Main function to run the complete GPT detector pipeline
    """
    print("=== SigLLM GPT Detector Pipeline ===")
    print("Using OpenAI GPT models for time series anomaly detection\n")
    
    # Check API key
    if not check_openai_api_key():
        return None
    
    # 1. Data Loading
    print("\n1. Loading data...")
    data = load_signal('exchange-2_cpm_results')
    print(f"Data shape: {data.shape}")
    
    # Quick visualization of the data
    plt.figure(figsize=(12, 4))
    plt.plot(data['value'])
    plt.title('Original Time Series Data - GPT Detector')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Optional: Use smaller segment for quick testing (recommended for API cost control)
    print("\n💡 TIP: Uncomment the lines below for quick testing to reduce API costs")
    # start = 900
    # end = start + 200
    # data = data.iloc[start: end]
    # print(f"Using subset: {data.shape}")
    
    # 2. Pipeline Setup
    print("\n2. Setting up GPT detector pipeline...")
    pipeline_name = 'gpt_detector'
    pipeline = MLPipeline(pipeline_name)
    
    # Performance mode selection
    print("🎛️ Select performance mode:")
    print("1. 🚀 High Performance (GPT-4, more accurate but expensive)")
    print("2. 💰 Economy Mode (GPT-3.5-turbo, faster and cheaper)")
    print("3. 🔧 Custom Settings")
    
    mode = input("Choose mode (1/2/3): ").strip()
    
    if mode == "1":
        print("🚀 Using High Performance mode with GPT-4")
        hyperparameters = {
            "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
                "interval": 1800  # 30 min aggregation (more granular)
            },
            "sigllm.primitives.forecasting.gpt.GPT#1": {
                "name": "gpt-4",          # Best model for accuracy
                "samples": 3,             # More samples for stability
                "temp": 0.3,              # Balanced creativity
                "steps": 10               # Longer prediction horizon
            },
            "sigllm.primitives.transformation.format_as_integer#1": {
                "trunc": 1,
                "errors": "coerce"
            },
            "orion.primitives.timeseries_anomalies.find_anomalies#1": {
                "fixed_threshold": False,      # Dynamic threshold
                "window_size_portion": 0.1,    # More sensitive detection
                "window_step_size_portion": 0.05
            }
        }
    elif mode == "3":
        print("🔧 Custom Settings mode")
        model_name = input("GPT model (gpt-3.5-turbo/gpt-4) [gpt-4]: ").strip() or "gpt-4"
        samples = int(input("Number of samples [3]: ").strip() or "3")
        temp = float(input("Temperature (0.0-1.0) [0.3]: ").strip() or "0.3")
        steps = int(input("Forecast steps [10]: ").strip() or "10")
        interval = int(input("Time interval in seconds [1800]: ").strip() or "1800")
        
        hyperparameters = {
            "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
                "interval": interval
            },
            "sigllm.primitives.forecasting.gpt.GPT#1": {
                "name": model_name,
                "samples": samples,
                "temp": temp,
                "steps": steps
            },
            "sigllm.primitives.transformation.format_as_integer#1": {
                "trunc": 1,
                "errors": "coerce"
            },
            "orion.primitives.timeseries_anomalies.find_anomalies#1": {
                "fixed_threshold": False,
                "window_size_portion": 0.1,
                "window_step_size_portion": 0.05
            }
        }
    else:
        print("💰 Using Economy mode with GPT-3.5-turbo")
        hyperparameters = {
            "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
                "interval": 3600  # 1 hour aggregation
            },
            "sigllm.primitives.forecasting.gpt.GPT#1": {
                "name": "gpt-3.5-turbo",  # Economical choice
                "samples": 2,             # Moderate samples
                "temp": 0.2,              # Slightly more creative
                "steps": 7                # Medium prediction horizon
            },
            "sigllm.primitives.transformation.format_as_integer#1": {
                "trunc": 1,
                "errors": "coerce"
            },
            "orion.primitives.timeseries_anomalies.find_anomalies#1": {
                "fixed_threshold": False,      # Dynamic threshold
                "window_size_portion": 0.2,    # Moderate sensitivity
                "window_step_size_portion": 0.1
            }
        }
    pipeline.set_hyperparameters(hyperparameters)
    
    print(f"Pipeline primitives: {len(pipeline.primitives)}")
    print("Pipeline primitives:", pipeline.primitives)
    
    # 3. Step-by-step execution
    print("\n3. Step-by-step pipeline execution...")
    
    # Step 0: Time segments aggregate
    print("\n--- Step 0: Time segments aggregate ---")
    print("Creates equi-spaced time series by aggregating values over fixed interval")
    step = 0
    context = pipeline.fit(data, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # Show first 5 entries
    for i, x in list(zip(context['index'], context['X']))[:5]:
        print(f"Entry at {i} has value {x}")
    
    print(f"X shape: {context['X'].shape}")
    print("Note: dimension changed from original shape")
    
    # Step 1: SimpleImputer
    print("\n--- Step 1: SimpleImputer ---")
    print("Fills missing values")
    step = 1
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # Step 2: Float2Scalar
    print("\n--- Step 2: Float2Scalar ---")
    print("Converts float values into scalar up to certain decimal points")
    step = 2
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # Show first 5 entries after conversion
    for i, x in list(zip(context['index'], context['X']))[:5]:
        print(f"Entry at {i} has value {x}")
    
    print(f"Minimum value: {context['minimum']}")
    
    # Step 3: Rolling window sequences
    print("\n--- Step 3: Rolling window sequences ---")
    print("Generates sub-sequences using rolling window approach")
    step = 3
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # Show shapes after slicing
    print(f"X shape = {context['X'].shape}")
    print(f"y shape = {context['y'].shape}")
    print(f"X index shape = {context['index'].shape}")
    print(f"y index shape = {context['target_index'].shape}")
    
    # Step 4: Format as string
    print("\n--- Step 4: Format as string ---")
    print("Converts scalar values into string representation")
    step = 4
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # Show first sequence
    first_sequence = (context['X']).flatten().tolist()[0]
    print(f"First sequence (first few values): {first_sequence[:10]}...")
    print("Note: Now string type, ready for GPT input")
    
    # Step 5: GPT (OpenAI GPT model)
    print("\n--- Step 5: GPT (OpenAI GPT model) ---")
    print("🚨 WARNING: This step calls OpenAI API and will incur costs!")
    print("Prompts OpenAI GPT model to forecast next steps")
    
    # Calculate expected costs
    num_windows = len(context['X'])
    gpt_params = hyperparameters["sigllm.primitives.forecasting.gpt.GPT#1"]
    total_calls = num_windows * gpt_params["samples"]
    model_name = gpt_params["name"]
    
    print(f"📊 API Call Estimates:")
    print(f"   • Windows: {num_windows}")
    print(f"   • Samples per window: {gpt_params['samples']}")
    print(f"   • Total API calls: {total_calls}")
    print(f"   • Model: {model_name}")
    
    if "gpt-4" in model_name:
        estimated_cost = total_calls * 0.03  # Rough estimate for GPT-4
        print(f"   • Estimated cost: ~${estimated_cost:.2f}")
    else:
        estimated_cost = total_calls * 0.002  # Rough estimate for GPT-3.5
        print(f"   • Estimated cost: ~${estimated_cost:.2f}")
    
    print("\n💡 TIP: For testing, uncomment the data subset lines above to reduce costs!")
    
    # Ask user confirmation for API calls
    response = input("Continue with GPT API calls? (y/n): ")
    if response.lower() != 'y':
        print("Execution stopped by user.")
        return None
    
    step = 5
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # Show first few predictions
    print(f"First 5 predictions: {context['y_hat'][:5]}")
    
    # Step 6: Format as integer
    print("\n--- Step 6: Format as integer ---")
    print("Converts string values into integers")
    step = 6
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    print(f"First 5 predictions: {context['y_hat'][:5]}")
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 7: Scalar2Float
    print("\n--- Step 7: Scalar2Float ---")
    print("Converts integers to float and adds minimum value")
    step = 7
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    print(f"First 5 predictions: {context['y_hat'][:5]}")
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 8: Aggregate rolling window
    print("\n--- Step 8: Aggregate rolling window ---")
    print("Aggregates multiple horizon predictions into single representation")
    step = 8
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 9: Reshape
    print("\n--- Step 9: Reshape ---")
    print("Reshapes y_hat sequences")
    step = 9
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 10: Regression errors
    print("\n--- Step 10: Regression errors ---")
    print("Computes point-wise difference between y and y_hat")
    step = 10
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    print(f"y_hat shape: {context['y_hat'].shape}")
    
    # Step 11: Find anomalies
    print("\n--- Step 11: Find anomalies ---")
    print("Extracts anomalies from error sequences")
    step = 11
    context = pipeline.fit(**context, start_=step, output_=step)
    print(f"Context keys: {list(context.keys())}")
    
    # 4. Results
    print("\n4. Results...")
    
    # Check if anomalies were detected
    if context['anomalies'] is not None and len(context['anomalies']) > 0:
        # Display anomalies as DataFrame
        anomalies_df = pd.DataFrame(context['anomalies'], columns=['start', 'end', 'score'])
        print("🎯 Detected Anomalies (by GPT):")
        print(anomalies_df)
    else:
        print("🎯 No anomalies detected by GPT")
        # Create empty DataFrame with correct structure
        anomalies_df = pd.DataFrame(columns=['start', 'end', 'score'])
        print("Created empty anomalies DataFrame for further processing")
    
    # 4.5. Accuracy Evaluation
    print("\n4.5. Accuracy Evaluation...")
    
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
    
    # Calculate accuracy metrics
    try:
        if len(anomalies_df) > 0:
            tp, fp, fn, tn = contextual_confusion_matrix(truth_anomalies, anomalies_df, eval_data)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            print("\n=== GPT Detector Accuracy Metrics ===")
            print(f"True Positives (TP): {tp}")
            print(f"False Positives (FP): {fp}")
            print(f"False Negatives (FN): {fn}")
            print(f"True Negatives (TN): {tn}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Store metrics in context for later use
            context['accuracy_metrics'] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
            }
        else:
            print("\n=== GPT Detector Accuracy Metrics ===")
            print("No anomalies detected - calculating metrics with zero detections")
            # When no anomalies are detected, all ground truth anomalies become false negatives
            tp, fp = 0, 0
            fn = len(truth_anomalies)
            tn = len(eval_data) - fn  # Approximate, should be calculated properly
            
            precision = 0.0  # No detections means no precision
            recall = 0.0     # No detections means no recall
            f1 = 0.0
            accuracy = tn / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            print(f"True Positives (TP): {tp}")
            print(f"False Positives (FP): {fp}")
            print(f"False Negatives (FN): {fn}")
            print(f"True Negatives (TN): {tn}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            context['accuracy_metrics'] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
            }
        
    except Exception as e:
        print(f"Accuracy calculation failed: {e}")
        context['accuracy_metrics'] = None
    
    # 5. Visualization
    print("\n5. Creating visualizations...")
    
    # Extract data for plotting
    index, y, yhat, errors, anomalies = list(map(context.get, ['target_index', 'y', 'y_hat', 'errors', 'anomalies']))
    
    # Main plot
    plt.figure(figsize=(15, 12))
    
    # Get accuracy metrics for title
    f1_score = context.get('accuracy_metrics', {}).get('f1', 0) if context.get('accuracy_metrics') else 0
    
    # Original plot
    plt.subplot(3, 1, 1)
    plt.plot(index, y, label='original', linewidth=2)
    plt.plot(index, yhat, label='GPT forecast', linewidth=2)
    plt.plot(index, errors, label='error', linewidth=1, alpha=0.7)
    
    # Mark anomalies
    if len(anomalies) > 0:
        plt.axvspan(*anomalies[0][:2], color='r', alpha=0.2, label='detected anomalies')
    
    plt.legend()
    plt.title(f'GPT-based Time Series Anomaly Detection Results (F1: {f1_score:.4f})')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Value')
    
    # Zoomed view
    plt.subplot(3, 1, 2)
    plt.plot(index, y, label='original', linewidth=2)
    plt.plot(index, yhat, label='GPT forecast', linewidth=2)
    plt.plot(index, errors, label='error', linewidth=1, alpha=0.7)
    
    if len(anomalies) > 0:
        plt.axvspan(*anomalies[0][:2], color='r', alpha=0.2, label='detected anomalies')
    
    # Zoom to specific time range
    plt.xlim(1310000000, 1311000000)
    plt.legend()
    plt.title('Zoomed View (1.31e9 ± 500000)')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Value')
    
    # Accuracy metrics visualization
    plt.subplot(3, 1, 3)
    if context.get('accuracy_metrics'):
        metrics = context['accuracy_metrics']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        metric_values = [metrics['precision'], metrics['recall'], metrics['f1'], metrics['accuracy']]
        
        bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange'])
        plt.title('GPT Detector Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'Accuracy metrics not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Performance Metrics')
    
    plt.xlabel('Time Index')
    plt.tight_layout()
    plt.show()
    
    print("\n=== GPT Detector Pipeline execution completed successfully! ===")
    
    # Print final accuracy summary
    if context.get('accuracy_metrics'):
        metrics = context['accuracy_metrics']
        print(f"\n=== Final GPT Detector Accuracy Summary ===")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print(f"\n💰 API Usage Note: This run made approximately {len(context.get('y_hat', []))} API calls to OpenAI")
    
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