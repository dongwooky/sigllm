# -*- coding: utf-8 -*-
import numpy as np


def outliers(predictions):
    """Remove outliers from predictions.

    Find the outliers according to Q1 and Q3 of the drawn samples. If
    the prediction value exceeds these bounds, remove it from the output.

    Args:
        predictions (ndarray):
            Predicted sequences.

    Return:
        ndarray:
            Predictions after removing outliers.
    """
    Q1, Q3 = np.percentile(predictions, [25, 75])

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    predictions[(predictions < lower_bound) | (predictions > upper_bound)] = np.nan

    return predictions


def aggregate_rolling_window(y, step_size=1, agg='median', remove_outliers=False):
    """Aggregate a rolling window sequence.

    Convert a rolling window sequence into a flattened time series.
    Use the aggregation specified to make each timestamp a single value.

    Args:
        y (ndarray):
            Windowed sequences. Each timestamp has multiple predictions.
        step_size (int):
            Stride size used when creating the rolling windows.
        agg (string):
            String denoting the aggregation method to use. Default is "median".
        remove_outliers (bool):
            Indicator to whether remove outliers from the predictions.

    Return:
        ndarray:
            Flattened sequence.
    """
    # Handle different input shapes
    print(f"🔍 Debug: y.shape = {y.shape}")
    
    if len(y.shape) == 3:
        # Expected 3D shape: (num_windows, num_samples, pred_length)
        num_windows, num_samples, pred_length = y.shape
    elif len(y.shape) == 2:
        # Handle 2D shape: assume (num_windows, pred_length) with num_samples=1
        num_windows, pred_length = y.shape
        num_samples = 1
        # Reshape to 3D
        y = y.reshape(num_windows, num_samples, pred_length)
        print(f"🔧 Reshaped y from 2D to 3D: {y.shape}")
    else:
        raise ValueError(f"Unexpected y.shape: {y.shape}. Expected 2D or 3D array.")
    
    num_errors = pred_length + step_size * (num_windows - 1)

    if remove_outliers:
        y = outliers(y)

    method = getattr(np, agg)
    signal = []

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            for k in range(num_samples):
                value = y[i - j, k, j]
                # 스칼라 값으로 변환하여 균일한 형태 보장
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        intermediate.append(float(value[0]))
                    else:
                        intermediate.append(0.0)
                else:
                    intermediate.append(float(value))

        if intermediate:  # 빈 리스트가 아닌 경우에만 처리
            signal.append(method(np.asarray(intermediate, dtype=float)))
        else:
            signal.append(0.0)  # 기본값

    return np.array(signal)
