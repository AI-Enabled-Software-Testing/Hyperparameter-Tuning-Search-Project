import numpy as np

def _exp_smooth(values: np.ndarray, smoothing_slider: float = 0.9) -> np.ndarray:
    """Apply exponential smoothing to a sequence of values."""
    smoothed = np.empty_like(values, dtype=np.float32)
    last = values[0]
    for idx, point in enumerate(values):
        last = smoothing_slider * last + (1.0 - smoothing_slider) * point
        smoothed[idx] = last
    return smoothed