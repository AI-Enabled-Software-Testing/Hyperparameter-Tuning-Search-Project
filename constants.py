# Exponential smoothing parameters
SMOOTHING_SLIDER_BEST_FITNESS = 0.05
SMOOTHING_SLIDER_CURRENT_FITNESS = 0.75
SMOOTHING_SLIDER_INDIVIDUAL_METRICS = 0.3

# Metric weights from our composite fitness
METRIC_WEIGHTS = {
    "f1_macro": 0.30,
    "recall_macro": 0.20,
    "roc_auc": 0.20,
    "precision_macro": 0.15,
    "accuracy": 0.10,
    "f1_micro": 0.05,
}