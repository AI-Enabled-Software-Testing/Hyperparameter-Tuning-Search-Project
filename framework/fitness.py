def calculate_composite_fitness(metrics: dict[str, float]) -> float:
    """Calculate composite fitness score from evaluation metrics."""
    # Extract metrics
    f1_macro = metrics.get("f1_macro", 0.0)
    recall_macro = metrics.get("recall_macro", 0.0)
    roc_auc = metrics.get("roc_auc", 0.0)
    precision_macro = metrics.get("precision_macro", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    f1_micro = metrics.get("f1_micro", 0.0)
    
    # Composite fitness
    composite_fitness = (
        0.30 * f1_macro +
        0.20 * recall_macro +
        0.20 * roc_auc +
        0.15 * precision_macro +
        0.10 * accuracy +
        0.05 * f1_micro
    )
    
    return composite_fitness

