def calculate_composite_fitness(metrics: dict[str, float]) -> float:
    """Calculate composite fitness score from evaluation metrics."""
    # Extract metrics
    f1_macro = metrics["f1_macro"]
    recall_macro = metrics["recall_macro"]
    roc_auc = metrics["roc_auc"]
    precision_macro = metrics["precision_macro"]
    accuracy = metrics["accuracy"]
    f1_micro = metrics["f1_micro"]
    
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

