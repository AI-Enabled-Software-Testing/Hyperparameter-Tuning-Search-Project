"""Analyze experiment results and generate visualization figures."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / ".cache" / "experiment"
FIGURES_ROOT = REPO_ROOT / ".cache" / "experiment_figures"

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


def load_experiment_runs(experiment_name: str, include_history: bool = False) -> List[Dict[str, Any]]:
    """Load all run data for an experiment.
    
    Args:
        experiment_name: Name of the experiment to load
        include_history: If True, load history.json instead of convergence.json
    
    Returns:
        List of run dictionaries containing run_dir, summary, and either convergence or history data
    """
    experiment_dir = EXPERIMENT_ROOT / experiment_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}\n"
            f"Available experiments: {[d.name for d in EXPERIMENT_ROOT.iterdir() if d.is_dir()]}"
        )
    
    runs = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        
        summary_path = run_dir / "summary.json"
        
        if include_history:
            data_path = run_dir / "history.json"
            data_key = "history"
        else:
            data_path = run_dir / "convergence.json"
            data_key = "convergence"
        
        if not data_path.exists() or not summary_path.exists():
            print(f"Warning: Skipping incomplete run {run_dir.name}")
            continue
        
        with open(data_path) as f:
            data = json.load(f)
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        runs.append({
            "run_dir": run_dir.name,
            data_key: data,
            "summary": summary,
        })
    
    if not runs:
        raise ValueError(f"No valid runs found in {experiment_dir}")
    
    suffix = " with history" if include_history else ""
    print(f"Loaded {len(runs)} runs{suffix} from {experiment_name}")
    return runs


def extract_metric_convergence(history: List[Dict[str, Any]], metric_name: str) -> Dict[str, list]:
    """Extract best-so-far convergence trace for a specific metric from history.
    
    Args:
        history: List of trial entries from history.json
        metric_name: Name of the metric to extract (e.g., 'accuracy', 'f1_macro')
    
    Returns:
        Dictionary with 'evaluations', 'best_metric', 'current_metric' lists
    """
    best_so_far = float("-inf")
    convergence = []
    
    for entry in history:
        metric_value = entry["metrics"][metric_name]
        if metric_value > best_so_far:
            best_so_far = metric_value
        convergence.append({
            "evaluation": entry["trial"],
            "best_metric": best_so_far,
            "current_metric": metric_value,
        })
    
    return {
        "evaluations": [c["evaluation"] for c in convergence],
        "best_metric": [c["best_metric"] for c in convergence],
        "current_metric": [c["current_metric"] for c in convergence],
    }

def plot_plateaus(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str, threshold: float | None):
    """Plot plateau detection for each run."""
    for i, run in enumerate(runs):
        convergence = run["convergence"]
        evaluations = np.array(convergence["evaluations"])
        best_fitness = np.array(convergence["best_fitness"])

        # Plateau detection: where best_fitness does not increase (or below threshold)
        diffs = np.diff(best_fitness)
        if threshold is not None:
            thresh = threshold
        else:
            thresh = 0.0  # exactly zero diff means strict plateau

        plateau_starts = np.where(np.abs(diffs) <= thresh)[0]
        escapes = np.where(diffs > thresh)[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(evaluations, best_fitness, label="Best Fitness", color="blue")
        # Highlight plateau regions
        for idx in plateau_starts:
            ax.axvspan(evaluations[idx], evaluations[idx + 1], color="red", alpha=0.2)
        # Mark escapes from plateaus
        if len(escapes) > 0:
            ax.scatter(evaluations[escapes + 1], best_fitness[escapes + 1], color='red', s=30, label="Plateau Escapes")

        ax.set_xlabel("Evaluation", fontsize=12)
        ax.set_ylabel("Best Fitness", fontsize=12)
        ax.set_title(f"Plateau Detection - {experiment_name.upper()} Run {i+1}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        output_path = output_dir / f"plateau_run_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

def plot_convergence(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot convergence curves for each run (one plot per run)."""
    for i, run in enumerate(runs):
        convergence = run["convergence"]
        evaluations = np.array(convergence["evaluations"])
        best_fitness = np.array(convergence["best_fitness"])
        current_fitness = np.array(convergence["current_fitness"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(evaluations, best_fitness, label="Best Fitness", color="blue")
        ax.plot(evaluations, current_fitness, label="Current Fitness", color="orange", alpha=0.7)
        
        ax.set_xlabel("Evaluation", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(f"Convergence Curve - {experiment_name.upper()} Run {i+1}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_path = output_dir / f"convergence_run_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

def plot_convergence_comparison(experiment_names: List[str], model: str, output_path: Path):
    """
    Plot convergence curves for multiple experiments (e.g., GA Standard vs Memetic) on the same plot.
    Uses only the latest run from each experiment.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name in experiment_names:
        runs = load_experiment_runs(exp_name)
        
        # Sort runs by directory name (which includes timestamp) and take the latest
        runs_sorted = sorted(runs, key=lambda r: r["run_dir"], reverse=True)
        latest_run = runs_sorted[0]
        
        evaluations = np.array(latest_run["convergence"]["evaluations"])
        best_fitness = np.array(latest_run["convergence"]["best_fitness"])
        
        # Extract algorithm name for cleaner label
        algo_name = exp_name.split("-", 1)[1] if "-" in exp_name else exp_name
        ax.plot(evaluations, best_fitness, linewidth=2, label=algo_name.upper())
    
    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Convergence Comparison: {model.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_models_comparison_by_optimizer(optimizer: str, output_path: Path):
    """
    Plot average best fitness across all models (dt, knn, cnn) for a specific optimizer.
    Shows mean convergence curve for each model without std deviation bands.
    """
    models = ["dt", "knn", "cnn"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        exp_name = f"{model}-{optimizer}"
        exp_dir = EXPERIMENT_ROOT / exp_name
        
        if not exp_dir.exists():
            print(f"Skipping {exp_name}: directory not found")
            continue
        
        try:
            runs = load_experiment_runs(exp_name)
            
            # Collect all runs' best fitness
            base_evaluations = np.array(runs[0]["convergence"]["evaluations"])
            all_fitnesses = []
            
            for run in runs:
                convergence = run["convergence"]
                evaluations = np.array(convergence["evaluations"])
                best_fitness = np.array(convergence["best_fitness"])
                
                # Interpolate if needed
                if len(evaluations) != len(base_evaluations) or not np.allclose(evaluations, base_evaluations):
                    best_fitness = np.interp(base_evaluations, evaluations, best_fitness)
                
                all_fitnesses.append(best_fitness)
            
            # Calculate mean fitness across all runs
            mean_fitness = np.mean(np.array(all_fitnesses), axis=0)
            
            # Plot mean line for this model
            ax.plot(base_evaluations, mean_fitness, linewidth=2, label=model.upper())
            
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Average Best Fitness", fontsize=12)
    ax.set_title(f"Model Comparison - {optimizer.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_experiments_comparison(model: str, output_path: Path):
    """
    Plot average best fitness for all available model-optimizer combinations for a specific model.
    Discovers all existing experiments dynamically and plots them.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Find all experiments for this model
    experiments_found = []
    for exp_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if exp_dir.is_dir() and exp_dir.name.startswith(f"{model}-"):
            experiments_found.append(exp_dir.name)
    
    if not experiments_found:
        print(f"No experiments found for model {model}")
        return
    
    print(f"Found {len(experiments_found)} experiments for {model}: {experiments_found}")
    
    for exp_name in experiments_found:
        try:
            runs = load_experiment_runs(exp_name)
            
            # Collect all runs' best fitness
            base_evaluations = np.array(runs[0]["convergence"]["evaluations"])
            all_fitnesses = []
            
            for run in runs:
                convergence = run["convergence"]
                evaluations = np.array(convergence["evaluations"])
                best_fitness = np.array(convergence["best_fitness"])
                
                # Interpolate if needed
                if len(evaluations) != len(base_evaluations) or not np.allclose(evaluations, base_evaluations):
                    best_fitness = np.interp(base_evaluations, evaluations, best_fitness)
                
                all_fitnesses.append(best_fitness)
            
            # Calculate mean fitness across all runs
            mean_fitness = np.mean(np.array(all_fitnesses), axis=0)
            
            # Extract optimizer name for label
            optimizer_name = exp_name.split("-", 1)[1] if "-" in exp_name else exp_name
            
            # Plot mean line for this experiment
            ax.plot(base_evaluations, mean_fitness, linewidth=2, label=optimizer_name.upper())
            
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Average Best Fitness", fontsize=12)
    ax.set_title(f"All Experiments Comparison - {model.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def _exp_smooth(values: np.ndarray, smoothing_slider: float = 0.9) -> np.ndarray:
    """Apply exponential smoothing to a sequence of values."""
    smoothed = np.empty_like(values, dtype=np.float32)
    last = values[0]
    for idx, point in enumerate(values):
        last = smoothing_slider * last + (1.0 - smoothing_slider) * point
        smoothed[idx] = last
    return smoothed


def plot_best_fitness_overlay(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot best fitness over evaluations showing mean and std deviation across runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all fitness values at each evaluation point
    # Use evaluation points from the first run (assuming they're all the same)
    base_evaluations = np.array(runs[0]["convergence"]["evaluations"])
    all_fitnesses = []
    
    for run in runs:
        convergence = run["convergence"]
        evaluations = np.array(convergence["evaluations"])
        best_fitness = np.array(convergence["best_fitness"])
        
        # If evaluations differ, interpolate to base grid
        if len(evaluations) != len(base_evaluations) or not np.allclose(evaluations, base_evaluations):
            best_fitness = np.interp(base_evaluations, evaluations, best_fitness)
        
        all_fitnesses.append(best_fitness)
    
    # Calculate mean and std at each evaluation point
    all_fitnesses = np.array(all_fitnesses)
    mean_fitness = np.mean(all_fitnesses, axis=0)
    std_fitness = np.std(all_fitnesses, axis=0)
    
    # Apply exponential smoothing to the mean and std
    mean_fitness_smoothed = _exp_smooth(mean_fitness, SMOOTHING_SLIDER_BEST_FITNESS)
    std_fitness_smoothed = _exp_smooth(std_fitness, SMOOTHING_SLIDER_BEST_FITNESS)
    
    # Plot shaded region for std deviation (using smoothed values)
    ax.fill_between(base_evaluations, mean_fitness_smoothed - std_fitness_smoothed, 
                    mean_fitness_smoothed + std_fitness_smoothed,
                    alpha=0.3, color='blue', label='±1 Std Dev')
    
    # Plot smoothed mean line
    ax.plot(base_evaluations, mean_fitness_smoothed, linewidth=2, color='blue', label='Mean (smoothed)')
    
    # Set y-axis from worst to best with padding
    min_fitness = np.min(mean_fitness_smoothed - std_fitness_smoothed)
    max_fitness = np.max(mean_fitness_smoothed + std_fitness_smoothed)
    fitness_range = max_fitness - min_fitness
    padding = fitness_range * 0.1  # 10% padding on both sides
    ax.set_ylim(bottom=min_fitness - padding, top=max_fitness + padding)
    
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Best Fitness Over Evaluations - {experiment_name.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "best_fitness_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_current_fitness_overlay(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot current fitness over evaluations showing mean and std deviation across runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all fitness values at each evaluation point
    # Use evaluation points from the first run (assuming they're all the same)
    base_evaluations = np.array(runs[0]["convergence"]["evaluations"])
    all_fitnesses = []
    
    for run in runs:
        convergence = run["convergence"]
        evaluations = np.array(convergence["evaluations"])
        current_fitness = np.array(convergence["current_fitness"])
        
        # If evaluations differ, interpolate to base grid
        if len(evaluations) != len(base_evaluations) or not np.allclose(evaluations, base_evaluations):
            current_fitness = np.interp(base_evaluations, evaluations, current_fitness)
        
        all_fitnesses.append(current_fitness)
    
    # Calculate mean and std at each evaluation point
    all_fitnesses = np.array(all_fitnesses)
    mean_fitness = np.mean(all_fitnesses, axis=0)
    std_fitness = np.std(all_fitnesses, axis=0)
    
    # Apply exponential smoothing to the mean and std
    mean_fitness_smoothed = _exp_smooth(mean_fitness, SMOOTHING_SLIDER_CURRENT_FITNESS)
    std_fitness_smoothed = _exp_smooth(std_fitness, SMOOTHING_SLIDER_CURRENT_FITNESS)
    
    # Plot shaded region for std deviation (using smoothed values)
    ax.fill_between(base_evaluations, mean_fitness_smoothed - std_fitness_smoothed, 
                    mean_fitness_smoothed + std_fitness_smoothed,
                    alpha=0.3, color='green', label='±1 Std Dev')
    
    # Plot smoothed mean line
    ax.plot(base_evaluations, mean_fitness_smoothed, linewidth=2, color='green', label='Mean (smoothed)')
    
    # Set y-axis from worst to best with padding
    min_fitness = np.min(mean_fitness_smoothed - std_fitness_smoothed)
    max_fitness = np.max(mean_fitness_smoothed + std_fitness_smoothed)
    fitness_range = max_fitness - min_fitness
    padding = fitness_range * 0.1  # 10% padding on both sides
    ax.set_ylim(bottom=min_fitness - padding, top=max_fitness + padding)
    
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Current Fitness", fontsize=12)
    ax.set_title(f"Current Fitness Over Evaluations - {experiment_name.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "current_fitness_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_boxplot(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot box plot of total time across runs."""
    times = [run["summary"]["total_time_sec"] for run in runs]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(times, vert=True, patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][0].set_alpha(0.7)
    
    # Add mean point
    mean_time = np.mean(times)
    ax.scatter([1], [mean_time], color="red", s=100, zorder=3, label=f"Mean: {mean_time:.2f}s")
    
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title(f"Execution Time Distribution - {experiment_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticklabels([experiment_name.upper()])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    
    # Add statistics text
    stats_text = f"N={len(times)}\nMedian: {np.median(times):.2f}s\nStd: {np.std(times):.2f}s"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "time_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_fitness_boxplot(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot box plot of final fitness values across runs."""
    final_fitnesses = [run["summary"]["final_fitness"] for run in runs]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(final_fitnesses, vert=True, patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][0].set_alpha(0.7)
    
    # Add mean point
    mean_fitness = np.mean(final_fitnesses)
    max_fitness = max(final_fitnesses)
    ax.scatter([1], [mean_fitness], color="red", s=100, zorder=3, label=f"Mean: {mean_fitness:.4f}")
    
    ax.set_ylabel("Final Fitness", fontsize=12)
    ax.set_title(f"Final Fitness Distribution - {experiment_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0, top=max_fitness * 1.1)  # 10% padding at top
    ax.set_xticklabels([experiment_name.upper()])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    
    # Add statistics text
    stats_text = f"N={len(final_fitnesses)}\nMedian: {np.median(final_fitnesses):.4f}\nStd: {np.std(final_fitnesses):.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "final_fitness_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_individual_metrics_multi_panel(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot all 6 individual metrics in a multi-panel figure showing convergence with mean ± std.
    
    Args:
        runs: List of runs loaded with load_experiment_runs(include_history=True)
        output_dir: Directory to save the plot
        experiment_name: Name of the experiment for the title
    """
    metrics_to_plot = list(METRIC_WEIGHTS.keys())
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Collect convergence data for this metric across all runs
        base_evaluations = None
        all_metric_values = []
        
        for run in runs:
            metric_convergence = extract_metric_convergence(run["history"], metric_name)
            evaluations = np.array(metric_convergence["evaluations"])
            best_metric = np.array(metric_convergence["best_metric"])
            
            if base_evaluations is None:
                base_evaluations = evaluations
            
            # Interpolate if needed
            if len(evaluations) != len(base_evaluations) or not np.allclose(evaluations, base_evaluations):
                best_metric = np.interp(base_evaluations, evaluations, best_metric)
            
            all_metric_values.append(best_metric)
        
        # Calculate mean and std
        all_metric_values = np.array(all_metric_values)
        mean_metric = np.mean(all_metric_values, axis=0)
        std_metric = np.std(all_metric_values, axis=0)
        
        # Apply exponential smoothing
        mean_metric_smoothed = _exp_smooth(mean_metric, SMOOTHING_SLIDER_INDIVIDUAL_METRICS)
        std_metric_smoothed = _exp_smooth(std_metric, SMOOTHING_SLIDER_INDIVIDUAL_METRICS)
        
        # Plot shaded region for std deviation
        ax.fill_between(base_evaluations, mean_metric_smoothed - std_metric_smoothed,
                        mean_metric_smoothed + std_metric_smoothed,
                        alpha=0.3, label='±1 Std Dev')
        
        # Plot mean line
        ax.plot(base_evaluations, mean_metric_smoothed, linewidth=2, label='Mean (smoothed)')
        
        # Set y-axis limits with padding
        min_val = np.min(mean_metric_smoothed - std_metric_smoothed)
        max_val = np.max(mean_metric_smoothed + std_metric_smoothed)
        val_range = max_val - min_val
        padding = val_range * 0.1 if val_range > 0 else 0.1
        ax.set_ylim(bottom=max(0, min_val - padding), top=min(1, max_val + padding))
        
        # Add weight annotation
        weight = METRIC_WEIGHTS[metric_name]
        metric_label = metric_name.replace("_", " ").title()
        ax.set_title(f"{metric_label} (weight: {weight:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    
    plt.suptitle(f"Individual Metrics Convergence - {experiment_name.upper()}", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / "individual_metrics_multi_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_metric_weight_impact(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str):
    """Plot weighted contributions of each metric to composite fitness over evaluations.
    
    This visualization shows how much each metric contributes to the final composite fitness
    score, helping understand if the optimization is balanced across all metrics or dominated
    by specific ones.
    
    Args:
        runs: List of runs loaded with load_experiment_runs(include_history=True)
        output_dir: Directory to save the plot
        experiment_name: Name of the experiment for the title
    """
    metrics_to_plot = list(METRIC_WEIGHTS.keys())
    
    # Collect convergence data for all metrics across all runs
    base_evaluations = None
    all_weighted_contributions = {metric: [] for metric in metrics_to_plot}
    
    for run in runs:
        run_weighted = {metric: [] for metric in metrics_to_plot}
        
        for metric_name in metrics_to_plot:
            metric_convergence = extract_metric_convergence(run["history"], metric_name)
            evaluations = np.array(metric_convergence["evaluations"])
            best_metric = np.array(metric_convergence["best_metric"])
            
            if base_evaluations is None:
                base_evaluations = evaluations
            
            # Interpolate if needed
            if len(evaluations) != len(base_evaluations) or not np.allclose(evaluations, base_evaluations):
                best_metric = np.interp(base_evaluations, evaluations, best_metric)
            
            # Calculate weighted contribution
            weighted = best_metric * METRIC_WEIGHTS[metric_name]
            run_weighted[metric_name] = weighted
        
        # Store this run's weighted contributions
        for metric_name in metrics_to_plot:
            all_weighted_contributions[metric_name].append(run_weighted[metric_name])
    
    # Calculate mean weighted contributions across runs
    mean_weighted = {}
    for metric_name in metrics_to_plot:
        mean_weighted[metric_name] = np.mean(np.array(all_weighted_contributions[metric_name]), axis=0)
    
    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for stacking
    bottom = np.zeros(len(base_evaluations))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_to_plot)))
    
    for idx, metric_name in enumerate(metrics_to_plot):
        metric_label = metric_name.replace("_", " ").title()
        weight = METRIC_WEIGHTS[metric_name]
        values = mean_weighted[metric_name]
        
        ax.fill_between(base_evaluations, bottom, bottom + values,
                       label=f"{metric_label} (w={weight:.2f})",
                       alpha=0.8, color=colors[idx])
        bottom += values
    
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Weighted Contribution to Composite Fitness", fontsize=12)
    ax.set_title(f"Metric Weight Impact - {experiment_name.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1.01, 1))
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / "metric_weight_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def diagnose_pso_hyperparameters(runs: List[Dict[str, Any]], experiment_name: str):
    """Diagnose PSO hyperparameter issues from convergence patterns."""
    if "pso" not in experiment_name.lower():
        print("Note: This diagnostic is for PSO experiments only.")
        return
    
    print("\n" + "=" * 80)
    print("PSO HYPERPARAMETER DIAGNOSTIC")
    print("=" * 80)
    
    for i, run in enumerate(runs):
        convergence = run["convergence"]
        evaluations = np.array(convergence["evaluations"])
        best_fitness = np.array(convergence["best_fitness"])
        current_fitness = np.array(convergence["current_fitness"])
        
        total_evals = len(evaluations)
        early_portion = int(total_evals * 0.2)  # First 20%
        late_portion = int(total_evals * 0.8)    # Last 20%
        
        # Calculate metrics
        early_improvement = best_fitness[early_portion] - best_fitness[0]
        late_improvement = best_fitness[-1] - best_fitness[late_portion]
        total_improvement = best_fitness[-1] - best_fitness[0]
        
        # Variance in current fitness (indicates exploration)
        fitness_variance = np.std(current_fitness)
        
        # Convergence rate (how quickly it plateaus)
        # Find when 90% of improvement is achieved
        target = best_fitness[0] + 0.9 * total_improvement
        convergence_point = np.where(best_fitness >= target)[0]
        convergence_ratio = convergence_point[0] / total_evals if len(convergence_point) > 0 else 1.0
        
        print(f"\n--- Run {i+1} ---")
        print(f"Total improvement: {total_improvement:.4f}")
        print(f"Early improvement (first 20%): {early_improvement:.4f} ({early_improvement/total_improvement*100:.1f}% of total)")
        print(f"Late improvement (last 20%): {late_improvement:.4f} ({late_improvement/total_improvement*100:.1f}% of total)")
        print(f"Fitness variance: {fitness_variance:.4f}")
        print(f"Convergence point: {convergence_ratio*100:.1f}% of evaluations")
        
        # Diagnose issues
        recommendations = []
        
        if convergence_ratio < 0.3:
            recommendations.append("⚠️  PREMATURE CONVERGENCE: Best fitness plateaus early")
            recommendations.append("   → Try: Lower c2 (e.g., 1.0-1.2) to reduce swarm attraction")
            recommendations.append("   → Try: Increase w (e.g., 0.7-0.9) for more exploration")
            recommendations.append("   → Try: Increase c1 (e.g., 2.0-2.5) for more personal exploration")
        
        if late_improvement / total_improvement > 0.3 and total_improvement > 0.01:
            recommendations.append("⚠️  SLOW CONVERGENCE: Still improving at the end")
            recommendations.append("   → Try: Increase c2 (e.g., 2.0-2.5) for faster swarm convergence")
            recommendations.append("   → Try: Decrease w (e.g., 0.3-0.5) for more exploitation")
        
        if fitness_variance > np.mean(current_fitness) * 0.3:
            recommendations.append("⚠️  HIGH VARIANCE: Particles exploring widely")
            recommendations.append("   → Try: Decrease w (e.g., 0.3-0.5) to reduce oscillation")
            recommendations.append("   → Try: Increase c2 (e.g., 2.0-2.5) for more coordination")
        
        if fitness_variance < np.mean(current_fitness) * 0.05 and convergence_ratio < 0.5:
            recommendations.append("⚠️  LOW DIVERSITY: Particles clustering too quickly")
            recommendations.append("   → Try: Increase w (e.g., 0.7-0.9) for more exploration")
            recommendations.append("   → Try: Increase c1 (e.g., 2.0-2.5) for more personal exploration")
            recommendations.append("   → Try: Decrease c2 (e.g., 1.0-1.2) to reduce swarm pressure")
        
        if early_improvement / total_improvement < 0.2 and total_improvement > 0.01:
            recommendations.append("⚠️  SLOW START: Little improvement in early stages")
            recommendations.append("   → Try: Increase c1 (e.g., 2.0-2.5) for better initial exploration")
            recommendations.append("   → Try: Increase w (e.g., 0.7-0.9) for more momentum")
        
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(rec)
        else:
            print("\n✓ No obvious issues detected. Parameters appear well-tuned.")
    
    print("\n" + "=" * 80)


def main(experiment: str = None, diagnose_pso: bool | str = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate visualization figures."
    )
    if experiment is None:
        parser.add_argument(
            "--experiment",
            type=str,
            required=True,
            help="Experiment name: 'cnn-rs', 'dt-ga', 'knn-pso')",
        )
    if diagnose_pso is None:
        parser.add_argument(
            "--diagnose-pso",
            action="store_true",
            help="Run PSO hyperparameter diagnostics",
        )
    
    args = parser.parse_args()
    
    # Load experiment data
    runs = load_experiment_runs(args.experiment)
    
    # Create output directory
    output_dir = FIGURES_ROOT / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_best_fitness_overlay(runs, output_dir, args.experiment)
    plot_current_fitness_overlay(runs, output_dir, args.experiment)
    plot_time_boxplot(runs, output_dir, args.experiment)
    plot_final_fitness_boxplot(runs, output_dir, args.experiment)
    # Plot convergence curves for each run
    plot_convergence(runs, output_dir, args.experiment)
    # Plot plateau detection for each run
    plot_plateaus(runs, output_dir, args.experiment, threshold=None)
    
    # Generate individual metrics analysis plots (requires history.json)
    print("\nGenerating individual metrics analysis...")
    try:
        runs_with_history = load_experiment_runs(args.experiment, include_history=True)
        plot_individual_metrics_multi_panel(runs_with_history, output_dir, args.experiment)
        plot_metric_weight_impact(runs_with_history, output_dir, args.experiment)
    except Exception as e:
        print(f"Warning: Could not generate individual metrics plots: {e}")
    # Plot comparison specifically for GA Standard vs Memetic
    if "ga" in args.experiment.lower():
        model = args.experiment.split("-")[0]
        comparison_experiments = [f"{model}-ga-standard", f"{model}-ga-memetic"]
        comparison_dir = output_dir.resolve().parent / f"GA_Comparison_{model}"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_output_path = comparison_dir / "GA_convergence_comparison.png"
        
        # Only create comparison if both experiments exist
        all_exist = all((EXPERIMENT_ROOT / exp).exists() for exp in comparison_experiments)
        if all_exist and not os.path.exists(comparison_output_path):
            plot_convergence_comparison(comparison_experiments, model, comparison_output_path)
        elif not all_exist:
            print(f"\nSkipping comparison plot: Not all experiments exist yet.")
            print(f"  Required: {comparison_experiments}")
            print(f"  Available: {[d.name for d in EXPERIMENT_ROOT.iterdir() if d.is_dir()]}")
    
    # Generate cross-model and cross-optimizer comparison plots
    print("\nGenerating cross-comparison plots...")
    
    # Extract model and optimizer from experiment name
    parts = args.experiment.split("-")
    if len(parts) >= 2:
        current_model = parts[0]
        current_optimizer = "-".join(parts[1:])
        
        # Plot models comparison for this optimizer
        models_comparison_dir = FIGURES_ROOT / f"Models_Comparison_{current_optimizer}"
        models_comparison_dir.mkdir(parents=True, exist_ok=True)
        models_comparison_path = models_comparison_dir / f"models_comparison_{current_optimizer}.png"
        
        if not os.path.exists(models_comparison_path):
            try:
                plot_models_comparison_by_optimizer(current_optimizer, models_comparison_path)
            except Exception as e:
                print(f"Could not generate models comparison: {e}")
        
        # Plot optimizers comparison for this model
        optimizers_comparison_dir = FIGURES_ROOT / f"Optimizers_Comparison_{current_model}"
        optimizers_comparison_dir.mkdir(parents=True, exist_ok=True)
        all_experiments_comparison_path = optimizers_comparison_dir / f"all_experiments_{current_model}.png"
        
        plot_all_experiments_comparison(current_model, all_experiments_comparison_path)
    
    # Run PSO diagnostics if requested
    if args.diagnose_pso:
        diagnose_pso_hyperparameters(runs, args.experiment)
    
    print(f"\n✓ Analysis complete! Figures saved to: {output_dir}")


if __name__ == "__main__":
    exit(main())

