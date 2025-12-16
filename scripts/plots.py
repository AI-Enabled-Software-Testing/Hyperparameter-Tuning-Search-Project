from matplotlib import pyplot as plt
from typing import Callable, List, Dict, Any
from pathlib import Path
import numpy as np
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / ".cache" / "experiment"
FIGURES_ROOT = REPO_ROOT / ".cache" / "experiment_figures"

sys.path.insert(0, str(REPO_ROOT))

from utils import _exp_smooth
from constants import (
    SMOOTHING_SLIDER_BEST_FITNESS,
    SMOOTHING_SLIDER_CURRENT_FITNESS,
    SMOOTHING_SLIDER_INDIVIDUAL_METRICS,
    METRIC_WEIGHTS,
)

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

def plot_convergence_comparison(experiment_names: List[str], model: str, output_path: Path, load_experiment_runs_fn: Callable):
    """
    Plot convergence curves for multiple experiments (e.g., GA Standard vs Memetic) on the same plot.
    Uses only the latest run from each experiment.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name in experiment_names:
        runs = load_experiment_runs_fn(exp_name)
        
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

def plot_models_comparison_by_optimizer(optimizer: str, output_path: Path, load_experiment_runs_fn: Callable):
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
            runs = load_experiment_runs_fn(exp_name)
            
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

def plot_all_experiments_comparison(model: str, output_path: Path, load_experiment_runs_fn: Callable):
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
            runs = load_experiment_runs_fn(exp_name)
            
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

def plot_individual_metrics_multi_panel(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str, extract_metric_convergence_fn: Callable):
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
            metric_convergence = extract_metric_convergence_fn(run["history"], metric_name)
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

def plot_metric_weight_impact(runs: List[Dict[str, Any]], output_dir: Path, experiment_name: str, extract_metric_convergence_fn: Callable):
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
            metric_convergence = extract_metric_convergence_fn(run["history"], metric_name)
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