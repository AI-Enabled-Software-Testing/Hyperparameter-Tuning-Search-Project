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

from plots import *


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
        plot_individual_metrics_multi_panel(runs_with_history, output_dir, args.experiment, extract_metric_convergence)
        plot_metric_weight_impact(runs_with_history, output_dir, args.experiment, extract_metric_convergence)
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
            plot_convergence_comparison(comparison_experiments, model, comparison_output_path, load_experiment_runs)
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
                plot_models_comparison_by_optimizer(current_optimizer, models_comparison_path, load_experiment_runs)
            except Exception as e:
                print(f"Could not generate models comparison: {e}")
        
        # Plot optimizers comparison for this model
        optimizers_comparison_dir = FIGURES_ROOT / f"Optimizers_Comparison_{current_model}"
        optimizers_comparison_dir.mkdir(parents=True, exist_ok=True)
        all_experiments_comparison_path = optimizers_comparison_dir / f"all_experiments_{current_model}.png"
        
        plot_all_experiments_comparison(current_model, all_experiments_comparison_path, load_experiment_runs)
    
    # Run PSO diagnostics if requested
    if args.diagnose_pso:
        diagnose_pso_hyperparameters(runs, args.experiment)
    
    print(f"\n✓ Analysis complete! Figures saved to: {output_dir}")


if __name__ == "__main__":
    exit(main())

