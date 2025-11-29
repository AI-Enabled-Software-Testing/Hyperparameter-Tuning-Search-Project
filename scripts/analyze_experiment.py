"""Analyze experiment results and generate visualization figures."""

import argparse
import json
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


def load_experiment_runs(experiment_name: str) -> List[Dict[str, Any]]:
    """Load all run data for an experiment."""
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
        
        convergence_path = run_dir / "convergence.json"
        summary_path = run_dir / "summary.json"
        
        if not convergence_path.exists() or not summary_path.exists():
            print(f"Warning: Skipping incomplete run {run_dir.name}")
            continue
        
        with open(convergence_path) as f:
            convergence = json.load(f)
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        runs.append({
            "run_dir": run_dir.name,
            "convergence": convergence,
            "summary": summary,
        })
    
    if not runs:
        raise ValueError(f"No valid runs found in {experiment_dir}")
    
    print(f"Loaded {len(runs)} runs from {experiment_name}")
    return runs

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
    """
    for exp_name in experiment_names:
        runs = load_experiment_runs(f"{model}-{exp_name}")
        for run in runs:
            evaluations = np.array(run["convergence"]["evaluations"])
            best_fitness = np.array(run["convergence"]["best_fitness"])
            plt.plot(evaluations, best_fitness, label=f"{exp_name} {run['run_dir']}")
    plt.xlabel("Evaluations")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence: {model.upper()} - " + " vs ".join(experiment_names))
    plt.legend()
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate visualization figures."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., 'cnn-rs', 'dt-ga', 'knn-pso')",
    )
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
    # Plot comparison specifically for GA Standard vs Memetic
    if "ga" in args.experiment.lower():
        model = args.experiment.split("-")[0]
        comparison_experiments = [f"{model}-ga-standard", f"{model}-ga-memetic"]
        comparison_output_path = output_dir / "convergence_comparison.png"
        plot_convergence_comparison(comparison_experiments, model, comparison_output_path)
    
    # Run PSO diagnostics if requested
    if args.diagnose_pso:
        diagnose_pso_hyperparameters(runs, args.experiment)
    
    print(f"\n✓ Analysis complete! Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

