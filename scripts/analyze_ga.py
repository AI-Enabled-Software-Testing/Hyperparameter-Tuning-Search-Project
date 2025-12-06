import subprocess
from pathlib import Path
import os
import sys
import gc
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# Import codefiles
from run_experiment import run_experiment, EVALUATIONS_PER_RUN
from analyze_experiment import main

def ga_comp_run_experiment(model, optimizer):
    try:
        if optimizer.startswith("ga"): # Only try GA optimizers
            run_experiment(
                model_key=model,
                optimizer_name=optimizer,
                num_runs=1,
                n_jobs=1,
                base_seed=42,
                evaluations=EVALUATIONS_PER_RUN
            )
    except Exception as e:
        print(f"Error running experiment for {model} with {optimizer}: {e}")
        return False
    return True

def ga_comp_analyze_experiment(model, optimizer):
    experiment_name = f"{model}_{optimizer}_experiment"
    try:
        if optimizer.startswith("ga"): # Only try GA optimizers
            main(
                experiment=experiment_name,
                diagnose_pso=False
            )
    except Exception as e:
        print(f"Error analyzing experiment for {model} with {optimizer}: {e}")
        return False
    return True

def experiment_exists(modelName: str, optimizerName: str) -> bool:
    """Check if experiment results already exist."""
    experiment_name = f"{modelName}_{optimizerName}_experiment"
    experiment_file = REPO_ROOT / ".cache" / "experiment" / f"{experiment_name}.json"
    return experiment_file.exists()

if __name__ == "__main__":
    # Dynamically adjust models based on available hardware
    models = ["dt", "knn"]
    if torch.cuda.is_available():
        models.append("cnn")
        print("GPU detected: Including CNN model")
    else:
        print("CPU-only detected: Excluding CNN model for faster execution")
    
    optimizers = ["ga-standard", "ga-memetic"]

    failed_runs = []
    failed_analyses = []
    completed = []
    skipped = []
    
    try:
        # Run each experiment sequentially, analyze immediately after each successful run
        for model in models:
            for optimizer in optimizers:
                print(f"\n{'='*80}")
                print(f"Processing: {model.upper()} with {optimizer.upper()}")
                print(f"{'='*80}\n")\
                
                # Check if experiment already exists
                if experiment_exists(model, optimizer):
                    print(f"Experiment {model}-{optimizer} already exists. Skipping...")
                    skipped.append(f"{model}-{optimizer}")
                    continue
                
                # Run experiment
                run_success = ga_comp_run_experiment((model, optimizer))
                
                if run_success:
                    # Analyze immediately after successful run
                    analyze_success = ga_comp_analyze_experiment(model, optimizer)
                    if not analyze_success:
                        print(f"WARNING: Analysis failed for {model}-{optimizer}")
                        failed_analyses.append(f"{model}-{optimizer}")
                else:
                    print(f"ERROR: Experiment run failed for {model}-{optimizer}")
                    failed_runs.append(f"{model}-{optimizer}")
                
                # Small cleanup after each iteration
                gc.collect()
    except KeyboardInterrupt:
        print("\n{'='*80}")
        print("Process interrupted by user.")
        print(f"{'='*80}\n")
        
        # Report progress instead of discarding
        print(f"Completed: {len(completed)}/{len(models) * len(optimizers)} experiments")
        if completed:
            print("Successfully completed:")
            for exp in completed:
                print(f"  ✓ {exp}")
        if skipped:
            print("Skipped (already existed):")
            for exp in skipped:
                print(f"  ↷ {exp}")
        if failed_runs:
            print("Failed experiment runs:")
            for exp in failed_runs:
                print(f"  ✗ {exp}")
        if failed_analyses:
            print("Failed analyses:")
            for exp in failed_analyses:
                print(f"  ✗ {exp}")
        
        print(f"\n{'='*80}")
        print("Partial results preserved. Re-run the script to resume.")
        print(f"{'='*80}\n")
        exit(130)  # Standard exit code for SIGINT


    # Report any failures
    if failed_runs or failed_analyses:
        print(f"\n{'='*80}")
        if failed_runs:
            print("Failed experiment runs:")
            for exp in failed_runs:
                print(f"  - {exp}")
        if failed_analyses:
            print("Failed analyses:")
            for exp in failed_analyses:
                print(f"  - {exp}")
        print(f"{'='*80}\n")
        exit(-1)

    # Check Paths
    EXPERIMENT_ROOT = REPO_ROOT / ".cache" / "experiment"
    FIGURES_ROOT = REPO_ROOT / ".cache" / "experiment_figures"

    print("All experiments and analyses completed successfully.")
    if EXPERIMENT_ROOT.exists() and FIGURES_ROOT.exists():
        print(f"Check Exported Plots in '{EXPERIMENT_ROOT}' and '{FIGURES_ROOT}' directories.")
    else:
        print("Experiment or Figures directories do not exist. Consider re-running the scripts.")
        exit(-1)
    # Final cleanup
    del models, optimizers, EXPERIMENT_ROOT, FIGURES_ROOT
    gc.collect()