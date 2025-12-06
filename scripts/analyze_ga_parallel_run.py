import subprocess
from pathlib import Path
import os
import sys
import multiprocessing
import psutil
import gc
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# Import codefiles
from run_experiment import run_experiment, EVALUATIONS_PER_RUN
from analyze_experiment import main
import argparse

def analyze_experiment(experiment_name) -> None:
    main(experiment=experiment_name, diagnose_pso=False)

def ga_comp_run_experiment(model, optimizer):
    runs = n_jobs = 1
    seed = 42
    evaluations = EVALUATIONS_PER_RUN

    try:
        if optimizer.startswith("ga"): # Only try GA optimizers
            run_experiment(
                model_key=model,
                optimizer_name=optimizer,
                num_runs=runs,
                n_jobs=n_jobs,
                base_seed=seed,
                evaluations=evaluations
            )
    except Exception as e:
        print(f"Error running experiment for {model} with {optimizer}: {e}")
        return False
    return True

def ga_comp_analyze_experiment(model, optimizer):
    experiment_name = f"{model}_{optimizer}_experiment"

    try:
        if optimizer.startswith("ga"): # Only try GA optimizers
            analyze_experiment(
                experiment_name=experiment_name
            )
    except Exception as e:
        print(f"Error analyzing experiment for {model} with {optimizer}: {e}")
        return False
    return True

def get_available_memory_gb():
    mem = psutil.virtual_memory()
    # Use available memory (includes reclaimable cache/buffers)
    # Apply 80% safety margin to leave headroom for OS
    available_memory = mem.available * 0.8
    return available_memory / (1024 ** 3)  # in gb

def allocate_job(memory_per_job_gb):
    available_memory_gb = get_available_memory_gb()
    max_jobs = int(available_memory_gb // memory_per_job_gb)
    cpu_count = multiprocessing.cpu_count()
    # Ensure at least 1 job, but don't exceed CPU count
    return max(1, min(max_jobs, cpu_count))

if __name__ == "__main__":
    # Dynamically adjust models based on available hardware
    models = ["dt", "knn"]
    if torch.cuda.is_available():
        models.append("cnn")
        print("GPU detected: Including CNN model")
    else:
        print("CPU-only detected: Excluding CNN model for faster execution")
    
    optimizers = ["ga-standard", "ga-memetic"]

    run_jobs = []
    # Experiment Runner
    for model in models:
        for optimizer in optimizers:
            run_jobs.append((model, optimizer))
        
    with multiprocessing.Pool(processes=allocate_job(1)) as pool:
        results = pool.map(run_experiment, run_jobs)

    if not all(results):
        print("Some experiments failed.")
        exit(-1)

    # Clean up experiment run variables
    del run_jobs, results, pool
    gc.collect()

    # Analyze Experiments
    analyze_jobs = []
    for model in models:
        for optimizer in optimizers:
            analyze_jobs.append((model, optimizer))
        
    with multiprocessing.Pool(processes=allocate_job(1)) as pool:
        analyze_results = pool.map(analyze_experiment, analyze_jobs)

    if not all(analyze_results):
        print("Some analyses failed.")
        exit(-1)

    # Clean up analysis variables
    del analyze_jobs, analyze_results, pool
    gc.collect()

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