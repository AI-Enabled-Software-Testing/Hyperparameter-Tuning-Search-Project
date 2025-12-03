import subprocess
from pathlib import Path
import os
import sys
import multiprocessing
import psutil
import gc

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

def run_experiment(args):
    model, optimizer, venv_python_path, venv_path = args
    cmd = [
        str(venv_python_path) if venv_path else "python",
        "./scripts/run_experiment.py",
        "--model",
        model,
        "--optimizer",
        optimizer
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        return False
    return True

def venv_setup():
    # Check and Use a Venv if Available
    # Find a venv directory containing 'venv' in its name
    venv_dirs = [d for d in REPO_ROOT.iterdir() if d.is_dir() and "venv" in d.name.lower()]
    if venv_dirs:
        venv_path = venv_dirs[0] / "Scripts" / "activate"
        print(f"Found venv: {venv_dirs[0]}")
    else:
        venv_path = None
        print("No venv directory found.")
    if venv_path:
        activate_command = str(venv_path)
        if os.name == 'nt':  # Windows
            command = f"{activate_command} && python"
        else:  # Unix or MacOS
            command = f"source {activate_command} && python"
        print(f"Using venv activation command: {command}")

    venv_python_path = venv_dirs[0] / "Scripts" / "python.exe" if os.name == 'nt' else venv_dirs[0] / "bin" / "python"
    return venv_python_path, venv_path

def analyze_experiment(model, optimizer, venv_python_path, venv_path):
    experiment_name = f"{model}_{optimizer}_experiment"
    cmd = [
        str(venv_python_path) if venv_path else "python",
        "./scripts/analyze_experiment.py",
        "--experiment-name",
        experiment_name
    ]
    print(f"Analyzing by script: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        return False
    return True

def get_available_memory_gb():
    mem = psutil.virtual_memory()
    avaialable_memory = mem.available or mem.total or 0
    return avaialable_memory / (1024 ** 3) # in gb

def allocate_job(memory_per_job_gb):
    available_memory_gb = get_available_memory_gb()
    max_jobs = round(available_memory_gb // memory_per_job_gb)
    cpu_count = multiprocessing.cpu_count()
    return min(max_jobs, cpu_count) if max_jobs and max_jobs > 0 else 1

if __name__ == "__main__":
    venv_python_path, venv_path = venv_setup()

    models = ["dt", "knn", "cnn"]
    optimizers = ["ga-standard", "ga-memetic"]

    run_jobs = []
    # Experiment Runner
    for model in models:
        for optimizer in optimizers:
            run_jobs.append((model, optimizer, venv_python_path, venv_path))
        
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
            analyze_jobs.append((model, optimizer, venv_python_path, venv_path))
        
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
    del venv_python_path, venv_path, models, optimizers, EXPERIMENT_ROOT, FIGURES_ROOT
    gc.collect()