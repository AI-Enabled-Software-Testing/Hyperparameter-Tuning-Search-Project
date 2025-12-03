import subprocess
from pathlib import Path
import os
import sys
import gc
import shutil

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
        optimizer,
        "--evaluations",
        str(20), # LIMIT BUDGET: much less than 50 for quick experiment (comparison)
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

def experiment_exists(modelName: str, optimizerName: str) -> bool:
    """Check if experiment results already exist."""
    experiment_name = f"{modelName}_{optimizerName}_experiment"
    experiment_file = REPO_ROOT / ".cache" / "experiment" / f"{experiment_name}.json"
    return experiment_file.exists()

if __name__ == "__main__":
    venv_python_path, venv_path = venv_setup()

    models = ["dt", "knn", "cnn"]
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
                run_success = run_experiment((model, optimizer, venv_python_path, venv_path))
                
                if run_success:
                    # Analyze immediately after successful run
                    analyze_success = analyze_experiment(model, optimizer, venv_python_path, venv_path)
                    
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
    del venv_python_path, venv_path, models, optimizers, EXPERIMENT_ROOT, FIGURES_ROOT
    gc.collect()