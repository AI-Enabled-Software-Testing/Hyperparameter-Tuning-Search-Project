import subprocess
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

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

models = ["dt", "knn", "cnn"]
optimizers = ["ga-standard", "ga-memetic"]

# Experiment Runner
for model in models:
    for optimizer in optimizers:
        experiment_name = f"{model}_{optimizer}_experiment"
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
            exit(-1)

# Analysis Runner
for model in models:
    for optimizer in optimizers:
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