import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
BASE_DATASETS_DIR = ROOT / ".cache" / "base_datasets"
PROCESSED_DATASETS_DIR = ROOT / ".cache" / "processed_datasets"
BASE_CIFAR10_DIR = BASE_DATASETS_DIR / "cifar10"
PROCESSED_CIFAR10_DIR = PROCESSED_DATASETS_DIR / "cifar10"
sys.path.append(str(ROOT))
sys.path.append(str(SCRIPTS_DIR))

# Import Scripts
from scripts.data_download import download_dataset
from scripts.data_process import main as preprocess
from scripts.run_experiment import main as run_experiment
from scripts.run_final_training import main as run_final_training

def main():
    parser = argparse.ArgumentParser(description="Main Pipeline: Run Experiments and Final Training")
    parser.add_argument("--force", default=False, help="Force re-download of datasets") # DEFAULT: do not re-download if already exists
    parser.add_argument("--model", type=str, default=None, help="Optional Model type to run experiments on. Otherwise runs all models.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optional Optimizer type to run experiments on. Otherwise runs all optimizers.")
    args = parser.parse_args()
    if not BASE_DATASETS_DIR.exists() or args.force:
        download_dataset(
            repo_id="uoft-cs/cifar10",
            destination=BASE_CIFAR10_DIR,
            force=args.force,
        )
    if not PROCESSED_CIFAR10_DIR.exists() or args.force:
        preprocess()

    # Remove '--force' and its value from sys.argv if present
    for item in ['--force', str(args.force)]:
        if item in sys.argv:
            sys.argv.remove(item)

    models = [args.model] if args.model else ['dt', 'knn', 'cnn']
    
    for modelName in models:
        sys.argv += ['--model', modelName]
        if args.optimizer:
            sys.argv += ['--optimizer', args.optimizer]

        run_experiment()

        # Clean up arguments
        if '--model' in sys.argv:
            sys.argv.remove('--model')
        if modelName in sys.argv:
            sys.argv.remove(modelName)
        if args.optimizer and '--optimizer' in sys.argv:
            sys.argv.remove('--optimizer')
        if args.optimizer and args.optimizer in sys.argv:
            sys.argv.remove(args.optimizer)
    
    return run_final_training()


if __name__ == "__main__":
    exit(main())
