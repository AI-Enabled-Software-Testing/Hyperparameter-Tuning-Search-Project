import argparse
import shutil
from pathlib import Path
from datasets import load_dataset


def download_dataset(repo_id: str, destination: Path, force: bool) -> None:
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo_id)
    ds.save_to_disk(str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if they already exist",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    datasets_root = repo_root / ".cache" / "base_datasets"

    # CIFAR-10
    cifar10_dir = datasets_root / "cifar10"
    print(f"Downloading CIFAR-10 (uoft-cs/cifar10) to {cifar10_dir}...")
    download_dataset("uoft-cs/cifar10", cifar10_dir, args.force)

    # MNIST
    mnist_dir = datasets_root / "mnist"
    print(f"Downloading MNIST (ylecun/mnist) to {mnist_dir}...")
    download_dataset("ylecun/mnist", mnist_dir, args.force)
