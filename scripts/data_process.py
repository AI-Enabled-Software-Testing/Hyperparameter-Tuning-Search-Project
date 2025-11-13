import argparse
import shutil
from pathlib import Path
from typing import List
from typing import Callable

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import img_as_float


def build_transform_pipeline():
    def transform_one(image) -> np.ndarray:
        arr = np.asarray(image)
        arr = rgb2gray(arr) if arr.ndim == 3 else arr
        arr = img_as_float(arr)
        arr = resize(arr, (32, 32), anti_aliasing=True, preserve_range=True)
        return arr

    def transform_batch(images: List) -> List:
        return [transform_one(img) for img in images]

    return transform_batch


def process_dataset(
    src_dir: Path, dst_dir: Path, transform_batch: Callable[[List], List]
) -> None:
    ds_dict = load_from_disk(str(src_dir))
    assert isinstance(ds_dict, DatasetDict)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # cifar10 uses 'img', we standardize to 'image'
    for split, ds in ds_dict.items():
        ds: Dataset = ds  # type annotation for ds
        if "img" in ds.features:
            ds = ds.rename_column("img", "image")
        ds_dict[split] = ds

    for split, ds in ds_dict.items():
        ds_dict[split] = ds.map(
            lambda batch: {"image": transform_batch(batch["image"])},
            batched=True,
            num_proc=4,
        )

    ds_dict.save_to_disk(str(dst_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing postprocessed datasets if present",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_root = repo_root / ".cache" / "base_datasets"
    out_root = repo_root / ".cache" / "processed_datasets"
    out_root.mkdir(parents=True, exist_ok=True)

    datasets_to_process = ["cifar10"]
    transform_batch = build_transform_pipeline()
    for name in datasets_to_process:
        src = base_root / name
        dst = out_root / name
        if dst.exists() and args.force:
            shutil.rmtree(dst, ignore_errors=True)
        print(f"Processing {name}: {src} -> {dst}")
        process_dataset(src, dst, transform_batch)
    print("Done.")


if __name__ == "__main__":
    main()
