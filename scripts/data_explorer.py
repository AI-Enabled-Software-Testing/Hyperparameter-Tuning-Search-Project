import argparse
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from PIL import Image
import uvicorn
from typing import Literal


# Global variables to store dataset info
base_datasets_info: Dict[str, Any] = {}
processed_datasets_info: Dict[str, Any] = {}


def make_serializable(obj):
    """Recursively convert non-JSON serializable objects to serializable types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def pil_to_np(image_pil):
    """Convert PIL Image to numpy array."""
    if isinstance(image_pil, Image.Image):
        return np.array(image_pil)
    return image_pil


def load_dataset_info(dataset_path: Path) -> Dict[str, Any]:
    ds_dict: DatasetDict = load_from_disk(str(dataset_path))
    info = {
        "name": dataset_path.name,
        "splits": list(ds_dict.keys()),
        "total_samples": sum(len(ds) for ds in ds_dict.values()),
        "features": {},
    }

    first_split_name = list(ds_dict.keys())[0]
    ds = ds_dict[first_split_name]

    info["features"] = {name: str(feature) for name, feature in ds.features.items()}

    image_col = None
    if "image" in ds.features:
        image_col = "image"
    elif "img" in ds.features:
        image_col = "img"
    assert image_col is not None, "No image column found"

    info["dimensions"] = list(pil_to_np(ds[0][image_col]).shape)

    # Sample min and max over first up to 100 samples
    sample_images = [pil_to_np(ds[i][image_col]) for i in range(min(100, len(ds)))]
    sample_images_np = np.stack(sample_images)
    info["min_value"] = make_serializable(np.min(sample_images_np))
    info["max_value"] = make_serializable(np.max(sample_images_np))
    return info


def image_to_base64(image_data) -> str:
    arr = pil_to_np(image_data)

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
                np.uint8
            )
        pil_image = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
                np.uint8
            )
        pil_image = Image.fromarray(arr, mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load dataset information on startup and cleanup on shutdown."""
    global base_datasets_info, processed_datasets_info

    repo_root = Path(__file__).resolve().parents[1]

    base_root = repo_root / ".cache" / "base_datasets"
    assert base_root.exists(), "Base datasets not found"
    for dataset_dir in base_root.iterdir():
        if dataset_dir.is_dir():
            base_datasets_info[dataset_dir.name] = load_dataset_info(dataset_dir)

    processed_root = repo_root / ".cache" / "processed_datasets"
    assert processed_root.exists(), "Processed datasets not found"
    for dataset_dir in processed_root.iterdir():
        if dataset_dir.is_dir():
            processed_datasets_info[dataset_dir.name] = load_dataset_info(dataset_dir)

    yield

    base_datasets_info.clear()
    processed_datasets_info.clear()


app = FastAPI(
    title="Dataset Explorer",
    description="Explore base and processed datasets",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    ui_path = Path(__file__).resolve().parents[1] / "web" / "ds_explorer.html"
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/datasets")
async def list_datasets():
    return {
        "base_datasets": base_datasets_info,
        "processed_datasets": processed_datasets_info,
    }


@app.get("/api/datasets/{dataset_type}/{dataset_name}")
async def get_dataset_info(
    dataset_type: Literal["base", "processed"],
    dataset_name: Literal["mnist", "cifar10"],
):
    assert dataset_type in ["base", "processed"], "Invalid dataset type"
    return (
        base_datasets_info[dataset_name]
        if dataset_type == "base"
        else processed_datasets_info[dataset_name]
    )


@app.get("/api/datasets/{dataset_type}/{dataset_name}/{split}")
async def get_dataset_samples(
    dataset_type: Literal["base", "processed"],
    dataset_name: Literal["mnist", "cifar10"],
    split: Literal["train", "test"],
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    assert dataset_type in ["base", "processed"], "Invalid dataset type"
    assert (
        dataset_name in base_datasets_info or dataset_name in processed_datasets_info
    ), f"Dataset {dataset_name} in {dataset_type} not found"

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / f"{dataset_type}_datasets" / dataset_name

    ds_dict: DatasetDict = load_from_disk(str(dataset_path))

    if split not in ds_dict:
        raise HTTPException(status_code=404, detail="Split not found")

    ds: Dataset = ds_dict[split]

    image_col = None
    if "img" in ds.features:
        ds = ds.rename_column("img", "image")
        image_col = "image"
    elif "image" in ds.features:
        image_col = "image"

    total_samples = len(ds)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_samples)

    samples = []
    for i in range(start_idx, end_idx):
        sample = ds[i]
        sample = make_serializable(sample)

        if image_col and image_col in sample:
            sample["image_base64"] = image_to_base64(sample[image_col])
            del sample[image_col]

        samples.append(sample)

    return {
        "samples": samples,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_samples": total_samples,
            "total_pages": (total_samples + page_size - 1) // page_size,
            "has_next": end_idx < total_samples,
            "has_prev": page > 1,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset Explorer Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    uvicorn.run("data_explorer:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
