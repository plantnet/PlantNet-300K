"""Stage the six Pl@ntNet-300K-v2 classifiers on the Hugging Face Hub."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("cnxt", "cnxs", "cnxb", "vits", "vitsplus", "vitb")


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    base_model: str
    embedding_dim: int
    val_accuracy: float
    test_accuracy: float


MODEL_SPECS = {
    "cnxt": ModelSpec(
        "DINOv3 ConvNeXt-Tiny",
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        768,
        0.8248,
        0.8216,
    ),
    "cnxs": ModelSpec(
        "DINOv3 ConvNeXt-Small",
        "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        768,
        0.8303,
        0.8296,
    ),
    "cnxb": ModelSpec(
        "DINOv3 ConvNeXt-Base",
        "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        1024,
        0.8358,
        0.8331,
    ),
    "vits": ModelSpec(
        "DINOv3 ViT-S/16",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        384,
        0.8296,
        0.8297,
    ),
    "vitsplus": ModelSpec(
        "DINOv3 ViT-S+/16",
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        384,
        0.8328,
        0.8293,
    ),
    "vitb": ModelSpec(
        "DINOv3 ViT-B/16",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
        0.8462,
        0.8457,
    ),
}


MODEL_CARD_TEMPLATE = """---
library_name: transformers
pipeline_tag: image-classification
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
base_model: {base_model}
tags:
- vision
- plants
- plantnet
- dinov3
- image-classification
- species-classification
- 1000-classes
inference: false
---

# Pl@ntNet-300K-v2 — {display_name}

This is a **1,000-species plant classifier** obtained by fine-tuning
[{base_model}](https://huggingface.co/{base_model}) on
[Pl@ntNet-300K-v2](https://zenodo.org/records/10419064).

| Property | Value |
|---|---:|
| Architecture | {display_name} |
| Classes | 1,000 |
| Embedding dimension | {embedding_dim} |
| Best validation accuracy | {val_accuracy:.4f} |
| Test accuracy | {test_accuracy:.4f} |
| Input size | 224 × 224 RGB |

## Usage

The repository stores the fine-tuned encoder and classifier in `model.pth`.
The included loader returns both image embeddings and classifier logits.

```python
import sys

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image

model_id = "{repo_id}"
model_dir = snapshot_download(model_id)
sys.path.insert(0, model_dir)

from model import load_model
from transforms import ValTransform

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("{model_name}", f"{{model_dir}}/model.pth", device=device)
transform = ValTransform()

image = transform(Image.open(f"{{model_dir}}/example.jpg").convert("RGB"))
with torch.inference_mode():
    embeddings, logits = model(image.unsqueeze(0).to(device))

species_id = logits.argmax(dim=1).item()
metadata = pd.read_csv(f"{{model_dir}}/species_metadata.csv")
species_name = metadata.loc[
    metadata["species_id"] == species_id, "full_species"
].item()
print(species_name, embeddings.shape)
```

The DINOv3 base model is license-gated. Accept its license and authenticate with
Hugging Face before the first load.

## Preprocessing

Images are resized/cropped to 224 × 224 and normalized with the Pl@ntNet
statistics used during fine-tuning:

```text
mean = (0.4425, 0.4695, 0.3266)
std  = (0.2353, 0.2219, 0.2325)
```

## Citation

```bibtex
@inproceedings{{plantnet-300k,
  author    = {{Garcin, Camille and Joly, Alexis and Bonnet, Pierre and
               Lombardo, Jean-Christophe and Affouard, Antoine and
               Chouet, Mathias and Servajean, Maximilien and
               Lorieul, Titouan and Salmon, Joseph}},
  booktitle = {{NeurIPS Datasets and Benchmarks 2021}},
  title     = {{{{Pl@ntNet-300K}}: a plant image dataset with high label
               ambiguity and a long-tailed distribution}},
  year      = {{2021}}
}}

@misc{{moummad2026,
  title         = {{Energy-Efficient Plant Monitoring via Knowledge Distillation}},
  author        = {{Ilyass Moummad and Reda Bensaid and Kawtar Zaher and
                   Hervé Goëau and Jean-Christophe Lombardo and Joseph Salmon and
                   Pierre Bonnet and Alexis Joly}},
  year          = {{2026}},
  eprint        = {{2604.27178}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.CV}},
  url           = {{https://arxiv.org/abs/2604.27178}}
}}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing plantnet300kv2_<model>.pth checkpoints.",
    )
    parser.add_argument(
        "--species-metadata",
        type=Path,
        required=True,
        help="Path to Pl@ntNet-300K-v2 species_metadata.csv.",
    )
    parser.add_argument("--organization", default="PlantNet")
    parser.add_argument("--repo-prefix", default="plantnet-300k-v2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Models to publish; defaults to all six.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/publish public repositories. The safe default is private.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the release plan without contacting the Hub.",
    )
    return parser.parse_args()


def validate_species_metadata(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Species metadata not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    required_columns = {"species_id", "full_species"}
    missing = required_columns.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Species metadata is missing columns: {sorted(missing)}")

    species_ids = sorted(int(row["species_id"]) for row in rows)
    if species_ids != list(range(1000)):
        raise ValueError(
            "species_metadata.csv must contain exactly one row for every "
            "species_id from 0 through 999."
        )


def validate_checkpoint(path: Path, model_name: str, spec: ModelSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary.")
    if "encoder" not in checkpoint or "classifier" not in checkpoint:
        raise ValueError(f"{path} must contain encoder and classifier state dictionaries.")

    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model is not None and checkpoint_model != model_name:
        raise ValueError(
            f"{path} declares model_name={checkpoint_model!r}, expected {model_name!r}."
        )

    classifier = checkpoint["classifier"]
    weight = classifier.get("weight")
    bias = classifier.get("bias")
    expected_weight_shape = (1000, spec.embedding_dim)
    if weight is None or tuple(weight.shape) != expected_weight_shape:
        actual_shape = None if weight is None else tuple(weight.shape)
        raise ValueError(
            f"{path} classifier.weight has shape {actual_shape}; "
            f"expected {expected_weight_shape}."
        )
    if bias is None or tuple(bias.shape) != (1000,):
        actual_shape = None if bias is None else tuple(bias.shape)
        raise ValueError(
            f"{path} classifier.bias has shape {actual_shape}; expected (1000,)."
        )


def checkpoint_path(checkpoint_dir: Path, model_name: str) -> Path:
    return checkpoint_dir / f"plantnet300kv2_{model_name}.pth"


def render_model_card(repo_id: str, model_name: str, spec: ModelSpec) -> str:
    return MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        model_name=model_name,
        display_name=spec.display_name,
        base_model=spec.base_model,
        embedding_dim=spec.embedding_dim,
        val_accuracy=spec.val_accuracy,
        test_accuracy=spec.test_accuracy,
    )


def publish_model(
    api: HfApi,
    repo_id: str,
    model_name: str,
    checkpoint: Path,
    species_metadata: Path,
    public: bool,
) -> None:
    spec = MODEL_SPECS[model_name]
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    repo_info = api.model_info(repo_id)
    if not public and not repo_info.private:
        raise RuntimeError(
            f"{repo_id} already exists and is public. Refusing a private staging upload."
        )
    with tempfile.TemporaryDirectory(prefix=f"plantnet-{model_name}-") as temp_dir:
        temp_path = Path(temp_dir)
        readme_path = temp_path / "README.md"
        config_path = temp_path / "plantnet_config.json"
        runtime_requirements_path = temp_path / "requirements.txt"

        readme_path.write_text(
            render_model_card(repo_id, model_name, spec), encoding="utf-8"
        )
        config_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "base_model": spec.base_model,
                    "num_classes": 1000,
                    "embedding_dim": spec.embedding_dim,
                    "image_size": 224,
                    "image_mean": [0.4425, 0.4695, 0.3266],
                    "image_std": [0.2353, 0.2219, 0.2325],
                    "best_validation_accuracy": spec.val_accuracy,
                    "test_accuracy": spec.test_accuracy,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        runtime_requirements_path.write_text(
            "pandas>=2.0\nPillow>=10.0\ntorch>=2.5\n"
            "torchvision>=0.20\ntransformers>=4.57.3\n",
            encoding="utf-8",
        )

        files = {
            "README.md": readme_path,
            "model.pth": checkpoint,
            "model.py": PROJECT_ROOT / "model.py",
            "transforms.py": PROJECT_ROOT / "transforms.py",
            "species_metadata.csv": species_metadata,
            "example.jpg": PROJECT_ROOT / "images" / "1.jpg",
            "plantnet_config.json": config_path,
            "requirements.txt": runtime_requirements_path,
        }
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            for repo_path, local_path in files.items()
        ]
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Add Pl@ntNet-300K-v2 {spec.display_name} classifier",
        )

    expected_files = set(files) | {".gitattributes"}
    remote_files = set(api.list_repo_files(repo_id, repo_type="model"))
    missing_files = expected_files.difference(remote_files)
    if missing_files:
        raise RuntimeError(f"Remote audit failed for {repo_id}: missing {missing_files}")

    if public and repo_info.private:
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)

    visibility = "public" if public else "private"
    print(f"Uploaded {repo_id} as a {visibility} model repository.")


def main() -> None:
    args = parse_args()
    validate_species_metadata(args.species_metadata)

    plan = []
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        checkpoint = checkpoint_path(args.checkpoint_dir, model_name)
        validate_checkpoint(checkpoint, model_name, spec)
        repo_id = f"{args.organization}/{args.repo_prefix}-{model_name}"
        plan.append((repo_id, model_name, checkpoint))

    visibility = "public" if args.public else "private"
    print(f"Validated {len(plan)} model release(s); target visibility: {visibility}")
    for repo_id, _, checkpoint in plan:
        print(f"  {checkpoint} -> https://huggingface.co/{repo_id}")

    if args.dry_run:
        return

    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as {user['name']}")
    for repo_id, model_name, checkpoint in plan:
        publish_model(
            api,
            repo_id,
            model_name,
            checkpoint,
            args.species_metadata,
            args.public,
        )


if __name__ == "__main__":
    main()
