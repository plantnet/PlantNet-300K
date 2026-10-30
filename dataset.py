from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 1000


class PlantNet300KDataset(Dataset):
    def __init__(self, metadata, image_root, class_to_idx, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        species_id = int(row["species_id"])
        image_name = str(row["PN_hash"])
        if not image_name.lower().endswith(".jpg"):
            image_name += ".jpg"

        image_path = (
            self.image_root
            / str(row["split"])
            / f"{species_id:04d}"
            / image_name
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.class_to_idx[species_id], dtype=torch.long)
        return image, label


def get_dataloaders(args, train_transform, eval_transform):
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "plantnet300K_metadata.csv"
    image_root = data_dir / "images"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"species_id", "split", "PN_hash"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing metadata columns: {missing}")

    species_ids = sorted(int(value) for value in metadata["species_id"].unique())
    expected_species_ids = list(range(NUM_CLASSES))
    if species_ids != expected_species_ids:
        raise ValueError(
            "Expected the Pl@ntNet-300K-v2 species IDs to span 0 through "
            f"{NUM_CLASSES - 1}; found {len(species_ids)} unique IDs."
        )
    class_to_idx = {species_id: index for index, species_id in enumerate(species_ids)}

    split_metadata = {
        split: metadata[metadata["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, rows in split_metadata.items():
        if rows.empty:
            raise ValueError(f"No rows found for the '{split}' split.")

    train_dataset = PlantNet300KDataset(
        split_metadata["train"], image_root, class_to_idx, train_transform
    )
    val_dataset = PlantNet300KDataset(
        split_metadata["val"], image_root, class_to_idx, eval_transform
    )
    test_dataset = PlantNet300KDataset(
        split_metadata["test"], image_root, class_to_idx, eval_transform
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, val_loader, test_loader, len(class_to_idx)
