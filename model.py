from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from transformers import AutoModel


NUM_CLASSES = 1000

MODEL_REGISTRY = {
    "cnxt": ("facebook/dinov3-convnext-tiny-pretrain-lvd1689m", 768),
    "cnxs": ("facebook/dinov3-convnext-small-pretrain-lvd1689m", 768),
    "cnxb": ("facebook/dinov3-convnext-base-pretrain-lvd1689m", 1024),
    "vits": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    "vitsplus": ("facebook/dinov3-vits16plus-pretrain-lvd1689m", 384),
    "vitb": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
}


class PlantNetOutput(NamedTuple):
    embeddings: torch.Tensor
    logits: torch.Tensor


class PlantNetClassifier(nn.Module):
    """DINOv3 image encoder with a Pl@ntNet-300K-v2 classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor) -> PlantNetOutput:
        encoder_output = self.encoder(pixel_values=images)
        embeddings = encoder_output.pooler_output
        logits = self.classifier(embeddings)
        return PlantNetOutput(embeddings=embeddings, logits=logits)


def build_model(model_name: str, num_classes: int = NUM_CLASSES) -> PlantNetClassifier:
    """Construct a classifier initialized with a pretrained DINOv3 backbone."""
    try:
        backbone_name, embedding_dim = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: {choices}."
        ) from exc

    encoder = AutoModel.from_pretrained(backbone_name)
    return PlantNetClassifier(encoder, embedding_dim, num_classes)


def load_pretrained_weights(
    model: PlantNetClassifier,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> PlantNetClassifier:
    """Load a training checkpoint into an existing model.

    Released checkpoints store separate ``encoder`` and ``classifier`` state
    dictionaries. Combined state dictionaries are accepted as well.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if "encoder" in checkpoint and "classifier" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        model.classifier.load_state_dict(checkpoint["classifier"], strict=True)
    else:
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=True)

    return model


def load_model(
    model_name: str,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    num_classes: int = NUM_CLASSES,
) -> PlantNetClassifier:
    """Construct a model, load fine-tuned weights, and put it in eval mode."""
    model = build_model(model_name, num_classes=num_classes)
    load_pretrained_weights(model, checkpoint_path, map_location="cpu")
    return model.to(device).eval()
