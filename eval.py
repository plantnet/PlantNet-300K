import torch
from tqdm import tqdm


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    device = torch.device(device)
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for images, labels in tqdm(data_loader, desc="Evaluation", leave=False):
            images = images.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(images)
                loss = criterion(output.logits, labels)

            total_loss += loss.item()
            correct_predictions += (output.logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(data_loader), correct_predictions / total_samples
