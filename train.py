import torch
from tqdm import tqdm


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    lr_schedule,
    epoch,
    device,
):
    model.train()
    total_loss = 0.0
    device = torch.device(device)
    use_amp = device.type == "cuda"

    for batch_index, (images, labels) in enumerate(
        tqdm(train_loader, desc="Training", leave=False)
    ):
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)

        iteration = len(train_loader) * epoch + batch_index
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr_schedule[iteration])

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(images)
            loss = criterion(output.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(train_loader)
