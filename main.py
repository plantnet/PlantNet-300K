from pathlib import Path

import torch
import torch.nn as nn

from args import parse_args
from dataset import get_dataloaders
from eval import evaluate
from model import build_model
from train import train_epoch
from transforms import TrainTransform, ValTransform
from utils import cosine_scheduler


def state_dict_to_cpu(module):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.state_dict().items()
    }


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    run_name = f"plantnet300kv2_{args.model}"
    if args.wandb:
        import wandb

        Path(args.wandb_tmpdir).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=args.wandb_projname,
            entity=args.wandb_entity,
            dir=args.wandb_tmpdir,
            name=run_name,
            config=vars(args),
        )

    train_transform = TrainTransform(
        crop_size=args.crop_size,
        scale_min=args.scale_min,
    )
    eval_transform = ValTransform(
        resize_size=args.resize_size,
        crop_size=args.crop_size,
    )
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args, train_transform, eval_transform
    )

    model = build_model(args.model, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        args.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_accuracy = float("-inf")
    best_checkpoint = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            lr_schedule,
            epoch,
            device,
        )
        metrics = {"train_loss": train_loss}
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {train_loss:.4f}")

        should_evaluate = (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs
        if should_evaluate:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            metrics.update({"val_loss": val_loss, "val_accuracy": val_accuracy})
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_checkpoint = {
                    "model_name": args.model,
                    "num_classes": num_classes,
                    "encoder": state_dict_to_cpu(model.encoder),
                    "classifier": state_dict_to_cpu(model.classifier),
                }

        if args.wandb:
            wandb.log(metrics, step=epoch + 1)

    model.encoder.load_state_dict(best_checkpoint["encoder"])
    model.classifier.load_state_dict(best_checkpoint["classifier"])
    _, test_accuracy = evaluate(model, test_loader, criterion, device)

    print(f"Best Val Acc: {best_val_accuracy:.4f}")
    print(f"Test Acc: {test_accuracy:.4f}")

    checkpoint_dir = Path(args.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    torch.save(best_checkpoint, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

    if args.wandb:
        wandb.log(
            {"best_val_accuracy": best_val_accuracy, "test_accuracy": test_accuracy}
        )
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
