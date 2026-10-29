import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a DINOv3 backbone on Pl@ntNet-300K-v2."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset root containing images/ and plantnet300K_metadata.csv.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnxt", "cnxs", "cnxb", "vits", "vitsplus", "vitb"],
        help="DINOv3 backbone to fine-tune.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device, for example cuda, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="Directory in which to save the best checkpoint.",
    )

    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--scale_min", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval_freq", type=int, default=1)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_tmpdir", type=str, default="./tmp")
    parser.add_argument("--wandb_projname", type=str, default="plantnet300kv2")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    if args.eval_freq < 1:
        parser.error("--eval_freq must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup_epochs must be between 0 and --epochs")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    return args
