from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from .data import get_data_loaders, load_dataframe, validate_dataframe
from .models.static_neural_bkt import BKTConfig, BKTransformer
from .train import NeuralBKTLightning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a static neural BKT model.")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to processed CSV.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=818)
    parser.add_argument("--max_guess", type=float, default=0.5)
    parser.add_argument("--max_slip", type=float, default=0.5)
    parser.add_argument("--lambda_consistency", type=float, default=0.25)
    parser.add_argument("--lambda_guess", type=float, default=0.25)
    parser.add_argument("--lambda_slip", type=float, default=0.25)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Directory for Lightning checkpoints and exported .pt files.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("artifacts/lightning_logs"),
        help="Directory for Lightning CSV logs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, n_skills: int) -> BKTConfig:
    return BKTConfig(
        n_skills=n_skills,
        n_embd=args.n_embd,
        hidden_dim=args.hidden_dim,
        block_size=args.block_size,
        dropout=args.dropout,
        max_guess=args.max_guess,
        max_slip=args.max_slip,
        lambda_consistency=args.lambda_consistency,
        lambda_guess=args.lambda_guess,
        lambda_slip=args.lambda_slip,
    )


def export_torch_checkpoint(model: BKTransformer, config: BKTConfig, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "static-neural-bkt",
            "model_class": "BKTransformer",
            "config": asdict(config),
            "state_dict": model.state_dict(),
        },
        export_path,
    )


def main(args: argparse.Namespace):
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    df = load_dataframe(args.data_path)
    validate_dataframe(df)

    config = build_config(args, n_skills=int(df["skill_id"].nunique()) + 1)
    train_loader, val_loader, test_loader = get_data_loaders(
        df=df,
        block_size=config.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = BKTransformer(config)
    lightning_model = NeuralBKTLightning(model=model, lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="static-neural-bkt-best-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        verbose=True,
    )
    logger = CSVLogger(save_dir=str(args.log_dir), name="static-neural-bkt")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        raise RuntimeError("No best checkpoint was created.")

    trainer.test(lightning_model, test_loader, ckpt_path=best_checkpoint)

    export_path = args.checkpoint_dir / "static-neural-bkt-best.pt"
    export_torch_checkpoint(lightning_model.model, config, export_path)
    print(f"Exported PyTorch checkpoint: {export_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    pl.seed_everything(cli_args.seed, workers=True)
    main(cli_args)
