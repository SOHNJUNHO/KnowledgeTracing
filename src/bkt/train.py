from __future__ import annotations

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC


class NeuralBKTLightning(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

    def forward(self, obs: torch.Tensor, output: torch.Tensor):
        return self.model(obs, output)

    def training_step(self, batch, batch_idx):
        obs, output, _ = batch
        corrects, _, _, loss = self(obs, output)
        valid_preds, valid_targets = self._flatten_valid_predictions(corrects, output)
        train_acc = ((valid_preds > 0.5).float() == valid_targets).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=obs.size(0))
        self.log("train_acc", train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=obs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        obs, output, _ = batch
        corrects, _, _, loss = self(obs, output)
        valid_preds, valid_targets = self._flatten_valid_predictions(corrects, output)
        val_acc = ((valid_preds > 0.5).float() == valid_targets).float().mean()
        self.val_auc.update(valid_preds, valid_targets.long())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=obs.size(0))
        self.log("val_acc", val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=obs.size(0))
        self.log("val_auc", self.val_auc, prog_bar=True, on_step=False, on_epoch=True, batch_size=obs.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        obs, output, _ = batch
        corrects, _, _, loss = self(obs, output)
        valid_preds, valid_targets = self._flatten_valid_predictions(corrects, output)
        test_acc = ((valid_preds > 0.5).float() == valid_targets).float().mean()
        self.test_auc.update(valid_preds, valid_targets.long())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=obs.size(0))
        self.log("test_acc", test_acc, on_step=False, on_epoch=True, batch_size=obs.size(0))
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, batch_size=obs.size(0))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @staticmethod
    def _flatten_valid_predictions(corrects: torch.Tensor, output: torch.Tensor):
        mask = output[..., 1] != -1000
        preds = corrects.squeeze(-1)[mask]
        targets = output[..., 1][mask]
        if preds.numel() == 0:
            preds = torch.zeros(1, device=corrects.device)
            targets = torch.zeros(1, device=corrects.device)
        return preds, targets
