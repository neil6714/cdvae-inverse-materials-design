
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from cdvae.models.cdvae import CDVAE, BetaAnnealer
from cdvae.data.datamodule import CrystalDataModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = CrystalDataModule(
        root="/kaggle/working/cdvae-inverse-materials-design/data",
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cutoff=cfg.data.cutoff,
        num_rbf=cfg.data.num_rbf,
    )

    model = CDVAE(cfg)

    callbacks = [
        BetaAnnealer(
            beta_max=cfg.training.beta_max,
            warmup_steps=cfg.training.warmup_steps,
        ),
        ModelCheckpoint(
            dirpath="/kaggle/working/checkpoints",
            monitor="val/loss",
            save_top_k=5,
            mode="min",
            filename="cdvae-{epoch:02d}-{val_loss:.4f}",
            save_last=True,
            every_n_epochs=1,  # save every single epoch
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=1,
        enable_progress_bar=True,
        default_root_dir="/kaggle/working/cdvae-inverse-materials-design",
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
