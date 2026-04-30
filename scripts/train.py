import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from cdvae.models.cdvae import CDVAE, BetaAnnealer
from cdvae.data.datamodule import CrystalDataModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    # Hydra changes cwd to outputs/; resolve data path relative to project root
    cfg.data.root = os.path.join(get_original_cwd(), cfg.data.root)

    datamodule = CrystalDataModule(
        root=cfg.data.root,
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
            monitor="val/l_recon",
            save_top_k=3,
            mode="min",
            filename="cdvae-{epoch:02d}-{val_l_recon:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()