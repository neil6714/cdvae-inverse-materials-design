import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from cdvae.models.encoder import EquivariantEncoder
from cdvae.models.decoder import ScoreNetwork
from cdvae.models.property_predictor import PropertyPredictor


def cosine_noise_schedule(T: int, s: float = 0.008):
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    Destroys structure more gradually than linear, improving sample quality.
    """
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = (f / f[0]).float()
    betas = torch.clamp(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)
    return betas, alphas_cumprod[1:]


class BetaAnnealer(Callback):
    """
    Linearly anneals the KL weight β from 0 to beta_max over warmup_steps.
    Prevents posterior collapse by allowing the encoder to learn before regularization kicks in.
    """

    def __init__(self, beta_max: float, warmup_steps: int):
        self.beta_max = beta_max
        self.warmup_steps = warmup_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.beta = min(
            self.beta_max,
            self.beta_max * trainer.global_step / self.warmup_steps,
        )


class CDVAE(pl.LightningModule):
    """
    Crystal Diffusion Variational Autoencoder.
    Combines a VAE with a denoising diffusion decoder for inverse crystal design.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.encoder = EquivariantEncoder(
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_encoder_layers,
            num_rbf=cfg.model.num_rbf,
            cutoff=cfg.model.cutoff,
            latent_dim=cfg.model.latent_dim,
            max_atomic_num=cfg.model.max_atomic_num,
        )
        self.decoder = ScoreNetwork(
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_decoder_layers,
            num_rbf=cfg.model.num_rbf,
            cutoff=cfg.model.cutoff,
            latent_dim=cfg.model.latent_dim,
            time_dim=cfg.model.time_dim,
            max_atomic_num=cfg.model.max_atomic_num,
        )
        self.predictor = PropertyPredictor(
            latent_dim=cfg.model.latent_dim,
            hidden_dim=cfg.model.hidden_dim,
            max_atomic_num=cfg.model.max_atomic_num,
            max_atoms=cfg.model.max_atoms,
        )

        # Precompute and register cosine schedule buffers for forward diffusion
        betas, alphas_cumprod = cosine_noise_schedule(cfg.training.T)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt()
        )

        # KL weight; updated each step by BetaAnnealer callback
        self.beta = 0.0

    def reparameterize(self, mu, log_var):
        return mu + (0.5 * log_var).exp() * torch.randn_like(mu)

    def training_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()},
            on_step=True, on_epoch=False, batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in logs.items()},
            on_step=False, on_epoch=True, batch_size=batch.num_graphs,
        )

    def _shared_step(self, batch):
        mu, log_var = self.encoder(batch)
        z = self.reparameterize(mu, log_var)

        lattice_pred, comp_logits, natoms_logits = self.predictor(z)

        l_recon = self._recon_loss(batch, lattice_pred, comp_logits, natoms_logits)
        # Closed-form KL divergence for diagonal Gaussian vs standard normal N(0,I)
        l_kl = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
        )
        l_score = self._score_loss(batch, z)

        loss = l_recon + self.beta * l_kl + self.cfg.training.lambda_score * l_score

        return loss, {
            "loss": loss, "l_recon": l_recon,
            "l_kl": l_kl, "l_score": l_score, "beta": self.beta,
        }

    def _recon_loss(self, batch, lattice_pred, comp_logits, natoms_logits):
        l_lattice = F.mse_loss(lattice_pred, self._lattice_to_params(batch))

        target_comp = self._compute_composition(batch)
        # Soft cross-entropy against normalized element histogram
        l_comp = -(target_comp * F.log_softmax(comp_logits, dim=-1)).sum(-1).mean()

        natoms = torch.bincount(batch.batch, minlength=batch.num_graphs)
        l_natoms = F.cross_entropy(
            natoms_logits,
            (natoms - 1).long().clamp(0, natoms_logits.shape[-1] - 1),
        )

        return l_lattice + l_comp + l_natoms

    def _score_loss(self, batch, z):
        B = batch.num_graphs
        t = torch.randint(0, self.cfg.training.T, (B,), device=self.device)

        sqrt_a = self.sqrt_alphas_cumprod[t][batch.batch].unsqueeze(-1)
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t][batch.batch].unsqueeze(-1)
        eps = torch.randn_like(batch.frac_coords)
        # Forward diffusion on fractional coords; modulo wraps coordinates into [0,1)
        noisy_coords = (sqrt_a * batch.frac_coords + sqrt_1a * eps) % 1.0

        # Corrupt atom types by replacing with uniform random element with prob (1 - ᾱ_t)
        corrupt_prob = 1 - self.alphas_cumprod[t][batch.batch]
        mask = torch.bernoulli(corrupt_prob).bool()
        noisy_types = batch.x.clone().long()
        noisy_types[mask] = torch.randint(
            1, self.cfg.model.max_atomic_num + 1, (mask.sum(),), device=self.device
        )

        coord_pred, type_logits = self.decoder(noisy_coords, noisy_types, z, t, batch)

        l_coord = F.mse_loss(coord_pred, eps)
        l_type = F.cross_entropy(type_logits, batch.x.long())

        return l_coord + l_type

    def _lattice_to_params(self, batch):
        """Extracts (a, b, c, cos α, cos β, cos γ) from the 3×3 lattice matrix."""
        L = batch.lattice.view(batch.num_graphs, 3, 3)
        a = torch.linalg.norm(L[:, 0], dim=-1)
        b = torch.linalg.norm(L[:, 1], dim=-1)
        c = torch.linalg.norm(L[:, 2], dim=-1)
        cos_alpha = (L[:, 1] * L[:, 2]).sum(-1) / (b * c).clamp(min=1e-8)
        cos_beta  = (L[:, 0] * L[:, 2]).sum(-1) / (a * c).clamp(min=1e-8)
        cos_gamma = (L[:, 0] * L[:, 1]).sum(-1) / (a * b).clamp(min=1e-8)
        return torch.stack([a, b, c, cos_alpha, cos_beta, cos_gamma], dim=-1)

    def _compute_composition(self, batch):
        """Computes normalized element histogram per crystal in the batch."""
        B = batch.num_graphs
        max_Z = self.cfg.model.max_atomic_num + 1
        comp = torch.zeros(B, max_Z, device=self.device)
        one_hot = F.one_hot(batch.x.long().clamp(0, max_Z - 1), max_Z).float()
        comp.scatter_add_(0, batch.batch.unsqueeze(1).expand(-1, max_Z), one_hot)
        return comp / comp.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.training.lr, weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.cfg.training.T_0, T_mult=self.cfg.training.T_mult
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }