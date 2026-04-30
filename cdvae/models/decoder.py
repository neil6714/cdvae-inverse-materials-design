import math
import torch
import torch.nn as nn
from cdvae.models.encoder import PaiNNLayer


class RBFExpansion(nn.Module):
    """Expands scalar distances into radial basis function features."""

    def __init__(self, num_rbf: int = 64, cutoff: float = 6.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = num_rbf / cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep t, identical to transformer positional encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        emb = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ScoreNetwork(nn.Module):
    """
    Noise-conditional score network used as the CDVAE diffusion decoder.
    Predicts the coordinate denoising direction and atom type logits per atom,
    conditioned on the latent vector z and the current noise level t.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_rbf: int = 64,
        cutoff: float = 6.0,
        latent_dim: int = 256,
        time_dim: int = 64,
        max_atomic_num: int = 100,
    ):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf, cutoff)
        self.atom_embed = nn.Embedding(max_atomic_num + 1, hidden_dim, padding_idx=0)
        self.time_embed = SinusoidalTimestepEmbedding(time_dim)

        # Project global conditioning signals to hidden_dim for per-atom injection
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.t_proj = nn.Linear(time_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)]
        )

        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atomic_num + 1),
        )

    def forward(self, noisy_coords, noisy_types, z, t, data):
        s = self.atom_embed(noisy_types.long())
        v = torch.zeros(s.shape[0], 3, s.shape[-1], device=s.device)

        t_emb = self.time_embed(t)
        # Expand batch-level z and t_emb to per-atom conditioning vectors
        cond = self.z_proj(z[data.batch]) + self.t_proj(t_emb[data.batch])

        edge_rbf, edge_vec_unit = self._recompute_edges(noisy_coords, data)

        for layer in self.layers:
            s = s + cond  # re-inject conditioning at every layer for strong guidance
            s, v = layer(s, v, data.edge_index, edge_rbf, edge_vec_unit)

        return self.coord_head(s), self.type_head(s)

    def _recompute_edges(self, frac_coords, data):
        """Recomputes edge geometry from current noisy fractional coordinates."""
        src, dst = data.edge_index
        lattice = data.lattice.view(data.num_graphs, 3, 3)
        frac_diff = frac_coords[dst] - frac_coords[src] + data.offsets
        cart_vec = torch.einsum("bi,bij->bj", frac_diff, lattice[data.batch[src]])
        distances = torch.linalg.norm(cart_vec, dim=-1).clamp(min=1e-8)
        return self.rbf(distances), nn.functional.normalize(cart_vec, dim=-1)