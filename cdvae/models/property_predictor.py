import torch.nn as nn


class PropertyPredictor(nn.Module):
    """
    MLP predicting macroscopic crystal descriptors from the latent vector z.
    Outputs lattice parameters, element composition, and atom count — the structural
    skeleton decoded before atomic positions are refined by the diffusion decoder.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        max_atomic_num: int = 100,
        max_atoms: int = 20,
    ):
        super().__init__()
        self.lattice_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # a, b, c, cos(α), cos(β), cos(γ)
        )
        self.composition_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atomic_num + 1),
        )
        # Classify atom count as discrete variable over {1, ..., max_atoms}
        self.natoms_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atoms),
        )

    def forward(self, z):
        return (
            self.lattice_head(z),
            self.composition_head(z),
            self.natoms_head(z),
        )