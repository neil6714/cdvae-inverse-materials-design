import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class PaiNNLayer(nn.Module):
    """
    Single PaiNN equivariant message passing layer.
    Maintains coupled scalar (invariant) and vector (SE(3)-equivariant) channels.
    """

    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        # Produces three gating signals from source scalar features and edge RBF
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        # Mixes aggregated scalar features with vector channel norms for node update
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, s, v, edge_index, edge_rbf, edge_vec_unit):
        src, dst = edge_index
        N = s.shape[0]

        phi = self.message_net(torch.cat([s[src], edge_rbf], dim=-1))
        a_ss, a_sv, a_vv = phi.chunk(3, dim=-1)

        msg_s = a_ss
        # Equivariant vector message: direction-scaled scalar + source vector gate
        msg_v = (
            a_sv.unsqueeze(1) * edge_vec_unit.unsqueeze(-1)
            + a_vv.unsqueeze(1) * v[src]
        )

        agg_s = torch.zeros(N, s.shape[-1], device=s.device)
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg_s), msg_s)

        agg_v = torch.zeros_like(v)
        agg_v.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg_v), msg_v)

        v_new = v + agg_v
        # L2 norm over spatial dim converts equivariant vectors to invariant scalars
        v_norm = torch.linalg.norm(v_new, dim=1)

        update = self.update_net(torch.cat([s + agg_s, v_norm], dim=-1))
        delta_s, gate_v = update.chunk(2, dim=-1)

        return self.layer_norm(s + delta_s), gate_v.unsqueeze(1) * v_new


class EquivariantEncoder(nn.Module):
    """
    SE(3)-equivariant GNN encoder for periodic crystal structures.
    Maps a crystal graph to posterior parameters (mu, log_var) of q(z|x).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_rbf: int = 64,
        cutoff: float = 6.0,
        latent_dim: int = 256,
        max_atomic_num: int = 100,
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(max_atomic_num + 1, hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)]
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        s = self.atom_embed(data.x.long())
        v = torch.zeros(s.shape[0], 3, s.shape[-1], device=s.device)

        edge_vec_unit = self._edge_directions(data)

        for layer in self.layers:
            # data.edge_attr contains pre-computed RBF features from Phase 1
            s, v = layer(s, v, data.edge_index, data.edge_attr, edge_vec_unit)

        h = global_mean_pool(s, data.batch)
        return self.mu_head(h), self.logvar_head(h)

    def _edge_directions(self, data):
        src, dst = data.edge_index
        lattice = data.lattice.view(data.num_graphs, 3, 3)
        frac_diff = data.frac_coords[dst] - data.frac_coords[src] + data.offsets
        # Convert fractional displacement to Cartesian via per-edge lattice matrix
        cart_vec = torch.einsum("bi,bij->bj", frac_diff, lattice[data.batch[src]])
        return nn.functional.normalize(cart_vec, dim=-1)