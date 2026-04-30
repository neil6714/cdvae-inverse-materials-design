import torch
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data


def rbf_expansion(distances: torch.Tensor, num_rbf: int = 64, cutoff: float = 6.0) -> torch.Tensor:
    centers = torch.linspace(0, cutoff, num_rbf, device=distances.device)
    gamma = num_rbf / cutoff
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)


def build_crystal_graph(structure: Structure, cutoff: float = 6.0, num_rbf: int = 64) -> Data:
    # atomic numbers as node features
    atom_types = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

    all_neighbors = structure.get_all_neighbors(r=cutoff)

    src_indices, dst_indices, distances, offsets = [], [], [], []

    for i, neighbors in enumerate(all_neighbors):
        for nbr in neighbors:
            src_indices.append(i)
            dst_indices.append(nbr.index)
            distances.append(nbr.nn_distance)
            offsets.append(list(nbr.image))

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    dist_tensor = torch.tensor(distances, dtype=torch.float32)
    offset_tensor = torch.tensor(offsets, dtype=torch.float32)

    # expand scalar distances to RBF feature vectors
    edge_attr = rbf_expansion(dist_tensor, num_rbf=num_rbf, cutoff=cutoff)

    return Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        frac_coords=frac_coords,
        lattice=lattice,
        distances=dist_tensor,
        offsets=offset_tensor,
        num_atoms=torch.tensor(len(structure)),
    )