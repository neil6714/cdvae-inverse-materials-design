import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdvae.data.crystal_dataset import CrystalDataset

for split in ["train", "val", "test"]:
    dataset = CrystalDataset(root="data", split=split)
    sample = dataset[0]
    print(f"\n{split}: {len(dataset)} structures")
    print(f"  Atoms: {sample.num_atoms.item()}, Edges: {sample.edge_index.shape[1]}")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Lattice shape: {sample.lattice.shape}")
    print(f"  Formation energy: {sample.formation_energy.item():.4f} eV/atom")