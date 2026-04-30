import os
import json
import torch
from tqdm import tqdm
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset
from cdvae.data.utils import build_crystal_graph


class CrystalDataset(Dataset):
    def __init__(self, root: str, split: str = "train", cutoff: float = 6.0,
                 num_rbf: int = 64, transform=None):
        self.split = split
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        return ["mp20_structures.json"]

    @property
    def processed_file_names(self):
        raw_path = os.path.join(self.raw_dir, "mp20_structures.json")
        with open(raw_path) as f:
            all_data = json.load(f)
        n = len(all_data)
        splits = {
            "train": range(0, int(0.6 * n)),
            "val":   range(int(0.6 * n), int(0.8 * n)),
            "test":  range(int(0.8 * n), n)
        }
        return [f"{self.split}_{i}.pt" for i in splits[self.split]]

    def process(self):
        raw_path = os.path.join(self.raw_dir, "mp20_structures.json")
        with open(raw_path) as f:
            all_data = json.load(f)

        n = len(all_data)
        splits = {
            "train": all_data[:int(0.6 * n)],
            "val":   all_data[int(0.6 * n):int(0.8 * n)],
            "test":  all_data[int(0.8 * n):]
        }
        subset = splits[self.split]

        for i, entry in enumerate(tqdm(subset, desc=f"Building {self.split} graphs")):
            save_path = os.path.join(self.processed_dir, f"{self.split}_{i}.pt")
            if os.path.exists(save_path):
                continue
            try:
                structure = Structure.from_dict(entry["structure"])
                graph = build_crystal_graph(structure, self.cutoff, self.num_rbf)
                graph.formation_energy = torch.tensor(
                    [entry["formation_energy_per_atom"] or 0.0], dtype=torch.float32
                )
                graph.band_gap = torch.tensor(
                    [entry["band_gap"] or 0.0], dtype=torch.float32
                )
                graph.material_id = entry["material_id"]
                torch.save(graph, save_path)
            except Exception as e:
                print(f"Skipping {entry.get('material_id', '?')}: {e}")

    def len(self):
        return min(100,len(self.processed_file_names))

    def get(self, idx):
        path = os.path.join(self.processed_dir, f"{self.split}_{idx}.pt")
        return torch.load(path,weights_only=False)
    