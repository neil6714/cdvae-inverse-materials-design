import os
import json
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()
API_KEY = os.environ.get("MP_API_KEY")


def download_mp20(save_dir: str = "data/raw", max_atoms: int = 20, e_hull_max: float = 0.08):
    os.makedirs(save_dir, exist_ok=True)

    with MPRester(API_KEY) as mpr:
        docs = mpr.summary.search(
            num_sites=(1, max_atoms),
            energy_above_hull=(0, e_hull_max),
            fields=["material_id", "structure", "formation_energy_per_atom", "band_gap"]
        )

    entries = []
    for doc in docs:
        if doc.structure is None:
            continue
        entries.append({
            "material_id": str(doc.material_id),
            "structure": doc.structure.as_dict(),
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "band_gap": doc.band_gap,
        })

    save_path = os.path.join(save_dir, "mp20_structures.json")
    with open(save_path, "w") as f:
        json.dump(entries, f)

    print(f"Saved {len(entries)} structures to {save_path}")


if __name__ == "__main__":
    download_mp20()