import os
import json
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()
API_KEY = os.environ.get("MP_API_KEY")


def download_mp20(save_dir: str = "data/raw", limit: int = 5000):
    os.makedirs(save_dir, exist_ok=True)

    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            num_sites=(1, 20),
            energy_above_hull=(0, 0.08),
            fields=["material_id", "structure", "formation_energy_per_atom", "band_gap"],
        )

    docs = docs[:limit]

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