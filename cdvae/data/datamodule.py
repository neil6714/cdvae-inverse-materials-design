import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from cdvae.data.crystal_dataset import CrystalDataset


class CrystalDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping the mp_20 crystal graph dataset."""

    def __init__(self, root, batch_size, num_workers, cutoff, num_rbf):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cutoff = cutoff
        self.num_rbf = num_rbf

    def setup(self, stage=None):
        self.train_data = CrystalDataset(self.root, "train", self.cutoff, self.num_rbf)
        self.val_data   = CrystalDataset(self.root, "val",   self.cutoff, self.num_rbf)
        self.test_data  = CrystalDataset(self.root, "test",  self.cutoff, self.num_rbf)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)