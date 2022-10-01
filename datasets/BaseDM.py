import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
	def __init__(self, dataset_path, batch_size, num_workers):
		super().__init__()
		self.dataset_path = dataset_path
		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage=None):
		self.train_ds = None
		self.test_ds = None

	def train_dataloader(self):
		return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)