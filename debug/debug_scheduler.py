# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/bug_report/bug_report_model.py
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from args.setup import set_logger, set_trainer, parse_args
from funcs.module_funcs import setup_scheduler, setup_optimizer

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.args = args

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        lr = self.optimizer.param_groups[0]["lr"]
        print(lr)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()


    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=self.args.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs, eta_min=1e-5),
            "interval": self.args.scheduler_interval,
            "frequency": 1,  # other small numbers may also cause this issue.
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    args = parse_args()
    args.epochs = 64

    args.scheduler_interval = 'step'
    args.lr = 0.01
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel(args)
    root_dir = 'results'
    wandb_logger = set_logger(args, root_dir)
    # wandb_logger = None
    save_path = root_dir + '/test'
    trainer = set_trainer(args, wandb_logger, save_path)

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    main()