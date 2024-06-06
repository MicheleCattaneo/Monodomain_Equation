import pytorch_lightning as pl
import torch
import torch.nn as nn

from monodomain import loss_pde, loss_neumann
from data import MonodomainDataset
from pinn import PINN


class Monodomain(pl.LightningModule):
    def __init__(self, lr=1e-3, hidden_sizes=[16]*8):
        super(Monodomain, self).__init__()

        self.model = PINN(3, hidden_sizes, 1)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, t, xbc, tbc, sigma_d = batch
        u = self.model(x, t)

        loss_domain = loss_pde(u, x, t, sigma_d)
        ubc = self.model(xbc, tbc)
        loss_bc = loss_neumann(ubc, xbc)

        loss = loss_domain + loss_bc * 5

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MonodomainDataset(num_cp=10_000, num_b_cp=5_000), batch_size=1000, shuffle=True)


if __name__ == '__main__':

    model = Monodomain()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu'
    )

    trainer.fit(model)