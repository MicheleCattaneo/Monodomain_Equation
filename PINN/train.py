import pytorch_lightning as pl
import torch
import torch.nn as nn

from monodomain import loss_pde, loss_neumann
from data import MonodomainDataset, get_test_points, get_data
from pinn import PINN


class Monodomain(pl.LightningModule):
    def __init__(self, lr=1e-3, hidden_sizes=[16] * 8):
        super(Monodomain, self).__init__()

        self.model = PINN(3, hidden_sizes, 1)
        self.lr = lr

        self.w_pde = nn.Parameter(torch.Tensor([5]))
        self.w_bc = nn.Parameter(torch.Tensor([1]))

        self.ip_t, self.ip_x, self.bc_t, self.bc_x, self.sigmas = get_data(num_cp=10_000, num_b_cp=500, dim=2)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        t, x, tbc, xbc, sigma = batch

        x, t, xbc, tbc, sigma = x[0], t[0], xbc[0], tbc[0], sigma[0]

        # t, x, tbc, xbc, sigma = self.ip_t, self.ip_x, self.bc_t, self.bc_x, self.sigmas

        u = self.model(x, t)

        loss_domain = loss_pde(u, x, t, sigma)
        ubc = self.model(xbc, tbc)
        loss_bc = loss_neumann(ubc, xbc)

        loss = self.w_pde * loss_domain + self.w_bc * loss_bc

        opt1, opt2 = self.optimizers()

        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_pde', loss_domain, prog_bar=True)
        self.log('loss_bc', loss_bc, prog_bar=True)

        self.manual_backward(loss)
        opt1.step()

        self.w_pde.grad = -self.w_pde.grad
        self.w_bc.grad = -self.w_bc.grad

        opt2.step()

        opt1.zero_grad()
        opt2.zero_grad()

        # print(self.w_pde.item())
        # print(self.w_bc.item())

        return loss

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def configure_optimizers(self):
        # Optimizer for model parameters
        optimizer1 = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Optimizer for ascent parameters
        optimizer2 = torch.optim.Adam([self.w_pde, self.w_bc], lr=0.01)
        return [optimizer1, optimizer2]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MonodomainDataset(num_cp=5_000, num_b_cp=5_000), batch_size=1000000,
                                           shuffle=True)


if __name__ == '__main__':
    model = Monodomain(lr=1e-4, hidden_sizes=[128] * 2)

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu'
    )

    trainer.fit(model)

    test_data, meshgrid_shape = get_test_points(100)
    model.model.visualize(test_data, grid_shape=meshgrid_shape, timestep_indx=50)
