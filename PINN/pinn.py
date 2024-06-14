import torch
import torch.nn as nn
from monodomain import u0, Tf
from data import get_test_points
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime
from monodomain import pde
import os
import rff


class LearnableTanh(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.tanh(self.alpha * x)


class PINN(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, rff_size=32) -> None:
        super().__init__()

        rff_encoding = rff.layers.GaussianEncoding(sigma=.125, input_size=in_size, encoded_size=rff_size)
        in_size = rff_size * 2

        layers = [rff_encoding]

        for hs in hidden_sizes:
            layers.append(nn.Linear(in_size, hs))
            layers.append(LearnableTanh())
            in_size = hs

        layers.append(nn.Linear(hidden_sizes[-1], out_size))

        self.layers = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, t):

        out = self.layers(torch.cat([x, t], dim=1))
        return torch.exp(-10 * t) * u0(x) + (1.0 - torch.exp(-10 * t)) * out
        # return out

    def visualize(self, x, grid_shape, timestep_indx=0):
        with torch.no_grad():
            out = self.forward(x=x[:, 1:], t=x[:, 0:1])
            # out = self.layers(torch.cat([x[:,1:],x[:,0:1]], dim=1))

            out = out.reshape(grid_shape)
            #
            grid_arrays = [np.linspace(0, 1, grid_shape[1]),
                           np.linspace(0, 1, grid_shape[2])]

            X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            ax.plot_surface(X_grid, Y_grid, out[timestep_indx, :, :].cpu().detach(), cmap='viridis')
            plt.show()

    def visualize_loss_pde(self, x, grid_shape, sigma, timestep_indx=0):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x.requires_grad = True
        x_ = x[:, 1:]
        t_ = x[:, 0:1]

        out = self.forward(x=x_, t=t_)

        pde_residual = pde(u=out, x=x_, t=t_, sigma=sigma)
        pde_residual = pde_residual ** 2
        pde_residual = torch.max(pde_residual, dim=-1)[0]
        pde_residual = pde_residual.reshape(grid_shape)  # 50x50x50
        #
        grid_arrays = [np.linspace(0, 1, grid_shape[1]),
                       np.linspace(0, 1, grid_shape[2])]

        X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')

        out = out.reshape(grid_shape)

        def update_surface(timestep_indx):
            colors = pde_residual[timestep_indx, :, :]
            norm = plt.Normalize(colors.min().item(), colors.max().item())  # current relative min and max residual^2
            # norm = plt.Normalize(0, 1) # plot colors between 0,1
            cmap = plt.cm.viridis(norm(colors.cpu().detach()))

            ax.clear()
            ax.plot_surface(X_grid, Y_grid, out[timestep_indx, :, :].cpu().detach(), facecolors=cmap, shade=False)
            ax.set_zlim(-.2, 1)

            plt.title(f'Timestep = {timestep_indx}')

        ani = FuncAnimation(fig, update_surface, frames=out.shape[0], repeat=True)

        out_dir = './outputs'
        os.makedirs(out_dir, exist_ok=True)
        filename = f'{out_dir}/pinn_residual_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
        ani.save(filename=filename, writer='pillow')

        plt.show()

    def visualize_animate(self, x, grid_shape, savevideo=False, nn_only=False):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        with torch.no_grad():
            if nn_only:
                out = self.layers(torch.cat([x[:, 1:], x[:, 0:1]], dim=1))
            else:
                out = self.forward(x=x[:, 1:], t=x[:, 0:1])

            out = out.reshape(grid_shape)

            grid_arrays = [np.linspace(0, 1, grid_shape[1]),
                           np.linspace(0, 1, grid_shape[2])]

            X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')

            def update_surface(timestep_indx):
                ax.clear()
                ax.plot_surface(X_grid, Y_grid, out[timestep_indx, :, :].cpu().detach(), cmap='viridis')
                ax.set_zlim(-.2, 1)

                if nn_only:
                    plt.title(f'(nn only) Timestep = {timestep_indx}')
                else:
                    plt.title(f'Timestep = {timestep_indx}')

            ani = FuncAnimation(fig, update_surface, frames=out.shape[0], repeat=True)

            if savevideo:
                out_dir = './outputs'
                os.makedirs(out_dir, exist_ok=True)
                filename = f'{out_dir}/pinn_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
                ani.save(filename=filename, writer='pillow')

            plt.show()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    test_data, meshgrid_shape, _ = get_test_points(10)

    pinn = PINN(3, [16] * 8, 1)

    ic_x = torch.Tensor([[0.91, 0.91], [0.5, 0.9], [0.3, 0.3], [0.2, 0.9]])
    ic_t = torch.Tensor([[0], [0], [0], [0]])

    print(pinn(x=ic_x, t=ic_t))

    print(test_data)
    pinn.visualize(test_data, grid_shape=meshgrid_shape, timestep_indx=0)
