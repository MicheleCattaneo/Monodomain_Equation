import torch
import torch.nn as nn
from monodomain import u0, Tf
from data import get_test_points
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime
from monodomain import pde
from monodomain import e_ds


class LearnableTanh(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.nn.functional.tanh(self.alpha * x)


class PINN(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super().__init__()

        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_size, hs))
            layers.append(LearnableTanh())
            in_size = hs

        layers.append(nn.Linear(hidden_sizes[-1],out_size))

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
        return torch.exp(-10*t) * u0(x) + (1.0 - torch.exp(-10*t)) * out
        # return out
    

    def visualize(self, x, grid_shape, timestep_indx=0):
        with torch.no_grad():
            out = self.forward(x=x[:,1:], t=x[:,0:1])
            # out = self.layers(torch.cat([x[:,1:],x[:,0:1]], dim=1))

            out = out.reshape(grid_shape)
#     
            grid_arrays = [np.linspace(0, 1, grid_shape[1]),
                            np.linspace(0, 1, grid_shape[2])]

            X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')


            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            ax.plot_surface(X_grid, Y_grid, out[timestep_indx,:,:].cpu().detach(), cmap='viridis') 
            plt.show()


    def visualize_loss_pde(self, x, grid_shape, timestep_indx=0):
        with torch.no_grad():
            out = self.forward(x=x[:,1:], t=x[:,0:1])

            out = out.reshape(grid_shape)

            pde_residual = pde(u=out, x=x[:,1:], t=x[:,0:1], sigma_d=e_ds)
#     
            # grid_arrays = [np.linspace(0, 1, grid_shape[1]),
            #                 np.linspace(0, 1, grid_shape[2])]

            # X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')


            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # ax.plot_surface(X_grid, Y_grid, out[timestep_indx,:,:].cpu().detach(), cmap='viridis') 
            # plt.show()

    def visualize_animate(self, x, grid_shape, savevideo=False):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


        



        with torch.no_grad():
            out = self.forward(x=x[:,1:], t=x[:,0:1])
            # out = self.layers(torch.cat([x[:,1:],x[:,0:1]], dim=1))

            out = out.reshape(grid_shape)
#     
            grid_arrays = [np.linspace(0, 1, grid_shape[1]),
                            np.linspace(0, 1, grid_shape[2])]

            X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')

            def update_surface(timestep_indx):
                ax.clear()
                ax.plot_surface(X_grid, Y_grid, out[timestep_indx,:,:].cpu().detach(), cmap='viridis') 
                ax.set_zlim(-.2,1)

                plt.title(f'Timestep = {timestep_indx}')


            def update(indx):
                update_surface(indx)

            
            ani = FuncAnimation(fig, update, frames=out.shape[0], repeat=True)

            filename = f'./outputs/pinn_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
            ani.save(filename=filename, writer='pillow')

            plt.show()






if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    test_data, meshgrid_shape = get_test_points(10)


    pinn = PINN(3, [16]*8, 1)

    ic_x = torch.Tensor([[0.91,0.91],[0.5,0.9],[0.3,0.3],[0.2,0.9]])
    ic_t = torch.Tensor([[0],[0],[0],[0]])

    print(pinn(x=ic_x, t=ic_t))

    print(test_data)
    pinn.visualize(test_data, grid_shape=meshgrid_shape, timestep_indx=0)