import torch
import torch.nn as nn
from monodomain import u0, Tf
from data import get_test_points
import matplotlib.pyplot as plt
import numpy as np


class PINN(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super().__init__()

        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_size, hs))
            layers.append(nn.Tanh())
            in_size = hs

        layers.append(nn.Linear(hidden_sizes[-1],out_size))

        self.layers = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        out = self.layers(torch.cat([x, t], dim=1))
        return (1 - t/Tf) * u0(x) + t/Tf * out
    

    def visualize(self, x, grid_shape, timestep_indx=0):
        with torch.no_grad():
            out = self.forward(x=x[:,1:], t=x[:,0:1])

            out = out.reshape(grid_shape)
#     
            grid_arrays = [np.linspace(0, 1, grid_shape[1]),
                            np.linspace(0, 1, grid_shape[2])]

            X_grid, Y_grid = np.meshgrid(*grid_arrays, indexing='ij')


            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            ax.plot_surface(X_grid, Y_grid, out[timestep_indx,:,:].cpu().detach(), cmap='viridis') 
            plt.show()





if __name__ == '__main__':

    test_data, meshgrid_shape = get_test_points(10)


    pinn = PINN(3, [16]*8, 1)

    ic_x = torch.Tensor([[0.91,0.91],[0.5,0.9],[0.3,0.3],[0.2,0.9]])
    ic_t = torch.Tensor([[0],[0],[0],[0]])

    print(pinn(x=ic_x, t=ic_t))

    print(test_data)
    pinn.visualize(test_data, grid_shape=meshgrid_shape, timestep_indx=0)