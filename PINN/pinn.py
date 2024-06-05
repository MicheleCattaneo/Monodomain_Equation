import torch
import torch.nn as nn
from monodomain import u0, Tf


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


if __name__ == '__main__':

    pinn = PINN(3, [16]*8, 1)

    print(pinn(torch.rand(10, 2), torch.tensor([[0.]*5 + [1]*5]).T))

    print(pinn)