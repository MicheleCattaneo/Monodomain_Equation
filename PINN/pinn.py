import torch
import torch.nn as nn


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

    def forward(self, x):
        return self.layers(x)




if __name__ == '__main__':

    pinn = PINN(10, [16]*8, 1)

    print(pinn)