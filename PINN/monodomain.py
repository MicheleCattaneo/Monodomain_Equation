import torch
import numpy as np

Tf = 5

SIGMA_H = 9.5298e-4
A = 18.515
Fr = 0
Ft = 0.2383
Fd = 1

SIGMA_D = [9.5298e-3, 9.5298e-4, 9.5298e-5][1]
# SIGMA_D = np.array([9.5298e-4] * 4)

diseased_areas = [
    {'center': np.array([0.3, 0.7]), 'radius': 0.1},
    {'center': np.array([0.5, 0.5]), 'radius': 0.1},
    {'center': np.array([0.7, 0.3]), 'radius': 0.15}
]


def pde(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # Returns the residual between for approximate solution u

    # ∇·(Σ_h ∇u) + ∇·(Σ_d ∇u) - f(u) - du/dt = 0
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    uxx = torch.autograd.grad(ux * sigma[..., None], x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    return uxx.sum(dim=-1, keepdims=True) - f(u) - ut


def f(u: torch.Tensor) -> torch.Tensor:
    # Reaction term function
    return A * (u - Fr) * (u - Ft) * (u - Fd)


def u0(x: torch.Tensor) -> torch.Tensor:
    # Initial conditions functions, when t=0
    return (x >= 0.9).all(dim=1).double().unsqueeze(-1)


def neumann_bc(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Returns the gradient of u wrt to x, where x should be points on the 
    # domain boundaries.

    # ∇u = 0
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return ux


def loss_pde(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    # Returns the PDE loss value 

    residual = pde(u, x, t, sigma)
    if weights is None:
        return ((residual ** 2)).mean()

    return ((residual ** 2) * weights).mean()


def loss_neumann(u: torch.Tensor, x: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    # Returns the Neumann boundary conditions loss value

    # we wanna allow gradients along the boundary so we mask them out and only penalize
    # gradients perpendicular to the boundary.
    mask = (x == 0.) | (x == 1.)
    dudx = neumann_bc(u, x)
    masked_dudx = dudx * mask
    
    if weights is None:
        return ((masked_dudx ** 2) ).sum() / mask.sum()

    return ((masked_dudx ** 2) * weights ).sum() / mask.sum()


def loss_ic(u, x):
    # Returns the initial condition loss value (soft constraints)
    target = u0(x)

    return torch.nn.functional.mse_loss(u, target)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    xs = np.linspace(-.2, 1.2, 100)
    plt.title("f(u)")
    plt.xlabel("u")
    plt.ylabel("f(u)")
    plt.plot(xs, f(xs))
    plt.show()
