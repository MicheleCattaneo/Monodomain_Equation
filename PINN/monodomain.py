import torch
import numpy as np

Tf = 5

SIGMA_H = 9.5298e-4
A = 18.515
Fr = 0
Ft = 0.2383
Fd = 1

# e_ds = np.array([9.5298e-4, 9.5298e-3, 9.5298e-4, 9.5298e-5])
e_ds = np.array([9.5298e-4]*4)

diseased_areas = [
    {'center': np.array([0.3, 0.7]), 'radius': 0.1},
    {'center': np.array([0.5, 0.5]), 'radius': 0.1},
    {'center': np.array([0.7, 0.3]), 'radius': 0.15}
]


def pde(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, sigma_d: torch.Tensor) -> torch.Tensor:
    # ∇·(Σ_h ∇u) + ∇·(Σ_d ∇u) - f(u) - du/dt = 0
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    uxx_h = torch.autograd.grad(ux * SIGMA_H, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uxx_d = torch.autograd.grad(ux * sigma_d.unsqueeze(-1), x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    return uxx_h - f(u) - ut


def f(u: torch.Tensor) -> torch.Tensor:
    return A * (u - Fr) * (u - Ft) * (u - Fd)


def u0(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0.9).all(dim=1).double().unsqueeze(-1)


def neumann_bc(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # ∇u = 0
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return ux


def loss_pde(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, sigma_d: torch.Tensor) -> torch.Tensor:
    residual = pde(u, x, t, sigma_d)
    return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))


def loss_neumann(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # we wanna allow gradients along the boundary so we mask them out and only penalize
    # gradients perpendicular to the boundary.
    mask = (x == 0.) | (x == 1.)    
    dudx = neumann_bc(u, x)
    masked_dudx = dudx * mask

    return torch.nn.functional.mse_loss(masked_dudx, torch.zeros_like(x))


def loss_ic(u, x):
    target = u0(x)

    return torch.nn.functional.mse_loss(u, target)

