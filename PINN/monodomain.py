import torch


SIGMA_H = 9.5298e-4
A = 18.515
Fr = 0
Ft = 0.2383
Fd = 1


def pde(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, sigma_d: torch.Tensor) -> torch.Tensor:
    # ∇·(Σ_h ∇u) + ∇·(Σ_d ∇u) - f(u) - du/dt = 0
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    uxx_h = torch.autograd.grad(ux * SIGMA_H, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uxx_d = torch.autograd.grad(ux * sigma_d, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    return uxx_h + uxx_d - f(u) - ut


def f(u: torch.Tensor) -> torch.Tensor:
    return A * (u - Fr) * (u - Ft) * (u - Fd)


def u0(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0.9).all().float()


def neumann_bc(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # ∇u = 0
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return ux


def loss(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, sigma_d: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pde(u, x, t, sigma_d), torch.zeros_like(u))


def neumann_loss(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(neumann_bc(u, x), torch.zeros_like(u))