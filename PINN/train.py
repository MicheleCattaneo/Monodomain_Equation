import torch
from tqdm import tqdm
import random
import numpy as np


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = random.randint(0, 1000000000)
fix_seed(seed)

from monodomain import loss_pde, loss_neumann, loss_ic
from data import get_test_points, get_data, get_initial_conditions_collocation_points
from pinn import PINN
import matplotlib.pyplot as plt

if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------
    # defines the model
    model = PINN(in_size=3, hidden_sizes=[64] * 4, out_size=1)

    model = model.to(device)
    # -----------------
    # define the training data
    transform = lambda x: torch.tensor(x).to(device).to(torch.float64)

    t, x, tbc, xbc, sigma = get_data(num_cp=10_000, num_b_cp=1000)

    ic_t, ic_x = get_initial_conditions_collocation_points(1000)
    ic_t = transform(ic_t)
    ic_x = transform(ic_x)

    x = transform(x).requires_grad_(True)
    t = transform(t).requires_grad_(True)
    xbc = transform(xbc).requires_grad_(True)
    tbc = transform(tbc).requires_grad_(True)
    sigma = transform(sigma)

    w_pde = torch.nn.Parameter(torch.tensor([2.0], requires_grad=True))
    w_bc = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))
    w_ic = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

    num_points = t.shape[0] + tbc.shape[0]
    # point_wise_weights = torch.nn.Parameter(torch.ones(num_points, 1).to(device).to(torch.float64), requires_grad=True)

    optim = torch.optim.Adam(model.parameters(), lr=8e-5)

    # point_wise_optim = torch.optim.Adam([point_wise_weights], lr=0.01, maximize=True)

    # weight_optim = torch.optim.Adam([w_pde, w_bc], lr=0.001, maximize=True)

    # -----------------
    # Training loop
    epochs = 1000
    # Defines the amount of epochs when hard constraints are introduced.
    # Before that, only a soft loss is applied
    hard_ic_epochs = epochs // 2

    progress_bar = tqdm(total=epochs, position=0, leave=False)

    losses = []
    weights_pde = []
    weights_bc = []

    # Train with adam
    for e in range(epochs):
        model.train()
        optim.zero_grad()
        # weight_optim.zero_grad() # uncomment to train loss weights
        # point_wise_optim.zero_grad()

        use_hc = e > hard_ic_epochs

        u = model(x=x, t=t, hard_ic=use_hc)

        # loss_domain = loss_pde(u, x, t, sigma, weights=point_wise_weights[:t.shape[0]])
        loss_domain = loss_pde(u, x, t, sigma, weights=None)

        ubc = model(xbc, tbc, hard_ic=use_hc)
        # loss_bc = loss_neumann(ubc, xbc, weights=point_wise_weights[t.shape[0]:])
        loss_bc = loss_neumann(ubc, xbc, weights=None)

        loss = w_pde.to(device) * loss_domain + w_bc.to(device) * loss_bc
        # loss IC if needed (soft constraints)
        if not use_hc:
            u_ic = model(x=ic_x, t=ic_t, hard_ic=False)
            loss_init = loss_ic(u_ic, ic_x)
            loss += w_ic.to(device) * loss_init

        losses.append(loss.item())

        loss.backward()

        optim.step()
        # weight_optim.step() # uncomment to train loss weights 
        # point_wise_optim.step()

        weights_pde.append(w_pde.item())
        weights_bc.append(w_bc.item())

        # print training stats
        if e % 10 == 0:
            progress_bar.set_description(
                f'Loss: {loss.item()} | Pde loss {loss_domain.item()} | Bc loss {loss_bc.item()} |')
        progress_bar.update(1)

    # -----------------
    # Test (adam) solution on uniform grid

    test_data, meshgrid_shape, test_sigmas = get_test_points(50)
    test_sigmas = torch.tensor(test_sigmas).to(device).to(torch.float64)
    test_data = test_data.to(device)

    # model.visualize_loss_pde(test_data, grid_shape=meshgrid_shape,
    #                          sigma=test_sigmas, 
    #                          savevideo=False)

    model.visualize(test_data, grid_shape=meshgrid_shape, timestep_indx=0)
    model.visualize_animate(test_data, grid_shape=meshgrid_shape, savevideo=True)
    # model.visualize_animate(test_data, grid_shape=meshgrid_shape, nn_only=True, savevideo=False)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    ax[0].semilogy(losses, label='overall loss')

    ax[1].plot(weights_pde, label='w_pde')
    ax[2].plot(weights_bc, label='w_bc')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

    # -----------------
    # Train with BFGS
    bfgs = torch.optim.LBFGS(model.parameters(), history_size=40, max_iter=50, line_search_fn='strong_wolfe')
    bfgs_losses = []

    bfgs_epochs = 150


    # hard_ic_epochs = bfgs_epochs // 3

    def closure():
        # training step function used by BFGS

        bfgs.zero_grad()

        # use_hc = e > hard_ic_epochs
        use_hc = True
        u = model(x=x, t=t, hard_ic=use_hc)

        loss_domain = loss_pde(u, x, t, sigma)
        ubc = model(x=xbc, t=tbc, hard_ic=use_hc)
        loss_bc = loss_neumann(ubc, xbc)

        # u_ic = model(x=ic_x, t=ic_t)
        # loss_init = loss_ic(u_ic, ic_x)

        loss = w_pde.to(device) * loss_domain + w_bc.to(device) * loss_bc

        if not use_hc:
            u_ic = model(x=ic_x, t=ic_t, hard_ic=False)
            loss_init = loss_ic(u_ic, ic_x)
            loss += w_ic.to(device) * loss_init

        bfgs_losses.append(loss.item())

        loss.backward()

        return loss


    progress_bar2 = tqdm(total=bfgs_epochs, position=0, leave=False)

    for e in range(bfgs_epochs):
        bfgs.step(closure=closure)

        if e % 1 == 0:
            progress_bar2.set_description(f'BFGS Loss {bfgs_losses[-1]}')
        progress_bar2.update(1)

    model.visualize_animate(test_data, grid_shape=meshgrid_shape, savevideo=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.semilogy(bfgs_losses, label='BFGS loss')
    ax.legend()
    plt.show()
