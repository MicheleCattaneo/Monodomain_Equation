import torch
from tqdm import tqdm

from monodomain import loss_pde, loss_neumann
from data import MonodomainDataset, get_test_points, get_data
from pinn import PINN


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(in_size=3, hidden_sizes=[32]*2, out_size=1)

    model = model.to(device)

    t, x, tbc, xbc, sigma_d = get_data(num_cp=1_000, num_b_cp=100, dim=2)

    x = torch.tensor(x).to(device).to(torch.float32).requires_grad_(True)
    t = torch.tensor(t).to(device).to(torch.float32).requires_grad_(True)
    xbc = torch.tensor(xbc).to(device).to(torch.float32).requires_grad_(True)
    tbc = torch.tensor(tbc).to(device).to(torch.float32).requires_grad_(True)
    sigma_d = torch.tensor(sigma_d).to(device).to(torch.float32).requires_grad_(True)

    
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 10000

    w_pde = 1000
    w_bc = 500

    progress_bar = tqdm(total=epochs, position=0, leave=False)

    # Train with adam
    for e in range(epochs):
        optim.zero_grad()

        model.train()

        u = model(x=x, t=t)

        loss_domain = loss_pde(u, x, t, sigma_d)
        ubc = model(xbc, tbc)
        loss_bc = loss_neumann(ubc, xbc)

        loss = w_pde * loss_domain + w_bc* loss_bc

        loss.backward()

        optim.step()
        if e % 100 == 0:
            progress_bar.set_description(f'Loss: {loss.item()} | Pde loss {loss_domain.item()} | Bc loss {loss_bc.item()}')
        progress_bar.update(1)


    test_data, meshgrid_shape = get_test_points(50)
    test_data = test_data.to(device)
    model.visualize_animate(test_data, grid_shape=meshgrid_shape)

    # # Train with BFGS
    # bfgs = torch.optim.LBFGS(model.parameters(), history_size=50, max_iter=50, line_search_fn='strong_wolfe')
    # bfgs_losses = []
    # def closure():

    #     u = model(x, t)

    #     loss_domain = loss_pde(u, x, t, sigma_d)
    #     ubc = model(xbc, tbc)
    #     loss_bc = loss_neumann(ubc, xbc)

    #     loss = w_pde * loss_domain + w_bc* loss_bc

    #     bfgs_losses.append(loss.item())

    #     loss.backward()

    #     return loss
    
    # bfgs_epochs = 100
    # progress_bar2 = tqdm(total=bfgs_epochs, position=0, leave=False)

    # for e in range(bfgs_epochs):
    #     bfgs.step(closure=closure)

    #     if e % 1 == 0:
    #         progress_bar2.set_description(f'BFGS Loss {bfgs_losses[-1]}')
    #     progress_bar2.update(1)


    