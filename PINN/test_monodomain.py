import torch
from monodomain import f, pde, loss_neumann


def test_pde():
    x = torch.randn((200, 2), requires_grad=True)
    t = torch.randn((200, 1), requires_grad=True)
    sigma = torch.randn(1)

    u = (x ** 2).sum(dim=-1, keepdims=True) + t ** 2 + 2

    res = pde(u, x, t, sigma)

    # analytical residual
    expected = sigma*4 - 2*t - f(u)

    assert res.shape == t.shape
    assert torch.allclose(res, expected, atol=1e-6)


def test_neumann_loss():
    tmp = torch.randn((200, 1))
    zeros = torch.zeros_like(tmp)
    ones = torch.ones_like(tmp)
    t = torch.randn((200, 1), requires_grad=True)

    # analytical solutions
    expected = torch.tensor([0., 4, 0, 4])
    for i, x in enumerate([
        torch.cat([zeros, tmp], dim=1), torch.cat([ones, tmp], dim=1),
        torch.cat([tmp, zeros], dim=1), torch.cat([tmp, ones], dim=1)
    ]):
        x.requires_grad = True
        t.grad = None

        u = (x ** 2).sum(dim=-1, keepdims=True) + t**2 + 2
        res = loss_neumann(u, x)

        assert res.shape == torch.tensor(0).shape
        assert torch.allclose(res, expected[i], atol=1e-6)


if __name__ == '__main__':
    test_pde()
    test_neumann_loss()
    print("All tests passed")
