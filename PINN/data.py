import skopt
import numpy as np
import matplotlib.pyplot as plt
import torch

from monodomain import e_ds, diseased_areas, Tf


def pad_with_boundaries(x_bc):
    results = []

    for i in range(1, x_bc.shape[-1] + 1):
        left_part = x_bc[:, :i]
        right_part = x_bc[:, i:]
        zeros_tensor = np.ones_like(x_bc[:, 0:1])
        results.append(np.concatenate((left_part, zeros_tensor, right_part), axis=1))

    for i in range(1, x_bc.shape[-1] + 1):
        left_part = x_bc[:, :i]
        right_part = x_bc[:, i:]
        zeros_tensor = np.zeros_like(x_bc[:, 0:1])
        results.append(np.concatenate((left_part, zeros_tensor, right_part), axis=1))

    return np.concatenate(results)


def get_collocation_points(num_cp, num_b_cp):
    # internal collocation points 

    space = [(0., Tf), (0., 1.), (0., 1.)]
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    internal_points = np.array(sampler.generate(space, num_cp))

    boundary = [(0., Tf), (0., 1.)]
    bc_points = np.array(sampler.generate(boundary, num_b_cp))

    bc_points = pad_with_boundaries(bc_points)

    return internal_points[:, :1], internal_points[:, 1:], bc_points[:, :1], bc_points[:, 1:]


def get_mask(points, diseased_areas):
    def is_in(point, center, radius):
        distance = np.sqrt(np.sum((point - center) ** 2))
        return distance <= radius

    masks = []
    for i, des_area in enumerate(diseased_areas):
        center = des_area['center']
        radius = des_area['radius']
        mask = np.apply_along_axis(lambda x: i + 1 if is_in(x, center, radius) else 0, axis=1, arr=points)

        masks.append(mask)
    return np.sum(np.stack(masks), axis=0)


def get_electrical_diffusivity_mask(mask, e_ds):
    return e_ds[mask]


def plot_collocation_points(cp, mask):
    colors = ['lightgreen', 'darkorchid', 'orangered', 'crimson']
    labels = ['Healthy', 'Diseased 1', 'Diseased 2', 'Diseased 3']
    x = cp[:, 0]
    y = cp[:, 1]

    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    for i, color in enumerate(colors):
        plt.scatter(x[mask == i], y[mask == i], color=color, label=labels[i], s=5)

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Collocation points')
    plt.legend()

    plt.show()


def get_test_points(points_per_dim):
    grid_arrays = [np.linspace(0, Tf, points_per_dim),
                   np.linspace(0, 1, points_per_dim),
                   np.linspace(0, 1, points_per_dim)]

    meshgrid_arrays = np.meshgrid(*grid_arrays, indexing='ij')
    test_collocation = np.vstack([elem.ravel() for elem in meshgrid_arrays]).T
    test_collocation = torch.tensor(test_collocation).to(torch.float32)
    return test_collocation


def get_data(num_cp=10000, num_b_cp=100, dim=2):
    ip_t, ip_x, bc_t, bc_x = get_collocation_points(num_cp=num_cp, num_b_cp=num_b_cp)

    mask = get_mask(ip_x, diseased_areas=diseased_areas)

    e_d_masks = get_electrical_diffusivity_mask(mask=mask, e_ds=e_ds)

    return ip_t, ip_x, bc_t, bc_x, e_d_masks


class MonodomainDataset(torch.utils.data.Dataset):
    def __init__(self, num_cp=10000, num_b_cp=100, dim=2):
        self.ip_t, self.ip_x, self.bc_t, self.bc_x, self.e_d_masks = get_data(num_cp=num_cp, num_b_cp=num_b_cp, dim=dim)

    def __len__(self):
        return len(self.ip_t)

    def __getitem__(self, idx):
        return list(map(
            lambda x: torch.tensor(x).to(torch.float32).requires_grad_(True),
            (self.ip_x[idx], self.ip_t[idx], self.bc_x[idx], self.bc_t[idx], self.e_d_masks[idx])
        ))


if __name__ == '__main__':
    ip_t, ip_x, bc_t, bc_x = get_collocation_points(num_cp=10, num_b_cp=10)

    diseased_areas = [{'center': np.array([0.3, 0.7]), 'radius': 0.1},
                      {'center': np.array([0.5, 0.5]), 'radius': 0.1},
                      {'center': np.array([0.7, 0.3]), 'radius': 0.15}]

    mask = get_mask(ip_x, diseased_areas=diseased_areas)

    plot_collocation_points(ip_x, mask=mask)

    print(mask.shape)

    e_d_masks = get_electrical_diffusivity_mask(mask=mask, e_ds=e_ds)

    print(e_d_masks.shape)
