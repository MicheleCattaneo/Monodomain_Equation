import skopt
import numpy as np
import matplotlib.pyplot as plt
import torch

from monodomain import SIGMA_D, SIGMA_H, diseased_areas, Tf


def pad_with_boundaries(x_bc):
    '''
    Pads the input with (spatial) boundary values (0 and 1) given a tensor of D-1 dimensions
    where the first dimension is the batch, and second timension is time.
    '''

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


def get_initial_conditions_collocation_points(n):
    '''
    Return the time tensor and the spatial tensor for initial conditions, such that t is full of 0s
    and the spatial tensor contains sampled points.
    '''
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    boundary = [(0., 1.), (0., 1.)]
    points = np.array(sampler.generate(boundary, n))

    return np.zeros((n, 1)), points


def get_collocation_points(num_cp, num_b_cp):
    ''' 
    Returns randomly sampled collocation points for the internal domain and 
    the boundaries. Returns 4 tensors since because time and space are 
    kept in separate tensors.
    '''

    space = [(0., Tf), (0., 1.), (0., 1.)]
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    internal_points = np.array(sampler.generate(space, num_cp))

    boundary = [(0., Tf), (0., 1.)]
    bc_points = np.array(sampler.generate(boundary, num_b_cp))

    bc_points = pad_with_boundaries(bc_points)

    return internal_points[:, :1], internal_points[:, 1:], bc_points[:, :1], bc_points[:, 1:]


def get_sigmas(points, diseased_areas):
    # Returns a mask with the sigma value corresponding to each collocation point
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    masks = np.zeros(points.shape[0])
    for d in diseased_areas:
        dist_from_center = np.sqrt(np.sum((points - d['center']) ** 2, axis=-1))
        masks = np.logical_or(masks, dist_from_center < d['radius'])

    return np.where(masks, SIGMA_D, SIGMA_H)


def get_mask(points, diseased_areas):
    # Returns an identified for each collocation point where 0 is reserved for healthy tissue.
    # Diseased tissue have identifiers 1,2, ...
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
    # Displays collocation points with their identifier (healthy or diseased) as a color

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
    # Returns a uniform grid of collocation points to test and visualize the PINN solution 
    grid_arrays = [np.linspace(0, Tf, points_per_dim),
                   np.linspace(0, 1, points_per_dim),
                   np.linspace(0, 1, points_per_dim)]

    meshgrid_arrays = np.meshgrid(*grid_arrays, indexing='ij')
    test_collocation = np.vstack([elem.ravel() for elem in meshgrid_arrays]).T
    test_collocation = torch.tensor(test_collocation).to(torch.float64)

    sigmas = get_sigmas(test_collocation[:, 1:], diseased_areas=diseased_areas)

    return test_collocation, meshgrid_arrays[0].shape, sigmas


def get_data(num_cp=10000, num_b_cp=100):
    # Returns internal and boundary collocation points, together with their corresponding sigma value
    ip_t, ip_x, bc_t, bc_x = get_collocation_points(num_cp=num_cp, num_b_cp=num_b_cp)

    simgas = get_sigmas(ip_x, diseased_areas=diseased_areas)

    return ip_t, ip_x, bc_t, bc_x, simgas


class MonodomainDataset(torch.utils.data.Dataset):
    # Batchless  TorchLightning dataset 
    def __init__(self, num_cp=10000, num_b_cp=100, dim=2):
        self.ip_t, self.ip_x, self.bc_t, self.bc_x, self.e_d_masks = get_data(num_cp=num_cp, num_b_cp=num_b_cp, dim=dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return list(map(
            lambda x: torch.tensor(x).to(torch.float64).requires_grad_(True),
            (self.ip_x, self.ip_t, self.bc_x, self.bc_t, self.e_d_masks)
        ))


if __name__ == '__main__':
    ip_t, ip_x, bc_t, bc_x = get_collocation_points(num_cp=10, num_b_cp=10)

    diseased_areas = [{'center': np.array([0.3, 0.7]), 'radius': 0.1},
                      {'center': np.array([0.5, 0.5]), 'radius': 0.1},
                      {'center': np.array([0.7, 0.3]), 'radius': 0.15}]

    mask = get_mask(ip_x, diseased_areas=diseased_areas)

    plot_collocation_points(ip_x, mask=mask)

    print(mask.shape)
