import skopt
import numpy as np
import matplotlib.pyplot as plt

def u_0(x):
    return 1 if (x[0] >= 0.9 and x[1] >= 0.9) else 0


def pad_with_boundaries(x_bc):
    results = []
    
    for i in range(1,x_bc.shape[-1]+1):  
        left_part = x_bc[:,:i]
        right_part = x_bc[:,i:]
        zeros_tensor = np.ones_like(x_bc[:,0:1])
        results.append(np.concatenate((left_part, zeros_tensor, right_part), axis=1))
        
    for i in range(1,x_bc.shape[-1]+1):  
        left_part = x_bc[:,:i]
        right_part = x_bc[:,i:]
        zeros_tensor = np.zeros_like(x_bc[:,0:1])
        results.append(np.concatenate((left_part, zeros_tensor, right_part), axis=1))
        
    return np.concatenate(results)


def get_collocation_points(dim, num_cp, num_b_cp):
    # internal collocation points 

    space = [(0., 1.)] * (dim+1) # +1 for time
    sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    internal_points = np.array(sampler.generate(space, num_cp))


    boundary = [(0., 1.)]*dim
    bc_points = np.array(sampler.generate(boundary , num_b_cp))

    # ic_points = np.concatenate([np.zeros_like(bc_points[:,0:1]), bc_points],axis=1)

    bc_points = pad_with_boundaries(bc_points)

    return internal_points[:, :1], internal_points[:, 1:], internal_points[:, :1], bc_points[:, 1:]
    

def get_mask(points, deseased_areas):

    def is_in(point,center, radius):
        distance = np.sqrt(np.sum((point-center)**2))
        return distance <= radius

    masks = []
    for i, des_area in enumerate(deseased_areas):
        center = des_area['center']
        radius = des_area['radius']
        mask = np.apply_along_axis(lambda x: i+1 if is_in(x, center, radius) else 0, axis=1, arr=points)

        masks.append(mask)
    return np.sum(np.stack(masks),axis=0)

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


if __name__ == '__main__':

    ip_t, ip_x, bc_t, bc_x = get_collocation_points(dim=2, num_cp=10000, num_b_cp=100)


    deseased_areas=[{'center': np.array([0.3, 0.7]), 'radius': 0.1},
                    {'center': np.array([0.5, 0.5]), 'radius': 0.1},
                    {'center': np.array([0.7, 0.3]), 'radius': 0.15}]

    mask = get_mask(ip_x, deseased_areas=deseased_areas)

    plot_collocation_points(ip_x, mask=mask)

    print(mask.shape)

    e_d_masks = get_electrical_diffusivity_mask(mask=mask, e_ds=np.array([9.5298e-4, 
                                                                 9.5298e-3,
                                                                 9.5298e-4,
                                                                 9.5298e-5]))
    
    print(e_d_masks.shape)