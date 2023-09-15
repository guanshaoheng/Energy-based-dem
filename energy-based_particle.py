import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.general import check_mkdir, echo
from utils.utils_torch import findDevice


def main(
        packing='unstable',  # dense stable unstable disturbed_top, disturbed_bottom
        device=findDevice()
):
    # material constants
    E = 1e6
    r = 0.5  # radius of the particle
    rho = 2600
    g = 9.8

    # particle positions
    disturbed_index = None
    if packing == 'dense':
        X, Y = np.meshgrid(np.linspace(1, 10, 10), np.linspace(1, 5, 5))
        X += np.array([r, 0, r, 0, r]).reshape([-1, 1])
        xx = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    else:
        xx = []
        for i in range(3, 8):
            for j in range(i - 2, 8 - 2):
                if packing == 'stable':
                    xx.append([i + (j % 2) * r, j])
                else:
                    xx.append([i, j])
        xx = np.array(xx, dtype=float)
        if packing == 'disturbed_top':
            xx[4, 0] -= 0.01
            disturbed_index = 4
        elif packing == 'disturbed_bottom':
            xx[0, 0] += 0.01
            disturbed_index = 0

    plot_particles(xx, r, step=0, disturbed_index=disturbed_index, packing=packing)

    # set a torch tensor to save the particles' coordinates
    x = torch.tensor(xx, device=device).float().requires_grad_()

    # optimizer
    optimizer = torch.optim.Adam([x])

    for i in range(10000 + 1):
        # ---------------------------------------
        # Energy calculation

        # calculate the contact energy
        overlap = overlap_check(x=x, r=r)
        energy_contact = 0.5 * torch.sum(overlap ** 2 * E)  # f = E * \delta, W = f* \delta = E * \delta^2

        # gravity potential
        energy_potential = torch.sum(x[:, 1] * torch.pi * r ** 2 * rho * g)  # W_potential = mgh

        # boundary penalty
        energy_boundary_penalty = torch.sum(
            (
                    torch.where(x[:, 0] < 0 + r, torch.abs(x[:, 0] - r), 0.) +
                    torch.where(x[:, 0] > 11 - r, torch.abs(x[:, 0] + r - 11.), 0.) +
                    torch.where(x[:, 1] < r, torch.abs(x[:, 1] - r), 0.)) * torch.pi * r ** 2 * rho * g * 10.
        )

        energy = energy_contact + energy_potential + energy_boundary_penalty
        if i % 100 == 0:
            echo('Epoch %d Total energy: %.2e, contact energy: %.2e, potential: %.2e' %
                 (i, energy.item(), energy_contact.item(), energy_potential.item()))

        # ----------------------------------------
        # torch.autograd.grad(overlap, x, torch.ones_like(overlap), retain_graph=True)
        # update
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        if i % 1000 == 0 and i != 0:
            x_numpy = x.cpu().detach().numpy()
            plot_particles(
                xx=x_numpy, r=r, step=i, disturbed_index=disturbed_index, packing=packing,
                energy=energy.item(), energy_potential=energy_potential.item(), energy_contact=energy_contact.item())
    return


def overlap_check(x: torch.Tensor, r):
    """
            x in shape of (num_samples, ndim)
    """
    device_num = x.get_device()
    device = torch.device('cuda:%d' % device_num) if device_num >= 0 else torch.device('cpu')
    x0, x1 = x[:, 0:1], x[:, 1:2]
    x0_diff = torch.add(x0, x0.T, alpha=-1)
    x1_diff = torch.add(x1, x1.T, alpha=-1)
    # here is important to add a small value to make sure the gradient calculation do not come to NAN
    distance = torch.sqrt(x0_diff ** 2 + x1_diff ** 2 + 1e-8) + torch.eye(len(x), device=device) * 2. * r
    overlap = 2. * r - distance
    overlap = torch.where(overlap < 0., 0., overlap)  # set the un-contacted overlap as 0
    return overlap


def plot_particles(xx: np.ndarray, r: float, step: int, packing: str,
                   energy=None, energy_potential=None, energy_contact=None,
                   disturbed_index=None):
    fig, ax = plt.subplots()
    for i in range(len(xx)):
        if disturbed_index != i:
            c = plt.Circle(xx[i], r, edgecolor='k')
        else:
            c = plt.Circle(xx[i], r, color='r')
        ax.add_patch(c)
    plt.xlim([0, 11])
    plt.ylim([-1, 6])
    plt.plot([0, 0], [-1, 6], linewidth=3, c='k')
    plt.plot([11, 11], [-1, 6], linewidth=3, c='k')
    plt.plot([0, 11], [0, 0], linewidth=3, c='k')
    plt.axis('equal')
    plt.axis('off')
    if step == 0:
        plt.title(r'$ \mathrm{Step} %d $' % step)
    else:
        plt.title(r'$ \mathrm{Step} \  %d \ E=%.1e \ E_{potential}=%.1e \ E_{contact}=%.1e $' %
                  (step, energy, energy_potential, energy_contact))
    # plt.show()
    dir_name = './img/%s' % packing
    check_mkdir(dir_name)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, 'step_%d.png' % step), dpi=100)
    plt.close()


if __name__ == '__main__':
    main()
