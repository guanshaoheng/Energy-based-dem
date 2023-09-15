import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils_particle import write_down_infor, plot_particles, overlap_check, plot_E_history
from utils.general import check_mkdir, echo
from utils.utils_torch import findDevice


def main(
        packing='dense',  # dense triangle
        device=findDevice()
):
    # load information
    save_dir_name = './img/%s_PSD' % packing
    check_mkdir(save_dir_name,
                os.path.join(save_dir_name, 'laod_infor'))

    # material constants
    youngs = 1e6
    r = 0.5  # radius of the particle (or the mean radius if graded)
    rho = 2600
    g = 9.8
    E, E_contact, E_potential = [], [], []
    patience = 10

    # particle positions
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
    rr = np.random.rand(len(xx))*0.5+0.2

    plot_particles(xx=xx, r=rr, step=0, save_dir_name=None, force_flag=False)

    # set a torch tensor to save the particles' coordinates
    x = torch.tensor(xx, device=device).float().requires_grad_()
    r = torch.tensor(rr, device=device, requires_grad=False).float()

    # boundary
    left_boundary = torch.nn.Parameter(torch.ones(1) * 0., requires_grad=False)
    right_boundary = torch.nn.Parameter(torch.ones(1) * 11., requires_grad=False)
    top_boundary = torch.nn.Parameter(torch.ones(1) * 5.5, requires_grad=False)
    bottom_boundary = torch.nn.Parameter(torch.ones(1) * 0., requires_grad=True)

    # optimizer
    optimizer = torch.optim.Adam([x])

    energy_lowest, trial_num = 1e32, 0
    for i in range(10000 + 1):
        # ---------------------------------------
        # Energy calculation

        # calculate the contact energy
        overlap = overlap_check(x=x, r=r)[0]
        energy_contact = 0.5 * torch.sum(overlap ** 2 * youngs)

        # gravity potential
        energy_potential = torch.sum(x[:, 1] * torch.pi * r ** 2 * rho * g)

        energy = energy_contact + energy_potential

        # ----------------------------------------
        # torch.autograd.grad(overlap, x, torch.ones_like(overlap), retain_graph=True)
        # update
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        # add the boundary condition
        with torch.no_grad():
            x[:, 0] = torch.where(x[:, 0] - r - left_boundary[0] < 0., left_boundary[0] + r, x[:, 0])   # left
            x[:, 0] = torch.where(x[:, 0] + r - right_boundary[0] > 0., right_boundary[0] - r, x[:, 0]) # right
            x[:, 1] = torch.where(x[:, 1] - r - bottom_boundary[0] < 0., bottom_boundary[0] + r, x[:, 1]) # bottom
            x[:, 1] = torch.where(x[:, 1] + r - top_boundary[0] > 0., top_boundary[0] - r, x[:, 1])    # top

        E.append(energy.item())
        E_potential.append(energy_potential.item())
        E_contact.append(energy_contact.item())

        if i % 1000 == 0 and i != 0:
            x_numpy = x.cpu().detach().numpy()
            plot_particles(
                xx=x_numpy, r=rr, step=i,
                energy=energy.item(), energy_potential=energy_potential.item(), energy_contact=energy_contact.item(),
                save_dir_name=save_dir_name, force_flag=True)

        if i % 100 == 0:
            echo('Epoch %d Total energy: %.2e, contact energy: %.2e, potential: %.2e' %
                 (i, energy.item(), energy_contact.item(), energy_potential.item()))
            write_down_infor(xx=x, rr=rr, save_dir_name=save_dir_name, step=i,
                energy=energy.item(), energy_potential=energy_potential.item(), energy_contact=energy_contact.item())

            # check if the engergy is still decreasing
            if energy.item() < energy_lowest:
                trial_num = 0
                energy_lowest = energy.item()
            else:
                trial_num += 1
                if trial_num > patience:
                    echo('No improvement, optimization stop!')
                    break

    # plot the particles at the final state
    plot_particles(
        xx=x.cpu().detach().numpy(), r=rr, step=-1,
        energy=energy.item(), energy_potential=energy_potential.item(), energy_contact=energy_contact.item(),
        save_dir_name=save_dir_name, force_flag=True)

    # plot the training history
    plot_E_history(E=E, E_contact=E_contact, E_potential=E_potential, save_dir_name=save_dir_name)
    return


if __name__ == '__main__':
    main()
