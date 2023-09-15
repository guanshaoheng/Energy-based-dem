import math
import os.path
import numpy as np
import torch
from utils.general import check_mkdir, echo
from utils.utils_torch import findDevice
from utils_particle import write_down_infor, plot_particles, plot_E_history, plot_f_history, Contact


# material constants
youngs = 1e6
r = 0.5  # radius of the particle (or the mean radius if graded)
rho = 2600
g = 9.8
thick = 1.0
device = findDevice()


def main(
        packing='dense',  # dense triangle
        device=device, r=r,
):
    # load information
    save_dir_name = './img/%s_PSD_servo' % packing
    check_mkdir(save_dir_name,
                os.path.join(save_dir_name, 'load_infor'))
    confining = 1e5

    E, E_contact, E_potential = [], [], []
    fx_list, fy_list = [], []
    np.random.seed(10001)

    patience = 20

    # particle positions
    if packing == 'dense':
        h_num = 5
        X, Y = np.meshgrid(np.linspace(1, 10, 10), np.linspace(1, 20, h_num))
        X += np.array(([r, 0]*math.ceil(h_num/2))[:h_num]).reshape([-1, 1])
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
    rr = np.random.rand(len(xx)) * 0.5 + 0.2

    plot_particles(xx=xx, r=rr, step=0, save_dir_name=None, force_flag=False)

    # set a torch tensor to save the particles' coordinates
    left_boundary = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    right_boundary = torch.nn.Parameter(torch.ones(1) * 11., requires_grad=False)
    top_boundary = torch.nn.Parameter(torch.ones(1) * 5.5, requires_grad=False)
    bottom_boundary = torch.nn.Parameter(torch.ones(1) * 0., requires_grad=True)
    x = torch.tensor(xx, device=device).float().requires_grad_()
    r = torch.tensor(rr, device=device, requires_grad=False).float()

    total_gravity = torch.sum(thick * torch.pi * r ** 2 * rho * g)

    # optimizer
    optimizer = torch.optim.Adam(params=[
            {'params': [x], }, ])
    # optimizer = torch.optim.LBFGS(params=[
    #     {'params': x, },])

    energy_lowest, trial_num = 1e32, 0
    for i in range(int(1e5) + 1):
        # ---------------------------------------
        # Energy calculation
        def closure():
            # calculate the contact energy
            overlap = overlap_check(x=x, r=r)

            energy_contact = 0.5 * youngs * torch.sum(torch.tril(overlap, diagonal=1) ** 2)

            # gravity potential
            energy_potential = torch.sum(x[:, 1] * thick * torch.pi * r ** 2 * rho * g)

            loss = energy_contact + energy_potential  # + energy_boundary_penalty
            return loss

        # ----------------------------------------
        # update
        optimizer.zero_grad()
        closure().backward()
        optimizer.step()
        # optimizer.step(closure=closure)

        # add the boundary condition
        with torch.no_grad():
            x[:, 0] = torch.where(x[:, 0] - r - left_boundary[0] < 0., left_boundary[0] + r, x[:, 0])      # left
            x[:, 0] = torch.where(x[:, 0] + r - right_boundary[0] > 0., right_boundary[0] - r, x[:, 0])    # right
            x[:, 1] = torch.where(x[:, 1] - r - bottom_boundary[0] < 0., bottom_boundary[0] + r, x[:, 1])  # bottom
            # x[:, 1] = torch.where(x[:, 1] + r - top_boundary[0] > 0., top_boundary[0] - r, x[:, 1])        # top

        if i % 100 == 0:
            # calculate the contact energy
            overlap = overlap_check(x=x, r=r)
            height = max(x[:, 1]) - min(x[:, 1])
            width = right_boundary[0] - left_boundary[0]
            stress_tensor = get_stress(height=height, width=width, x=x, overlap=overlap)

            force_x = stress_tensor[0, 0] * thick * height
            force_y = stress_tensor[1, 1] * thick * width

            #
            energy_contact = 0.5 * youngs * torch.sum(torch.tril(overlap, diagonal=1) ** 2)

            # gravity potential
            energy_potential = torch.sum(x[:, 1] * thick * torch.pi * r ** 2 * rho * g)

            energy = energy_contact + energy_potential  # + energy_boundary_penalty

            E.append(energy.item())
            E_potential.append(energy_potential.item())
            E_contact.append(energy_contact.item())
            fx_list.append(force_x.item())
            fy_list.append(force_y.item())
            echo('Epoch %d Total energy: %.2e, contact energy: %.2e, potential: %.2e fx_:%.2e fy_:%.2e gravity:%.2e' %
                 (i, energy.item(), energy_contact.item(), energy_potential.item(),
                  force_x.item(), force_y.item(), total_gravity.item()))
            write_down_infor(xx=x, rr=rr, save_dir_name=save_dir_name, step=i,
                             energy=energy.item(), energy_potential=energy_potential.item(),
                             energy_contact=energy_contact.item(),
                             fx=force_x.item(), fy=force_y.item(), gravity=total_gravity.item())

            # check if the engergy is still decreasing
            if energy.item() < energy_lowest:
                trial_num = 0
                energy_lowest = energy.item()
            else:
                trial_num += 1
                if trial_num > patience:
                    echo('No improvement, optimization stop!')
                    break

            if i % 1000 == 0 and i != 0:
                plot_particles(
                    xx=x.cpu().detach().numpy(), r=rr, step=i,
                    energy=energy.item(), energy_potential=energy_potential.item(), energy_contact=energy_contact.item(),
                    save_dir_name=save_dir_name, force_flag=True,
                    left_boundary=left_boundary[0].item(), right_boundary=right_boundary[0].item(),
                    top_boundary=top_boundary[0].item(), bottom_boundary=bottom_boundary[0].item(),)

    # plot the particles at the final state
    plot_particles(
        xx=x.cpu().detach().numpy(), r=rr, step=-1,
        energy=energy.item(), energy_potential=energy_potential.item(), energy_contact=energy_contact.item(),
        save_dir_name=save_dir_name, force_flag=False,
        left_boundary=left_boundary[0].item(), right_boundary=right_boundary[0].item(),
        top_boundary=top_boundary[0].item(), bottom_boundary=bottom_boundary[0].item(),)

    # plot the training history
    plot_E_history(E=E, E_contact=E_contact, E_potential=E_potential, save_dir_name=save_dir_name)
    plot_f_history(fx_list=fx_list, fy_list=fy_list, save_dir_name=save_dir_name)
    return


def overlap_check(x: torch.Tensor, r: torch.Tensor):
    diag_r = torch.diag(r)
    r = r.reshape(-1, 1)
    # device_num = x.get_device()
    # device = torch.device('cuda:%d' % device_num) if device_num >= 0 else torch.device('cpu')
    x0, x1 = x[:, 0:1], x[:, 1:2]
    x0_diff = x0 - x0.T
    x1_diff = x1 - x1.T
    r_0_1 = r+r.T
    # here is important to add a small value to make sure the gradient calculation do not come to NAN
    distance = torch.sqrt(x0_diff ** 2 + x1_diff ** 2 + 1e-16)
    overlap = r_0_1 - distance - diag_r*2.

    overlap = torch.where(overlap > 0., overlap, 0)  # set the un-contacted overlap as 0

    return overlap


def get_stress(height: float, width: float, x: torch.Tensor, overlap: torch.Tensor):
    # 添加所有接触的列表

    contact_list = get_contact(x, overlap, youngs=youngs)
    stress = torch.zeros(size=[2, 2], device=device)

    for contact in contact_list:
        # a, b = contact.force_vector(), contact.branch()
        stress += torch.outer(contact.force_vector(), contact.branch())

    # overlap_uper = torch.triu(overlap, diagonal=1)
    #
    # # x0_diff/distance, x1_diff/distance 分别为 x 和 y 方向 单位向量
    # # 将两者转换为
    # overlap_0, overlap_1 = x0_diff/distance*overlap_uper, x1_diff/distance*overlap_uper
    #
    # overlap_xy = torch.stack((overlap_0, overlap_1))
    #
    # # d为支向量 branch vector
    # d = torch.stack((torch.triu(x0_diff, diagonal=1), torch.triu(x1_diff, diagonal=1)))
    #
    # """
    #         Reference: https://www.sciencedirect.com/science/article/pii/S0020768313001492
    #         eq 15
    # """
    # stress = torch.einsum('imn, jmn->ij', overlap_xy, d)

    # area = (max(x[:, 0]) - min(x[:, 0])) * (max(x[:, 1]) - min(x[:, 1]))
    return stress / height/ width /thick


def get_contact(x: torch.Tensor, overlap: torch.Tensor, youngs: float):
    contact_list = []
    shape = overlap.shape
    for i in range(shape[0]):
        for j in range(i+1, shape[1]):
            if overlap[i, j]>0:
                contact_list.append(
                    Contact(
                        x1=x[i], x2=x[j], num1=i, num2=j, force=overlap[i, j]*youngs, overlap=overlap[i, j]))
    return contact_list



# def overlap_check(x: torch.Tensor, r: torch.Tensor):
#     r = r.reshape(-1, 1)
#     device_num = x.get_device()
#     device = torch.device('cuda:%d' % device_num) if device_num >= 0 else torch.device('cpu')
#     x0, x1 = x[:, 0:1], x[:, 1:2]
#     x0_diff = torch.add(x0, x0.T, alpha=-1)
#     x1_diff = torch.add(x1, x1.T, alpha=-1)
#     r_0_1 = r+r.T
#     # here is important to add a small value to make sure the gradient calculation do not come to NAN
#     distance = torch.sqrt(x0_diff ** 2 + x1_diff ** 2 + 1e-16)
#     overlap = r_0_1 - distance - torch.eye(len(x), device=device) * r_0_1
#     overlap = torch.where(overlap < 0., 0., overlap)  # set the un-contacted overlap as 0
#
#     overlap_0, overlap_1 = x0_diff/distance*overlap, x1_diff/distance*overlap
#     return overlap, overlap_0, overlap_1


if __name__ == '__main__':
    main()
