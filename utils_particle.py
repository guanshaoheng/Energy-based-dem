import os.path

import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt


class Contact:
    def __init__(
            self,
            x1: torch.Tensor, x2: torch.Tensor, num1: int, num2: int, force: torch.Tensor,
            overlap:torch.Tensor
    ):
        """
        force 为 第一个球施加在第二个球上的力的向量
        """
        self.x1, self.x2, self.num1, self.num2, self.force = x1, x2, num1, num2, force
        self.overlap = overlap

    def norm(self):
        """ norm 单位向量，该单位向量的方向与力的方向相同，即第1个指向第2个"""
        branch = self.branch()
        return branch/torch.linalg.norm(branch)

    def branch(self):
        """branch 为第1个指向第2个的向量"""
        return self.x2 - self.x1

    def force_vector(self):
        """ 将力分解为张量"""
        return self.force*self.norm()

    def energy(self):
        return self.overlap * self.force * 0.5


def write_down_infor(
        xx: torch.Tensor, rr: numpy.ndarray, save_dir_name: str,  step:int,
        energy: float, energy_potential: float, energy_contact: float, fx:float = None, fy:float=None, gravity:float=None):
    save_dir_name = os.path.join(save_dir_name, 'load_infor')
    xx_numpy = xx.cpu().detach().numpy()
    data = np.concatenate((xx_numpy, rr.reshape(-1, 1)), axis=1)
    if fx is None:
        header = 'x y radius \n E:%.2e E_potential:%.2e E_contact:%.2e' % (energy, energy_potential, energy_contact)
    else:
        header = 'x y radius \n E:%.2e E_potential:%.2e E_contact:%.2e fx:%.2e fy:%.2e gravity:%.2e' % (
        energy, energy_potential, energy_contact, fx, fy, gravity)
    np.savetxt(
        fname=os.path.join(save_dir_name, 'pos_particles_%d.txt' % step), X=data, fmt='%.5e',
        header=header)
    return


def overlap_check(x: torch.Tensor, r: torch.Tensor):
    r = r.reshape(-1, 1)
    device_num = x.get_device()
    device = torch.device('cuda:%d' % device_num) if device_num >= 0 else torch.device('cpu')
    x0, x1 = x[:, 0:1], x[:, 1:2]
    x0_diff = torch.add(x0, x0.T, alpha=-1)
    x1_diff = torch.add(x1, x1.T, alpha=-1)
    r_0_1 = r+r.T
    # here is important to add a small value to make sure the gradient calculation do not come to NAN
    distance = torch.sqrt(x0_diff ** 2 + x1_diff ** 2 + 1e-16)
    overlap = r_0_1 - distance - torch.eye(len(x), device=device) * r_0_1
    overlap = torch.where(overlap < 0., 0., overlap)  # set the un-contacted overlap as 0

    overlap_0, overlap_1 = x0_diff/distance*overlap, x1_diff/distance*overlap
    return overlap, overlap_0, overlap_1


def plot_particles(xx: np.ndarray, r: np.ndarray, step: int,
                   energy=None, energy_potential=None, energy_contact=None,
                   disturbed_index=None,
                   save_dir_name=None, force_flag=False,
                   left_boundary=0., right_boundary=11.0, top_boundary=None, bottom_boundary=None,
                   unbalanced_x_index: np.ndarray=None, unbalanced_y_index: np.ndarray=None):
    num_particles = len(xx)
    fig, ax = plt.subplots()
    for i in range(len(xx)):
        if disturbed_index != i:
            c = plt.Circle(xx[i], r[i], edgecolor='k')
        else:
            c = plt.Circle(xx[i], r[i], color='r')
        if unbalanced_x_index is not None and unbalanced_x_index[i]==1:
            c = plt.Circle(xx[i], r[i], edgecolor='r')
        # if unbalanced_y_index is not None and unbalanced_y_index[i]==1:
        #     c = plt.Circle(xx[i], r[i], edgecolor='blue')
        ax.add_patch(c)
    # plot the forces
    if force_flag:
        overlap = plot_forces(xx=xx, r=r).cpu().detach().numpy()
        for i in range(num_particles):
            for j in range(i, num_particles):
                if overlap[i, j] > 1e-5:
                    plot_line(xx[i], xx[j], overlap=overlap[i, j])
    plt.xlim([0, 11])
    plt.ylim([-1, 6])

    # plot the boundary
    plt.plot([left_boundary, left_boundary], [-1, 6], linewidth=3, c='k')
    plt.plot([right_boundary, right_boundary], [-1, 6], linewidth=3, c='k')
    if top_boundary is not None:
        plt.plot([left_boundary, right_boundary], [top_boundary, top_boundary], linewidth=3, c='k')
    if bottom_boundary is None:
        plt.plot([left_boundary, right_boundary], [0, 0], linewidth=3, c='k')
    else:
        plt.plot([left_boundary, right_boundary], [bottom_boundary, bottom_boundary], linewidth=3, c='k')
    plt.axis('equal')
    # plt.axis('off')
    if step == 0:
        plt.title(r'$ \mathrm{Step}\  %d $' % step)
    else:
        if step > 0:
            plt.title(r'$ \mathrm{Step} \  %d \ E=%.1e \ E_{potential}=%.1e \ E_{contact}=%.1e $' %
                  (step, energy, energy_potential, energy_contact))
        else:
            plt.title(r'$ Final\ state \ E=%.1e \ E_{potential}=%.1e \ E_{contact}=%.1e $' %
                      (energy, energy_potential, energy_contact))
    # plt.show()
    if save_dir_name:
        plt.tight_layout()
        if step != -1:
            plt.savefig(os.path.join(save_dir_name, 'step_%d.png' % step), dpi=100)
        else:
            plt.savefig(os.path.join(save_dir_name, 'step_final.png'), dpi=100)
    else:
        plt.show()
    plt.close()


def plot_forces(xx: np.ndarray, r: np.ndarray,
                device=torch.device('cuda:0')):
    x = torch.tensor(xx, device=device).float()
    r = torch.tensor(r, device=device).float()
    overlap = overlap_check(x, r)[0]
    return overlap


def plot_line(x0, x1, overlap):
    plt.plot([x0[0], x1[0]], [x0[1], x1[1]], 'm', zorder=10, linewidth=100*overlap)


def plot_E_history(E, E_potential, E_contact, save_dir_name):
    plt.plot(E, linewidth=3,  c = 'r', label='$E$')
    plt.plot(E_potential, linewidth=3,  c = 'b', label='$E_{potential}$')
    plt.plot(E_contact, linewidth=3,  c = 'g', label='$E_{contact}$')
    plt.legend(fontsize='medium')
    plt.xlabel('Epoch', fontsize='medium')
    plt.ylabel('Energy', fontsize='medium')
    plt.tight_layout()
    if save_dir_name:
        plt.savefig(os.path.join(save_dir_name, 'history.png'), dpi=200)
    else:
        plt.show()
    plt.close()


def plot_f_history(fx_list, fy_list, save_dir_name):
    plt.plot(fx_list, linewidth=3, c='b', label='$f_{x}$')
    plt.plot(fy_list, linewidth=3, c='g', label='$f_{y}$')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Force')
    plt.tight_layout()
    if save_dir_name:
        plt.savefig(os.path.join(save_dir_name, 'force.png'), dpi=200)
    else:
        plt.show()
    plt.close()

