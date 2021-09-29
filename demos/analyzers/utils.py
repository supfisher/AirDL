import os
import matplotlib.pyplot as plt
import re
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

default_colors = ['r-^', 'k-^', 'g:^', 'g--^', 'g-^', 'b:^', 'b--^',  'b-^']

def plot_x_y(x, y):
    """
    :param x: the x index list
    :param y: the y index list
    :return:
    """
    plt.plot(x, y)
    plt.show()


def plot_xs_ys(xs, ys, xlabel=None, ylabel=None, title=None, legends=None, save_path=None, colors=default_colors, loc=4, markersize=0, width=3, height=3.5, show=True, logy=False):
    """
    :param xs: an iterable of x index list
    :param ys: an iterable of  y index list
    :return:
    """
    # plt.figure(figsize=(width, height))
    for i, (x, y) in enumerate(zip(xs, ys)):
        if logy:
            plt.semilogy(x, y, colors[i], markersize=markersize)
        else:
            plt.plot(x, y, colors[i], markersize=markersize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legends, loc=loc, fontsize='xx-small')
    plt.minorticks_on()
    plt.grid(axis='both', alpha=0.2, linestyle='--', which='minor')
    # plt.xlim(0, 25)
    plt.tight_layout()

    if show:
        # plt.savefig(save_path)
        plt.show()


def plot_xs_ys1_ys2(xs, ys1, ys2, xlabel=None, ylabel1=None, ylabel2=None, title=None, legends=None, loc=4, save_path=None, colors=default_colors, show=True):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    for i, (x, y) in enumerate(zip(xs, ys1)):
        ax1.plot(x, y, colors[i], markersize=0)
    plt.xlabel(xlabel)

    ax1.set_ylabel(ylabel1)
    # plt.legend(loc=2)

    ax2 = ax1.twinx()
    for i, (x, y) in enumerate(zip(xs, ys2)):
        ax2.plot(x, y, colors[i], markersize=5)
    ax2.set_ylabel(ylabel2)
    # plt.legend(loc=1)  # 设置图例在右上方
    plt.title(title)
    plt.legend(legends, loc=loc)
    plt.grid()
    if show:
        plt.savefig(save_path)
        plt.show()



def parse_time_acc_loss(time_acc_loss_path, stop_epoch=100, stop_acc=1000, stop_loss=0, time_ratio=1):
    """
    :param time_acc_loss_path: the path of time_acc_loss.txt
    :return: epochs, times, wall_clocks, losses, accs, before a stop_epoch
    """
    epochs, times, wall_clocks, losses, accs = [], [], [], [], []

    with open(time_acc_loss_path) as f:
        for line in f.readlines():
            numbers = re.findall(r"\d+\.?\d*", line)
            epoch, time, wall_clock, loss, acc = int(numbers[0]), float(numbers[1]), float(numbers[2]), float(
                numbers[3]), float(numbers[4])

            epochs.append(epoch)
            times.append(time*time_ratio)
            wall_clocks.append(wall_clock)
            losses.append(loss)
            accs.append(acc)
            if epoch > stop_epoch or acc > stop_acc or loss < stop_loss:
                break

    return epochs, times, wall_clocks, losses, accs



def parse_energy(dir, time_limit):
    """
    :param dir: the dir of trace
    :param time_limit: the given time limit
    :return: the values which happened before time limit
    """
    times, energys = [], []
    for log_path in os.listdir(dir):
        if "energy" in log_path and "ml_energy" not in log_path:
            log_path = os.path.join(dir, log_path)
            time_per, energy_per = [], []
            with open(log_path) as f:
                for line in f.readlines():
                    numbers = re.findall(r"\d+\.?\d*", line)
                    time, energy = float(numbers[0]), float(numbers[1])
                    if time >= time_limit:
                        break
                    time_per.append(time)
                    energy_per.append(energy)

            times.append(time_per)
            energys.append(energy_per)

    index = min([len(_) for _ in times])

    times = [t[0:index] for t in times]
    energys = [e[0:index] for e in energys]

    times = np.array(times)
    energys = np.array(energys)
    avg_time = np.average(times, axis=0)
    avg_energy = np.average(energys, axis=0)
    sum_energy = np.sum(energys, axis=0)
    return avg_time, avg_energy, energys



def parse_energy_per_file(log_path, time_limit):
    time, energy = 0, 0
    with open(log_path) as f:
        for line in f.readlines():
            numbers = re.findall(r"\d+\.?\d*", line)
            time, energy = float(numbers[0]), float(numbers[1])
            if time >= time_limit:
                break
    return time, energy


def parse_all_energy(dir, time_limit):
    """
    :param dir: the dir of trace
    :param time_limit: the given time limit
    :return: the values which happened before time limit
    """

    times, energys, ml_to_wifi_ratio = [], [], []
    for log_path in os.listdir(dir):
        if "ml_energy" in log_path:
            number = int(re.findall("\d+", log_path)[0])
            log_path = os.path.join(dir, log_path)
            time_per, energy_per, ml_to_wifi_ratio_per = [], [], []
            with open(log_path) as f:
                for line in f.readlines():
                    numbers = re.findall(r"\d+\.?\d*", line)
                    time_ml, energy_ml = float(numbers[0]), float(numbers[1])

                    time_wifi, energy_wifi = parse_energy_per_file(os.path.join(dir, "energy{}.txt".format(number)), time_ml)

                    if time_ml >= time_limit:
                        break
                    time_per.append(time_ml)
                    energy_per.append(energy_ml + energy_wifi)
                    ml_to_wifi_ratio_per.append(energy_ml / energy_wifi)

            times.append(time_per)
            energys.append(energy_per)
            ml_to_wifi_ratio.append(ml_to_wifi_ratio_per)

    index = min([len(_) for _ in times])

    times = [t[0:index] for t in times]
    energys = [e[0:index] for e in energys]

    times = np.array(times)
    energys = np.array(energys)
    ml_to_wifi_ratio = np.array(ml_to_wifi_ratio)
    avg_time = np.average(times, axis=0)
    avg_energy = np.average(energys, axis=0)
    sum_energy = np.sum(energys, axis=0)
    return avg_time, avg_energy, energys, ml_to_wifi_ratio
