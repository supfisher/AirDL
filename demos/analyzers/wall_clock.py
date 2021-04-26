import os
import numpy as np
from utils import plot_xs_ys, parse_time_acc_loss, parse_energy, plot_xs_ys1_ys2, parse_all_energy

current_path = '/Users/mag0a/Desktop/Github/FLinMEN/ns-3-allinone/ns-3-dev/contrib/distributed-ml-test/demos/'


def parse_traffic_wall_clock():
    dir = os.path.join(current_path, "saved_traffic/outputs-" + str(32))

    print("nCells \t 1 \t 2 \t 4 \t 8 \t 16 \t 32 ")

    for nCells in [32, 64, 128, 256]:
        wall_clock_list = []
        wall_clock_str = ""
        base = 1
        for sysCount in [1, 2, 4, 8, 16, 32]:
            tf_dir = os.path.join(dir, "tf-systemCount-" + str(sysCount) + "-nCells-" + str(nCells))

            if os.path.exists(tf_dir):
                epoch, time, wall_clock, loss, mse = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                                         stop_epoch=3)
                if sysCount == 1:
                    base = int(wall_clock[-1] - wall_clock[-2])
                wall_clock_list.append(int(wall_clock[-1] - wall_clock[-2]))
                wall_clock_str += " \t {:.0f} ".format(wall_clock[-1] - wall_clock[-2])

        wall_clock_str += "\n"
        for v in wall_clock_list:
            wall_clock_str += " \t {:.1f}\% ".format(v / base * 100)

        print("{}{}".format(nCells, wall_clock_str))


def parse_minist_wall_clock():
    dir = os.path.join(current_path, "saved_minist/outputs-" + str(32))

    print("nCells \t 1 \t 2 \t 4 \t 8 \t 16 \t 32 ")

    for nCells in [32, 64]:
        wall_clock_list = []
        wall_clock_str = ""
        base = 1
        for sysCount in [1, 2, 4, 8, 16, 32]:
            tf_dir = os.path.join(dir, "tf-systemCount-" + str(sysCount) + "-nCells-" + str(nCells))

            if os.path.exists(tf_dir):
                epoch, time, wall_clock, loss, mse = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_epoch=3)
                if sysCount==1:
                    base =  int(wall_clock[-1] - wall_clock[-2] )
                wall_clock_list.append(int(wall_clock[-1] - wall_clock[-2]))
                wall_clock_str += " \t {:.0f} ".format(wall_clock[-1] - wall_clock[-2])

        wall_clock_str += "\n"
        for v in wall_clock_list:
            wall_clock_str += " \t {:.1f}\% ".format(v/base*100)

        print("{}{}".format(nCells, wall_clock_str))





if __name__ == "__main__":
    # parse_minist_wall_clock()
    print()
    print()
    parse_traffic_wall_clock()
