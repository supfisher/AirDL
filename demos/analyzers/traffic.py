import os
import pickle
import numpy as np
from utils import plot_xs_ys, parse_time_acc_loss, parse_energy, plot_xs_ys1_ys2, parse_all_energy
import matplotlib.pyplot as plt



current_path = '/Users/mag0a/Desktop/Github/FLinMEN/ns-3-allinone/ns-3-dev/contrib/distributed-ml-test/demos/'


plot_1 = False
plot_2 = False

plot_error1 = False
plot_error2 = False

plot_naughty1 = False
plot_naughty2 = False

plot_activeratio1 = False
plot_activeratio2 = False

plot_epoch1 = False
plot_epoch2 = False

plot_bs1 = False
plot_bs2 = False

plot_all1 = False
plot_all2 = False

plot_partition = True

global_time_w = {}
global_loss_w = {}
global_energy_w = {}
global_loss1_w = {}
global_legends = []
global_colors = []

stop_epoch = 60


def process_loss_energy(energy, loss):
    energy_list = []
    loss_list = []
    for e, l in zip(energy, loss):
        set_e = sorted(list(set(e)))
        l = [l[e.index(v)] for v in set_e]
        e = list(set_e)
        energy_list.append(e)
        loss_list.append(l)
    return energy_list, loss_list


if __name__=="__main__":

    if plot_1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, mse_w = {}, {}, {}, {}, {}
        legends = []

        for sysCount in [1, 2, 8]:
            for nActivePerCell in [1, 2, 4]:
                if sysCount <= 2 and nActivePerCell < 4:
                    continue

                local_epochs = 1
                batch_size = 8
                error_rate = 0
                noise_type = 'add'
                noise_ratio = 0

                dir_ = os.path.join(current_path, "saved_traffic/outputs-"+str(sysCount))

                record_prefix = "-local_epochs-" + str(local_epochs) + \
                                "-batch_size-" + str(batch_size) + \
                                "-error_rate-" + str(error_rate) + \
                                "-nActivePerCell-" + str(nActivePerCell) + \
                                "-noise_type-" + str(noise_type) + \
                                "-noise_ratio-" + str(noise_ratio)

                tf_dir = os.path.join(dir_, "tf"+record_prefix)
                if os.path.exists(tf_dir):
                    legends.append(r"$C={:}$, $M={:}$".format(sysCount, nActivePerCell))

                    print("Read from ", tf_dir)
                    time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                    epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                    epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], mse_w[iters] = epochs, times, wall_clocks, losses, mses

                    iters += 1

        plt.figure(figsize=[6.4, 2.4])
        plt.subplot(121)
        # plt.figure(1)
        plot_xs_ys(epoch_w.values(), loss_w.values(), xlabel='Communication Round', ylabel="MSE Loss", loc=1,
                   legends=legends, show=False)
        plt.subplot(122)
        # plt.figure(2)
        plot_xs_ys(epoch_w.values(), time_w.values(), xlabel='Communication Round', ylabel="Elapsed Time (s)", loc=4,
                   legends=legends, logy=True, show=False)

        plt.savefig(
            os.path.join(current_path, 'saved_results/traffic-epoch-vs-loss-time-stopEpoch-{}.pdf'.format(stop_epoch)))
        plt.show()



    if plot_2:
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        loss_w, loss_avg_energy_w, loss_sum_energy_w = {}, {}, {}
        legends = []
        for sysCount in [1, 2, 4, 8]:
            for nActivePerCell in [1, 2, 4]:

                local_epochs = 1
                batch_size = 8
                error_rate = 0
                noise_type = 'add'
                noise_ratio = 0

                dir = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

                record_prefix = "-local_epochs-" + str(local_epochs) + \
                                "-batch_size-" + str(batch_size) + \
                                "-error_rate-" + str(error_rate) + \
                                "-nActivePerCell-" + str(nActivePerCell) + \
                                "-noise_type-" + str(noise_type) + \
                                "-noise_ratio-" + str(noise_ratio)

                trace_dir = os.path.join(dir, "trace" + record_prefix)

                tf_dir = os.path.join(dir, "tf" + record_prefix)

                if os.path.exists(tf_dir):
                    legends.append("sysCount_" + str(sysCount) + "_nActivePerCell_" + str(nActivePerCell))
                    epoch, time, wall_clock, loss, mse = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_loss=0.25)
                    avg_time, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    time_w[iters] = avg_time
                    time_avg_energy_w[iters] = avg_energy
                    time_sum_energy_w[iters] = sum_energy

                    loss_avg_energy_w[iters] = []
                    loss_sum_energy_w[iters] = []
                    loss_w[iters] = []
                    for loss in np.arange(0.275, 0.255, -0.001):
                        _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_loss=loss)
                        _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                        loss_avg_energy_w[iters].append(avg_energy[-1])
                        loss_sum_energy_w[iters].append(sum_energy[-1])
                        loss_w[iters].append(loss)

                    iters += 1

        plot_xs_ys(time_w.values(), time_avg_energy_w.values(), xlabel="time", ylabel="energy", title="time-vs-avg_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/traffic-time-vs-avg_energy.png'))
        plot_xs_ys(loss_w.values(), loss_avg_energy_w.values(), xlabel="loss", ylabel="energy", title="loss-vs-avg_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/traffic-loss-vs-avg_energy.png'))

        plot_xs_ys(time_w.values(), time_sum_energy_w.values(), xlabel="time", ylabel="energy", title="time-vs-sum_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/traffic-time-vs-sum_energy.png'))
        plot_xs_ys(loss_w.values(), loss_sum_energy_w.values(), xlabel="loss", ylabel="energy", title="loss-vs-sum_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/traffic-loss-vs-sum_energy.png'))


    if plot_error1:
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, mse_w = {}, {}, {}, {}, {}
        legends = []
        for error_rate in [1e-4, 1e-5, 1e-6, 1e-7]:
            sysCount = 2
            nActivePerCell = 4
            local_epochs = 1
            batch_size = 8
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)
            if os.path.exists(tf_dir):
                legends.append(r"$PER={:}$".format(error_rate))

                print("Read from ", tf_dir)
                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=60)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], mse_w[
                    iters] = epochs, times, wall_clocks, losses, mses

                iters += 1

        global_time_w['error_rate'] = list(time_w.values())
        global_loss_w['error_rate'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), loss_w.values(), xlabel="time", ylabel="loss", title="time-vs-loss", colors=colors,
        #            legends=legends, save_path=os.path.join(current_path,
        #                                                    'saved_results/traffic-error-time-vs-loss-stopEpoch-{}.png'.format(stop_epoch)))
        # plot_xs_ys1_ys2(epoch_w.values(), loss_w.values(), time_w.values(), xlabel="epoch", ylabel1="loss",
        #                 ylabel2='time', loc=2, title="epoch-vs-loss-time", legends=legends, colors=colors,
        #                 save_path=os.path.join(current_path,
        #                                        'saved_results/traffic-error-epoch-vs-loss-time-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_error2:
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        loss_w, loss_avg_energy_w, loss_sum_energy_w = {}, {}, {}
        legends = []
        for error_rate in [1e-4, 1e-5, 1e-6, 0]:
            sysCount = 2
            nActivePerCell = 4

            local_epochs = 1
            batch_size = 8

            noise_type = 'add'
            noise_ratio = 0

            dir = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            trace_dir = os.path.join(dir, "trace" + record_prefix)

            tf_dir = os.path.join(dir, "tf" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$PER={:}$".format(error_rate))

                epoch, time, wall_clock, loss, mse = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                                         stop_loss=0.25)
                avg_time, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                time_w[iters] = avg_time
                time_avg_energy_w[iters] = avg_energy
                time_sum_energy_w[iters] = sum_energy

                loss_avg_energy_w[iters] = []
                loss_sum_energy_w[iters] = []
                loss_w[iters] = []
                for loss in np.arange(0.3, 0.24, -0.001):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                           stop_loss=loss)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    loss_avg_energy_w[iters].append(avg_energy[-1])
                    loss_sum_energy_w[iters].append(sum_energy[-1])
                    loss_w[iters].append(loss)

                iters += 1

        global_energy_w['error_rate'] = list(loss_avg_energy_w.values())
        global_loss1_w['error_rate'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), time_avg_energy_w.values(), xlabel="time", ylabel="energy", colors=colors,
        #            title="time-vs-avg_energy",
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-error-time-vs-avg_energy.png'))
        # plot_xs_ys(loss_w.values(), loss_avg_energy_w.values(), xlabel="loss", ylabel="energy", colors=colors,
        #            title="loss-vs-avg_energy",
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-error-loss-vs-avg_energy.png'))
        #
        # plot_xs_ys(time_w.values(), time_sum_energy_w.values(), xlabel="time", ylabel="energy", colors=colors,
        #            title="time-vs-sum_energy",
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-error-time-vs-sum_energy.png'))
        # plot_xs_ys(loss_w.values(), loss_sum_energy_w.values(), xlabel="loss", ylabel="energy", colors=colors,
        #            title="loss-vs-sum_energy",
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-error-loss-vs-sum_energy.png'))



    if plot_naughty1:
        plt.figure(figsize=[6.4, 4.8])
        noise_character = {'add': "NIS_a", 'multi': 'NIS_m'}

        noise_type_ratio = {'add': [1e-2, 5e-2, 1e-1, 0.15],
                            'multi':[1e-1, 2e-1, 3e-1, 4e-1]}
        for i, noise_type in enumerate(['add', 'multi']):
            iters = 0
            colors = ['g-^', 'k-^', 'r-^', 'b-^', 'y-^']
            epoch_w, time_w, wall_clock_w, loss_w, mse_w = {}, {}, {}, {}, {}
            legends = []

            for noise_ratio in noise_type_ratio[noise_type]:
                sysCount = 2
                nActivePerCell = 4
                local_epochs = 1
                batch_size = 8
                error_rate = 0

                dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

                record_prefix = "-local_epochs-" + str(local_epochs) + \
                                "-batch_size-" + str(batch_size) + \
                                "-error_rate-" + str(error_rate) + \
                                "-nActivePerCell-" + str(nActivePerCell) + \
                                "-noise_type-" + str(noise_type) + \
                                "-noise_ratio-" + str(noise_ratio) + \
                                "-part_ratio-" + str('1,1,1,1')

                tf_dir = os.path.join(dir_, "tf" + record_prefix)
                if os.path.exists(tf_dir):
                    legends.append(r"${:}={:}$".format(noise_character[noise_type], noise_ratio))

                    print("Read from ", tf_dir)
                    time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                    epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=60, time_ratio=10)
                    epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], mse_w[
                        iters] = epochs, times, wall_clocks, losses, mses

                    iters += 1

            plt.subplot(221 + i)
            plot_xs_ys(time_w.values(), loss_w.values(), xlabel="Elapsed Time (s)", ylabel="MSE Loss", colors=colors, show=False,
                       legends=legends, loc=1)

            # plot_xs_ys1_ys2(epoch_w.values(), loss_w.values(), time_w.values(), xlabel="epoch", ylabel1="loss", colors=colors,
            #                 ylabel2='time', loc=2,
            #                 title="epoch-vs-loss-time", legends=legends,
            #                 save_path=os.path.join(current_path,
            #                                        'saved_results/traffic-naughty-epoch-vs-loss-time-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_naughty2:
        noise_character = {'add': "NIS_a", 'multi': 'NIS_m'}

        noise_type_ratio = {'add': [1e-2, 5e-2, 1e-1, 0.15],
                            'multi':[1e-1, 2e-1, 3e-1, 4e-1]}
        for i, noise_type in enumerate(['add', 'multi']):
            iters = 0
            colors = ['g-^', 'k-^', 'r-^', 'b-^', 'y-^']
            time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
            loss_w, loss_avg_energy_w, loss_sum_energy_w = {}, {}, {}
            legends = []

            for noise_ratio in noise_type_ratio[noise_type]:
                sysCount = 2
                nActivePerCell = 4
                local_epochs = 1
                batch_size = 8
                error_rate = 0

                dir = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

                record_prefix = "-local_epochs-" + str(local_epochs) + \
                                "-batch_size-" + str(batch_size) + \
                                "-error_rate-" + str(error_rate) + \
                                "-nActivePerCell-" + str(nActivePerCell) + \
                                "-noise_type-" + str(noise_type) + \
                                "-noise_ratio-" + str(noise_ratio) + \
                                "-part_ratio-" + str('1,1,1,1')

                trace_dir = os.path.join(dir, "trace" + record_prefix)

                tf_dir = os.path.join(dir, "tf" + record_prefix)

                if os.path.exists(tf_dir):
                    legends.append(r"${:}={:}$".format(noise_character[noise_type], noise_ratio))

                    epoch, time, wall_clock, loss, mse = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                                             stop_loss=0.25)
                    avg_time, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    time_w[iters] = avg_time
                    time_avg_energy_w[iters] = avg_energy
                    time_sum_energy_w[iters] = sum_energy

                    loss_avg_energy_w[iters] = []
                    loss_sum_energy_w[iters] = []
                    loss_w[iters] = []
                    for loss in np.arange(0.3, 0.245, -0.001):
                        epochs, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                               stop_loss=loss)
                        if epochs[-1]==stop_epoch:
                            break
                        _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                        loss_avg_energy_w[iters].append(avg_energy[-1])
                        loss_sum_energy_w[iters].append(sum_energy[-1])
                        loss_w[iters].append(loss)

                    iters += 1

            plt.subplot(221 + i + 2)
            # plot_xs_ys(time_w.values(), time_avg_energy_w.values(), xlabel="time", ylabel="energy", colors=colors,
            #            title="time-vs-avg_energy",
            #            legends=legends,
            #            save_path=os.path.join(current_path,
            #                                   'saved_results/traffic-naughty-time-vs-avg_energy.png'))

            loss_avg_energy_w, loss_w = process_loss_energy(list(loss_avg_energy_w.values()), list(loss_w.values()))

            plot_xs_ys(loss_avg_energy_w, loss_w, xlabel="Consumed Energy (J)", ylabel="MSE Loss", colors=colors,
                       show=False, legends=legends, loc=1)

            # plot_xs_ys(time_w.values(), time_sum_energy_w.values(), xlabel="time", ylabel="energy", colors=colors,
            #            title="time-vs-sum_energy",
            #            legends=legends,
            #            save_path=os.path.join(current_path,
            #                                   'saved_results/traffic-naughty-time-vs-sum_energy.png'))
            # plot_xs_ys(loss_w.values(), loss_sum_energy_w.values(), xlabel="loss", ylabel="energy", colors=colors,
            #            title="loss-vs-sum_energy",
            #            legends=legends,
            #            save_path=os.path.join(current_path,
            #                                   'saved_results/traffic-naughty-loss-vs-sum_energy.png'))

        plt.savefig(os.path.join(current_path,
                                 'saved_results/traffic-naughty.pdf'))
        plt.show()


    if plot_activeratio1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, mse_w = {}, {}, {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']

        for nActivePerCell in [1, 2, 4]:
            sysCount = 2
            local_epochs = 1
            batch_size = 8
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_traffic/outputs-"+str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf"+record_prefix)
            if os.path.exists(tf_dir):
                legends.append(r"$r={:}$".format(nActivePerCell/4))

                print("Read from ", tf_dir)
                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=60)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], mse_w[iters] = epochs, times, wall_clocks, losses, mses

                iters += 1

        global_time_w['num_clients'] = list(time_w.values())
        global_loss_w['num_clients'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), loss_w.values(), xlabel="time", ylabel="loss", title="time-vs-loss",
        #            legends=legends, save_path=os.path.join(current_path, 'saved_results/traffic-active_ratio-time-vs-loss-stopEpoch-{}.png'.format(stop_epoch)))
        # plot_xs_ys1_ys2(epoch_w.values(), loss_w.values(), time_w.values(), xlabel="epoch", ylabel1="loss", ylabel2='time', loc=2,
        #                 title="epoch-vs-loss-time", legends=legends,
        #                 save_path=os.path.join(current_path, 'saved_results/traffic-active_ratio-epoch-vs-loss-time-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_activeratio2:
        iters = 0
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        loss_w, loss_avg_energy_w, loss_sum_energy_w = {}, {}, {}
        legends = []

        for nActivePerCell in [1, 2, 4]:
            sysCount = 2
            local_epochs = 1
            batch_size = 8
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            trace_dir = os.path.join(dir, "trace" + record_prefix)

            tf_dir = os.path.join(dir, "tf" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$r={:}$".format(nActivePerCell/4))

                loss_avg_energy_w[iters] = []
                loss_sum_energy_w[iters] = []
                loss_w[iters] = []
                for loss in np.arange(0.3, 0.245, -0.001):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_loss=loss)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    loss_avg_energy_w[iters].append(avg_energy[-1])
                    loss_sum_energy_w[iters].append(sum_energy[-1])
                    loss_w[iters].append(loss)

                iters += 1

        global_energy_w['num_clients'] = list(loss_avg_energy_w.values())
        global_loss1_w['num_clients'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(loss_w.values(), loss_avg_energy_w.values(), xlabel="loss", ylabel="energy", title="loss-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-active_ratio-loss-vs-avg_energy.png'))


    if plot_epoch1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, mse_w = {}, {}, {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        for local_epochs in [2, 4, 8, 16]:
            batch_size = 8
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)
            if os.path.exists(tf_dir):
                legends.append(r"$E={:}$".format(local_epochs))

                print("Read from ", tf_dir)
                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], mse_w[
                    iters] = epochs, times, wall_clocks, losses, mses

                iters += 1

        global_time_w['local_epochs'] = list(time_w.values())
        global_loss_w['local_epochs'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), loss_w.values(), xlabel="time", ylabel="loss", title="time-vs-loss", markersize=4, colors=colors,
        #            legends=legends, save_path=os.path.join(current_path,
        #                                                    'saved_results/traffic-local_epoch-epoch-time-vs-loss-stopEpoch-{}.png'.format(stop_epoch)))



    if plot_epoch2:
        iters = 0
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        loss_w, loss_avg_energy_w, loss_sum_energy_w = {}, {}, {}
        legends = []
        for local_epochs in [2, 4, 8, 16]:
            batch_size = 8
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$E={:}$".format(local_epochs))

                loss_avg_energy_w[iters] = []
                loss_sum_energy_w[iters] = []
                loss_w[iters] = []
                for loss in np.arange(0.3, 0.245, -0.001):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_loss=loss)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    loss_avg_energy_w[iters].append(avg_energy[-1])
                    loss_sum_energy_w[iters].append(sum_energy[-1])
                    loss_w[iters].append(loss)

                iters += 1

        global_energy_w['local_epochs'] = list(loss_avg_energy_w.values())
        global_loss1_w['local_epochs'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(loss_w.values(), loss_avg_energy_w.values(), xlabel="loss", ylabel="energy",
        #            title="loss-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-local_epoch-loss-vs-avg_energy.png'))



    if plot_bs1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, mse_w = {}, {}, {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        for batch_size in [8, 64, 128, 256]:
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0
            local_epochs = 1

            dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)
            if os.path.exists(tf_dir):
                legends.append(r"$b_s={:}$".format(batch_size))

                print("Read from ", tf_dir)
                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=60)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], mse_w[
                    iters] = epochs, times, wall_clocks, losses, mses

                iters += 1

        global_time_w['local_bs'] = list(time_w.values())
        global_loss_w['local_bs'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), loss_w.values(), xlabel="time", ylabel="loss", title="time-vs-loss", markersize=4, colors=colors,
        #            legends=legends, save_path=os.path.join(current_path,
        #                                                    'saved_results/traffic-bs-time-vs-loss-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_bs2:
        iters = 0
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        loss_w, loss_avg_energy_w, loss_sum_energy_w = {}, {}, {}
        legends = []
        for batch_size in [8, 64, 128, 256]:
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0
            local_epochs = 1

            dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$b_s={:}$".format(batch_size))

                loss_avg_energy_w[iters] = []
                loss_sum_energy_w[iters] = []
                loss_w[iters] = []
                for loss in np.arange(0.3, 0.245, -0.001):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_loss=loss)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    loss_avg_energy_w[iters].append(avg_energy[-1])
                    loss_sum_energy_w[iters].append(sum_energy[-1])
                    loss_w[iters].append(loss)

                iters += 1

        global_energy_w['local_bs'] = list(loss_avg_energy_w.values())
        global_loss1_w['local_bs'] = list(loss_w.values())
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(loss_w.values(), loss_avg_energy_w.values(), xlabel="loss", ylabel="energy",
        #            title="loss-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/traffic-bs-loss-vs-avg_energy.png'))


    if plot_all1:
        plt.figure(figsize=[6.4, 4.8])
        for i, (time_w, loss_w) in enumerate(zip(global_time_w.values(), global_loss_w.values())):
            plt.subplot(221+i)
            plot_xs_ys(time_w, loss_w, xlabel="Elapsed Time (s)", ylabel="MSE Loss", markersize=0, colors=global_colors[i], show=False, legends=global_legends[i], loc=1)

        plt.savefig(os.path.join(current_path, 'saved_results/traffic-all-time-vs-loss.pdf'))
        plt.show()


    if plot_all2:
        plt.figure(figsize=[6.4, 4.8])
        for i, (energy_w, loss_w) in enumerate(zip(global_energy_w.values(), global_loss1_w.values())):
            energy_w, loss_w = process_loss_energy(energy_w, loss_w)
            plt.subplot(221+i)
            plot_xs_ys(energy_w, loss_w, xlabel="Consumed Energy (J)", ylabel="MSE Loss", markersize=0, colors=global_colors[i], show=False, legends=global_legends[i], loc=1)

        plt.savefig(os.path.join(current_path,
                                 'saved_results/traffic-all-energy-vs-loss.pdf'))
        plt.show()


    if plot_partition:
        iters = 0
        colors = ['r--', 'r-^', 'b-^', 'g-^', 'k-^', 'g--', 'k--']
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        loss_w, loss_avg_energy_w, loss_energy_ratio = {}, {}, {}
        ml_to_wifi_ratio_w = {}
        legends = []
        partitions = ['1,1,1,1', '8,1,1,1', '64,1,1,1', '512,1,1,1', '4096,1,1,1', "512,512,512,1", "4096,4096,4096,1"]
        for part_ratio in partitions:
            sysCount = 1
            nActivePerCell = 4
            batch_size = 8
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0
            local_epochs = 1

            dir_ = os.path.join(current_path, "saved_traffic/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio) + \
                            "-part_ratio-" + part_ratio

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append("partition ratio: {}".format(part_ratio))

                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, mses = parse_time_acc_loss(time_acc_loss_path, stop_epoch=25)
                time_w[iters] = times
                loss_w[iters] = losses
                avg_time, avg_energy, energys, ml_to_wifi_ratio = parse_all_energy(trace_dir, times[-1])

                loss_avg_energy_w[iters] = avg_energy
                loss_energy_ratio[iters] = energys[0]/energys[-1]
                ml_to_wifi_ratio_w[iters] = ml_to_wifi_ratio
                iters += 1

        plot_xs_ys(time_w.values(), loss_energy_ratio.values(), xlabel="time", ylabel="energy_ratio",
                   title="energy_ratio-vs-loss", colors=colors,
                   legends=legends, loc=2,
                   save_path=os.path.join(current_path,
                                          'saved_results/traffic-partition-energy_ratio-vs-loss.png'))

        print("partition\t energy_sumed\t time_consumed\t loss \t energy_ratio\t")

        for iters in range(len(partitions)):
            print("{}\t & {:.3f}\t & {:.3f}\t & {:.3f}\t & {:.3f}\t".format(partitions[iters], loss_avg_energy_w[iters][-1].item()/loss_avg_energy_w[0][-1].item(),
                                               time_w[iters][-1]/time_w[0][-1], loss_w[iters][-1], loss_energy_ratio[iters][-1].item()))

        print()
        print()


        for iters in range(len(partitions)):
            ml_to_wifi_ratio = ml_to_wifi_ratio_w[iters]
            print("{}\t & {:.3f}\t & {:.3f}".format(partitions[iters], ml_to_wifi_ratio[0,-1], ml_to_wifi_ratio[-1,-1]))

