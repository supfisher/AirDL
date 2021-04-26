import os
import numpy as np
from utils import plot_xs_ys, parse_time_acc_loss, parse_energy, plot_xs_ys1_ys2, parse_all_energy
import matplotlib.pyplot as plt

current_path = '/Users/mag0a/Desktop/Github/FLinMEN/ns-3-allinone/ns-3-dev/contrib/distributed-ml-test/demos/'

plot_1 = False
plot_2 = False

plot_error1 = False
plot_error2 = True

plot_naughty1 = False
plot_naughty2 = False

plot_activeratio1=False
plot_activeratio2=True

plot_epoch1 = False
plot_epoch2 = True

plot_bs1 = False
plot_bs2 = True

plot_all1 = False
plot_all2 = True

plot_partition = False

global_time_w = {}
global_acc_w = {}

global_acc1_w = {}
global_energy_w = {}
global_legends = []
global_colors = []

stop_epoch = 24

def process_acc_energy(energy, acc):
    energy_list = []
    acc_list = []
    for e, a in zip(energy, acc):
        set_e = sorted(list(set(e)))
        a = [a[e.index(v)] for v in set_e]
        e = list(set_e)
        energy_list.append(e)
        acc_list.append(a)
    return energy_list, acc_list


if __name__=="__main__":
    if plot_1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, acc_w = {}, {}, {}, {}, {}
        legends = []
        for sysCount in [1, 2, 8]:
            for nActivePerCell in [1, 2, 4]:

                local_epochs = 1
                batch_size = 128
                error_rate = 0

                dir_ = os.path.join(current_path, "saved_minist/outputs-"+str(sysCount))

                record_prefix = "-local_epochs-" + str(local_epochs) + \
                                "-batch_size-" + str(batch_size) + \
                                "-error_rate-" + str(error_rate) + \
                                "-nActivePerCell-" + str(nActivePerCell)

                tf_dir = os.path.join(dir_, "tf"+record_prefix)
                if os.path.exists(tf_dir):
                    legends.append(r"$C={:}$, $M={:}$".format(sysCount, nActivePerCell))

                    print("Read from ", tf_dir)
                    time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                    epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                    epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], acc_w[iters] = epochs, times, wall_clocks, losses, accs

                    iters += 1

        plt.subplot(121)
        plot_xs_ys(epoch_w.values(), acc_w.values(), xlabel='Communication Round', ylabel="Acc (\%)", loc=4, show=False,
                   legends=legends)
        plt.subplot(122)
        plot_xs_ys(epoch_w.values(), time_w.values(), xlabel='Communication Round', ylabel="Elapsed Time (s)", loc=4, show=False,
                   legends=legends, logy=True)

        plt.savefig(os.path.join(current_path, 'saved_results/minist-epoch-vs-acc-time-stopEpoch-{}.pdf'.format( stop_epoch)))
        plt.show()


    if plot_2:
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        acc_w, acc_avg_energy_w, acc_sum_energy_w = {}, {}, {}
        legends = []
        for sysCount in [1, 2, 4, 8]:
            for nActivePerCell in [1, 2, 4]:

                local_epochs = 1
                batch_size = 128
                error_rate = 0

                dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

                record_prefix = "-local_epochs-" + str(local_epochs) + \
                                "-batch_size-" + str(batch_size) + \
                                "-error_rate-" + str(error_rate) + \
                                "-nActivePerCell-" + str(nActivePerCell)

                trace_dir = os.path.join(dir_, "trace" + record_prefix)

                tf_dir = os.path.join(dir_, "tf" + record_prefix)

                if os.path.exists(tf_dir):
                    legends.append("sysCount_" + str(sysCount) + "_nActivePerCell_" + str(nActivePerCell))
                    epoch, time, wall_clock, loss, acc = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_acc=0.955)

                    avg_time, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])

                    time_w[iters] = avg_time
                    time_avg_energy_w[iters] = avg_energy
                    time_sum_energy_w[iters] = sum_energy

                    acc_avg_energy_w[iters] = []
                    acc_sum_energy_w[iters] = []
                    acc_w[iters] = []
                    for acc in np.arange(0.9, 0.955, 0.005):
                        _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_acc=acc)
                        _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                        acc_avg_energy_w[iters].append(avg_energy[-1])
                        acc_sum_energy_w[iters].append(sum_energy[-1])
                        acc_w[iters].append(acc)

                    iters += 1

        plot_xs_ys(time_w.values(), time_avg_energy_w.values(), xlabel="time", ylabel="energy", title="time-vs-avg_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/minist-time-vs-avg_energy.png'))
        plot_xs_ys(acc_w.values(), acc_avg_energy_w.values(), xlabel="acc", ylabel="energy", title="acc-vs-avg_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/minist-acc-vs-avg_energy.png'))

        plot_xs_ys(time_w.values(), time_sum_energy_w.values(), xlabel="time", ylabel="energy", title="time-vs-sum_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/minist-time-vs-sum_energy.png'))
        plot_xs_ys(acc_w.values(), acc_sum_energy_w.values(), xlabel="acc", ylabel="energy", title="acc-vs-sum_energy",
                   legends=legends,
                   save_path=os.path.join(current_path, 'saved_results/minist-acc-vs-sum_energy.png'))


    if plot_error1:
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, acc_w = {}, {}, {}, {}, {}
        legends = []
        for error_rate in [1e-4, 1e-5, 1e-6, 1e-7]:
            sysCount = 2
            nActivePerCell = 4
            local_epochs = 1
            batch_size = 128

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)
            if os.path.exists(tf_dir):
                legends.append(r"$PER={:}$".format(error_rate))

                print("Read from ", tf_dir)
                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], acc_w[
                    iters] = epochs, times, wall_clocks, losses, accs

                iters += 1


        global_time_w['error_rate'] = time_w.values()
        global_acc_w['error_rate'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), acc_w.values(), xlabel="time", ylabel="acc", title="time-vs-acc", colors=colors,
        #            legends=legends, save_path=os.path.join(current_path,
        #                                                    'saved_results/minist-error-time-vs-acc-stopEpoch-{}.png'.format(stop_epoch)))
        # plot_xs_ys1_ys2(epoch_w.values(), acc_w.values(), time_w.values(), xlabel="epoch", ylabel1="acc",
        #                 ylabel2='time', loc=2, title="epoch-vs-acc-time", legends=legends, colors=colors,
        #                 save_path=os.path.join(current_path,
        #                                        'saved_results/minist-error-epoch-vs-acc-time-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_error2:
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        acc_w, acc_avg_energy_w, acc_sum_energy_w = {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        for error_rate in [1e-4, 1e-5, 1e-6, 1e-7]:
            sysCount = 2
            nActivePerCell = 4
            local_epochs = 1
            batch_size = 128

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell)

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$PER={:}$".format(error_rate))

                acc_avg_energy_w[iters] = []
                acc_sum_energy_w[iters] = []
                acc_w[iters] = []
                for acc in np.arange(0.9, 0.98, 0.002):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_acc=acc)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    acc_avg_energy_w[iters].append(avg_energy[-1])
                    acc_sum_energy_w[iters].append(sum_energy[-1])
                    acc_w[iters].append(acc)

                iters += 1

        global_energy_w['error_rate'] = acc_avg_energy_w.values()
        global_acc1_w['error_rate'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(acc_w.values(), acc_avg_energy_w.values(), xlabel="acc", ylabel="energy", title="acc-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path, 'saved_results/minist-error-acc-vs-avg_energy.png'))


    if plot_naughty1:
        plt.figure(figsize=(5, 5))
        noise_type_ratio = {'add': [2e-2, 4e-2, 8e-2, 1e-1],
                            'multi': [1e-1, 2e-1, 5e-1, 8e-1]}
        noise_character = {'add': "NIS_a", 'multi': 'NIS_m'}
        for i, noise_type in enumerate(['add', 'multi']):
            iters = 0
            colors = ['g-^', 'k-^', 'r-^', 'b-^', 'y-^']
            epoch_w, time_w, wall_clock_w, loss_w, acc_w = {}, {}, {}, {}, {}
            legends = []

            for noise_ratio in noise_type_ratio[noise_type]:
                sysCount = 2
                nActivePerCell = 4
                local_epochs = 1
                batch_size = 128
                error_rate = 0

                dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

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

                    epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                    epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], acc_w[
                        iters] = epochs, times, wall_clocks, losses, accs

                    iters += 1

            plt.subplot(221 + i)
            plot_xs_ys(time_w.values(), acc_w.values(),  xlabel="Elapsed Time (s)", ylabel="Acc (\%)", colors=colors, show=False,
                       legends=legends, loc=4)

            # plot_xs_ys1_ys2(epoch_w.values(), acc_w.values(), time_w.values(), xlabel="epoch", ylabel1="acc", colors=colors,
            #                 ylabel2='time', loc=4,
            #                 title="epoch-vs-acc-time", legends=legends,
            #                 save_path=os.path.join(current_path,
            #                                        'saved_results/minist-naughty-{}-epoch-vs-acc-time-stopEpoch-{}.png'.format(noise_type, stop_epoch)))


    if plot_naughty2:
        noise_type_ratio = {'add': [2e-2, 4e-2, 8e-2, 1e-1],
                            'multi': [1e-1, 2e-1, 5e-1, 8e-1]}
        noise_character = {'add': "NIS_a", 'multi': 'NIS_m'}
        for i, noise_type in enumerate(['add', 'multi']):
            iters = 0
            colors = ['g-^', 'k-^', 'r-^', 'b-^', 'y-^']
            time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
            acc_w, acc_avg_energy_w, acc_sum_energy_w = {}, {}, {}
            legends = []

            for noise_ratio in noise_type_ratio[noise_type]:
                sysCount = 2
                nActivePerCell = 4
                local_epochs = 1
                batch_size = 128
                error_rate = 0

                dir = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

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

                    epoch, time, wall_clock, loss, acc = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                                             stop_acc=0.98)
                    avg_time, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    time_w[iters] = avg_time
                    time_avg_energy_w[iters] = avg_energy
                    time_sum_energy_w[iters] = sum_energy

                    acc_avg_energy_w[iters] = []
                    acc_sum_energy_w[iters] = []
                    acc_w[iters] = []
                    for acc in np.arange(0.9, 0.98, 0.002):
                        epochs, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'),
                                                               stop_acc=acc)
                        if epochs[-1]==stop_epoch:
                            print("reach stop epoch")
                            break
                        _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                        acc_avg_energy_w[iters].append(avg_energy[-1])
                        acc_sum_energy_w[iters].append(sum_energy[-1])
                        acc_w[iters].append(acc)

                    iters += 1

            # plot_xs_ys(time_w.values(), time_avg_energy_w.values(), xlabel="time", ylabel="energy", colors=colors,
            #            title="time-vs-avg_energy",
            #            legends=legends,
            #            save_path=os.path.join(current_path,
            #                                   'saved_results/minist-naughty-{}-time-vs-avg_energy.png'.format(noise_type)))
            plt.subplot(221 + i + 2)
            acc_avg_energy_w, acc_w = process_acc_energy(list(acc_avg_energy_w.values()), list(acc_w.values()))

            plot_xs_ys(acc_avg_energy_w, acc_w, xlabel="Consumed Energy (J)", ylabel="Acc (\%)", colors=colors, show=False,
                       legends=legends, loc=4)

            # plot_xs_ys(time_w.values(), time_sum_energy_w.values(), xlabel="time", ylabel="energy", colors=colors,
            #            title="time-vs-sum_energy",
            #            legends=legends,
            #            save_path=os.path.join(current_path,
            #                                   'saved_results/minist-naughty-time-vs-sum_energy.png'))
            # plot_xs_ys(acc_w.values(), acc_sum_energy_w.values(), xlabel="acc", ylabel="energy", colors=colors,
            #            title="acc-vs-sum_energy",
            #            legends=legends,
            #            save_path=os.path.join(current_path,
            #                                   'saved_results/minist-naughty-acc-vs-sum_energy.png'))


        plt.savefig(os.path.join(current_path,
                                 'saved_results/minist-naughty.pdf'))
        plt.show()



    if plot_activeratio1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, acc_w = {}, {}, {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']

        for nActivePerCell in [1, 2, 4]:
            sysCount = 2
            local_epochs = 1
            batch_size = 128
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_minist/outputs-"+str(sysCount))

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

                epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], acc_w[iters] = epochs, times, wall_clocks, losses, accs

                iters += 1

        global_time_w['num_clients'] = time_w.values()
        global_acc_w['num_clients'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), acc_w.values(), xlabel="time", ylabel="acc", title="time-vs-acc",
        #            legends=legends, save_path=os.path.join(current_path, 'saved_results/minist-active_ratio-time-vs-acc-stopEpoch-{}.png'.format(stop_epoch)))
        #
        # plot_xs_ys1_ys2(epoch_w.values(), acc_w.values(), time_w.values(), xlabel="epoch", ylabel1="acc", ylabel2='time', title="epoch-vs-acc-time",
        #            legends=legends, save_path=os.path.join(current_path, 'saved_results/minist-active_ratio-epoch-vs-acc-time-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_activeratio2:
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        acc_w, acc_avg_energy_w, acc_sum_energy_w = {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']

        for nActivePerCell in [1, 2, 4]:
            sysCount = 2
            local_epochs = 1
            batch_size = 128
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_minist/outputs-"+str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf"+record_prefix)

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$r={:}$".format(nActivePerCell/4))

                acc_avg_energy_w[iters] = []
                acc_sum_energy_w[iters] = []
                acc_w[iters] = []
                for acc in np.arange(0.9, 0.98, 0.002):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_acc=acc)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    acc_avg_energy_w[iters].append(avg_energy[-1])
                    acc_sum_energy_w[iters].append(sum_energy[-1])
                    acc_w[iters].append(acc)

                iters += 1

        global_energy_w['num_clients'] = acc_avg_energy_w.values()
        global_acc1_w['num_clients'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(acc_w.values(), acc_avg_energy_w.values(), xlabel="acc", ylabel="energy", title="acc-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/minist-active_ratio-acc-vs-avg_energy.png'))


    if plot_epoch1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, acc_w = {}, {}, {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        for local_epochs in [2, 4, 8, 16]:
            batch_size = 128
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

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

                epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], acc_w[
                    iters] = epochs, times, wall_clocks, losses, accs

                iters += 1

        global_time_w['local_epochs'] = time_w.values()
        global_acc_w['local_epochs'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), acc_w.values(), xlabel="time", ylabel="acc", title="time-vs-acc", markersize=4, colors=colors,
        #            legends=legends, save_path=os.path.join(current_path,
        #                                                    'saved_results/minist-local_epoch-time-vs-acc-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_epoch2:
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        acc_w, acc_avg_energy_w, acc_sum_energy_w = {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']

        for local_epochs in [2, 4, 8, 16]:
            batch_size = 128
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$E={:}$".format(local_epochs))

                acc_avg_energy_w[iters] = []
                acc_sum_energy_w[iters] = []
                acc_w[iters] = []
                for acc in np.arange(0.9, 0.98, 0.002):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_acc=acc)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    acc_avg_energy_w[iters].append(avg_energy[-1])
                    acc_sum_energy_w[iters].append(sum_energy[-1])
                    acc_w[iters].append(acc)

                iters += 1

        global_energy_w['local_epochs'] = acc_avg_energy_w.values()
        global_acc1_w['local_epochs'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(acc_w.values(), acc_avg_energy_w.values(), xlabel="acc", ylabel="energy", title="acc-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/minist-local_epoch-acc-vs-avg_energy.png'))


    if plot_bs1:
        iters = 0
        epoch_w, time_w, wall_clock_w, loss_w, acc_w = {}, {}, {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']
        for batch_size in [8, 32, 128, 512]:
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0
            local_epochs = 1

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

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

                epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=stop_epoch)
                epoch_w[iters], time_w[iters], wall_clock_w[iters], loss_w[iters], acc_w[
                    iters] = epochs, times, wall_clocks, losses, accs

                iters += 1

        global_time_w['local_bs'] = time_w.values()
        global_acc_w['local_bs'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(time_w.values(), acc_w.values(), xlabel="time", ylabel="acc", title="time-vs-acc", markersize=4, colors=colors,
        #            legends=legends, save_path=os.path.join(current_path,
        #                                                    'saved_results/minist-bs-time-vs-acc-stopEpoch-{}.png'.format(stop_epoch)))


    if plot_bs2:
        iters = 0
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        acc_w, acc_avg_energy_w, acc_sum_energy_w = {}, {}, {}
        legends = []
        colors = ['r-^', 'b-^', 'g-^', 'k-^']

        for batch_size in [8, 32, 128, 512]:
            sysCount = 2
            nActivePerCell = 4
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0
            local_epochs = 1

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

            record_prefix = "-local_epochs-" + str(local_epochs) + \
                            "-batch_size-" + str(batch_size) + \
                            "-error_rate-" + str(error_rate) + \
                            "-nActivePerCell-" + str(nActivePerCell) + \
                            "-noise_type-" + str(noise_type) + \
                            "-noise_ratio-" + str(noise_ratio)

            tf_dir = os.path.join(dir_, "tf" + record_prefix)

            trace_dir = os.path.join(dir_, "trace" + record_prefix)

            if os.path.exists(tf_dir):
                legends.append(r"$b_s={:}$".format(batch_size))

                acc_avg_energy_w[iters] = []
                acc_sum_energy_w[iters] = []
                acc_w[iters] = []
                for acc in np.arange(0.9, 0.98, 0.002):
                    _, time, _, _, _ = parse_time_acc_loss(os.path.join(tf_dir, 'time-acc-loss.txt'), stop_acc=acc)
                    _, avg_energy, sum_energy = parse_energy(trace_dir, time[-1])
                    acc_avg_energy_w[iters].append(avg_energy[-1])
                    acc_sum_energy_w[iters].append(sum_energy[-1])
                    acc_w[iters].append(acc)

                iters += 1

        global_energy_w['local_bs'] = acc_avg_energy_w.values()
        global_acc1_w['local_bs'] = acc_w.values()
        global_legends.append(legends)
        global_colors.append(colors)

        # plot_xs_ys(acc_w.values(), acc_avg_energy_w.values(), xlabel="acc", ylabel="energy", title="acc-vs-avg_energy", colors=colors,
        #            legends=legends,
        #            save_path=os.path.join(current_path,
        #                                   'saved_results/minist-bs-acc-vs-avg_energy.png'))


    if plot_all1:
        plt.figure(figsize=(5, 5))
        for i, (time_w, acc_w) in enumerate(zip(global_time_w.values(), global_acc_w.values())):
            plt.subplot(221+i)
            plot_xs_ys(time_w, acc_w, xlabel="Elapsed Time (s)", ylabel="Acc (\%)", markersize=0, colors=global_colors[i], show=False, legends=global_legends[i])

        plt.savefig(os.path.join(current_path,
                                 'saved_results/minist-all-time-vs-acc.pdf'))
        plt.show()


    if plot_all2:
        plt.figure(figsize=(5, 5))
        for i, (energy_w, acc_w) in enumerate(zip(global_energy_w.values(), global_acc1_w.values())):
            energy_w, acc_w = process_acc_energy(energy_w, acc_w)
            plt.subplot(221+i)
            plot_xs_ys(energy_w, acc_w, xlabel="Consumed Energy (J)", ylabel="Acc (\%)", markersize=0, colors=global_colors[i], show=False, legends=global_legends[i])

        plt.savefig(os.path.join(current_path,
                                 'saved_results/minist-all-energy-vs-acc.pdf'))
        plt.show()




    if plot_partition:
        iters = 0
        colors = ['r--', 'r-^', 'b-^', 'g-^', 'k-^', 'g--', 'k--']
        time_w, time_avg_energy_w, time_sum_energy_w = {}, {}, {}  # This variables are for time vs energy
        acc_w, acc_avg_energy_w, acc_energy_ratio = {}, {}, {}
        legends = []
        partitions = ['1,1,1,1', '8,1,1,1', '64,1,1,1', '512,1,1,1', '4096,1,1,1', "512,512,512,1", "4096,4096,4096,1"]
        for part_ratio in partitions:
            sysCount = 1
            nActivePerCell = 4
            batch_size = 128
            error_rate = 0
            noise_type = 'add'
            noise_ratio = 0
            local_epochs = 1

            dir_ = os.path.join(current_path, "saved_minist/outputs-" + str(sysCount))

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
                legends.append("partition ratio: " + str(part_ratio))

                time_acc_loss_path = os.path.join(tf_dir, "time-acc-loss.txt")

                epochs, times, wall_clocks, losses, accs = parse_time_acc_loss(time_acc_loss_path, stop_epoch=9)
                time_w[iters] = times
                acc_w[iters] = accs
                avg_time, avg_energy, energys = parse_all_energy(trace_dir, times[-1])

                acc_avg_energy_w[iters] = avg_energy
                acc_energy_ratio[iters] = energys[0]/energys[-1]
                iters += 1

        plot_xs_ys(time_w.values(), acc_energy_ratio.values(), xlabel="time", ylabel="energy_ratio",
                   title="energy_ratio-vs-loss", colors=colors,
                   legends=legends, loc=2,
                   save_path=os.path.join(current_path,
                                          'saved_results/minist-partition-energy_ratio-vs-loss.png'))

        print("partition\t energy_sumed\t time_consumed\t acc \t energy_ratio\t")

        for iters in range(len(partitions)):
            print("{}\t & {:.3f}\t & {:.3f}\t & {:.3f}\t & {:.3f}\t \\\\hline".format(partitions[iters], acc_avg_energy_w[iters][-1].item()/acc_avg_energy_w[0][-1].item(),
                                               time_w[iters][-1]/time_w[0][-1], acc_w[iters][-1], acc_energy_ratio[iters][-1].item()))
