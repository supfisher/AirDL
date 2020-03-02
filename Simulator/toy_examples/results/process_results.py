import matplotlib.pyplot as plt
import pandas as pd


def avg(x): return sum(x)/len(x)


def analyze_one_file(file_path, loss_threshold, max_tolerance):
    tolerance = 0

    throughput, goodput, energy_cost, time_cost, pakcet_loss = 0, 0, 0, 0, 0
    avg_loss = []
    with open(file_path, 'r') as f:
        iter_times = 0
        for report in f.readlines():
            report = eval(report)
            if report['rank'] == 0:
                throughput = (throughput*iter_times +
                              report['throughput'])/(iter_times+1)
                goodput = (goodput*iter_times +
                              report['goodput'])/(iter_times+1)
                energy_cost = report['energy_cost']
                time_cost = report['time_cost']
                pakcet_loss = (pakcet_loss*iter_times +
                               report['packet_loss'])/(iter_times+1)
                iter_times += 1
                loss = avg(report['loss'])
                avg_loss.append(loss)
                if loss <= loss_threshold:
                    tolerance += 1
                    if tolerance >= max_tolerance:
                        print(iter_times)
                        break
                else:
                    tolerance = 0

    return throughput, goodput, energy_cost, time_cost, pakcet_loss, avg_loss


def myplot(x, y, title, xlabel, ylabel, lengend):
    plt.semilogx(x, y[0], 'r', linestyle="--", marker="^", markersize=4, linewidth=1.0)
    plt.semilogx(x, y[1], 'b', linestyle="--", marker="s", markersize=4, linewidth=1.0)
    plt.semilogx(x, y[2], 'k', linestyle="--", marker="o", markersize=4, linewidth=1.0)
    plt.xticks(x, [str(xx) for xx in x])
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(lengend)
    plt.grid(color="0.9", linestyle="-")


def mymultiplot(fig_id, x, data_dict, xlabel, lengend):
    plt.figure(fig_id)
    plt.subplot(221)
    myplot(x, data_dict['throughput'], 'throughput', xlabel, 'Throughput (Mb/s)', lengend)
    plt.subplot(222)
    myplot(x, data_dict['energy_cost'], 'energy_cost', xlabel, 'Energy (J)', lengend)
    plt.subplot(223)
    myplot(x, data_dict['time_cost'], 'time_cost', xlabel, 'Time (s)', lengend)
    plt.subplot(224)
    myplot(x, data_dict['pakcet_loss'], 'pakcet_loss', xlabel, 'Packet loss ratio', lengend)
    # plt.title(title)
    plt.show()

def analyze_loss(file_path):
    avg_loss = []
    with open(file_path, 'r') as f:
        for report in f.readlines():
            report = eval(report)
            loss = avg(report['loss'])
            avg_loss.append(loss)
    return avg_loss


if __name__ == '__main__':
    epsilon = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    loss_threshold = [0.1, 0.05, 0.01]
    throughputs, goodputs, energy_costs, time_costs, pakcet_losss = \
        [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    for i, l in enumerate(loss_threshold):
        for e in epsilon:
            file_path = 'results_40_epsilon_%s.json'%e
            throughput, goodput, energy_cost, time_cost, pakcet_loss, avg_loss = analyze_one_file(file_path, l, 10)
            print(file_path, '---throughput: ', throughput, 'goodput: ', goodput,
              'energy_cost: ', energy_cost, 'time_cost: ', time_cost, 'average_pakcet_loss: ', pakcet_loss)

            throughputs[i].append(throughput)
            goodputs[i].append(goodput)
            energy_costs[i].append(energy_cost)
            time_costs[i].append(time_cost)
            pakcet_losss[i].append(pakcet_loss)

    epsilon_data_dict = {'throughput': throughputs,
                               'goodput': goodputs,
                               'energy_cost': energy_costs,
                               'time_cost': time_costs,
                               'pakcet_loss': pakcet_losss,
                               'loss_threshold': loss_threshold}

    data_frame = pd.DataFrame(epsilon_data_dict)
    data_frame.to_csv('different_epsilon.csv')

    mymultiplot(1, epsilon, epsilon_data_dict, 'epsilon value', ['loss<0.1', 'loss<0.05', 'loss<0.01'])


    clients = [10, 20, 40, 80, 160]
    loss_threshold = [0.2, 0.15, 0.1]
    throughputs, goodputs, energy_costs, time_costs, pakcet_losss = \
        [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    for i, l in enumerate(loss_threshold):
        for c in clients:
            file_path = 'results_%s_epsilon_0.2.json'%c
            throughput, goodput, energy_cost, time_cost, pakcet_loss, avg_loss = analyze_one_file(file_path, l, 10)
            print(file_path, '---throughput: ', throughput, 'goodput: ', goodput,
              'energy_cost: ', energy_cost, 'time_cost: ', time_cost, 'average_pakcet_loss: ', pakcet_loss)

            throughputs[i].append(throughput)
            goodputs[i].append(goodput)
            energy_costs[i].append(energy_cost)
            time_costs[i].append(time_cost)
            pakcet_losss[i].append(pakcet_loss)

    clients_data_dict = {'throughput': throughputs,
                               'goodput': goodputs,
                               'energy_cost': energy_costs,
                               'time_cost': time_costs,
                               'pakcet_loss': pakcet_losss,
                               'loss_threshold': loss_threshold}

    data_frame = pd.DataFrame(clients_data_dict)
    data_frame.to_csv('different_clients.csv')
    mymultiplot(2, clients, clients_data_dict, 'number of clients', ['loss<0.2', 'loss<0.15', 'loss<0.1'])


    clients = [10, 20, 40, 80, 160]
    loss_threshold = [0.2, 0.15, 0.1]
    throughputs, goodputs, energy_costs, time_costs, pakcet_losss = \
        [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    for i, l in enumerate(loss_threshold):
        for c in clients:
            file_path = 'results_%s_bs_64.json'%c
            throughput, goodput, energy_cost, time_cost, pakcet_loss, avg_loss = analyze_one_file(file_path, l, 10)
            print(file_path, '---throughput: ', throughput, 'goodput: ', goodput,
              'energy_cost: ', energy_cost, 'time_cost: ', time_cost, 'average_pakcet_loss: ', pakcet_loss)

            throughputs[i].append(throughput)
            goodputs[i].append(goodput)
            energy_costs[i].append(energy_cost)
            time_costs[i].append(time_cost)
            pakcet_losss[i].append(pakcet_loss)

    bs_data_dict = {'throughput': throughputs,
                               'goodput': goodputs,
                               'energy_cost': energy_costs,
                               'time_cost': time_costs,
                               'pakcet_loss': pakcet_losss,
                               'loss_threshold': loss_threshold}

    data_frame = pd.DataFrame(bs_data_dict)
    data_frame.to_csv('different_batch_size.csv')
    mymultiplot(3, clients, bs_data_dict, 'number of clients', ['loss<0.2', 'loss<0.15', 'loss<0.1'])
