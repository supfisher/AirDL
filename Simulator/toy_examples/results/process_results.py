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


def myplot(x, y, title):
    plt.plot(x, y[0], 'r')
    plt.plot(x, y[1], 'g')
    plt.plot(x, y[2], 'b')
    plt.title(title)
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
    # epsilon = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    epsilon = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    clients = [10, 20, 40, 80, 160]
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

    myplot(epsilon, throughputs, 'epsilon: throughput')
    myplot(epsilon, goodputs, 'epsilon: goodput')
    myplot(epsilon, energy_costs, 'epsilon: energy_cost')
    myplot(epsilon, time_costs, 'epsilon: time_cost')
    myplot(epsilon, pakcet_losss, 'epsilon: pakcet_loss')

    data_frame = pd.DataFrame({'throughput': throughputs,
                               'goodput': goodputs,
                               'energy_cost': energy_costs,
                               'time_cost': time_costs,
                               'pakcet_loss': pakcet_losss,
                               'loss_threshold': loss_threshold})
    data_frame.to_csv('different_epsilon.csv')



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

    myplot(clients, throughputs, 'clients: throughput')
    myplot(clients, goodputs, 'clients: goodput')
    myplot(clients, energy_costs, 'clients: energy_cost')
    myplot(clients, time_costs, 'clients: time_cost')
    myplot(clients, pakcet_losss, 'clients: pakcet_loss')

    data_frame = pd.DataFrame({'throughput': throughputs,
                               'goodput': goodputs,
                               'energy_cost': energy_costs,
                               'time_cost': time_costs,
                               'pakcet_loss': pakcet_losss,
                               'loss_threshold': loss_threshold})
    data_frame.to_csv('different_clients.csv')
