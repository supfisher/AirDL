import matplotlib.pyplot as plt


def avg(x): return sum(x)/len(x)


def analyze_one_file(file_path, loss_threshold):
    throughput, goodput, energy_cost, time_cost, pakcet_loss = 0, 0, 0, 0, 0

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
                if avg(report['loss']) <= loss_threshold:
                    break
    return throughput, goodput, energy_cost, time_cost, pakcet_loss


def myplot(x, y, title):
    plt.plot(x, y[0], 'r')
    plt.plot(x, y[1], 'g')
    plt.plot(x, y[2], 'b')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    epsilon = [0.05, 0.1, 0.2, 0.3, 0.4]
    clients = [10, 20, 40, 80, 320]
    loss_threshold = [0.02, 0.015, 0.01]
    throughputs, goodputs, energy_costs, time_costs, pakcet_losss = \
        [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    for i, l in enumerate(loss_threshold):
        for e in epsilon:
            file_path = 'results_20_epsilon_%s.json'%e
            throughput, goodput, energy_cost, time_cost, pakcet_loss = analyze_one_file(file_path, l)
            print(file_path, '---throughput: ', throughput, 'goodput: ', goodput,
              'energy_cost: ', energy_cost, 'time_cost: ', time_cost, 'average_pakcet_loss: ', pakcet_loss)

            throughputs[i].append(throughput)
            goodputs[i].append(goodput)
            energy_costs[i].append(energy_cost)
            time_costs[i].append(time_cost)
            pakcet_losss[i].append(pakcet_loss)

    myplot(epsilon, throughputs, 'throught')
    myplot(epsilon, goodputs, 'goodput')
    myplot(epsilon, energy_costs, 'energy_cost')
    myplot(epsilon, time_costs, 'time_cost')
    myplot(epsilon, pakcet_losss, 'pakcet_loss')
