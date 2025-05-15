import numpy as np
import matplotlib.pyplot as plt


# Experience :
# 0 : Influence de la longueur de pas
# 1 : Influence du bruit de processus
# 2 : Influence de la méthode de resampling

experience = 1

if experience == 0:
    step_size = np.loadtxt("./data/step_size.csv", delimiter=";", skiprows=1)
    
    h = step_size[:, 0]
    mean_distance = step_size[:, 1]
    std_distance = step_size[:, 2]
    var = step_size[:, 3]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 5)
    ax2 = ax1.twinx()
    ax1.plot(h, mean_distance, label='Mean Distance', marker='o')
    ax2.plot(h, var, label='Variance', marker='o', color='orange')
    ax1.set_xlabel('Step Size (h)')
    ax1.set_ylabel('Mean Distance')
    ax2.set_ylabel('Variance')
    ax1.set_title('Influence of Step Size on Distance')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()

    plt.savefig("./figures/step_size.png")
    plt.show()

if experience == 1:
    process_noise = np.loadtxt("./data/process_noise.csv", delimiter=";", skiprows=1)
    noise = process_noise[:, 0]
    mean_distance = process_noise[:, 1]
    std_distance = process_noise[:, 2]
    var = process_noise[:, 3]

    # two scales
    fig, ax1 = plt.subplots()
    fig.set_size_inches(7, 4)
    ax2 = ax1.twinx()
    ax1.plot(noise, mean_distance, label='Mean Distance', marker='o')
    ax2.plot(noise, var, label='Variance', marker='o', color='orange')
    ax1.set_xlabel('Measurement Noise')
    ax1.set_ylabel('Mean Distance')
    ax2.set_ylabel('Variance')
    ax1.set_title('Influence of Measurement Noise on Distance')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper center')
    ax1.grid()

    plt.savefig("./figures/measurement_noise.png")
    plt.show()

if experience == 2:
    resampling_methods = ['temoin', 'multinomial', 'residual', 'systematic']
    resampling_data = np.loadtxt("./data/resampling_mes10_N100.csv", delimiter=";", skiprows=1)
    mean_distance = resampling_data[:, 0]
    std_distance = resampling_data[:, 1]
    var = resampling_data[:, 2]
    time = resampling_data[:, 3]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 5)
    ax2 = ax1.twinx()
    ax1.plot(resampling_methods, mean_distance, label='Mean Distance', marker='o')
    ax1.plot(resampling_methods, var, label='Variance', marker='o', color='orange')
    ax2.plot(resampling_methods, time, label='Time', marker='o', color='green')
    ax1.set_xlabel('Resampling Method')
    ax1.set_ylabel('Mean Distance')
    ax2.set_ylabel('Time')
    ax1.set_title('Influence of Resampling Method on Distance and execution Time')
    ax1.legend(loc='center right')
    ax2.legend(loc='center left')

    
    ax1.grid()
    plt.savefig("./figures/resampling_mes10_N100.png")
    plt.show()

