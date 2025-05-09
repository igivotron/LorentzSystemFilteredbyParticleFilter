import numpy as np
import matplotlib.pyplot as plt


# Experience :
# 0 : Influence de la longueur de pas
# 1 : Influence du bruit de processus
# 2 : Influence de la méthode de resampling

experience = 0

if experience == 0:
    step_size = np.loadtxt("./data/step_size.csv", delimiter=";", skiprows=1)
    h = step_size[:, 0]
    mean_distance = step_size[:, 1]
    std_distance = step_size[:, 2]
    var = step_size[:, 3]

    # two scales
    plt.figure(figsize=(10, 5))
    plt.plot(h, mean_distance, label='Mean Distance', marker='o')
    # plt.plot(h, std_distance, label='Standard Deviation', marker='o')
    plt.plot(h, var, label='Variance', marker='o')

    plt.xlabel('Step Size (h)')
    plt.ylabel('Distance')
    plt.title('Influence of Step Size on Distance')
    plt.legend()
    plt.grid()
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
    fig.set_size_inches(10, 5)
    ax2 = ax1.twinx()
    ax1.plot(noise, mean_distance, label='Mean Distance', marker='o')
    ax2.plot(noise, var, label='Variance', marker='o', color='orange')
    ax1.set_xlabel('Process Noise')
    ax1.set_ylabel('Mean Distance')
    ax2.set_ylabel('Variance')
    ax1.set_title('Influence of Process Noise on Distance')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()


    plt.xlabel('Process Noise')
    plt.ylabel('Distance')
    plt.title('Influence of Process Noise on Distance')
    plt.legend()
    plt.grid()
    plt.savefig("./figures/process_noise.png")
    plt.show()

if experience == 2:
    pass

