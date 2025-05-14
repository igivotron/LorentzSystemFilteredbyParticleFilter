import importlib
# Tout ça à cause du tiret dans le nom du fichier
particle_filter = importlib.import_module("particle-filter")
ParticleFilter = particle_filter.LorentzParticlesFilter
LorentzSystem = particle_filter.LorentzSystem
MeasuringTools = particle_filter.MeasuringTools
import numpy as np
import time


# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [1.0, 1.0, 1.0]
tmax = 100
h = 0.02

measurement_noise = 1
process_noise = 10
N = 100

# Experiences :
# 0 : Influence de la longueur de pas
# 1 : Influence du bruit de processus
# 2 : Influence de la méthode de resampling
experience = 0


toolBox = MeasuringTools(None, None)
LorentzSystem = LorentzSystem(sigma, rho, beta, initial_state, tmax, 1)

if experience == 0:
    h= np.arange(0.0001, 0.04, 0.001)
    mean_distance = np.zeros(len(h))
    std_distance = np.zeros(len(h))
    states = LorentzSystem.compute()
    toolBox.real_states = states


    for i in range(len(h)):
        print(f"Computing for h={h[i]}")
        # LorentzSystem.h = h[i]
        # states = LorentzSystem.compute()
        # toolBox.real_states = states
        observations = np.random.normal(states, measurement_noise, (len(states), 3))
        filtered_observation = ParticleFilter(observations, N, h[i], measurement_noise, process_noise, initial_state, sigma, rho, beta).compute()
        toolBox.approx_states = filtered_observation
        _, mean, std = toolBox.distance_p2p()
        mean_distance[i] = mean
        std_distance[i] = std
        print(f"Mean distance for h={h[i]}: {mean}, Std distance: {std}")

    var = np.square(std_distance)
    data = np.array([h, mean_distance, std_distance, var]).T
    np.savetxt("./data/step_sizeN100.csv", data, delimiter=";", header="Step_Size,Mean,Std,Var", comments='')
    # plt.figure(figsize=(10, 5))
    # plt.plot(h, mean_distance, label='Mean Distance', marker='o')
    # plt.plot(h, std_distance, label='Standard Deviation', marker='o')
    # plt.grid()
    # plt.xlabel('Step Size (h)')
    # plt.ylabel('Distance')
    # plt.title('Influence of Step Size on Distance')
    # plt.legend()
    # plt.savefig("./figures/step_size.png")
    # plt.show() 

if experience == 1:
    process_noise = np.arange(0.5, 50, 0.5)
    mean_distance = np.zeros(len(process_noise))
    std_distance = np.zeros(len(process_noise))
    states = LorentzSystem.compute()
    observations = np.random.normal(states, measurement_noise, (len(states), 3))
    toolBox.real_states = states
    filter = ParticleFilter(observations, N, h, measurement_noise, process_noise[0], initial_state, sigma, rho, beta)

    for i in range(len(process_noise)):
        print(f"Computing for process noise={process_noise[i]}")
        filter.process_noise = process_noise[i]
        filtered_observation = filter.compute()
        toolBox.approx_states = filtered_observation
        _, mean, std = toolBox.distance_p2p()
        mean_distance[i] = mean
        std_distance[i] = std
        print(f"Mean distance for process noise={process_noise[i]}: {mean}, Std distance: {std}")
    
    var = np.square(std_distance)

    data = np.array([process_noise, mean_distance, std_distance, var]).T
    np.savetxt("./data/process_noise.csv", data, delimiter=";", header="Process_Noise,Mean,Std,Var,Time", comments='')

    # plt.figure(figsize=(10, 5))
    # plt.plot(process_noise, mean_distance, label='Mean Distance', marker='o')
    # # plt.plot(process_noise, std_distance, label='Standard Deviation', marker='o')
    # plt.plot(process_noise, var, label='Variance', marker='o')
    
    # plt.xlabel('Process Noise')
    # plt.ylabel('Distance')
    # plt.title('Influence of Process Noise on Distance')
    # plt.legend()
    # plt.grid()
    # plt.savefig("process_noise.png")
    # plt.show()


if experience == 2:
    LorentzSystem.h = 0.02
    measurement_noise = 1
    resampling_methods = ['temoin', 'multinomial', 'residual', 'systematic']
    states = LorentzSystem.compute()
    observations = np.random.normal(states, measurement_noise, (len(states), 3))
    toolBox.real_states = states
    filter = ParticleFilter(observations, N, h, measurement_noise, process_noise, initial_state, sigma, rho, beta)

    loop = 10
    mean_distance_methods = np.zeros(len(resampling_methods))
    std_distance_methods = np.zeros(len(resampling_methods))
    var_methods = np.zeros(len(resampling_methods))
    temps = np.zeros(len(resampling_methods))

    for i in range(len(resampling_methods)):
        filter.change_resampling(resampling_methods[i])
        l1, l2 = [], []
        for j in range(loop):
            start = time.time()
            filtered_observation = filter.compute()
            l1.append(time.time() - start)
            toolBox.approx_states = filtered_observation
            distance, _, _ = toolBox.distance_p2p()
            l2.append(distance)
        
        temps[i] = np.mean(l1)
        mean_distance_methods[i] = np.mean(l2)
        std_distance_methods[i] = np.std(l2)
        var_methods[i] = np.var(l2)

    
    data = np.array([mean_distance_methods, std_distance_methods, var_methods, temps]).T
    np.savetxt("./data/resampling.csv", data, delimiter=";", header="Mean,Std,Var", comments='')
    


