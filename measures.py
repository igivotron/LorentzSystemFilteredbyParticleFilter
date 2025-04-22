import importlib
# Tout ça à cause du tiret dans le nom du fichier
particle_filter = importlib.import_module("particle-filter")
ParticleFilter = particle_filter.LorentzParticlesFilter
LorentzSystem = particle_filter.LorentzSystem
MeasuringTools = particle_filter.MeasuringTools
import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [1.0, 1.0, 1.0]
tmax = 100
h= [0.01, 0.02, 0.05, 0.1]

measurement_noise = 1
process_noise = 0.1
N = 100

toolBox = MeasuringTools(None, None)

mean_distance = np.zeros(len(h))
std_distance = np.zeros(len(h))

LorentzSystem = LorentzSystem(sigma, rho, beta, initial_state, tmax, 1)

for i in range(len(h)):
    print(f"Computing for h={h[i]}")
    LorentzSystem.put_h(h[i])
    states = LorentzSystem.compute()
    observations = np.random.normal(states, measurement_noise, (len(states), 3))
    filtered_observation = ParticleFilter(observations, N, h[i], measurement_noise, process_noise, initial_state, sigma, rho, beta).compute()
    toolBox.real_states = states
    toolBox.approx_states = filtered_observation
    _, mean, std = toolBox.distance_p2p()
    mean_distance[i] = mean
    std_distance[i] = std
    print(f"Mean distance for h={h[i]}: {mean}, Std distance: {std}")

plt.figure(figsize=(10, 5))
plt.plot(h, mean_distance, label='Mean Distance', marker='o')
plt.plot(h, std_distance, label='Standard Deviation', marker='o')
plt.grid()
plt.show()

