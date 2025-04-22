import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio


'''
Si tu te demandes pourquoi j'ai mis des classes, c'est parce que je le voulais. 
Il n'y a pas d'autres raisons 
(à part la modularité lors des tests et le fait qu'on pourra faire les plots depuis un autre fichier)

La dégénéréscence est assez vénère donc il va falloir faire du resampling (je te laisse le faire, au pire on le fait à deux).
(Ne regarde pas trop la première partie du projet, j'en suis pas fier (moins de 24h pour le faire...) mais il y a des sources intéressantes)


Oui j'ai utilisé ce mode de commentaires pour te faire chier
'''

class LorentzParticlesFilter:
    def __init__(self, observations, N, h, measurement_noise, process_noise, initial_state, sigma, rho, beta, resampling_algorithm="multinomial"):
        self.size = len(observations)
        self.observations = observations
        self.N = N
        self.h = h
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.initial_state = initial_state
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.filtered_states = np.zeros((self.size, 3))
        self.filtered_states[0] = initial_state
        self.resampling_treshold = self.N / 10
        self.resampling = None
        if resampling_algorithm == "multinomial":
            self.resampling = self.multinommial_resampling
        elif resampling_algorithm == "residual":
            self.resampling = self.residual_resampling
        elif resampling_algorithm == "systematic":
            self.resampling = self.systematic_resampling
        else:
            raise NotImplementedError()
    
    def f(self, state):
        x, y, z = state
        a = self.sigma * (y - x)
        b = x * (self.rho - z) - y
        c = x * y - self.beta * z
        return np.array([a, b, c])
    
    def RK_discretize(self, state):
        f1 = self.f(state)
        f2 = self.f(state + self.h * f1 / 2)
        f3 = self.f(state + self.h * f2 / 2)
        f4 = self.f(state + self.h * f3)
        return state + self.h * (f1 + 2 * f2 + 2 * f3 + f4) / 6 + np.random.normal(0, self.process_noise, 3)
                    
    def compute(self):
        # Step 1 and 2 (We draw N samples from the prior and initialize the weights)
        samples = np.random.normal(self.initial_state, self.measurement_noise, (self.N, 3))
        weights = np.ones(self.N) / self.N

        # Step 3 and 4 (le step 3 sert juste à mettre n à 0 mais on s'en fout, Draw N samples from the importance distribution)
        for n in range(self.size):
            for i in range(self.N): samples[i] = self.RK_discretize(samples[i])
            # Step 5 (Update the weights) (merci à la loi de Bayes)
            for i in range(self.N):
                diff = self.observations[n] - samples[i]
                weights[i] *= np.exp(-np.dot(diff, diff) / (2 * self.measurement_noise ** 2))
            # Step 6 (Ocus Pocos we juste normalize the weights -us)
            weights /= np.sum(weights)
            # Step 7 (Dot c'est quand même plus rapide que de faire une boucle for)
            self.filtered_states[n] = np.dot(weights, samples)

            if 1 / np.dot(weights, weights) < self.resampling_treshold:
                samples = self.resampling(samples, weights)
                weights = np.ones(self.N) / self.N
        
        return self.filtered_states

    def multinommial_resampling(self, samples, weights): # chaque point est choisis avec une probabilité égale à son poids
        return samples[np.random.choice(self.N, self.N, p=weights)]
    
    def residual_resampling(self, samples, weights):
        res = np.zeros_like(samples)

        # on regarde les proba qui ont un poids w >= n/N où n est un entier (le plus grand pour que ce soit vrai)
        floors = np.floor(weights * self.N)
        index = 0
        for i in range(self.N):
            if floors[i] != 0:
                # on rajoute à notre resampling n fois le point sachant que w >= n/N
                res[index:index+int(floors[i])] = samples[i]
                index += int(floors[i])
        
        # pour tout échantillon r = w - n/N donnant le résidu du poids avec n le nombre de fois que l'échantillon fais déjà parti du resampling
        residuals = weights - floors/self.N
        residuals /= np.sum(residuals)

        # on choisis les points restants en appliquant multinomial sur base des résidus
        res[index:] = samples[np.digitize(np.random.rand(self.N-index), bins=np.cumsum(residuals))]
        
        return res
    
    def systematic_resampling(self, samples, weights): # tous les points sont choisis de manière à être séparés de n/N où n est un entier
        uniforms = np.random.rand() / self.N + np.arange(self.N) / self.N 
        return samples[np.digitize(uniforms, bins=np.cumsum(weights))]


class LorentzSystem:
    def __init__(self, sigma, rho, beta, initial_state, N, h):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.state0 = initial_state
        self.t = np.arange(0.0, N, h)
        self.states = None
    
    def f(self, state, t):
        x, y, z = state
        a = self.sigma * (y - x)
        b = x * (self.rho - z) - y
        c = x * y - self.beta * z
        return np.array([a, b, c])
    
    def compute(self, t):
        self.states = odeint(self.f, self.state0, t)
        return self.states
    

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [1.0, 1.0, 1.0]
tmax = 100
h = 0.02

measurement_noise = 1
process_noise = 0.1
N = 100


    
# Lorentz system
lorentzSystem = LorentzSystem(sigma, rho, beta, initial_state, tmax, h)
states = lorentzSystem.compute(lorentzSystem.t)
observations = np.random.normal(states, measurement_noise, (len(states), 3))

# Particle filter
particleFilter = LorentzParticlesFilter(observations, N, h, measurement_noise, process_noise, initial_state, sigma, rho, beta)
filtered_observation = particleFilter.compute()

fig = plt.figure()
plt.rcParams['font.family'] = 'serif'
ax = fig.add_subplot(projection="3d")
ax.scatter(states[:, 0], states[:, 1], states[:, 2], marker='o', s=1, alpha=0.8)
ax.scatter(filtered_observation[:, 0], filtered_observation[:, 1], filtered_observation[:, 2], marker='o', s=1, alpha=0.8)
# ax.plot(states[:, 0], states[:, 1], states[:, 2])
# ax.plot(filtered_observation[:, 0], filtered_observation[:, 1], filtered_observation[:, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['True system', 'Filtered system'])
plt.draw()
plt.show()