import numpy as np
import scipy as sp
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

rd.seed(1089)

def SEIR_model(IC, t, parameters):
    """
    SEIR model differential equations.
    """

    S, E, I, R = IC
    beta, sigma, gamma = parameters

    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]

def solve_SEIR(IC, parameters, t):
    """
    Solves the SEIR model using the odeint function from scipy.
    """

    solution = sp.integrate.odeint(SEIR_model, IC, t, args=(parameters,))
    return solution

def plot_SEIR(solution, t):
    """
    Plots the SEIR model solution.
    """

    S, E, I, R = solution.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.grid()
    plt.show()

def synthetic_data(IC, parameters, t, noise_level=0.01,
                   output_path='SEIR_noisy_data.csv'):
    """
    Generates synthetic data by adding noise to the SEIR model solution.
    """

    solution = solve_SEIR(IC, parameters, t)
    noise = np.random.normal(0, noise_level, solution.shape)
    noisy_solution = np.zeros_like(solution)
    for i in range(solution.shape[0]):
        for j in range(solution.shape[1]):
            noisy_solution[i, j] = solution[i, j] + noise[i, j]
            if noisy_solution[i, j] < 0:
                noisy_solution[i, j] = min(-noisy_solution[i, j], 1)
            if noisy_solution[i, j] > 1:
                noisy_solution[i, j] = max(2 - noisy_solution[i, j], 0)

    noisy_data = pd.DataFrame(
        noisy_solution,
        columns=['Susceptible', 'Exposed', 'Infected', 'Recovered']
    )
    noisy_data.insert(0, 'Time', t)
    noisy_data.to_csv(output_path, index=False)

    return noisy_solution


IC = [0.99, 0.01, 0.0, 0.0]  # Initial conditions: S, E, I, R
parameters = [0.3, 0.1, 0.05]  # Parameters: beta, sigma, gamma
t = np.linspace(0, 160, 160) 

noisy_solution = synthetic_data(IC, parameters, t)
plot_SEIR(noisy_solution, t)
