import os
import numpy as np
import scipy as sp
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

dir = "C:/Users/wilfr/Documents/Influenza_forecasting/Data/CDC Data/"

rd.seed(1089)

def SEIR_model(IC, t, parameters):
    """
    SEIR model differential equations.
    """

    S, E, I, R = IC
    beta, sigma, gamma = parameters

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
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

    S, E, I, R = solution[:, 0:4].T

    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)

    ILI = solution[:, 4].T

    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, ILI, label='Daily ILI')
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.title('SEIR Model Simulation - Daily reported ILI')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)

def synthetic_data(IC, parameters, t,
                   output_path='CDC_SEIR_synthetic_data.csv'):
    """
    Generates synthetic data by adding noise to the SEIR model solution.
    """

    path = os.path.join(dir, output_path)

    solution = solve_SEIR(IC, parameters, t)
    infect_report_data = np.zeros(solution.shape[0])
    infect_report_data[0] = 0
    for i in range (1, solution.shape[0]):
        infect_report_data[i] = (solution[i][2] - solution[i-1][2]) + (solution[i][3] - solution[i-1][3])
    infect_report_data = infect_report_data.T

    print(infect_report_data)

    noise = np.random.normal(0, 82.8, solution.shape[0])
    noisy_infect_data = np.zeros_like(infect_report_data)
    for i in range(len(infect_report_data)):
        noisy_infect_data[i] = infect_report_data[i] + noise[i]
        if noisy_infect_data[i] < 0:
            noisy_infect_data[i] = 0

    print(noisy_infect_data)

    noisy_solution = np.column_stack((solution, noisy_infect_data))

    print(noisy_solution)

    noisy_data = pd.DataFrame(
        noisy_solution,
        columns=['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Daily ILI']
    )
    noisy_data.insert(0, 'Time', t)
    noisy_data.to_csv(path, index=False)

    return noisy_solution

N = 334906305
S = 334906305 - 26417 - 26896
I = 26417
E = 26896
R = 0

IC = [S, E, I, R]  # Initial conditions: S, E, I, R
parameters = [4.16, 2.59, 4.25]  # Parameters: beta, sigma, gamma
t = np.linspace(0, 52, 53) 

noisy_solution = synthetic_data(IC, parameters, t)
plot_SEIR(noisy_solution, t)
