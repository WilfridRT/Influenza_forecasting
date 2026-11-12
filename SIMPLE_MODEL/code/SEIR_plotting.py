import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def extract_regime_params(filepath):
    """
    Extracts the regime parameters from the MCMC chains CSV file.
    """

    chains = pd.read_csv(filepath)
    relevant_chains = np.asarray(chains[["beta", "sigma", "gamma"]])
    
    return relevant_chains

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

def plot_mean_SEIR(mean, CI, t):
    """
    Plots the SEIR model solution.
    """

    S, E, I, R = mean.T
    S_upper, E_upper, I_upper, R_upper = CI[0].T
    S_lower, E_lower, I_lower, R_lower = CI[1].T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.fill_between(t, S_lower, S_upper, color='blue', alpha=0.2)
    plt.fill_between(t, E_lower, E_upper, color='orange', alpha=0.2)
    plt.fill_between(t, I_lower, I_upper, color='red', alpha=0.2)
    plt.fill_between(t, R_lower, R_upper, color='green', alpha=0.2)
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_CI(list, confidence):

    upper = np.quantile(list, 0.5 + confidence / 2, axis=0)
    lower = np.quantile(list, 0.5 - confidence / 2, axis=0) 

    return upper, lower

if __name__ == "__main__":

    dir = "C:/Users/wilfr/Documents/Influenza_forecasting/Synthetic_model/"
    filepath = os.path.join(dir, "SEIR_MCMC_chains.csv")
    
    regime_params = extract_regime_params(filepath)

    IC = [0.99, 0.01, 0.0, 0.0]
    t = np.linspace(0, 160, 160)
    regime_num = int(regime_params.shape[0]/40)
    solution = np.zeros((regime_num, len(t), len(IC)))
    for r in range(regime_num):
        regime_parameters = regime_params[40*r]
        regime_solution = solve_SEIR(IC, regime_parameters, t)
        solution[r] = regime_solution
    mean_solution = np.mean(solution, axis=0)
    CI_solution = calculate_CI(solution, 0.95)
    plot_mean_SEIR(mean_solution, CI_solution, t)
    
