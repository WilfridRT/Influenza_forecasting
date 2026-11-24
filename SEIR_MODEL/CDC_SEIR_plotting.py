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
    relevant_param_chains = np.asarray(chains[["beta", "sigma", "gamma", "kappa"]])
    relevant_IC_chains = np.asarray(chains[["S0", "E0", "I0", "R0"]])
    
    return relevant_param_chains, relevant_IC_chains

def SEIR_model(IC, t, N, parameters):
    """
    SEIR model differential equations.
    """

    S, E, I, R, C = IC
    beta, sigma, gamma, kappa = parameters

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dCdt = beta * S * I / N  # New infections for ILI calculation

    return [dSdt, dEdt, dIdt, dRdt, dCdt]

def solve_SEIR(IC, N, parameters, t):
    """
    Solves the SEIR model using the odeint function from scipy.
    """

    solution = sp.integrate.odeint(SEIR_model, IC, t, args=(N,parameters,))
    return solution

def plot_SEIR_regimes(solution, t):
    """
    Plots the SEIR model solution.
    """

    plt.figure(figsize=(10, 6))

    for i in range(solution.shape[0]):

        regime_solution = solution[i]
        S, E, I, R, C = regime_solution.T

        plt.plot(t, S, label='Susceptible', color='blue', alpha=0.5)
        plt.plot(t, E, label='Exposed', color='orange', alpha=0.5)
        plt.plot(t, I, label='Infected', color='red', alpha=0.5)
        plt.plot(t, R, label='Recovered', color='green', alpha=0.5)
        plt.plot(t, C, label='New Infections', color='purple', alpha=0.5)

    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.title('SEIR Model Simul0ation')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

def plot_mean_SEIR(mean, CI, t):
    """
    Plots the SEIR model solution against the synthetic one
    """

    S, E, I, R, C = mean.T
    S_upper, E_upper, I_upper, R_upper, C_upper = CI[0].T
    S_lower, E_lower, I_lower, R_lower, C_lower = CI[1].T

    plt.figure(figsize=(10, 6))

    plt.plot(t, S, label='Susceptible', color='blue')
    plt.plot(t, E, label='Exposed', color='orange')
    plt.plot(t, I, label='Infected', color='red')
    plt.plot(t, R, label='Recovered', color='green')
    plt.plot(t, C, label='New Infections', color='purple')
    plt.fill_between(t, S_lower, S_upper, color='blue', alpha=0.2)
    plt.fill_between(t, E_lower, E_upper, color='orange', alpha=0.2)
    plt.fill_between(t, I_lower, I_upper, color='red', alpha=0.2)
    plt.fill_between(t, R_lower, R_upper, color='green', alpha=0.2)
    plt.fill_between(t, C_lower, C_upper, color='purple', alpha=0.2)

    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.grid()
    plt.savefig("C:/Users/wilfr/Documents/Influenza_forecasting/Data/CDC Data/CDC_SEIR_mean_CI_plot.png")
    plt.show()
    plt.close()

def plot_mean_ILI(mean, CI, ILI_counts, t):
    """
    Plots the mean ILI solution against the synthetic one
    """

    ILI = mean
    ILI_upper = CI[0]
    ILI_lower = CI[1]

    plt.figure(figsize=(10, 6))

    plt.plot(t, ILI, label='Mean Weekly ILI', color='purple')
    plt.fill_between(t, ILI_lower, ILI_upper, color='purple', alpha=0.2)
    plt.plot(t, ILI_counts, label='Reported Weekly ILI', color='black', linestyle='--')

    plt.xlabel('Time (Weeks since W30 2023)')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation - Weekly reported ILI')
    plt.legend()
    plt.grid()
    plt.savefig("C:/Users/wilfr/Documents/Influenza_forecasting/Data/CDC Data/CDC_ILI_mean_CI_plot.png")
    plt.show()
    plt.close()

def calculate_CI(solution, confidence):

    upper = np.quantile(solution, 0.5 + confidence / 2, axis=0)
    lower = np.quantile(solution, 0.5 - confidence / 2, axis=0) 

    return upper, lower

if __name__ == "__main__":

    dir = "C:/Users/wilfr/Documents/Influenza_forecasting/"

    regime_params, regime_IC = extract_regime_params(os.path.join(dir, "Data/CDC Data/CDC_SEIR_MCMC_chains.csv"))
    CDC_ILI_data = pd.read_csv(os.path.join(dir, "Data/CDC Data/CDC_Age_Structured_ILI_filtered.csv"))

    filtered_data = CDC_ILI_data[((CDC_ILI_data['year'] == 2023) & (CDC_ILI_data['week'] >= 30) | 
                      (CDC_ILI_data['year'] == 2024) & (CDC_ILI_data['week'] <= 30))]
    pivoted_data = (
        filtered_data
        .pivot_table(index=['year', 'week'], columns='age_group', values='value')
        .sort_index()
        )

    ILI_reported = pivoted_data["ILITOTAL"].to_numpy(dtype=float)
    ILI_counts = np.rint(ILI_reported).astype(int)  

    N = 334906305
    t = np.linspace(0, 52, 53)
    regime_num = int(regime_params.shape[0]/40)
    solution = np.zeros((regime_num, len(t), len(regime_IC[0]) + 1))
    ILI = np.zeros((regime_num, len(t)))

    for r in range(regime_num):
        IC = np.append(regime_IC[r*40], 0.0)
        parameters = regime_params[r*40]
        kappa = parameters[3]
        solution[r] = solve_SEIR(IC, N, parameters, t)
        ILI[r] = np.diff(solution[r][:, 4], prepend=np.nan) / kappa

    mean_solution = np.mean(solution, axis=0)
    CI_solution = calculate_CI(solution, 0.95)
    plot_mean_SEIR(mean_solution, CI_solution, t)

    mean_ILI = np.nanmean(ILI, axis=0)
    CI_ILI = calculate_CI(ILI, 0.95)
    plot_mean_ILI(mean_ILI, CI_ILI, ILI_counts, t)
