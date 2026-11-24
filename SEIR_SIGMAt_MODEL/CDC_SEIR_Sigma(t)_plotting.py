import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


# ==========================================================
# 1. Extract posterior samples and predetermined beta
# ==========================================================

def extract_regime_params(constant_filepath, time_filepath):
    """
    Load posterior samples and the predetermined
    time-dependent transmission rate.
    """

    constant_chains = pd.read_csv(constant_filepath)
    time_params = pd.read_csv(time_filepath)

    # Posterior samples of constant parameters
    parameter_chains = constant_chains[
        ["sigma", "gamma", "kappa"]
    ].to_numpy(dtype=float)

    # Posterior samples of initial conditions
    IC_chains = constant_chains[
        ["S0", "E0", "I0"]
    ].to_numpy(dtype=float)

    # Predetermined weekly beta
    beta_weekly = time_params[
        "beta_mean"
    ].to_numpy(dtype=float)

    return parameter_chains, beta_weekly, IC_chains


# ==========================================================
# 2. SEIR differential equations
# ==========================================================

def SEIR_model(t, y, N, parameters, beta_times, beta_weekly):
    """
    SEIR model with predetermined time-dependent beta.

    State variables:
        S: Susceptible
        E: Exposed
        I: Infectious
        R: Recovered
        C: Cumulative new infections
    """

    S, E, I, R, C = y

    sigma, gamma = parameters

    # Interpolate beta at the current solver time
    beta_t = np.interp(
        t,
        beta_times,
        beta_weekly
    )

    new_infections = beta_t * S * I / N

    dSdt = -new_infections
    dEdt = new_infections - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dCdt = new_infections

    return [
        dSdt,
        dEdt,
        dIdt,
        dRdt,
        dCdt
    ]


# ==========================================================
# 3. Solve SEIR model
# ==========================================================

def solve_SEIR(IC, N, parameters, beta_times, beta_weekly, t):
    """
    Solve the SEIR model at weekly time points.
    """

    solution = solve_ivp(
        fun=SEIR_model,
        t_span=(t[0], t[-1]),
        y0=IC,
        t_eval=t,
        args=(
            N,
            parameters,
            beta_times,
            beta_weekly
        ),
        method="RK45",
        rtol=1e-7,
        atol=1e-5
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y.T


# ==========================================================
# 4. Calculate credible intervals
# ==========================================================

def calculate_CI(solution, confidence=0.95):

    lower = np.quantile(
        solution,
        (1 - confidence) / 2,
        axis=0
    )

    upper = np.quantile(
        solution,
        (1 + confidence) / 2,
        axis=0
    )

    return lower, upper


# ==========================================================
# 5. Plot SEIR compartment trajectories
# ==========================================================

def plot_mean_SEIR(mean, CI, t, output_path):

    labels = [
        "Susceptible",
        "Exposed",
        "Infectious",
        "Recovered",
        "Cumulative Infections"
    ]

    colors = [
        "blue",
        "orange",
        "red",
        "green",
        "purple"
    ]

    lower, upper = CI

    fig, axes = plt.subplots(
        5, 1,
        figsize=(11, 14),
        sharex=True
    )

    for i, ax in enumerate(axes):

        ax.plot(
            t,
            mean[:, i],
            color=colors[i],
            label=f"Mean {labels[i]}"
        )

        ax.fill_between(
            t,
            lower[:, i],
            upper[:, i],
            color=colors[i],
            alpha=0.2,
            label="95% credible interval"
        )

        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Weeks since start")

    fig.suptitle(
        "Seasonal SEIR Model - Posterior Simulations"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# ==========================================================
# 6. Plot predicted weekly ILI
# ==========================================================

def plot_mean_ILI(mean, CI, ILI_counts, t, output_path):

    lower, upper = CI

    plt.figure(figsize=(11, 6))

    plt.plot(
        t,
        mean,
        label="Mean Predicted ILI",
        color="purple",
        linewidth=2
    )

    plt.fill_between(
        t,
        lower,
        upper,
        color="purple",
        alpha=0.2,
        label="95% credible interval"
    )

    plt.plot(
        t,
        ILI_counts,
        label="Observed Weekly ILI",
        color="black",
        linestyle="--"
    )

    plt.xlabel("Weeks since start")
    plt.ylabel("Weekly ILI")
    plt.title(
        "Seasonal SEIR Model - Weekly ILI Predictions"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# ==========================================================
# 7. Plot predetermined beta
# ==========================================================

def plot_beta(beta_weekly, output_path):

    t = np.arange(1, len(beta_weekly) + 1)

    plt.figure(figsize=(11, 5))

    plt.plot(
        t,
        beta_weekly,
        color="darkred",
        linewidth=2
    )

    plt.xlabel("Weeks since start")
    plt.ylabel("Transmission rate beta(t)")
    plt.title("Predetermined Seasonal Transmission Rate")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# ==========================================================
# 8. Main
# ==========================================================

if __name__ == "__main__":

    base_dir = (
        "C:/Users/wilfr/Documents/"
        "Influenza_forecasting/"
    )

    data_dir = os.path.join(
        base_dir,
        "Data/CDC Data"
    )

    # ------------------------------------------------------
    # Load posterior samples and beta
    # ------------------------------------------------------

    parameter_chains, beta_weekly, IC_chains = (
        extract_regime_params(
            os.path.join(
                data_dir,
                "CDC_SEIR_Sigma(t)_MCMC_chains.csv"
            ),
            os.path.join(
                data_dir,
                "CDC_SEIR_Sigma(t)_timedep_params.csv"
            )
        )
    )

    # ------------------------------------------------------
    # Load CDC ILI observations
    # ------------------------------------------------------

    CDC_ILI_data = pd.read_csv(
        os.path.join(
            data_dir,
            "CDC_Age_Structured_ILI_filtered.csv"
        )
    )

    filtered_data = CDC_ILI_data[
        (
            (CDC_ILI_data["year"] == 2023)
            & (CDC_ILI_data["week"] >= 30)
        )
        |
        (
            (CDC_ILI_data["year"] == 2024)
            & (CDC_ILI_data["week"] <= 30)
        )
    ]

    pivoted_data = (
        filtered_data
        .pivot_table(
            index=["year", "week"],
            columns="age_group",
            values="value"
        )
        .sort_index()
    )

    ILI_reported = pivoted_data[
        "ILITOTAL"
    ].to_numpy(dtype=float)

    ILI_counts = np.rint(
        ILI_reported
    ).astype(int)

    # ------------------------------------------------------
    # Time and population
    # ------------------------------------------------------

    N = 334906305

    T = len(ILI_counts)

    if len(beta_weekly) != T:
        raise ValueError(
            "Number of beta values must equal "
            "number of observed weeks."
        )

    # Weekly endpoints including initial time
    t = np.arange(T + 1, dtype=float)

    # beta_weekly[i] is the value at t = i + 1.
    # Extend the first value back to t = 0.
    beta_times = np.arange(T + 1, dtype=float)

    beta_forcing = np.concatenate((
        [beta_weekly[0]],
        beta_weekly
    ))

    # ------------------------------------------------------
    # Select posterior samples
    # ------------------------------------------------------

    n_available = len(parameter_chains)

    n_simulations = min(100, n_available)

    rng = np.random.default_rng(1089)

    sample_indices = rng.choice(
        n_available,
        size=n_simulations,
        replace=False
    )

    # ------------------------------------------------------
    # Allocate simulation arrays
    # ------------------------------------------------------

    solution = np.zeros(
        (n_simulations, len(t), 5)
    )

    ILI = np.zeros(
        (n_simulations, T)
    )

    # ------------------------------------------------------
    # Simulate posterior samples
    # ------------------------------------------------------

    for r, idx in enumerate(sample_indices):

        sigma, gamma, kappa = parameter_chains[idx]

        S0, E0, I0 = IC_chains[idx]

        R0 = N - S0 - E0 - I0

        if R0 < 0:
            raise ValueError(
                f"Invalid initial conditions at draw {idx}"
            )

        IC = [
            S0,
            E0,
            I0,
            R0,
            0.0
        ]

        parameters = [sigma, gamma]

        solution[r] = solve_SEIR(
            IC,
            N,
            parameters,
            beta_times,
            beta_forcing,
            t
        )

        # Weekly incident infections
        weekly_infections = np.diff(
            solution[r, :, 4]
        )

        # Convert to predicted reported ILI
        ILI[r] = weekly_infections / (kappa)

    # ------------------------------------------------------
    # Calculate posterior summaries
    # ------------------------------------------------------

    mean_solution = np.mean(
        solution,
        axis=0
    )

    CI_solution = calculate_CI(
        solution,
        confidence=0.95
    )

    mean_ILI = np.mean(
        ILI,
        axis=0
    )

    CI_ILI = calculate_CI(
        ILI,
        confidence=0.95
    )

    # ------------------------------------------------------
    # Plot results
    # ------------------------------------------------------

    plot_mean_SEIR(
        mean_solution,
        CI_solution,
        t,
        os.path.join(
            data_dir,
            "CDC_Seasonal_SEIR_mean_CI_plot.png"
        )
    )

    plot_mean_ILI(
        mean_ILI,
        CI_ILI,
        ILI_counts,
        t[1:],
        os.path.join(
            data_dir,
            "CDC_Seasonal_ILI_mean_CI_plot.png"
        )
    )

    plot_beta(
        beta_weekly,
        os.path.join(
            data_dir,
            "CDC_Seasonal_beta_plot.png"
        )
    )

    print("Simulations completed:", n_simulations)
    print("Mean predicted ILI:", mean_ILI.mean())
