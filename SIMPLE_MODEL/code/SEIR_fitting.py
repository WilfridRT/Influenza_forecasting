import os
import cmdstanpy as stan
import numpy as np
import pandas as pd

fitting_code = """
functions {
    vector SEIR(real t, vector y, real beta, real sigma, real gamma) {
       
        vector[4] dydt;
        
        real S = y[1];
        real E = y[2];
        real I = y[3];

        dydt[1] = -beta * S * I;
        dydt[2] = beta * S * I - sigma * E;
        dydt[3] = sigma * E - gamma * I;
        dydt[4] = gamma * I;

        return dydt; 
    }
}
data {
    int<lower=1> T;  // number of time points
    
    // Observation times
    real t0;
    array[T-1] real ts;

    // Observed infectious proportions
    vector[T] I_obs;

    // Initial conditions [S0, E0, I0, R0]
    vector[4] y0;
}
parameters {
    real<lower=0> beta;  // Transmission rate
    real<lower=0> sigma;  // Rate of progression from exposed to infected
    real<lower=0> gamma;  // Recovery rate
    real<lower=0> delta; // Observation noise
}
transformed parameters {
    // Solve the SEIR differential equations
    array[T] vector[4] y_hat =
        ode_rk45(SEIR, y0, t0, ts,
                 beta, sigma, gamma);

    // Extract predicted infectious proportions
    vector[T] I_pred;

    for (i in 1:T) {
        I_pred[i] = y_hat[i][3];
    }
}
model {
    // Priors - illustrative values for daily rates
    beta ~ normal(0.5, 0.5);
    sigma ~ normal(0.2, 0.2);
    gamma ~ normal(0.1, 0.1);

    // Prior for observation noise
    delta ~ normal(0, 0.05);

    // Likelihood
    I_obs ~ normal(I_pred, delta);
}
"""

dir = "C:/Users/wilfr/Documents/Influenza_forecasting/Synthetic_model/"

filepath = os.path.join(dir, "Influenza_forecasting/Synthetic_model/SEIR_noisy_data.csv")
stan_path = os.path.join(dir, "Influenza_forecasting/Synthetic_model/SEIR_fitting.stan")

solution = pd.read_csv(filepath)

times = solution["Time"].to_numpy(dtype=float)
I_obs = solution["Infected"].to_numpy(dtype=float)

y0 = [0.99, 0.01, 0.0, 0.0]

stan_data = {
    "T": len(times),
    "t0": times[0],
    "ts": times[1:].tolist(),
    "I_obs": I_obs.tolist(),
    "y0": y0
}

print("Stan path:", repr(stan_path))
print("Filename:", repr(os.path.basename(stan_path)))
print("Extension:", repr(os.path.splitext(stan_path)[1]))
print("File exists:", os.path.exists(stan_path))
print("Is a file:", os.path.isfile(stan_path))

model = stan.CmdStanModel(
    stan_file=stan_path
)

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=1089
)

chains = fit.draws_pd()
summary = fit.summary()

relevant_chains = chains[["chain__", "iter__", "beta", "sigma", "gamma", "delta"]]
relevant_summary = summary.loc[["beta", "sigma", "gamma", "delta"]]

relevant_chains.to_csv(
    os.path.join(dir, "SEIR_MCMC_chains.csv"), 
    index=True
)
relevant_summary.to_csv(
    os.path.join(dir, "SEIR_MCMC_summary.csv"),
    index=True
)

print(fit)
print(fit.summary())