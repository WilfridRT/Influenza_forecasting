import os
import cmdstanpy as stan
import numpy as np
import pandas as pd

fitting_code = """
functions {
    vector SEIR(real t, vector y, real N, real beta, real sigma, real gamma) {
       
        vector[5] dydt;
        
        real S = y[1];
        real E = y[2];
        real I = y[3];

        dydt[1] = -beta * S * I / N;
        dydt[2] = beta * S * I / N - sigma * E;
        dydt[3] = sigma * E - gamma * I;
        dydt[4] = gamma * I;
        dydt[5] = beta * S * I / N;  // New infections for ILI calculation

        return dydt; 
    }
}
data {
    int<lower=1> T;  // number of time points

    int<lower=0> N;  // total population size
    
    // Observation times
    real t0;
    array[T] real ts;

    // Observed infectious proportions
    array[T] int<lower=0> ILI_reported;

}
parameters {
    real<lower=0> Reff0; 
    real<lower=0> sigma; 
    real<lower=0> gamma; 
    real<lower=0> kappa; 

    // Initial susceptible fraction
    real<lower=0, upper=1> s0_frac;

    // Initial infectious population
    real<lower=0> I0;

    // Initial exposed/infectious ratio
    real<lower=0> E_I_ratio;

    // Negative binomial dispersion
    real<lower=0> phi;
}
transformed parameters {

    real S0 = N * s0_frac;

    real E0 = I0 * E_I_ratio;

    real R0 = N - S0 - E0 - I0;

    // Initial effective reproduction number:
    // Reff0 = beta * S0 / (gamma * N)
    real beta = Reff0 * gamma * N / S0;

    vector[5] y0;

    array[T] vector[5] y_solution;
    
    vector[T] infection_incidence;
    vector[T] ILI_pred;

    // Initial conditions
    y0[1] = S0;
    y0[2] = E0;
    y0[3] = I0;
    y0[4] = R0;
    y0[5] = 0;

    // Integrate to each weekly endpoint
    y_solution = ode_rk45(SEIR, y0, t0, ts, N, beta, sigma, gamma);

    // First week: cumulative infections since t0
    infection_incidence[1] = y_solution[1][5];

    // Subsequent weeks
    for (i in 2:T) {
        infection_incidence[i] =
            y_solution[i][5] -
            y_solution[i-1][5];
    }

    // Convert infections to expected observed ILI
    for (i in 1:T) {
        ILI_pred[i] =
            fmax(infection_incidence[i] / kappa, 1e-8);
    }
}
model {
    // Initial effective reproduction number
    Reff0 ~ lognormal(log(1.2), 0.20);

    // Latent period approximately 1.5 days
    sigma ~ lognormal(log(7.0 / 1.5), 0.15);

    // Infectious period approximately 3 days
    gamma ~ lognormal(log(7.0 / 3.0), 0.15);

    // Observation-to-incidence conversion
    // Median = 10, with substantial uncertainty
    kappa ~ lognormal(log(10), 0.6);

    // Initial susceptible fraction centred at 0.70
    s0_frac ~ beta(14, 6);

    // Initial infectious population
    I0 ~ lognormal(log(100000), 1.0);

    // Initial exposed/infectious ratio
    E_I_ratio ~ lognormal(log(1), 0.5);

    // Observation dispersion
    phi ~ lognormal(log(50), 1.0);

    // Valid initial compartment populations
    target += (R0 >= 0) ? 0 : negative_infinity();

    ILI_reported ~ neg_binomial_2(
        ILI_pred,
        phi
    );
}
generated quantities {

    array[T] int ILI_rep;

    for (i in 1:T) {
        ILI_rep[i] = neg_binomial_2_rng(
            ILI_pred[i],
            phi
        );
    }
}
"""

dir = "C:/Users/wilfr/Documents/Influenza_forecasting/"

filepath = os.path.join(dir, "Data/CDC Data/CDC_Age_Structured_ILI_filtered.csv")
stan_path = os.path.join(dir, "Synthetic_model/SEIR_fitting.stan")

data = pd.read_csv(filepath)
N = 334906305

filtered_data = data[((data['year'] == 2023) & (data['week'] >= 30) | 
                      (data['year'] == 2024) & (data['week'] <= 30))]
pivoted_data = (
    filtered_data
    .pivot_table(index=['year', 'week'], columns='age_group', values='value')
    .sort_index()
)

ILI_reported = pivoted_data["ILITOTAL"].to_numpy(dtype=float)
ILI_counts = np.rint(ILI_reported).astype(int)

N = 334906305
T = len(ILI_counts)

t0 = 0.0
ts = np.arange(1, T + 1, dtype=float)

stan_data = {
    "T": T,
    "N": N,
    "t0": t0,
    "ts": ts.tolist(),
    "ILI_reported": ILI_counts.tolist(),
}

with open(stan_path, "w", encoding="utf-8") as f:
    f.write(fitting_code)

model = stan.CmdStanModel(
    stan_file=stan_path,
    force_compile=True
)

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=1089,
    adapt_delta=0.95,
    max_treedepth=12
)

chains = fit.draws_pd()
summary = fit.summary()

parameters = ["Reff0", "beta", "sigma", "gamma", "kappa", "s0_frac", "I0", "E0", "S0", "R0", "phi"]

relevant_summary = summary.loc[parameters]
relevant_chains = chains[["chain__", "iter__"] + parameters]

relevant_chains.to_csv(
    os.path.join(dir, "Data/CDC Data/CDC_SEIR_MCMC_chains.csv"), 
    index=True
)
relevant_summary.to_csv(
    os.path.join(dir, "Data/CDC Data/CDC_SEIR_MCMC_summary.csv"),
    index=True
)

print(fit)
print(fit.summary())

ILI_predictions = fit.stan_variable("ILI_pred")

pred_mean = np.mean(ILI_predictions, axis=0)

pred_lower = np.quantile(
    ILI_predictions, 0.025, axis=0
)

pred_upper = np.quantile(
    ILI_predictions, 0.975, axis=0
)

results = pd.DataFrame({
    "year": pivoted_data.index.get_level_values("year"),
    "week": pivoted_data.index.get_level_values("week"),
    "Observed_ILI": ILI_reported,
    "Predicted_ILI": pred_mean,
    "Lower_95": pred_lower,
    "Upper_95": pred_upper
})

results.to_csv(
    os.path.join(
        dir,
        "Data/CDC Data/CDC_SEIR_fitted_ILI.csv"
    ),
    index=False
)

print(relevant_summary)

print("\nPosterior mean conversion factor:")
print(np.mean(fit.stan_variable("kappa")))

print("\nPosterior mean transmission rate:")
print(np.mean(fit.stan_variable("beta")))

print("\nPosterior mean initial Reff:")
print(np.mean(fit.stan_variable("Reff0")))