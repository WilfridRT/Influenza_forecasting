import os
import cmdstanpy as stan
import numpy as np
import pandas as pd

fitting_code = """
functions {

  /// Computes the seasonal reproduction number using a Fourier series expansion.
  real seasonal_R(
      real t,
      real phase0,
      real alpha,
      vector a,
      vector b
  ) {

    int K = num_elements(a);

    real phase = phase0 + 7.0 * t / 365.2425;
    real log_R = alpha;

    for (k in 1:K) {
      log_R +=
          a[k] * cos(2 * pi() * k * phase) +
          b[k] * sin(2 * pi() * k * phase);
    }

    return exp(log_R);
  }

  /// Computes the SEIR model dynamics.
  vector SEIR(
      real t,
      vector y,
      real N,
      real sigma,
      real gamma,
      real c,
      real phase0,
      real alpha,
      vector a,
      vector b
  ) {

    vector[3] dydt;

    real S = y[1];
    real E = y[2];
    real I = y[3];

    /// Computes the seasonal reproduction number.
    real Rt_season =
        seasonal_R(
            t,
            phase0,
            alpha,
            a,
            b
        );

    real beta_t =
        c * gamma * Rt_season;

    real new_infections =
        beta_t * S * I / N;

    dydt[1] = -new_infections;
    dydt[2] = new_infections - sigma * E;
    dydt[3] = sigma * E - gamma * I;

    return dydt;
  }
}

data {

  int<lower=1> T;

  real<lower=0> N;

  real t0;

  array[T] real ts;

  array[T] int<lower=0> ILI_reported;

  int<lower=1> K;

  real seasonal_alpha;

  vector[K] seasonal_a;

  vector[K] seasonal_b;

  real<lower=0,upper=1> phase0;
}

parameters {

  real<lower=0> c;

  real<lower=0> sigma;

  real<lower=0> gamma;

  real<lower=0> kappa;

  real<lower=0,upper=1> s0_frac;

  real log_I0;

  real log_EI_ratio;

  real<lower=0> phi;
}

transformed parameters {

  real I0 = exp(log_I0);

  real E_I_ratio = exp(log_EI_ratio);

  real S0 = N * s0_frac;

  real E0 = I0 * E_I_ratio;

  real R0 = N - S0 - E0 - I0;

  vector[3] y0;

  array[T] vector[3] y_solution;

  vector[T] infection_incidence;

  vector[T] ILI_pred;

  y0[1] = S0;
  y0[2] = E0;
  y0[3] = I0;

  y_solution =
      ode_rk45(
          SEIR,
          y0,
          t0,
          ts,
          N,
          sigma,
          gamma,
          c,
          phase0,
          seasonal_alpha,
          seasonal_a,
          seasonal_b
      );

  infection_incidence[1] =
      S0 - y_solution[1][1];

  for (i in 2:T) {

    infection_incidence[i] =
        y_solution[i - 1][1]
        - y_solution[i][1];
  }

  for (i in 1:T) {

    ILI_pred[i] =
        infection_incidence[i] / kappa
        + 1e-8;
  }
}

model {

  c ~ lognormal(0, 0.3);

  sigma ~ lognormal(
      log(7.0 / 1.5),
      0.15
  );

  gamma ~ lognormal(
      log(7.0 / 3.0),
      0.15
  );

  kappa ~ lognormal(
      log(10),
      0.6
  );

  s0_frac ~ beta(
      14,
      6
  );

  log_I0 ~ normal(
      log(100000),
      1.0
  );

  log_EI_ratio ~ normal(
      0,
      0.5
  );

  phi ~ lognormal(
      log(50),
      1.0
  );

  if (R0 < 0)
    target += negative_infinity();

  ILI_reported ~ neg_binomial_2(
      ILI_pred,
      phi
  );
}

generated quantities {

  array[T] int ILI_rep;

  vector[T] beta_weekly;

  vector[T] R_seasonal;

  vector[T] R_effective;

  for (i in 1:T) {

    R_seasonal[i] =
        seasonal_R(
            ts[i],
            phase0,
            seasonal_alpha,
            seasonal_a,
            seasonal_b
        );

    beta_weekly[i] =
        c * gamma * R_seasonal[i];

    R_effective[i] =
        c *
        R_seasonal[i] *
        y_solution[i][1] /
        N;

    ILI_rep[i] =
        neg_binomial_2_rng(
            ILI_pred[i],
            phi
        );
  }
}
"""


import os
import numpy as np
import pandas as pd
import cmdstanpy as stan
from epiweeks import Week

base_dir = (
    "C:/Users/wilfr/Documents/"
    "Influenza_forecasting/"
)

# ------------------------------------------
# 1. Load seasonal forcing coefficients
# ------------------------------------------

seasonal_filepath = os.path.join(
    base_dir,
    "Data/CDC Data/CDC_Renewal_MCMC_chains.csv"
)

seasonal_chains = pd.read_csv(seasonal_filepath)

# Detect number of Fourier harmonics
K = sum(
    col.startswith("a[")
    for col in seasonal_chains.columns
)

# Posterior mean coefficients
seasonal_alpha = seasonal_chains["alpha"].mean()

seasonal_a = np.array([
    seasonal_chains[f"a[{k}]"].mean()
    for k in range(1, K + 1)
])

seasonal_b = np.array([
    seasonal_chains[f"b[{k}]"].mean()
    for k in range(1, K + 1)
])

# ------------------------------------------
# 2. Load ILI data
# ------------------------------------------

filepath = os.path.join(
    base_dir,
    "Data/CDC Data/CDC_Age_Structured_ILI_filtered.csv"
)

data = pd.read_csv(filepath)

filtered_data = data[
    ((data["year"] == 2023) &
     (data["week"] >= 30))
    |
    ((data["year"] == 2024) &
     (data["week"] <= 30))
].copy()

pivoted_data = (
    filtered_data
    .pivot_table(
        index=["year", "week"],
        columns="age_group",
        values="value"
    )
    .sort_index()
)

ILI_reported = pivoted_data["ILITOTAL"].to_numpy(
    dtype=float
)

ILI_counts = np.rint(ILI_reported).astype(int)

# ------------------------------------------
# 3. Time indexing
# ------------------------------------------

N = 334906305
T = len(ILI_counts)

t0 = 0.0
ts = np.arange(1, T + 1, dtype=float)

# First observation week
first_year, first_week = pivoted_data.index[0]

first_date = pd.Timestamp(
    Week(int(first_year), int(first_week)).startdate()
)

# The ODE starts at the beginning of the first
# observation week, not at its endpoint.

# Week 40 start preceding the first observation
season_year = (
    int(first_year)
    if int(first_week) >= 40
    else int(first_year) - 1
)

season_start = pd.Timestamp(
    Week(season_year, 40).startdate()
)

elapsed_days = (first_date - season_start).days

phase0 = (elapsed_days / 365.2425) % 1.0

# ------------------------------------------
# 4. Stan data
# ------------------------------------------

stan_data = {
    "T": T,
    "N": float(N),
    "t0": t0,
    "ts": ts.tolist(),
    "ILI_reported": ILI_counts.tolist(),
    "K": K,
    "seasonal_alpha": float(seasonal_alpha),
    "seasonal_a": seasonal_a.tolist(),
    "seasonal_b": seasonal_b.tolist(),
    "phase0": float(phase0)
}

# ------------------------------------------
# 5. Compile and sample
# ------------------------------------------

stan_path = os.path.join(
    base_dir,
    "Data/CDC_SEIR_Sigma_t_fitting.stan"
)

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
    adapt_delta=0.90,
    max_treedepth=10
)

print(fit.diagnose())

chains = fit.draws_pd()
summary = fit.summary()

parameters = [
    "c",
    "sigma",
    "gamma",
    "kappa",
    "s0_frac",
    "I0",
    "E_I_ratio",
    "S0",
    "E0",
    "phi"
]

relevant_summary = summary.loc[parameters]
relevant_chains = chains[["chain__", "iter__"] + parameters]

relevant_chains.to_csv(
    os.path.join(base_dir, "Data/CDC Data/CDC_SEIR_Sigma(t)_MCMC_chains.csv"), 
    index=True
)
relevant_summary.to_csv(
    os.path.join(base_dir, "Data/CDC Data/CDC_SEIR_Sigma(t)_MCMC_summary.csv"),
    index=True
)


# Extract posterior samples
beta_samples = fit.stan_variable("beta_weekly")
R_eff_samples = fit.stan_variable("R_effective")
R_season_samples = fit.stan_variable("R_seasonal")

# Baseline reproduction number before susceptible depletion
c_samples = fit.stan_variable("c")

R_baseline_samples = (
    c_samples[:, None] * R_season_samples
)

def summarise_posterior(samples):
    return {
        "mean": np.mean(samples, axis=0),
        "lower_95": np.quantile(samples, 0.025, axis=0),
        "upper_95": np.quantile(samples, 0.975, axis=0)
    }

beta_summary = summarise_posterior(beta_samples)
R_eff_summary = summarise_posterior(R_eff_samples)
R_base_summary = summarise_posterior(R_baseline_samples)

# Save time-dependent results
time_results = pd.DataFrame({
    "year": pivoted_data.index.get_level_values("year"),
    "week": pivoted_data.index.get_level_values("week"),

    "beta_mean": beta_summary["mean"],
    "beta_lower_95": beta_summary["lower_95"],
    "beta_upper_95": beta_summary["upper_95"],

    "R_baseline_mean": R_base_summary["mean"],
    "R_baseline_lower_95": R_base_summary["lower_95"],
    "R_baseline_upper_95": R_base_summary["upper_95"],

    "R_eff_mean": R_eff_summary["mean"],
    "R_eff_lower_95": R_eff_summary["lower_95"],
    "R_eff_upper_95": R_eff_summary["upper_95"]
})

time_results.to_csv(
    os.path.join(
        base_dir,
        "Data/CDC Data/CDC_SEIR_Sigma(t)_timedep_params.csv"
    ),
    index=False
)
