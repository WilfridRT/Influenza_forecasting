import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from cmdstanpy import CmdStanModel
from epiweeks import Week

fitting_code = """

data {
  int<lower=1> Y;                 // Number of seasons/years
  int<lower=1> W;                 // Total observed weeks
  int<lower=1> D;                 // Total days
  int<lower=1> L;                 // Maximum generation lag (days)
  int<lower=1> K;                 // Fourier harmonics

  array[W] int<lower=0> ili;

  // Indexing of daily observations
  array[D] int<lower=1, upper=Y> day_year;
  array[D] int<lower=1, upper=W> day_week;

  // First daily index for each year
  array[Y] int<lower=1, upper=D> year_start;

  // Annual phase, from 0 to 1
  vector<lower=0, upper=1>[D] phase;

  // Discretised gamma generation-time distribution
  simplex[L] w;
}

parameters {
  // Shared seasonal forcing
  real alpha;
  vector[K] a;
  vector[K] b;

  // Year-specific initial incidence
  vector<lower=0>[Y] seed;

  // Observation overdispersion
  real<lower=0> phi;
}

transformed parameters {
  vector<lower=0>[D] Rt;
  vector<lower=0>[D] incidence;
  vector<lower=0>[W] weekly_mu;

  // Shared seasonal R(t)
  for (t in 1:D) {
    real log_R = alpha;

    for (k in 1:K) {
      log_R +=
        a[k] * cos(2 * pi() * k * phase[t]) +
        b[k] * sin(2 * pi() * k * phase[t]);
    }

    Rt[t] = exp(log_R);
  }

  // Independent daily renewal processes
  for (t in 1:D) {
    int yr = day_year[t];
    real lambda = 0;

    for (s in 1:L) {
      int previous_day = t - s;

      if (previous_day >= year_start[yr]) {
        lambda += w[s] * incidence[previous_day];
      } else {
        // Year-specific pre-observation history
        lambda += w[s] * seed[yr];
      }
    }

    incidence[t] =
      Rt[t] * lambda;
  }

  // Aggregate daily incidence to weekly observations
  weekly_mu = rep_vector(0.0, W);

  for (t in 1:D) {
    weekly_mu[day_week[t]] += incidence[t];
  }
}

model {
  // Shared seasonal parameters
  alpha ~ normal(0, 0.15);

  for (k in 1:K) {
    a[k] ~ normal(0, 0.15 / k);
    b[k] ~ normal(0, 0.15 / k);
  }

  // Year-specific seeds
  seed ~ lognormal(log(7500), 0.5);

  // Observation overdispersion
  phi ~ lognormal(log(30), 0.7);

  // Weekly likelihood
  ili ~ neg_binomial_2(weekly_mu, phi);
}

generated quantities {
  array[W] int ili_rep;

  // A common annual forcing curve
  vector[365] seasonal_R;

  for (d in 1:365) {
    real p = (d - 1.0) / 365.0;
    real log_R = alpha;

    for (k in 1:K) {
      log_R +=
        a[k] * cos(2 * pi() * k * p) +
        b[k] * sin(2 * pi() * k * p);
    }

    seasonal_R[d] = exp(log_R);
  }

  // Posterior predictive observations
  for (j in 1:W) {
    ili_rep[j] =
      neg_binomial_2_rng(weekly_mu[j], phi);
  }
}

"""

def extract_ILI_data(filepath):
    """
    Extracts Influenza-Like Illness (ILI) data from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file containing ILI data.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted ILI data.
    """
    try:
        # Read the CSV file into a DataFrame
        ili_data = pd.read_csv(filepath)
        
        filtered_data = ili_data[['year', 'week', 'age_group', 'value']]
        filtered_data = filtered_data[filtered_data['age_group'] == 'ILITOTAL']
        
        return filtered_data
    
    except Exception as e:
        print(f"An error occurred while extracting ILI data: {e}")
        return None

ili_data = extract_ILI_data('C:/Users/wilfr/Documents/Influenza_forecasting/Data/CDC Data/CDC_Age_Structured_ILI_filtered.csv')
stan_path = 'C:/Users/wilfr/Documents/Influenza_forecasting/Data/seasonal_renewal_multiyear.stan'

filtered_data = (
    ili_data.loc[
        ili_data["age_group"] == "ILITOTAL",
        ["year", "week", "value"]
    ]
    .copy()
)

filtered_data = filtered_data.iloc[:-4].copy()

filtered_data["year"] = filtered_data["year"].astype(int)
filtered_data["week"] = filtered_data["week"].astype(int)
filtered_data["value"] = pd.to_numeric(
    filtered_data["value"]
)


filtered_data["season"] = np.where(
    filtered_data["week"] >= 40,
    filtered_data["year"],
    filtered_data["year"] - 1
)

filtered_data["date"] = [
    Week(int(year), int(week)).startdate()
    for year, week in zip(
        filtered_data["year"],
        filtered_data["week"]
    )
]

filtered_data = filtered_data.sort_values(
    ["season", "date"]
).reset_index(drop=True)

# Generation-time distribution
mean_gen = 3.2
sd_gen = 2.1
L = 10

shape = (mean_gen / sd_gen)**2
scale = sd_gen**2 / mean_gen

edges = np.arange(L + 1)

w = np.diff(
    gamma.cdf(edges, a=shape, scale=scale)
)
w /= w.sum()

day_year = []
day_week = []
year_start = []
daily_dates = []

current_day = 1

for yr, (season, group) in enumerate(
    filtered_data.groupby("season", sort=True), 
    start=1
):
    dates = pd.to_datetime(group["date"])

    assert (
        dates.diff().dropna()
        == pd.Timedelta(days=7)
    ).all(), f"Missing weeks in season {season}"

    year_start.append(current_day)

    for week_idx, start in zip(group.index, dates):
        days = pd.date_range(start, periods=7, freq="D")

        daily_dates.extend(days)
        day_year.extend([yr] * 7)
        day_week.extend([week_idx + 1] * 7)

        current_day += 7

daily_dates = pd.DatetimeIndex(daily_dates)

phase = np.zeros(len(daily_dates))

for yr, (season, group) in enumerate(
    filtered_data.groupby("season", sort=True)
):
    # Find the start of epidemiological week 40
    season_start = pd.Timestamp(
        Week(int(season), 40).startdate()
    )

    mask = np.array(day_year) == yr + 1

    elapsed_days = (
        daily_dates[mask] - season_start
    ).days

    phase[mask] = (
        elapsed_days / 365.2425
    ) % 1.0

stan_data = {
    "Y": filtered_data["season"].nunique(),
    "W": len(filtered_data),
    "D": len(daily_dates),
    "L": L,
    "K": 3,
    "ili": filtered_data["value"].astype(int).tolist(),
    "day_year": day_year,
    "day_week": day_week,
    "year_start": year_start,
    "phase": phase,
    "w": w
}

with open(stan_path, "w", encoding="utf-8") as f:
    f.write(fitting_code)


model = CmdStanModel(
    stan_file=stan_path,
    force_compile=True
)


fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True
)

print(
    fit.summary().loc[
        ["alpha", "phi"]
        if "phi" in fit.stan_variables()
        else ["alpha"]
    ]
)

"""

PLOTTING RESULTS

"""

def generation_time_weights(mean_gen=3.2, sd_gen=2.1, L=10):
    """
    Discretised gamma generation-time distribution.
    Must match the weights used in Stan.
    """

    shape = (mean_gen / sd_gen)**2
    scale = sd_gen**2 / mean_gen

    edges = np.arange(L + 1)

    w = np.diff(
        gamma.cdf(edges, a=shape, scale=scale)
    )

    return w / w.sum()


# ============================================================
# 2. SEASONAL REPRODUCTION NUMBER
# ============================================================

def seasonal_Rt(phase, alpha, a, b):
    """
    Calculate shared seasonal R(t) using Fourier coefficients.

    phase: array of annual phases between 0 and 1
    """

    log_R = np.full(
        len(phase),
        alpha,
        dtype=float
    )

    for k in range(1, len(a) + 1):
        log_R += (
            a[k-1] * np.cos(2 * np.pi * k * phase)
            +
            b[k-1] * np.sin(2 * np.pi * k * phase)
        )

    return np.exp(log_R)


# ============================================================
# 3. DAILY RENEWAL EQUATION
# ============================================================

def solve_renewal(Rt, seed, w):
    """
    Solve the deterministic daily renewal equation:

    I(t) = R(t) * sum_s w(s) I(t-s)

    For times before the observation period:
    I(t) = seed
    """

    D = len(Rt)
    L = len(w)

    incidence = np.zeros(D)

    for t in range(D):

        infectiousness = 0.0

        for s in range(1, L + 1):

            previous_day = t - s

            if previous_day >= 0:
                previous_incidence = incidence[previous_day]
            else:
                previous_incidence = seed

            infectiousness += (
                w[s-1] * previous_incidence
            )

        incidence[t] = (
            Rt[t] * infectiousness
        )

    return incidence


# ============================================================
# 4. SIMULATE POSTERIOR TRAJECTORIES
# ============================================================

def simulate_posterior(
    fit,
    stan_data,
    n_samples=200,
    random_seed=42
):
    """
    Simulate daily renewal trajectories from posterior draws.

    Returns:
        weekly_predictions: (samples, weeks)
        daily_predictions:  (samples, days)
        Rt_predictions:     (samples, days)
    """

    rng = np.random.default_rng(random_seed)

    # Extract posterior draws
    alpha_samples = fit.stan_variable("alpha")
    a_samples = fit.stan_variable("a")
    b_samples = fit.stan_variable("b")

    seed_samples = fit.stan_variable("seed")

    n_available = len(alpha_samples)

    sample_indices = rng.choice(
        n_available,
        size=min(n_samples, n_available),
        replace=False
    )

    D = stan_data["D"]
    W = stan_data["W"]
    Y = stan_data["Y"]

    phase = np.asarray(stan_data["phase"])
    day_year = np.asarray(stan_data["day_year"])
    day_week = np.asarray(stan_data["day_week"])

    w = np.asarray(stan_data["w"])

    weekly_predictions = np.zeros(
        (len(sample_indices), W)
    )

    daily_predictions = np.zeros(
        (len(sample_indices), D)
    )

    Rt_predictions = np.zeros(
        (len(sample_indices), D)
    )

    for i, draw in enumerate(sample_indices):

        alpha = alpha_samples[draw]
        a = a_samples[draw]
        b = b_samples[draw]

        seeds = seed_samples[draw]

        # Shared seasonal forcing
        Rt = seasonal_Rt(
            phase, alpha, a, b
        )

        Rt_predictions[i] = Rt

        # Solve each season independently
        for yr in range(1, Y + 1):

            mask = day_year == yr

            Rt_year = Rt[mask]

            incidence_year = solve_renewal(
                Rt_year,
                seeds[yr-1],
                w
            )

            daily_predictions[i, mask] = incidence_year

        # Aggregate daily incidence into weeks
        weekly_predictions[i] = np.bincount(
            day_week - 1,
            weights=daily_predictions[i],
            minlength=W
        )

    return (
        weekly_predictions,
        daily_predictions,
        Rt_predictions
    )


# ============================================================
# 5. POSTERIOR SUMMARY
# ============================================================

def calculate_CI(solution, confidence=0.95):
    """
    Return upper and lower credible intervals.
    """

    upper = np.quantile(
        solution,
        0.5 + confidence / 2,
        axis=0
    )

    lower = np.quantile(
        solution,
        0.5 - confidence / 2,
        axis=0
    )

    return upper, lower


# ============================================================
# 6. PLOT WEEKLY ILI TRAJECTORIES
# ============================================================

def plot_renewal_ILI(
    weekly_predictions,
    ILI_observed,
    dates,
    save_path=None
):
    """
    Plot posterior renewal trajectories and observed ILI.
    """

    mean_ILI = np.mean(
        weekly_predictions,
        axis=0
    )

    upper, lower = calculate_CI(
        weekly_predictions,
        confidence=0.95
    )

    plt.figure(figsize=(13, 6))

    # Individual posterior trajectories
    for trajectory in weekly_predictions[:30]:

        plt.plot(
            dates,
            trajectory,
            color="purple",
            alpha=0.08
        )

    # Posterior mean
    plt.plot(
        dates,
        mean_ILI,
        color="purple",
        linewidth=2,
        label="Mean renewal prediction"
    )

    # 95% credible interval
    plt.fill_between(
        dates,
        lower,
        upper,
        color="purple",
        alpha=0.2,
        label="95% credible interval"
    )

    # Observed data
    plt.plot(
        dates,
        ILI_observed,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Observed weekly ILI"
    )

    plt.xlabel("Date")
    plt.ylabel("Weekly ILI")
    plt.title("Daily Renewal Model - Weekly ILI Fit")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()


# ============================================================
# 7. PLOT SEASONAL R(t)
# ============================================================

def plot_seasonal_Rt(
    Rt_predictions,
    phase,
    save_path=None
):
    """
    Plot shared seasonal reproduction number across one year.
    """

    # Evaluate each posterior trajectory on a common phase grid
    # rather than directly plotting concatenated seasons.

    phase_grid = np.linspace(0, 1, 365, endpoint=False)

    # For this function, Rt_predictions should already be
    # evaluated on phase_grid.

    mean_R = np.mean(Rt_predictions, axis=0)

    upper, lower = calculate_CI(
        Rt_predictions,
        confidence=0.95
    )

    days = np.arange(1, len(mean_R) + 1)

    plt.figure(figsize=(11, 5))

    plt.plot(
        days,
        mean_R,
        color="blue",
        linewidth=2,
        label="Mean seasonal R(t)"
    )

    plt.fill_between(
        days,
        lower,
        upper,
        color="blue",
        alpha=0.2,
        label="95% credible interval"
    )

    plt.axhline(
        1,
        color="black",
        linestyle="--",
        label="R(t) = 1"
    )

    plt.xlabel("Day of season")
    plt.ylabel("Reproduction number")
    plt.title("Shared Seasonal Reproduction Number")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()


# ============================================================
# 8. RUN USING COMPLETED STAN FIT
# ============================================================

# Assumes:
# - fit is your completed CmdStanMCMC object
# - stan_data is the dictionary used during fitting
# - filtered_data is the same ordered dataframe used in Stan

weekly_predictions, daily_predictions, Rt_predictions = (
    simulate_posterior(
        fit,
        stan_data,
        n_samples=200
    )
)

# If your likelihood uses weekly totals:
# weekly_predictions is already on the observation scale.

# If you fitted weekly ILI percentages/rates using
# log(weekly_mu / 7), divide predictions by 7 instead:
# weekly_predictions = weekly_predictions / 7

ILI_observed = filtered_data["value"].to_numpy(
    dtype=float
)

dates = pd.to_datetime(
    filtered_data["date"]
)

output_dir = (
    "C:/Users/wilfr/Documents/"
    "Influenza_forecasting/Data/CDC Data/"
)

plot_renewal_ILI(
    weekly_predictions,
    ILI_observed,
    dates,
    save_path=os.path.join(
        output_dir,
        "CDC_Renewal_ILI_mean_CI_plot.png"
    )
)

# Use the seasonal_R generated quantity from Stan.
# This is evaluated on a common 365-day phase grid.
seasonal_R_samples = fit.stan_variable("seasonal_R")

plot_seasonal_Rt(
    seasonal_R_samples,
    phase=None,
    save_path=os.path.join(
        output_dir,
        "CDC_Renewal_seasonal_Rt_plot.png"
    )
)

chains = fit.draws_pd()
summary = fit.summary()

K = stan_data["K"]
Y = stan_data["Y"]

# Shared seasonal forcing
parameters = ["alpha"]

parameters += [
    f"a[{k}]" for k in range(1, K + 1)
]

parameters += [
    f"b[{k}]" for k in range(1, K + 1)
]

# Year-specific initial conditions
parameters += [
    f"seed[{y}]" for y in range(1, Y + 1)
]

# Observation dispersion
parameters += ["phi"]

relevant_summary = summary.loc[parameters]
relevant_chains = chains[["chain__", "iter__"] + parameters]

relevant_chains.to_csv(
    os.path.join(output_dir, "CDC_Renewal_MCMC_chains.csv"), 
    index=True
)
relevant_summary.to_csv(
    os.path.join(output_dir, "CDC_Renewal_MCMC_summary.csv"),
    index=True
)