

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

