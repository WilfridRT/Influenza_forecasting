
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
