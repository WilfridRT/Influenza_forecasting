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
    real<lower=1e-6> delta; // Observation noise
}
transformed parameters {

    array[T-1] vector[4] y_hat =
        ode_rk45(SEIR, y0, t0, ts,
                 beta, sigma, gamma);

    vector[T] I_pred;

    // Initial infectious proportion
    I_pred[1] = y0[3];

    // Remaining predictions from the ODE solution
    for (i in 2:T) {
        I_pred[i] = y_hat[i-1][3];
    }
}
model {
    // Priors - illustrative values for daily rates
    beta ~ normal(0.5, 0.5);
    sigma ~ normal(0.2, 0.2);
    gamma ~ normal(0.1, 0.1);

    // Prior for observation noise
    delta ~ normal(0.1, 0.05);

    // Likelihood
    I_obs ~ normal(I_pred, delta);
}