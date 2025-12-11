
// src/advanced.rs

/// Local helper: mean and std for small slices.
fn mean_std_local(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    let denom = (n - 1.0).max(1.0);
    let var = data
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / denom;

    (mean, var.sqrt())
}

/* =======================
   KALMAN DRIFT FILTER
   ======================= */

#[derive(Debug, Copy, Clone)]
pub struct KalmanDriftState {
    pub mu: f64,
    pub var: f64,
}

/// Smooth drift from returns using a 1D Kalman filter.
///
/// Model:
///   mu_t = mu_{t-1} + q
///   r_t  = mu_t + eps_t
///
/// Where q is tuned from realized variance so it adapts to the regime:
/// - In calm regimes, drift moves slowly.
/// - In choppy regimes, drift is allowed to move more.
pub fn kalman_smooth_drift(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![];
    }

    let (m, s) = mean_std_local(returns);
    // Observation variance (noise level)
    let obs_var = (s * s).max(1e-8);

    // Process noise: small fraction of obs_var, but not zero.
    // Higher vol => we let drift move a bit more.
    let q = (0.02 * obs_var).max(1e-8);

    let mut mu = m;
    let mut var_post = obs_var;
    let mut out = Vec::with_capacity(returns.len());

    for &r in returns {
        // Predict
        let mu_prior = mu;
        let var_prior = var_post + q;

        // Update
        let k = var_prior / (var_prior + obs_var);
        mu = mu_prior + k * (r - mu_prior);
        var_post = (1.0 - k) * var_prior;

        out.push(mu);
    }

    out
}

/* =======================
   AUTOCORR OF SQUARED RETURNS
   ======================= */

/// Estimate max autocorrelation of squared returns up to `max_lag`.
/// Used to detect vol clustering (GARCH-like behavior).
pub fn estimate_autocorr_squared(returns: &[f64], max_lag: usize) -> f64 {
    if returns.len() < 4 || max_lag == 0 {
        return 0.0;
    }
    let n = returns.len();
    let xs: Vec<f64> = returns.iter().map(|r| r * r).collect();
    let (mean_x, std_x) = mean_std_local(&xs);
    let var_x = (std_x * std_x).max(1e-12);

    let mut max_rho = 0.0;

    for lag in 1..=max_lag {
        if lag >= n {
            break;
        }
        let mut num = 0.0;
        let denom = (n - lag) as f64 * var_x;

        for t in lag..n {
            let x_t = xs[t] - mean_x;
            let x_l = xs[t - lag] - mean_x;
            num += x_t * x_l;
        }

        if denom > 0.0 {
            let rho = num / denom;
            if rho.abs() > max_rho {
                max_rho = rho.abs();
            }
        }
    }

    max_rho.clamp(0.0, 1.0)
}

/* =======================
   GARCH(1,1) MLE (GRID)
   ======================= */

#[derive(Debug, Copy, Clone)]
pub struct GarchParams {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

/// Fit a simple GARCH(1,1) by crude grid-search MLE.
/// Returns (params, sigma_t series).
///
/// This is intentionally simple and robust:
/// - Coarse grid over (omega, alpha, beta)
/// - Enforces alpha + beta < 0.999 for stationarity-ish behavior
/// - Falls back to flat vol if returns are too short or too degenerate
pub fn fit_garch_mle_and_series(returns: &[f64]) -> (GarchParams, Vec<f64>) {
    if returns.len() < 30 {
        // Fallback: flat vol
        let (_, s) = mean_std_local(returns);
        let sigma = s.max(1e-4);
        return (
            GarchParams {
                omega: 1e-8,
                alpha: 0.05,
                beta: 0.9,
            },
            vec![sigma; returns.len()],
        );
    }

    let (mean_r, s) = mean_std_local(returns);
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_params = GarchParams {
        omega: 1e-8,
        alpha: 0.05,
        beta: 0.9,
    };

    // Coarse but reasonable grid.
    let omega_grid = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6];
    let alpha_grid = [0.02, 0.05, 0.1, 0.15, 0.2];
    let beta_grid = [0.5, 0.7, 0.8, 0.9, 0.95];

    // Initial variance estimate from returns.
    let mut init_var = s * s;
    if init_var <= 0.0 || !init_var.is_finite() {
        init_var = 1e-6;
    }

    for &omega in &omega_grid {
        for &alpha in &alpha_grid {
            for &beta in &beta_grid {
                if alpha + beta >= 0.999 {
                    continue;
                }

                let mut sigma2 = init_var.max(1e-8);
                let mut ll = 0.0;

                for &r in returns {
                    let dev = r - mean_r;
                    sigma2 = omega + alpha * dev * dev + beta * sigma2;
                    let s2 = sigma2.max(1e-8);
                    // Gaussian log-likelihood (ignoring constants)
                    ll += -0.5 * (s2.ln() + dev * dev / s2);
                }

                if ll > best_ll {
                    best_ll = ll;
                    best_params = GarchParams { omega, alpha, beta };
                }
            }
        }
    }

    // Build final sigma series with best params.
    let mut sigma2 = init_var.max(1e-8);
    let mut series = Vec::with_capacity(returns.len());

    for &r in returns {
        let dev = r - mean_r;
        sigma2 = best_params.omega
            + best_params.alpha * dev * dev
            + best_params.beta * sigma2;
        let sig = sigma2.sqrt().max(1e-4);
        series.push(sig);
    }

    (best_params, series)
}

/* =======================
   HMM-LIKE VOL STATE
   ======================= */

#[derive(Debug, Copy, Clone)]
pub enum VolState {
    Low,
    Medium,
    High,
}

/// Crude "HMM-lite": quantize |returns| into 3 states using quantiles.
/// This approximates a 3-state Markov vol regime.
///
/// - Uses |r| magnitudes
/// - 33%/66% quantiles for Low/Med/High split
pub fn hmm_infer_state(returns: &[f64]) -> VolState {
    let n = returns.len();
    if n < 20 {
        return VolState::Medium;
    }

    let mut mags: Vec<f64> = returns
        .iter()
        .map(|r| r.abs())
        .filter(|x| x.is_finite())
        .collect();

    if mags.len() < 10 {
        return VolState::Medium;
    }

    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_mag = mags.len();
    let idx33 = ((0.33 * n_mag as f64).floor() as usize).min(n_mag - 1);
    let idx66 = ((0.66 * n_mag as f64).floor() as usize).min(n_mag - 1);

    let q33 = mags[idx33];
    let q66 = mags[idx66];

    let last = returns[n - 1].abs();

    if last <= q33 {
        VolState::Low
    } else if last <= q66 {
        VolState::Medium
    } else {
        VolState::High
    }
}

/* =======================
   SIMPLE RNN-LIKE VOL MODEL
   ======================= */

/// Tiny 1D RNN-ish volatility model.
///
/// This is not a full LSTM, but behaves like a nonlinear smoother
/// over recent returns:
///
/// h_t = tanh(w_hh * h_{t-1} + w_xh * r_t)
/// vol ≈ |w_hy * h_T + b_y|
pub fn predict_vol_rnn(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 10 {
        return 0.01;
    }

    let (_m, s) = mean_std_local(returns);
    let s = s.max(1e-5);

    // Use the last ~40 bars (or fewer if shorter).
    let window = 40.min(n);
    let start = n - window;

    // Weights chosen to be scale-aware and stable.
    let w_hh = 0.6;
    let w_xh = 0.5 / s; // scale-invariant transformation of returns
    let w_hy = 0.8 * s;
    let b_h = 0.0;
    let b_y = 0.0;

    let mut h = 0.0;
    for &r in &returns[start..] {
        let x = w_hh * h + w_xh * r + b_h;
        h = x.tanh();
    }

    let raw = w_hy * h + b_y;

    // Clamp to a sane volatility band.
    raw.abs().clamp(0.001, 0.2)
}
