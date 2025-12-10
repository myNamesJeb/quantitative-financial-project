
// src/advanced.rs

use std::f64::consts::E;

/// Simple helper: mean and std
fn mean_std_local(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0).max(1.0);
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
/// Model:
///   mu_t = mu_{t-1} + q
///   r_t  = mu_t + eps_t
pub fn kalman_smooth_drift(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![];
    }

    let (m, s) = mean_std_local(returns);
    let obs_var = (s * s).max(1e-8);
    let q = 0.05 * obs_var; // process noise

    let mut mu = m;
    let mut var_post = obs_var;
    let mut out = Vec::with_capacity(returns.len());

    for &r in returns {
        // predict
        let mu_prior = mu;
        let var_prior = var_post + q;

        // update
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
/// Used to detect vol clustering.
pub fn estimate_autocorr_squared(returns: &[f64], max_lag: usize) -> f64 {
    if returns.len() < 4 {
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

    max_rho
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

    let (mean_r, _s) = mean_std_local(returns);
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_params = GarchParams {
        omega: 1e-8,
        alpha: 0.05,
        beta: 0.9,
    };

    // Very coarse grid – you can refine if you want
    let omega_grid = [1e-8, 1e-7, 1e-6, 5e-6];
    let alpha_grid = [0.02, 0.05, 0.1, 0.15, 0.2];
    let beta_grid = [0.5, 0.7, 0.8, 0.9, 0.95];

    for &omega in &omega_grid {
        for &alpha in &alpha_grid {
            for &beta in &beta_grid {
                if alpha + beta >= 0.999 {
                    continue;
                }

                let mut sigma2 = returns
                    .iter()
                    .map(|r| (r - mean_r).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                sigma2 = sigma2.max(1e-8);

                let mut ll = 0.0;
                for &r in returns {
                    sigma2 = omega + alpha * (r - mean_r).powi(2) + beta * sigma2;
                    let s2 = sigma2.max(1e-8);
                    // Gaussian log-likelihood
                    ll += -0.5 * ((s2.ln()) + (r - mean_r).powi(2) / s2);
                }

                if ll > best_ll {
                    best_ll = ll;
                    best_params = GarchParams { omega, alpha, beta };
                }
            }
        }
    }

    // Build final sigma series
    let mut sigma2 = returns
        .iter()
        .map(|r| (r - mean_r).powi(2))
        .sum::<f64>() / returns.len() as f64;
    sigma2 = sigma2.max(1e-8);

    let mut series = Vec::with_capacity(returns.len());
    for &r in returns {
        sigma2 = best_params.omega
            + best_params.alpha * (r - mean_r).powi(2)
            + best_params.beta * sigma2;
        series.push(sigma2.sqrt().max(1e-4));
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
pub fn hmm_infer_state(returns: &[f64]) -> VolState {
    let n = returns.len();
    if n < 20 {
        return VolState::Medium;
    }

    let mut mags: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx33 = ((0.33 * n as f64).floor() as usize).min(n - 1);
    let idx66 = ((0.66 * n as f64).floor() as usize).min(n - 1);

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
/// This is not a full LSTM, but behaves like a nonlinear smoother.
///
/// h_t = tanh(w_hh h_{t-1} + w_xh r_t)
/// vol ≈ |w_hy h_T + b_y|
pub fn predict_vol_rnn(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 10 {
        return 0.01;
    }

    let (_m, s) = mean_std_local(returns);
    let s = s.max(1e-5);

    let window = 40.min(n);
    let start = n - window;

    let w_hh = 0.6;
    let w_xh = 0.5 / s; // scale-invariant
    let w_hy = 0.8 * s;
    let b_h = 0.0;
    let b_y = 0.0;

    let mut h = 0.0;
    for &r in &returns[start..] {
        let x = w_hh * h + w_xh * r + b_h;
        h = x.tanh();
    }

    let raw = w_hy * h + b_y;
    raw.abs().max(0.001)
}
