

use crate::features::mean_std;
use crate::advanced::{hmm_infer_state, VolState}; // add this
#[derive(Debug, Copy, Clone)]
pub enum MarketRegime {
    TrendingUp,
    TrendingDown,
    SidewaysLowVol,
    SidewaysHighVol,
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn classify_regime_ml(pred_ret: f64, pred_vol: f64, returns: &[f64]) -> MarketRegime {
    use crate::advanced::{hmm_infer_state, VolState};
    use crate::features::mean_std;

    // Realized volatility window
    let look = returns.len().min(80);
    let (_, vol_realized) = mean_std(&returns[returns.len().saturating_sub(look)..]);

    // Clamp directional signal into {-1, 0, 1}
    let pv = pred_vol.max(1e-4);
    let raw_dir = if pred_ret > 1.5 * pv {
        1
    } else if pred_ret < -1.5 * pv {
        -1
    } else {
        0
    };

    // HMM-lite vol state
    let state = hmm_infer_state(returns);

    match (state, raw_dir) {
        // HIGH VOL
        (VolState::High, 1)  => MarketRegime::TrendingUp,
        (VolState::High, -1) => MarketRegime::TrendingDown,
        (VolState::High, 0)  => MarketRegime::SidewaysHighVol,

        // MEDIUM VOL
        (VolState::Medium, 1)  => MarketRegime::TrendingUp,
        (VolState::Medium, -1) => MarketRegime::TrendingDown,
        (VolState::Medium, 0) => {
            if vol_realized > 0.015 {
                MarketRegime::SidewaysHighVol
            } else {
                MarketRegime::SidewaysLowVol
            }
        }

        // LOW VOL (anything else)
        (VolState::Low, _) => {
            if vol_realized > 0.015 {
                MarketRegime::SidewaysHighVol
            } else {
                MarketRegime::SidewaysLowVol
            }
        }

        // FULL FALLBACK — satisfies Rust's exhaustiveness rules
        _ => MarketRegime::SidewaysLowVol,
    }
}

/// Solves Ax = b using Gaussian elimination. Includes singular fallback.
fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = a.len();
    let mut x = vec![0.0; n];

    for i in 0..n {
        // Pivot
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                max_row = k;
            }
        }

        if max_val < 1e-12 {
            // matrix is singular-ish
            return x.clone();
        }

        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        // eliminate
        for k in (i + 1)..n {
            let factor = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // back-substitution
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-12 {
            x[i] = 0.0;
        } else {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
    }

    x
}

/// Ensemble ridge regression (BAGGED)
pub fn fit_linear_regression_ensemble(
    features: &[Vec<f64>],
    targets: &[f64],
    lambda: f64,
    n_models: usize,
    sample_frac: f64,
) -> (Vec<f64>, f64) {
    let n = features.len();
    let d = features[0].len();

    let mut rng = rand::thread_rng();

    let mut beta_sum = vec![0.0; d];
    let mut rss_total = 0.0;

    for _ in 0..n_models {
        let m = ((n as f64 * sample_frac).round() as usize).clamp(d + 1, n);

        let mut xtx = vec![vec![0.0; d]; d];
        let mut xty = vec![0.0; d];

        for _ in 0..m {
            let idx = rand::Rng::gen_range(&mut rng, 0..n);
            let x = &features[idx];
            let y = targets[idx];

            for i in 0..d {
                xty[i] += x[i] * y;
                for j in 0..d {
                    xtx[i][j] += x[i] * x[j];
                }
            }
        }

        // ridge penalty
        for i in 0..d {
            xtx[i][i] += lambda;
        }

        let mut xtx_clone = xtx.clone();
        let mut xty_clone = xty.clone();
        let beta = solve_linear_system(&mut xtx_clone, &mut xty_clone);

        for j in 0..d {
            beta_sum[j] += beta[j];
        }

        // model residual
        let mut rss = 0.0;
        for (x, y) in features.iter().zip(targets.iter()) {
            let y_hat = dot(&beta, x);
            rss += (y - y_hat).powi(2);
        }
        rss_total += rss;
    }

    // average weights
    for j in 0..d {
        beta_sum[j] /= n_models as f64;
    }

    // FIXED: correct residual variance
    let resid_std = (rss_total / (n_models * n) as f64).sqrt();

    (beta_sum, resid_std)
}
