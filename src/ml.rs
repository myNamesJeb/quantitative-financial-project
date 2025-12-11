
// src/ml.rs

use rand::Rng;

use crate::features::mean_std;
use crate::advanced::{hmm_infer_state, VolState};

/// Market regime classification used by the router and forecaster.

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MarketRegime {
    TrendingUp,
    TrendingDown,
    SidewaysLowVol,
    SidewaysHighVol,
}

/// Simple dot product helper.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Classify the current regime using:
/// - ML prediction (pred_ret, pred_vol)
/// - Realized volatility of returns
/// - HMM-lite vol state (Low / Medium / High)
///
/// Goals:
/// - High-vol + strong positive dir → TrendingUp
/// - High-vol + strong negative dir → TrendingDown
/// - High-vol + weak dir           → SidewaysHighVol
/// - Low/med vol + weak dir        → SidewaysLowVol or SidewaysHighVol
pub fn classify_regime_ml(
    pred_ret: f64,
    pred_vol: f64,
    returns: &[f64],
) -> MarketRegime {
    if returns.len() < 20 {
        // Not enough info: assume churny low-vol.
        return MarketRegime::SidewaysLowVol;
    }

    // Realized volatility window.
    let look = returns.len().min(80);
    let window = &returns[returns.len().saturating_sub(look)..];
    let (_m, vol_realized) = mean_std(window);

    // Normalize directional signal by predicted vol (avoid unit issues).
    let pv = pred_vol.max(1e-4);
    let dir_z = (pred_ret / pv).clamp(-5.0, 5.0);

    // Quantize into {-1, 0, 1} for simpler logic.
    let raw_dir = if dir_z > 1.5 {
        1
    } else if dir_z < -1.5 {
        -1
    } else {
        0
    };

    // HMM-lite vol state from the advanced module.
    let state = hmm_infer_state(returns);

    // Thresholds for deciding "high" vs "low" realized volatility.
    let high_vol_thresh = 0.015;
    let very_high_vol_thresh = 0.03;

    match (state, raw_dir) {
        // HIGH VOL STATE
        (VolState::High, 1) => {
            // Very high realized vol => strong uptrend.
            if vol_realized > very_high_vol_thresh {
                MarketRegime::TrendingUp
            } else {
                MarketRegime::SidewaysHighVol
            }
        }
        (VolState::High, -1) => {
            if vol_realized > very_high_vol_thresh {
                MarketRegime::TrendingDown
            } else {
                MarketRegime::SidewaysHighVol
            }
        }
        (VolState::High, 0) => MarketRegime::SidewaysHighVol,

        // MEDIUM VOL STATE
        (VolState::Medium, 1) => {
            if vol_realized > high_vol_thresh {
                MarketRegime::TrendingUp
            } else {
                MarketRegime::SidewaysLowVol
            }
        }
        (VolState::Medium, -1) => {
            if vol_realized > high_vol_thresh {
                MarketRegime::TrendingDown
            } else {
                MarketRegime::SidewaysLowVol
            }
        }
        (VolState::Medium, 0) => {
            if vol_realized > high_vol_thresh {
                MarketRegime::SidewaysHighVol
            } else {
                MarketRegime::SidewaysLowVol
            }
        }

        // LOW VOL STATE
        (VolState::Low, 1) => {
            // Low vol but positive dir ⇒ soft uptrend.
            if vol_realized > high_vol_thresh {
                MarketRegime::TrendingUp
            } else {
                MarketRegime::SidewaysLowVol
            }
        }
        (VolState::Low, -1) => {
            if vol_realized > high_vol_thresh {
                MarketRegime::TrendingDown
            } else {
                MarketRegime::SidewaysLowVol
            }
        }
        (VolState::Low, 0) => {
            if vol_realized > high_vol_thresh {
                MarketRegime::SidewaysHighVol
            } else {
                MarketRegime::SidewaysLowVol
            }
        }
        _ => MarketRegime::SidewaysLowVol,
    }
}

/// Solves Ax = b using Gaussian elimination with partial pivoting.
/// Includes a simple singular fallback that returns a zero vector
/// if the system is too ill-conditioned to solve safely.
fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = a.len();
    let mut x = vec![0.0; n];

    if n == 0 {
        return x;
    }

    // Forward elimination
    for i in 0..n {
        // Pivot row selection
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                max_row = k;
            }
        }

        // If pivot is basically zero, treat as singular.
        if max_val < 1e-12 {
            return x.clone();
        }

        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        // Eliminate below pivot
        for k in (i + 1)..n {
            if a[i][i].abs() < 1e-12 {
                continue;
            }
            let factor = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back-substitution
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

/// Ensemble ridge regression (bagged).
///
/// Trains `n_models` separate ridge regressions on random subsamples
/// of the feature set, then averages the weights. Also returns a
/// residual std estimate (used as ML vol).
pub fn fit_linear_regression_ensemble(
    features: &[Vec<f64>],
    targets: &[f64],
    lambda: f64,
    n_models: usize,
    sample_frac: f64,
) -> (Vec<f64>, f64) {
    let n = features.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }

    let d = features[0].len();
    if d == 0 {
        return (vec![0.0; 0], 0.0);
    }

    let mut rng = rand::thread_rng();

    let mut beta_sum = vec![0.0; d];
    let mut rss_total = 0.0;

    let nm = n_models.max(1);

    for _ in 0..nm {
        let m = ((n as f64 * sample_frac).round() as usize).clamp(d + 1, n);

        let mut xtx = vec![vec![0.0; d]; d];
        let mut xty = vec![0.0; d];

        // Sample with replacement for bagging.
        for _ in 0..m {
            let idx = rng.gen_range(0..n);
            let x = &features[idx];
            let y = targets[idx];

            for i in 0..d {
                xty[i] += x[i] * y;
                for j in 0..d {
                    xtx[i][j] += x[i] * x[j];
                }
            }
        }

        // Ridge penalty on diagonal.
        for i in 0..d {
            xtx[i][i] += lambda;
        }

        let mut xtx_clone = xtx.clone();
        let mut xty_clone = xty.clone();
        let beta = solve_linear_system(&mut xtx_clone, &mut xty_clone);

        for j in 0..d {
            beta_sum[j] += beta[j];
        }

        // Model residuals across the FULL dataset (not just the subsample).
        let mut rss = 0.0;
        for (x, y) in features.iter().zip(targets.iter()) {
            let y_hat = dot(&beta, x);
            rss += (y - y_hat).powi(2);
        }
        rss_total += rss;
    }

    // Average weights
    for j in 0..d {
        beta_sum[j] /= nm as f64;
    }

    // Residual std over all models and all samples
    let denom = (nm * n) as f64;
    let resid_std = if denom > 0.0 {
        (rss_total / denom).sqrt()
    } else {
        0.0
    };

    (beta_sum, resid_std)
}
