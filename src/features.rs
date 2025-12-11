
// src/features.rs

use crate::advanced::fit_garch_mle_and_series;
use crate::data::MarketBar;

/// mean + std
pub fn mean_std(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    // Unbiased-ish sample variance with guard for n = 1.
    let denom = (n - 1.0).max(1.0);
    let var = data
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / denom;

    (mean, var.sqrt())
}

/// Log returns from close prices.
pub fn compute_log_returns(bars: &[MarketBar]) -> Vec<f64> {
    if bars.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(bars.len() - 1);
    for i in 1..bars.len() {
        let p0 = bars[i - 1].close;
        let p1 = bars[i].close;
        if p0 > 0.0 && p1 > 0.0 {
            out.push((p1 / p0).ln());
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Calls advanced GARCH MLE and returns the sigma_t series.
pub fn compute_garch_like_vol(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![];
    }
    let (_params, series) = fit_garch_mle_and_series(returns);
    series
}

/// Build ML dataset for one-step-ahead return prediction.
///
/// Each row of `features` corresponds to a time `t`, and the target
/// is `returns[t + 1]`.
///
/// Features include:
/// - bias term
/// - current return
/// - short/long cumulative returns (5, 20)
/// - short/long realized vol (5, 20)
/// - volume/trades z-scores
/// - relative bar range
/// - trend-run length (how many bars in a row we kept the same sign)
pub fn build_ml_dataset(
    bars: &[MarketBar],
    returns: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = returns.len();
    if n < 40 || bars.len() <= n {
        // Not enough history to build a decent dataset.
        return (vec![], vec![]);
    }

    // Windows for "short" and "long" horizons.
    let vol_s = 5usize;
    let vol_l = 20usize;

    // Basic volume/trades stats over the whole sample.
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let trades: Vec<f64> = bars.iter().map(|b| b.trades).collect();

    let (vm, vstd) = mean_std(&volumes);
    let (tm, tstd) = mean_std(&trades);

    let mut feats = Vec::new();
    let mut targs = Vec::new();

    // We need at least `vol_l` history behind each point, and we predict t+1.
    let start = vol_l;
    for t in start..(n - 1) {
        let end = t + 1;

        // Short/long windows on returns.
        let s_s = end.saturating_sub(vol_s);
        let s_l = end.saturating_sub(vol_l);

        let r_s = &returns[s_s..end];
        let r_l = &returns[s_l..end];

        let r_t = returns[t];
        let r5 = r_s.iter().sum::<f64>();
        let r20 = r_l.iter().sum::<f64>();

        let (_, vol5) = mean_std(r_s);
        let (_, vol20) = mean_std(r_l);

        // Volume & trades z-scores at bar[end].
        let vol_z = if vstd > 0.0 {
            (bars[end].volume - vm) / vstd
        } else {
            0.0
        };

        let trades_z = if tstd > 0.0 {
            (bars[end].trades - tm) / tstd
        } else {
            0.0
        };

        // Relative intrabar range at bar[end].
        let range = if bars[end].close > 0.0 {
            (bars[end].high - bars[end].low).abs() / bars[end].close
        } else {
            0.0
        };

        // Trend run-length: how many consecutive returns have the same sign as r_t.
        let mut run_len = 0usize;
        let sign_t = r_t.signum();
        if sign_t != 0.0 {
            // Look backwards from t-1 down to 0 while sign matches.
            let mut k = t;
            while k > 0 {
                let prev = returns[k - 1];
                if prev.signum() == sign_t {
                    run_len += 1;
                    if run_len >= 30 {
                        break; // cap to avoid absurd values
                    }
                    k -= 1;
                } else {
                    break;
                }
            }
        }

        // Construct feature vector.
        feats.push(vec![
            1.0,                     // bias term
            r_t,                     // current return
            r5,                      // short-horizon cum return
            r20,                     // long-horizon cum return
            vol5,                    // short-horizon vol
            vol20,                   // long-horizon vol
            vol_z,                   // volume z-score
            trades_z,                // trades z-score
            range,                   // relative range
            run_len as f64,          // trend run length
        ]);

        // One-step-ahead target.
        targs.push(returns[t + 1]);
    }

    (feats, targs)
}

/// Mean of a slice.
pub fn mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        a.iter().sum::<f64>() / (a.len() as f64)
    }
}
