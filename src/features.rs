
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
    let vol_z_window = 200usize;

    // Cumulative sums for rolling volume/trades stats.
    let mut cum_vol = Vec::with_capacity(bars.len() + 1);
    let mut cum_vol2 = Vec::with_capacity(bars.len() + 1);
    let mut cum_tr = Vec::with_capacity(bars.len() + 1);
    let mut cum_tr2 = Vec::with_capacity(bars.len() + 1);
    cum_vol.push(0.0);
    cum_vol2.push(0.0);
    cum_tr.push(0.0);
    cum_tr2.push(0.0);
    for b in bars {
        let v = b.volume;
        let t = b.trades;
        cum_vol.push(cum_vol.last().unwrap() + v);
        cum_vol2.push(cum_vol2.last().unwrap() + v * v);
        cum_tr.push(cum_tr.last().unwrap() + t);
        cum_tr2.push(cum_tr2.last().unwrap() + t * t);
    }

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

        // Volume & trades z-scores at bar[end] using rolling stats.
        let z_start = end.saturating_sub(vol_z_window);
        let z_len = (end - z_start).max(1) as f64;

        let vol_sum = cum_vol[end] - cum_vol[z_start];
        let vol_sum2 = cum_vol2[end] - cum_vol2[z_start];
        let vol_mean = vol_sum / z_len;
        let vol_var = (vol_sum2 / z_len) - vol_mean * vol_mean;
        let vol_std = vol_var.max(0.0).sqrt().max(1e-6);
        let vol_z = (bars[end].volume - vol_mean) / vol_std;

        let tr_sum = cum_tr[end] - cum_tr[z_start];
        let tr_sum2 = cum_tr2[end] - cum_tr2[z_start];
        let tr_mean = tr_sum / z_len;
        let tr_var = (tr_sum2 / z_len) - tr_mean * tr_mean;
        let tr_std = tr_var.max(0.0).sqrt().max(1e-6);
        let trades_z = (bars[end].trades - tr_mean) / tr_std;

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

        // Normalize some features by longer vol to stabilize scale.
        let vol20_safe = vol20.max(1e-6);
        let r_t_z = r_t / vol20_safe;
        let r5_z = r5 / vol20_safe;
        let r20_z = r20 / vol20_safe;
        let vol_ratio = (vol5 / vol20_safe).clamp(0.1, 3.0);

        // Clamp heavy tails to reduce outlier impact.
        let clamp = |x: f64, lo: f64, hi: f64| x.clamp(lo, hi);

        let r_t_z = clamp(r_t_z, -6.0, 6.0);
        let r5_z = clamp(r5_z, -10.0, 10.0);
        let r20_z = clamp(r20_z, -12.0, 12.0);
        let vol5_c = clamp(vol5, 0.0, 0.2);
        let vol20_c = clamp(vol20, 0.0, 0.2);
        let vol_z_c = clamp(vol_z, -6.0, 6.0);
        let trades_z_c = clamp(trades_z, -6.0, 6.0);
        let range_c = clamp(range, 0.0, 0.2);
        let run_len_c = clamp(run_len as f64, 0.0, 30.0);

        // Construct feature vector with nonlinear terms.
        feats.push(vec![
            1.0,                         // bias term
            r_t_z,                       // current return (z)
            r5_z,                        // short-horizon cum return (z)
            r20_z,                       // long-horizon cum return (z)
            vol5_c,                      // short-horizon vol
            vol20_c,                     // long-horizon vol
            vol_ratio,                   // vol ratio
            vol_z_c,                     // volume z-score
            trades_z_c,                  // trades z-score
            range_c,                     // relative range
            run_len_c,                   // trend run length
            r_t_z.abs(),                 // abs return
            r_t_z * r_t_z,               // squared return
            r_t_z * vol5_c,              // return-vol interaction
            r5_z * vol_ratio,            // momentum-vol interaction
            range_c * vol_z_c,           // range/volume interaction
        ]);

        // One-step-ahead target.
        targs.push(returns[t + 1]);
    }

    (feats, targs)
}

/// Same as build_ml_dataset but keeps the time index for walk-forward CV.
pub fn build_ml_dataset_indexed(
    bars: &[MarketBar],
    returns: &[f64],
) -> (Vec<(usize, Vec<f64>)>, Vec<f64>) {
    let (feats, targs) = build_ml_dataset(bars, returns);
    let n = returns.len();
    if n < 40 || bars.len() <= n {
        return (vec![], vec![]);
    }
    let start = 20usize; // matches build_ml_dataset's long window
    let mut out = Vec::with_capacity(feats.len());
    for (i, f) in feats.into_iter().enumerate() {
        let t = start + i;
        out.push((t, f));
    }
    (out, targs)
}

/// Mean of a slice.
pub fn mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        a.iter().sum::<f64>() / (a.len() as f64)
    }
}
