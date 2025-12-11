
// src/drift.rs

use crate::data::MarketBar;
use crate::ml::MarketRegime;
use crate::features::mean_std;
use crate::advanced::{
    kalman_smooth_drift,
    estimate_autocorr_squared,
    predict_vol_rnn,
};

/// Compute drift (expected log-return) and base volatility scale
/// for the next step, using:
/// - ML prediction (ml_pred_ret, ml_pred_vol)
/// - Kalman-smoothed drift from realized returns
/// - GARCH-like volatility series
/// - Volatility clustering (autocorr of squared returns)
/// - Microstructure proxies (volume, range, trades, close-location)
/// - Regime classification (MarketRegime)
pub fn compute_drift_and_vol_scale(
    bars: &[MarketBar],
    returns: &[f64],
    garch_vol: &[f64],
    regime: MarketRegime,
    ml_pred_ret: f64,
    ml_pred_vol: f64,
) -> (f64, f64) {
    // Safety guard: if we have basically no history, just fall back
    if bars.len() < 10 || returns.is_empty() {
        let drift = ml_pred_ret.clamp(-0.01, 0.01);
        let sigma = ml_pred_vol.abs().max(0.01);
        return (drift, sigma);
    }

    let last_ret = *returns.last().unwrap_or(&0.0);
    let last_bar = bars.last().unwrap();

    // ============================
    // Microstructure window
    // ============================
    //
    // Use up to the last 120 bars for microstructure averages.
    let n_micro = bars.len().min(120);
    let slice = &bars[bars.len() - n_micro..];

    // Average volume and trades
    let avg_vol = slice.iter().map(|b| b.volume).sum::<f64>() / n_micro as f64;
    let avg_tr = slice.iter().map(|b| b.trades).sum::<f64>() / n_micro as f64;

    // Average relative range
    let avg_range = slice
        .iter()
        .map(|b| {
            if b.close > 0.0 {
                (b.high - b.low).abs() / b.close
            } else {
                0.0
            }
        })
        .sum::<f64>() / n_micro as f64;

    // Last-bar relative range
    let last_range = if last_bar.close > 0.0 {
        (last_bar.high - last_bar.low).abs() / last_bar.close
    } else {
        0.0
    };

    // Volume z-score
    let vol_z = if avg_vol > 0.0 {
        (last_bar.volume - avg_vol) / avg_vol
    } else {
        0.0
    };

    // Trades z-score
    let trades_z = if avg_tr > 0.0 {
        (last_bar.trades - avg_tr) / avg_tr
    } else {
        0.0
    };

    // Range z-score
    let range_z = if avg_range > 0.0 {
        (last_range - avg_range) / avg_range
    } else {
        0.0
    };

    // ============================
    // Close-location microstructure
    // ============================
    //
    // Where did the close land inside the bar?
    //   - near high  -> aggressive buying
    //   - near low   -> aggressive selling
    //   - in middle  -> neutral
    let bar_range_abs = (last_bar.high - last_bar.low).abs();
    let close_loc = if bar_range_abs > 0.0 {
        (last_bar.close - last_bar.low) / bar_range_abs
    } else {
        0.5
    };

    // Average close-location over the micro window
    let mut sum_cl = 0.0;
    let mut cl_count = 0.0;
    for b in slice {
        let r = (b.high - b.low).abs();
        if r > 0.0 {
            let cl = (b.close - b.low) / r;
            sum_cl += cl;
            cl_count += 1.0;
        }
    }
    let avg_close_loc = if cl_count > 0.0 {
        sum_cl / cl_count
    } else {
        0.5
    };

    // Close-location z-score (rough)
    let close_loc_z = if avg_close_loc > 0.0 {
        (close_loc - avg_close_loc) / avg_close_loc.max(1e-3)
    } else {
        0.0
    };

    // ============================
    // Trend run-length
    // ============================
    //
    // How many bars in a row have we been moving in the same direction?
    let rn = returns.len();
    let mut run_len = 0usize;
    let sign_last = last_ret.signum();

    if sign_last != 0.0 && rn > 1 {
        // Count how many of the previous returns have same sign as the last.
        for &r in returns[..rn - 1].iter().rev() {
            if r.signum() == sign_last {
                run_len += 1;
                if run_len >= 30 {
                    break; // cap to avoid insane values
                }
            } else {
                break;
            }
        }
    }

    // ============================
    // Realized return volatility
    // ============================
    //
    // Use last ~200 returns as a realized vol anchor.
    let k = rn.min(200);
    let ret_slice = &returns[rn - k..];
    let (_ret_mean, ret_std) = mean_std(ret_slice);

    // ============================
    // Kalman-smoothed drift series
    // ============================
    let k_series = kalman_smooth_drift(returns);
    let k_last = *k_series.last().unwrap_or(&ml_pred_ret);

    // ============================
    // Base drift (directional)
    // ============================
    //
    // Blend ML prediction and Kalman drift,
    // then refine with flow, trend-run, and close-location.
    let mut drift = 0.4 * ml_pred_ret + 0.6 * k_last;

    // Flow term: if returns and volume both positive, drift nudges up; etc.
    let flow = last_ret.signum() * vol_z.signum();
    drift += 0.20 * flow * ml_pred_vol.abs();

    // Trend run length: longer runs make drift slightly more persistent
    if sign_last != 0.0 && run_len > 0 {
        let run_factor = (run_len as f64 / 10.0).min(2.0);
        drift += 0.0015 * run_factor * sign_last;
    }

    // Close-location skew: if we keep closing near highs/lows, drift nudges that way.
    drift += 0.05 * close_loc_z;

    // Normalize drift by realized volatility to avoid insane scaling.
    if ret_std > 0.0 {
        let drift_z = drift / ret_std;
        // Keep it within a reasonable z range, then rescale.
        let drift_z = drift_z.clamp(-2.0, 2.0);
        drift = drift_z * ret_std;
    }

    // ============================
    // Volatility base (sigma)
    // ============================
    //
    // Combine:
    // - latest GARCH-like vol
    // - ML residual vol estimate (ml_pred_vol)
    // - simple RNN-ish vol predictor
    // - clustering via autocorr of squared returns
    let last_garch = garch_vol.last().cloned().unwrap_or(ml_pred_vol);
    let mut sigma = 0.4 * last_garch + 0.3 * ml_pred_vol.abs();

    let nn_vol = predict_vol_rnn(returns);
    sigma = 0.4 * sigma + 0.6 * nn_vol;

    // Vol clustering: autocorr of squared returns
    let ac = estimate_autocorr_squared(returns, 5);
    if ac > 0.4 {
        sigma *= 1.5;
    } else if ac > 0.2 {
        sigma *= 1.2;
    }

    // ============================
    // Regime-based scaling
    // ============================
    //
    // Regime tells us whether to lean into drift and how much vol to expect.
    let (dr_mult, vol_mult) = match regime {
        MarketRegime::TrendingUp => (1.3, 1.15),
        MarketRegime::TrendingDown => (1.3, 1.15),
        MarketRegime::SidewaysLowVol => (0.6, 0.8),
        MarketRegime::SidewaysHighVol => (0.8, 1.4),
    };

    drift *= dr_mult;
    sigma *= vol_mult;

    // ============================
    // Microstructure vol boost
    // ============================
    //
    // If volume, range, trades, or close-location are extreme,
    // bump volatility accordingly.
    let micro = (1.0
        + 0.25 * vol_z.abs()
        + 0.20 * range_z.abs()
        + 0.15 * trades_z.abs()
        + 0.10 * close_loc_z.abs())
        .clamp(0.5, 3.0);

    sigma *= micro;

    // Final clamps to keep everything sane for Monte Carlo.
    let drift = drift.clamp(-0.05, 0.05);
    let sigma = sigma.max(0.005);

    (drift, sigma)
}
