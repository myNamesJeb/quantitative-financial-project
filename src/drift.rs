
use crate::data::MarketBar;
use crate::ml::MarketRegime;
use crate::features::mean_std;
use crate::advanced::{kalman_smooth_drift, estimate_autocorr_squared, predict_vol_rnn};

pub fn compute_drift_and_vol_scale(
    bars: &[MarketBar],
    returns: &[f64],
    garch_vol: &[f64],
    regime: MarketRegime,
    ml_pred_ret: f64,
    ml_pred_vol: f64,
) -> (f64, f64) {
    let last_ret = *returns.last().unwrap_or(&0.0);
    let last_bar = bars.last().unwrap();

    // ===== Microstructure stats (same as before) =====
    let n = bars.len().min(120);
    let slice = &bars[bars.len() - n..];

    let avg_vol = slice.iter().map(|b| b.volume).sum::<f64>() / n as f64;
    let avg_tr = slice.iter().map(|b| b.trades).sum::<f64>() / n as f64;
    let avg_range = slice
        .iter()
        .map(|b| {
            if b.close > 0.0 {
                (b.high - b.low).abs() / b.close
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n as f64;

    let last_range = if last_bar.close > 0.0 {
        (last_bar.high - last_bar.low).abs() / last_bar.close
    } else {
        0.0
    };

    let vol_z = if avg_vol > 0.0 {
        (last_bar.volume - avg_vol) / avg_vol
    } else {
        0.0
    };

    let trades_z = if avg_tr > 0.0 {
        (last_bar.trades - avg_tr) / avg_tr
    } else {
        0.0
    };

    let range_z = if avg_range > 0.0 {
        (last_range - avg_range) / avg_range
    } else {
        0.0
    };

    // ===== Kalman-smoothed drift =====
    let k_series = kalman_smooth_drift(returns);
    let k_last = *k_series.last().unwrap_or(&ml_pred_ret);

    // Base drift: blend ML + Kalman
    let mut drift = 0.5 * ml_pred_ret + 0.5 * k_last;

    // Flow adjustment (same idea as before)
    let flow = last_ret.signum() * vol_z.signum();
    drift += 0.25 * flow * ml_pred_vol.abs();

    // ===== Vol base: blend GARCH, ML, RNN, clustering =====
    let last_garch = garch_vol.last().cloned().unwrap_or(ml_pred_vol);
    let mut sigma = 0.4 * last_garch + 0.3 * ml_pred_vol.abs();

    // RNN-ish volatility estimate
    let nn_vol = predict_vol_rnn(returns);
    sigma = 0.4 * sigma + 0.6 * nn_vol;

    // Vol clustering via autocorr
    let ac = estimate_autocorr_squared(returns, 5);
    if ac > 0.4 {
        sigma *= 1.5;
    } else if ac > 0.2 {
        sigma *= 1.2;
    }

    // Regime scaling
    let (dr_mult, vol_mult) = match regime {
        MarketRegime::TrendingUp => (1.3, 1.15),
        MarketRegime::TrendingDown => (1.3, 1.15),
        MarketRegime::SidewaysLowVol => (0.6, 0.8),
        MarketRegime::SidewaysHighVol => (0.8, 1.4),
    };
    drift *= dr_mult;
    sigma *= vol_mult;

    // Microstructure vol boost
    let micro = (1.0
        + 0.3 * vol_z.abs()
        + 0.2 * range_z.abs()
        + 0.2 * trades_z.abs())
        .clamp(0.5, 3.0);
    sigma *= micro;

    (drift.clamp(-0.05, 0.05), sigma.max(0.01))
}
