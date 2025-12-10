
use crate::data::MarketBar;
use crate::ml::MarketRegime;
use crate::features::mean_std;

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

    // Microstructure stats
    let n = bars.len().min(120);
    let slice = &bars[bars.len() - n..];

    let avg_vol   = slice.iter().map(|b| b.volume).sum::<f64>() / n as f64;
    let avg_tr    = slice.iter().map(|b| b.trades).sum::<f64>() / n as f64;
    let avg_range = slice.iter().map(|b| {
        if b.close > 0.0 {
            (b.high - b.low).abs() / b.close
        } else { 0.0 }
    }).sum::<f64>() / n as f64;

    let last_range = if last_bar.close > 0.0 {
        (last_bar.high - last_bar.low).abs() / last_bar.close
    } else { 0.0 };

    let vol_z =
        if avg_vol > 0.0 { (last_bar.volume - avg_vol) / avg_vol } else { 0.0 };

    let trades_z =
        if avg_tr > 0.0 { (last_bar.trades - avg_tr) / avg_tr } else { 0.0 };

    let range_z =
        if avg_range > 0.0 { (last_range - avg_range) / avg_range } else { 0.0 };

    // Base drift (ML)
    let mut drift = ml_pred_ret * 0.5;

    // Flow adjustment
    let flow = last_ret.signum() * vol_z.signum();
    drift += 0.25 * flow * ml_pred_vol.abs();

    // Vol base
    let last_garch = garch_vol.last().cloned().unwrap_or(ml_pred_vol);
    let mut sigma = 0.5 * last_garch + 0.5 * ml_pred_vol.abs();

    // Regime scale
    let (dr_mult, vol_mult) = match regime {
        MarketRegime::TrendingUp      => (1.2, 1.15),
        MarketRegime::TrendingDown    => (1.2, 1.15),
        MarketRegime::SidewaysLowVol  => (0.5, 0.8),
        MarketRegime::SidewaysHighVol => (0.8, 1.4),
    };
    drift *= dr_mult;

    // Microstructure vol boost
    let micro = (1.0 + 0.3 * vol_z.abs() + 0.2 * range_z.abs() + 0.2 * trades_z.abs())
        .clamp(0.5, 3.0);
    sigma *= vol_mult * micro;

    (drift.clamp(-0.05, 0.05), sigma.max(0.01))
}
