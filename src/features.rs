
use crate::data::MarketBar;

/// sample mean + sample std
pub fn mean_std(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

pub fn compute_log_returns(bars: &[MarketBar]) -> Vec<f64> {
    let mut out = Vec::with_capacity(bars.len().saturating_sub(1));
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

pub fn compute_garch_like_vol(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![];
    }

    let (mean_r, var_r) = mean_std(returns);
    let mut sigma2 = var_r.max(1e-8);

    const OMEGA: f64 = 1e-7;
    const ALPHA: f64 = 0.1;
    const BETA: f64 = 0.85;

    let mut out = Vec::with_capacity(returns.len());
    for &r in returns {
        sigma2 = OMEGA + ALPHA * (r - mean_r).powi(2) + BETA * sigma2;
        out.push(sigma2.sqrt());
    }
    out
}

/// Build ML features (unchanged structure)
pub fn build_ml_dataset(
    bars: &[MarketBar],
    returns: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = returns.len();
    if n < 40 {
        return (vec![], vec![]);
    }

    let vol_s = 5;
    let vol_l = 20;

    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let trades: Vec<f64> = bars.iter().map(|b| b.trades).collect();
    let (_vm, vstd) = mean_std(&volumes);
    let (_tm, tstd) = mean_std(&trades);

    let mut feats = Vec::new();
    let mut targs = Vec::new();

    let start = vol_l;
    for t in start..(n - 1) {
        let end = t + 1;
        let s_s = end.saturating_sub(vol_s);
        let s_l = end.saturating_sub(vol_l);

        let r_s = &returns[s_s..end];
        let r_l = &returns[s_l..end];

        let r_t = returns[t];
        let r5 = r_s.iter().sum::<f64>();
        let r20 = r_l.iter().sum::<f64>();

        let (_, vol5) = mean_std(r_s);
        let (_, vol20) = mean_std(r_l);

        let vol_z =
            if vstd > 0.0 { (bars[end].volume - volumes.iter().sum::<f64>() / volumes.len() as f64) / vstd } else { 0.0 };

        let trades_z =
            if tstd > 0.0 { (bars[end].trades - trades.iter().sum::<f64>() / trades.len() as f64) / tstd } else { 0.0 };

        let range = if bars[end].close > 0.0 {
            (bars[end].high - bars[end].low).abs() / bars[end].close
        } else {
            0.0
        };

        feats.push(vec![
            1.0, r_t, r5, r20, vol5, vol20, vol_z, trades_z, range,
        ]);
        targs.push(returns[t + 1]);
    }

    (feats, targs)
}
