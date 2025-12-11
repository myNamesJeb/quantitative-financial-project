
// src/simulate.rs

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::data::MarketBar;
use crate::sabr::{SabrParams, sabr_implied_vol};
use crate::features::compute_log_returns;
use crate::advanced::estimate_autocorr_squared;

/// Result of a full Monte Carlo simulation.
pub struct SimulationResult {
    pub heatmap: Vec<Vec<f64>>,
    pub min_price: f64,
    pub max_price: f64,
}

/// Map price → bucket index within [min, max].
pub fn price_to_bucket(price: f64, min: f64, max: f64, buckets: usize) -> usize {
    if price <= min {
        return 0;
    }
    if price >= max {
        return buckets - 1;
    }
    let r = (price - min) / (max - min);
    let idx = (r * buckets as f64).floor() as isize;
    idx.clamp(0, buckets as isize - 1) as usize
}

/// Compute price bounds so that:
/// - they scale with vol * sqrt(horizon)
/// - they adapt to volatility clustering
/// - they contain ≈98% of terminal prices using empirical quantiles
fn compute_price_bounds(
    last: f64,
    base_sigma: f64,
    horizon: usize,
    terminal_prices: &[f64],
    vol_cluster: f64,
) -> (f64, f64) {
    // Analytic horizon vol scale
    let eff_sigma = (base_sigma * (horizon as f64).sqrt()).max(1e-6);

    // Vol clustering factor in [0, 1+] range, clamp for sanity
    let vc = vol_cluster.clamp(0.0, 1.0);

    // Spread multiplier in sigmas; allow 2–4σ based on clustering
    let spread_mult = 2.0 + 2.0 * vc; // 2σ when calm, up to 4σ when clustered

    let analytic_low = last * f64::exp(-spread_mult * eff_sigma);
    let analytic_high = last * f64::exp(spread_mult * eff_sigma);

    // If we somehow have no terminal prices, just use analytic band.
    if terminal_prices.is_empty() {
        return (analytic_low.max(1e-6), analytic_high.max(analytic_low * 1.5));
    }

    // Empirical quantiles of terminal prices
    let mut v = terminal_prices.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    let idx_low = ((0.01 * n as f64).floor() as usize).min(n - 1);
    let idx_high = ((0.99 * n as f64).ceil() as usize).min(n - 1);

    let q_low = v[idx_low].max(1e-6);
    let q_high = v[idx_high];

    // Combine analytic + empirical:
    let mut min_p = q_low.min(analytic_low);
    let mut max_p = q_high.max(analytic_high);

    // Pad a bit so edges are not saturated
    min_p *= 0.99;
    max_p *= 1.01;

    if min_p <= 0.0 {
        min_p = analytic_low.min(last * 0.5).max(1e-6);
    }
    if max_p <= min_p {
        max_p = min_p * 1.5;
    }

    (min_p, max_p)
}

/// Monte Carlo simulation:
/// - uses SABR local vol
/// - uses base_sigma as stochastic vol scale (adjusted by clustering)
/// - introduces liquidity-sensitive jump probability and size
/// - returns heatmap + the grid bounds
pub fn simulate_heatmap(
    bars: &[MarketBar],
    drift: f64,
    base_sigma_in: f64,
    sabr: &SabrParams,
    horizon: usize,
    buckets: usize,
    paths: usize,
) -> SimulationResult {
    let last_close = bars.last().unwrap().close;

    // Compute log-returns for volatility clustering diagnostics.
    let returns = compute_log_returns(bars);
    let vol_cluster = if returns.len() > 10 {
        estimate_autocorr_squared(&returns, 5).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Microstructure / liquidity proxy from recent bars.
    let n_micro = bars.len().min(200);
    let slice = &bars[bars.len() - n_micro..];

    let mut sum_range = 0.0;
    let mut sum_vol = 0.0;
    let mut cnt = 0.0;

    for b in slice {
        if b.close > 0.0 {
            let r = (b.high - b.low).abs() / b.close;
            sum_range += r;
        }
        sum_vol += b.volume.max(0.0);
        cnt += 1.0;
    }

    let avg_range = if cnt > 0.0 { sum_range / cnt } else { 0.0 };
    let avg_vol = if cnt > 0.0 { sum_vol / cnt } else { 0.0 };

    // Liquidity thinness proxy: larger when ranges are big relative to volume.
    let liq_thin = if avg_vol > 0.0 {
        (avg_range / avg_vol).abs()
    } else {
        avg_range.abs()
    };

    // Base jump probability and scaling based on liquidity thinness and clustering.
    let mut jump_prob_base = 0.01 + 0.20 * liq_thin;
    jump_prob_base = jump_prob_base.clamp(0.005, 0.08);

    let jump_prob = (jump_prob_base * (1.0 + 0.5 * vol_cluster)).clamp(0.005, 0.15);

    // Clamp base sigma to a realistic intraday band, then adjust by clustering.
    let mut base_sigma = base_sigma_in.clamp(0.009, 0.07);
    if vol_cluster > 0.4 {
        base_sigma *= 1.3;
    } else if vol_cluster < 0.1 {
        base_sigma *= 0.85;
    }

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Store all simulated prices for rebucketing later:
    // prices[t][path]
    let mut prices = vec![vec![0.0; paths]; horizon];

    // Per-step drift (log scale) – keep it mild across the horizon.
    let drift_step = drift / horizon as f64;

    // Vol clustering scale factor used inside the simulation.
    let vol_cluster_scale = 1.0 + 0.5 * vol_cluster;

    for path in 0..paths {
        let mut price = last_close;

        for t in 0..horizon {
            // SABR local vol: use F = K = price (ATM smile at current level)
            let sabr_vol = sabr_implied_vol(price, price, sabr);
            let smile_scale = (sabr_vol / sabr.atm_vol.max(1e-6)).clamp(0.5, 1.8);

            // Stochastic vol shock
            let vol_shock: f64 = normal.sample(&mut rng);

            let local_sigma_raw =
                base_sigma * smile_scale * vol_cluster_scale * (1.0 + 0.15 * vol_shock);
            let local_sigma = local_sigma_raw.clamp(0.002, 0.06);

            let z: f64 = normal.sample(&mut rng);
            price *= (drift_step + local_sigma * z).exp();

            // Liquidity-sensitive jump process: rare but meaningful.
            if rng.gen_bool(jump_prob) {
                let jump_z: f64 = normal.sample(&mut rng);
                let jump_scale = 0.15 + 0.30 * vol_cluster; // 15–45% of local_sigma
                price *= (jump_scale * local_sigma * jump_z).exp();
            }

            if !price.is_finite() || price <= 0.0 {
                // If something explodes, reset to last_close (very rare).
                price = last_close;
            }

            prices[t][path] = price;
        }
    }

    // Use terminal distribution to set bounds
    let terminal_prices = &prices[horizon - 1];
    let (min_p, max_p) =
        compute_price_bounds(last_close, base_sigma, horizon, terminal_prices, vol_cluster);

    println!("Price grid: {:.4} → {:.4}", min_p, max_p);

    // Build heatmap under the chosen bounds
    let mut heatmap = vec![vec![0.0; buckets]; horizon];

    for t in 0..horizon {
        let row_prices = &prices[t];
        let row = &mut heatmap[t];

        for &price in row_prices {
            if price > 0.0 && price.is_finite() {
                let b = price_to_bucket(price, min_p, max_p, buckets);
                row[b] += 1.0;
            }
        }

        // Normalize row to sum to 1
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for v in row.iter_mut() {
                *v /= s;
            }
        }
    }

    SimulationResult {
        heatmap,
        min_price: min_p,
        max_price: max_p,
    }
}
