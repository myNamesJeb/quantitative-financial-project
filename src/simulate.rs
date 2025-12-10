
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::data::MarketBar;
use crate::sabr::{SabrParams, sabr_implied_vol};

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
/// - they contain ≈98% of terminal prices (1%–99% quantiles)
fn compute_price_bounds(
    last: f64,
    base_sigma: f64,
    horizon: usize,
    terminal_prices: &[f64],
) -> (f64, f64) {
    // Analytic horizon vol scale
    let eff_sigma = (base_sigma * (horizon as f64).sqrt()).max(1e-6);

    // Analytic bounds from lognormal assumption
    let analytic_low = last * f64::exp(-4.0 * eff_sigma);
    let analytic_high = last * f64::exp(4.0 * eff_sigma);

    // If we somehow have no prices, just use analytic
    if terminal_prices.is_empty() {
        return (analytic_low, analytic_high);
    }

    // Empirical quantiles of terminal prices
    let mut v = terminal_prices.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    // Clamp indices to valid range
    let idx_low = ((0.01 * n as f64).floor() as usize).min(n - 1);
    let idx_high = ((0.99 * n as f64).ceil() as usize).min(n - 1);

    let q_low = v[idx_low].max(1e-6);
    let q_high = v[idx_high];

    // Combine analytic + empirical:
    // - lower = min(analytic_low, q_low) * slight pad
    // - upper = max(analytic_high, q_high) * slight pad
    let mut min_p = q_low.min(analytic_low);
    let mut max_p = q_high.max(analytic_high);

    // Pad a bit so edges are not saturated
    min_p *= 0.98;
    max_p *= 1.02;

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
/// - uses base_sigma as stochastic vol scale
/// - returns heatmap + the grid bounds
pub fn simulate_heatmap(
    bars: &[MarketBar],
    drift: f64,
    base_sigma: f64,
    sabr: &SabrParams,
    horizon: usize,
    buckets: usize,
    paths: usize,
) -> SimulationResult {
    let last_close = bars.last().unwrap().close;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Store all simulated prices for rebucketing later:
    // prices[t][path]
    let mut prices = vec![vec![0.0; paths]; horizon];

    // Per-step drift (log scale)
    let drift_step = drift / horizon as f64;

    for path in 0..paths {
        let mut price = last_close;

        for t in 0..horizon {
            // SABR local vol: use F = K = price (ATM smile at current level)
            let sabr_vol = sabr_implied_vol(price, price, sabr);
            let smile_scale = (sabr_vol / sabr.atm_vol).clamp(0.5, 2.0);

            // Stochastic vol shock
            let vol_shock: f64 = normal.sample(&mut rng);
            let local_sigma = (base_sigma * smile_scale * (1.0 + 0.3 * vol_shock))
                .max(0.001);

            let z: f64 = normal.sample(&mut rng);
            price *= (drift_step + local_sigma * z).exp();

            // Jump process: mild, two-sided
            if rng.gen_bool(0.02) {
                let jump_z: f64 = normal.sample(&mut rng);
                price *= (0.5 * local_sigma * jump_z).exp();
            }

            if !price.is_finite() || price <= 0.0 {
                // If something explodes, reset to last_close (very rare)
                price = last_close;
            }

            prices[t][path] = price;
        }
    }

    // Use terminal distribution to set bounds
    let terminal_prices = &prices[horizon - 1];
    let (min_p, max_p) = compute_price_bounds(last_close, base_sigma, horizon, terminal_prices);

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
