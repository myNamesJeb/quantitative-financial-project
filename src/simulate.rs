
// src/simulate.rs

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::data::MarketBar;
use crate::sabr::{SabrParams, sabr_implied_vol};

pub struct SimulationResult {
    pub heatmap: Vec<Vec<f64>>,
    pub min_price: f64,
    pub max_price: f64,
}

/// Convert price → bucket index
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

/// Compute min/max using analytic + empirical quantiles
pub fn compute_price_bounds(
    last: f64,
    base_sigma: f64,
    horizon: usize,
    terminal_prices: &[f64],
) -> (f64, f64) {
    let eff_sigma = (base_sigma * (horizon as f64).sqrt()).max(1e-6);

    let analytic_low = last * f64::exp(-4.0 * eff_sigma);
    let analytic_high = last * f64::exp(4.0 * eff_sigma);

    if terminal_prices.is_empty() {
        return (analytic_low, analytic_high);
    }

    let mut v = terminal_prices.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    let low = v[(0.01 * n as f64) as usize];
    let high = v[(0.99 * n as f64) as usize];

    let mut min_p = low.min(analytic_low) * 0.98;
    let mut max_p = high.max(analytic_high) * 1.02;

    if min_p <= 0.0 {
        min_p = analytic_low.max(1e-6);
    }
    if max_p <= min_p {
        max_p = min_p * 1.5;
    }

    (min_p, max_p)
}

/// Full Monte Carlo (Bates + Heston + SABR smile)
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

    let mut prices = vec![vec![0.0; paths]; horizon];

    // === Heston parameters ===
    let v0 = (base_sigma * base_sigma).max(1e-6);
    let theta = v0;
    let kappa = 2.0;
    let xi = (base_sigma * 1.5).max(0.05);
    let rho = sabr.rho.clamp(-0.8, 0.8);

    let dt = 1.0 / horizon.max(1) as f64;
    let sqrt_dt = dt.sqrt();

    for path in 0..paths {
        let mut price = last_close;
        let mut v = v0;

        for t in 0..horizon {
            let sabr_vol = sabr_implied_vol(price, price, sabr);
            let smile = (sabr_vol / sabr.atm_vol).clamp(0.5, 2.0);

            let z1: f64 = normal.sample(&mut rng);
            let z2: f64 = normal.sample(&mut rng);

            let z_v = z2;
            let z_p = rho * z2 + (1.0 - rho * rho).sqrt() * z1;

            v = (v + kappa * (theta - v) * dt + xi * v.sqrt() * sqrt_dt * z_v)
                .max(1e-8);

            let local_sigma = (v.sqrt() * smile).max(0.001);

            let drift_step = drift / horizon as f64;
            price *= (drift_step + local_sigma * z_p).exp();

            // Bates jumps
            if rng.gen_bool(0.02) {
                let jump_z: f64 = normal.sample(&mut rng);
                price *= (0.7 * local_sigma * jump_z).exp();
            }

            if !price.is_finite() || price <= 0.0 {
                price = last_close;
                v = v0;
            }

            prices[t][path] = price;
        }
    }

    let terminal = &prices[horizon - 1];
    let (min_p, max_p) = compute_price_bounds(last_close, base_sigma, horizon, terminal);

    println!("Price grid: {:.4} → {:.4}", min_p, max_p);

    let mut heatmap = vec![vec![0.0; buckets]; horizon];

    for t in 0..horizon {
        for &price in &prices[t] {
            let b = price_to_bucket(price, min_p, max_p, buckets);
            heatmap[t][b] += 1.0;
        }
        let sum: f64 = heatmap[t].iter().sum();
        if sum > 0.0 {
            for p in heatmap[t].iter_mut() {
                *p /= sum;
            }
        }
    }

    SimulationResult {
        heatmap,
        min_price: min_p,
        max_price: max_p,
    }
}
