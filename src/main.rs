
use std::error::Error;

mod data;
mod features;
mod ml;
mod sabr;
mod drift;
mod simulate;
mod viz;

use data::*;
use features::*;
use ml::*;
use sabr::*;
use drift::*;
use simulate::*;
use viz::*;

fn main() -> Result<(), Box<dyn Error>> {
    // Only argument is output PNG
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <output.png>", args[0]);
        std::process::exit(1);
    }
    let output_png = &args[1];

    // Load all timeframes
    let one_d  = read_bars("archive/sol_1d_data_2020_to_2025.csv")?;
    let four_h = read_bars("archive/sol_4h_data_2020_to_2025.csv")?;
    let one_h  = read_bars("archive/sol_1h_data_2020_to_2025.csv")?;
    let f15    = read_bars("archive/sol_15m_data_2020_to_2025.csv")?;

    println!("Loaded:");
    println!("  1D  = {}", one_d.len());
    println!("  4H  = {}", four_h.len());
    println!("  1H  = {}", one_h.len());
    println!("  15m = {}", f15.len());

    // For now, drive the MC model off 1H data
    let mut bars = one_h.clone();
    bars.sort_by(|a, b| a.unix.cmp(&b.unix));

    let returns = compute_log_returns(&bars);
    let garch_vol = compute_garch_like_vol(&returns);

    let (features, targets) = build_ml_dataset(&bars, &returns);
    if features.is_empty() {
        eprintln!("Not enough 1h data for ML dataset.");
        std::process::exit(1);
    }

    let (beta, ml_resid_std) = fit_linear_regression_ensemble(
        &features,
        &targets,
        1e-4,
        128,
        0.7,
    );

    let last_x = features.last().unwrap();
    let ml_pred_ret = dot(last_x, &beta);
    let ml_pred_vol = if ml_resid_std > 0.0 { ml_resid_std } else { 0.01 };

    let regime = classify_regime_ml(ml_pred_ret, ml_pred_vol, &returns);

    println!(
        "Regime (ML): {:?} | pred_ret {:.5} | pred_vol {:.5}",
        regime, ml_pred_ret, ml_pred_vol
    );

    let last_close = bars.last().unwrap().close;
    let (drift, base_sigma) = compute_drift_and_vol_scale(
        &bars,
        &returns,
        &garch_vol,
        regime,
        ml_pred_ret,
        ml_pred_vol,
    );

    println!("Drift = {:.6}", drift);
    println!("Base sigma = {:.6}", base_sigma);

    let sabr_params = estimate_sabr_params(&returns, base_sigma, last_close);
    println!("SABR = {:?}", sabr_params);

    // --- NEW: get heatmap + bounds from simulation ---
    let sim = simulate_heatmap(
        &bars,
        drift,
        base_sigma,
        &sabr_params,
        50,        // horizon
        80,        // buckets
        10_000,    // paths
    );

    print_bucket_summary(&sim.heatmap, sim.min_price, sim.max_price, &bars);
    save_heatmap_png(&sim.heatmap, sim.min_price, sim.max_price, output_png)?;

    println!("Saved {}", output_png);

    Ok(())
}
