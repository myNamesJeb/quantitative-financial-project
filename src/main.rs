
use std::error::Error;
mod advanced;
mod data;
mod features;
mod ml;
mod sabr;
mod drift;
mod simulate;
mod viz;
mod backtest;
mod strategy;

use data::read_bars;
use backtest::Backtester;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <output.png>", args[0]);
        std::process::exit(1);
    }
    let output_png = &args[1];

    // ------------------------------------------------------
    // 1) Load multiple timeframes
    // ------------------------------------------------------
    let one_d  = read_bars("archive/sol_1d_data_2020_to_2025.csv")?;
    let four_h = read_bars("archive/sol_4h_data_2020_to_2025.csv")?;
    let one_h  = read_bars("archive/sol_1h_data_2020_to_2025.csv")?;
    let f15    = read_bars("archive/sol_15m_data_2020_to_2025.csv")?;

    println!("Loaded:");
    println!("  1D  = {}", one_d.len());
    println!("  4H  = {}", four_h.len());
    println!("  1H  = {}", one_h.len());
    println!("  15m = {}", f15.len());

    // ------------------------------------------------------
    // 2) Run Backtester on the 1H timeframe with StrategyRouter
    // ------------------------------------------------------
    let mut bars_1h = one_h.clone();
    bars_1h.sort_by(|a, b| a.unix.cmp(&b.unix));

    let mut bt = Backtester::default();
    bt.initial_equity = 1_000.0;
    bt.max_drawdown_allowed = 0.30;

    let results = bt.run(&bars_1h);

    println!("\n=========== BACKTEST SUMMARY ===========");
    println!("Initial equity: {:.2}", bt.initial_equity);
    println!("Final equity:   {:.2}", results.final_equity);
    println!("Total return:   {:.2}%", results.total_return * 100.0);
    println!("Trades:         {}", results.trades.len());
    println!("Max drawdown:   {:.2}%", results.max_drawdown * 100.0);
    println!("Sharpe (ann.):  {:.2}", results.sharpe);
    println!("========================================\n");

    // ------------------------------------------------------
    // 3) Forecast engine for most recent bar
    // ------------------------------------------------------
    use features::{compute_log_returns, compute_garch_like_vol, build_ml_dataset};
    use ml::{fit_linear_regression_ensemble, classify_regime_ml, dot};
    use drift::compute_drift_and_vol_scale;
    use sabr::estimate_sabr_params;
    use simulate::simulate_heatmap;
    use viz::{print_bucket_summary, save_heatmap_png};

    let bars = bars_1h;
    let returns = compute_log_returns(&bars);
    let garch_vol = compute_garch_like_vol(&returns);

    // ---------- ML drift + vol predictor ----------
    let (features_x, targets_y) = build_ml_dataset(&bars, &returns);
    if features_x.is_empty() {
        eprintln!("Not enough data to build ML dataset.");
        std::process::exit(1);
    }

    let (beta, ml_resid_std) = fit_linear_regression_ensemble(
        &features_x,
        &targets_y,
        1e-4,
        64,
        0.7,
    );

    let last_x = features_x.last().unwrap();
    let ml_pred_ret = dot(last_x, &beta);
    let ml_pred_vol = if ml_resid_std > 0.0 { ml_resid_std } else { 0.01 };

    let regime = classify_regime_ml(ml_pred_ret, ml_pred_vol, &returns);

    println!(
        "Regime (ML): {:?} | pred_ret {:.5} | pred_vol {:.5}",
        regime, ml_pred_ret, ml_pred_vol
    );

    // ---------- Drift + microstructure ----------
    let last_close = bars.last().unwrap().close;
    let (drift, base_sigma) =
        compute_drift_and_vol_scale(&bars, &returns, &garch_vol, regime, ml_pred_ret, ml_pred_vol);

    println!("Drift = {:.6}", drift);
    println!("Base sigma = {:.6}", base_sigma);

    // ---------- SABR curvature ----------
    let sabr_params = estimate_sabr_params(&returns, base_sigma, last_close);
    println!("SABR = {:?}\n", sabr_params);

    // ------------------------------------------------------
    // 4) Monte Carlo heatmap (multi-threaded)
    // ------------------------------------------------------
    let sim = simulate_heatmap(
        &bars,
        drift,
        base_sigma,
        &sabr_params,
        50,      // horizon
        80,      // buckets
        3_000,   // paths (fast because Rayon)
    );

    print_bucket_summary(&sim.heatmap, sim.min_price, sim.max_price, &bars);
    save_heatmap_png(&sim.heatmap, sim.min_price, sim.max_price, output_png)?;

    println!("Saved heatmap to {}", output_png);

    Ok(())
}
