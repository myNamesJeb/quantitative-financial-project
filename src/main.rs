
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

// For multi-threaded, multi-timeframe summaries
use rayon::join;

use crate::data::MarketBar;
use crate::features::{compute_log_returns, compute_garch_like_vol, mean_std};
use crate::ml::classify_regime_ml;
use crate::drift::compute_drift_and_vol_scale;
use crate::sabr::estimate_sabr_params;
use crate::simulate::simulate_heatmap;
use crate::viz::{print_bucket_summary, save_heatmap_png};

/// Simple summary of a timeframe's signal: drift + vol.
struct TFSummary {
    pub name: &'static str,
    pub drift: f64,
    pub sigma: f64,
}

/// Build a crude but consistent drift + vol summary for a given timeframe.
/// This reuses your existing plumbing:
/// - log-returns
/// - GARCH-like vol
/// - ML-style regime classifier (but using mean/std as proxy for ML preds)
/// - drift + vol scaling from `compute_drift_and_vol_scale`
fn compute_tf_summary(name: &'static str, bars: &[MarketBar]) -> TFSummary {
    if bars.len() < 60 {
        // Not enough history to say anything intelligent.
        return TFSummary {
            name,
            drift: 0.0,
            sigma: 0.01,
        };
    }

    let returns = compute_log_returns(bars);
    let garch_vol = compute_garch_like_vol(&returns);

    // Use realized mean/std as a proxy "ML prediction" for this TF.
    let (m_ret, s_ret) = mean_std(&returns);
    let ml_pred_ret = m_ret;
    let ml_pred_vol = s_ret.max(1e-4);

    let regime = classify_regime_ml(ml_pred_ret, ml_pred_vol, &returns);

    let (drift, base_sigma) = compute_drift_and_vol_scale(
        bars,
        &returns,
        &garch_vol,
        regime,
        ml_pred_ret,
        ml_pred_vol,
    );

    TFSummary {
        name,
        drift,
        // Keep a sane floor so nothing degenerates.
        sigma: base_sigma.max(0.005),
    }
}

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
    let one_d = read_bars("archive/sol_1d_data_2020_to_2025.csv")?;
    let mut four_h = read_bars("archive/sol_4h_data_2020_to_2025.csv")?;
    let mut one_h = read_bars("archive/sol_1h_data_2020_to_2025.csv")?;
    let mut f15 = read_bars("archive/sol_15m_data_2020_to_2025.csv")?;

    // Sort intraday TFs by time just in case
    four_h.sort_by(|a, b| a.unix.cmp(&b.unix));
    one_h.sort_by(|a, b| a.unix.cmp(&b.unix));
    f15.sort_by(|a, b| a.unix.cmp(&b.unix));

    println!("Loaded:");
    println!("  1D  = {}", one_d.len());
    println!("  4H  = {}", four_h.len());
    println!("  1H  = {}", one_h.len());
    println!("  15m = {}", f15.len());

    // ------------------------------------------------------
    // 2) Run Backtester on the 1H timeframe with StrategyRouter
    // ------------------------------------------------------
    let mut bt = Backtester::default();
    bt.initial_equity = 1_000.0;
    bt.max_drawdown_allowed = 0.30;

    let results = bt.run(&one_h);

    println!("\n=========== BACKTEST SUMMARY ===========");
    println!("Initial equity: {:.2}", bt.initial_equity);
    println!("Final equity:   {:.2}", results.final_equity);
    println!("Total return:   {:.2}%", results.total_return * 100.0);
    println!("Trades:         {}", results.trades.len());
    println!("Max drawdown:   {:.2}%", results.max_drawdown * 100.0);
    println!("Sharpe (ann.):  {:.2}", results.sharpe);
    println!("========================================\n");

    // ------------------------------------------------------
    // 3) Multi-timeframe forecast engine for most recent bar
    // ------------------------------------------------------
    //
    // We:
    //  - Do the full ML+SABR+MC pipeline on 1H (as before).
    //  - In parallel, compute drift/vol summaries for 1D, 4H, and 15m.
    //  - Fuse all 4 TF signals into a single drift + sigma.
    //  - Run Monte Carlo on the 1H price path using the fused parameters.
    //
    let bars_1h = &one_h;
    let last_close = bars_1h.last().unwrap().close;

    // ---------- 1H: full ML pipeline (single-threaded here) ----------
    let returns_1h = compute_log_returns(bars_1h);
    let garch_vol_1h = compute_garch_like_vol(&returns_1h);

    let (features_x, targets_y) = features::build_ml_dataset(bars_1h, &returns_1h);
    if features_x.is_empty() {
        eprintln!("Not enough data to build ML dataset.");
        std::process::exit(1);
    }

    let (beta_1h, ml_resid_std_1h) = ml::fit_linear_regression_ensemble(
        &features_x,
        &targets_y,
        1e-4,
        64,
        0.7,
    );

    let last_x = features_x.last().unwrap();
    let ml_pred_ret_1h = ml::dot(last_x, &beta_1h);
    let ml_pred_vol_1h = if ml_resid_std_1h > 0.0 {
        ml_resid_std_1h
    } else {
        0.01
    };

    let regime_1h = classify_regime_ml(ml_pred_ret_1h, ml_pred_vol_1h, &returns_1h);

    println!(
        "Regime (1H, ML): {:?} | pred_ret {:.5} | pred_vol {:.5}",
        regime_1h, ml_pred_ret_1h, ml_pred_vol_1h
    );

    let (drift_1h, base_sigma_1h) = compute_drift_and_vol_scale(
        bars_1h,
        &returns_1h,
        &garch_vol_1h,
        regime_1h,
        ml_pred_ret_1h,
        ml_pred_vol_1h,
    );

    println!("1H drift = {:.6}", drift_1h);
    println!("1H base sigma = {:.6}", base_sigma_1h);

    // ---------- Multi-timeframe summaries (parallel over 1D, 4H, 15m) ----------
    //
    // We use 4 threads total:
    //  - main thread: 1H ML (already done)
    //  - 3 worker tasks: 1D, 4H, 15m summaries
    //
    let (tf_1d, (tf_4h, tf_15m)) = join(
        || compute_tf_summary("1D", &one_d),
        || {
            join(
                || compute_tf_summary("4H", &four_h),
                || compute_tf_summary("15m", &f15),
            )
        },
    );

    println!(
        "TF summary 1D : drift = {:.6}, sigma = {:.6}",
        tf_1d.drift, tf_1d.sigma
    );
    println!(
        "TF summary 4H : drift = {:.6}, sigma = {:.6}",
        tf_4h.drift, tf_4h.sigma
    );
    println!(
        "TF summary 15m: drift = {:.6}, sigma = {:.6}",
        tf_15m.drift, tf_15m.sigma
    );

    // Treat the 1H ML-based view as another timeframe summary.
    let tf_1h = TFSummary {
        name: "1H",
        drift: drift_1h,
        sigma: base_sigma_1h.max(0.005),
    };

    // ---------- Fuse the 4 timeframes ----------
    //
    // Weighting:
    //  - 1D  : slow macro drift anchor
    //  - 4H  : swing trend
    //  - 1H  : main trading TF (ML)
    //  - 15m : microstructure / short-term flow
    //
    let w_1d = 0.35;
    let w_4h = 0.30;
    let w_1h = 0.25;
    let w_15m = 0.10;

    let drift_fused =
        w_1d * tf_1d.drift +
        w_4h * tf_4h.drift +
        w_1h * tf_1h.drift +
        w_15m * tf_15m.drift;

    let sigma_fused_raw =
        w_1d * tf_1d.sigma +
        w_4h * tf_4h.sigma +
        w_1h * tf_1h.sigma +
        w_15m * tf_15m.sigma;

    // Keep volatility in a realistic band for intraday SOL.
    let sigma_fused = sigma_fused_raw.clamp(0.0075, 0.05);

    println!("\n===== MULTI-TF FUSED SIGNAL =====");
    println!("Fused drift = {:.6}", drift_fused);
    println!("Fused sigma = {:.6}", sigma_fused);
    println!("(1D, 4H, 1H, 15m all contributing)");
    println!("=================================\n");

    // ---------- SABR curvature with fused sigma ----------
    let sabr_params = estimate_sabr_params(&returns_1h, sigma_fused, last_close);
    println!("SABR (fused) = {:?}\n", sabr_params);

    // ------------------------------------------------------
    // 4) Monte Carlo heatmap using fused drift & sigma
    // ------------------------------------------------------
    let sim = simulate_heatmap(
        bars_1h,
        drift_fused,
        sigma_fused,
        &sabr_params,
        50,      // horizon
        80,      // buckets
        3_000,   // paths
    );

    print_bucket_summary(&sim.heatmap, sim.min_price, sim.max_price, bars_1h);
    save_heatmap_png(&sim.heatmap, sim.min_price, sim.max_price, output_png)?;

    println!("Saved heatmap to {}", output_png);

    Ok(())
}
