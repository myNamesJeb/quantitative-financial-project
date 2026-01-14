
// src/backtest.rs

use crate::data::MarketBar;
use crate::features::{
    build_ml_dataset,
    compute_garch_like_vol,
    compute_log_returns,
    mean,
    mean_std,
};
use crate::ml::{
    classify_regime_ml,
    fit_linear_regression_ensemble,
    dot,
    fit_mlp,
    fit_stump,
    predict_mlp,
    predict_stump,
};
use crate::drift::compute_drift_and_vol_scale;
use crate::sabr::estimate_sabr_params;
use crate::simulate::{price_to_bucket, simulate_heatmap};
use crate::strategy::{Context, ForecastSnapshot, StrategyRouter};
use crate::viz::print_bucket_summary;

#[derive(Debug)]
pub struct Trade {
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64, // fraction of capital, with sign (long +, short -)
    pub pnl: f64,
}

#[derive(Debug)]
pub struct BacktestResult {
    pub final_equity: f64,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
    pub max_drawdown: f64,
    pub annualized_return: f64,
    pub total_return: f64,
    pub winrate: f64,
    pub sharpe: f64,
    pub accuracy: AccuracyReport,
    pub stress_dd_5: f64,
    pub stress_dd_10: f64,
}

#[derive(Debug, Clone)]
struct ForecastDist {
    row: Vec<f64>,
    min_price: f64,
    max_price: f64,
}

#[derive(Debug, Copy, Clone)]
struct ExecutionEngine {
    fee_bps: f64,
    spread_bps: f64,
    slippage_bps: f64,
    impact_power: f64,
    participation_rate: f64,
    queue_penalty: f64,
    latency_bars: usize,
}

#[derive(Debug, Copy, Clone)]
enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AccuracyMetric {
    Brier,
    LogLoss,
    Quantile,
}

#[derive(Debug, Clone)]
pub struct AccuracyReport {
    pub horizon: usize,
    pub window: usize,
    pub brier_scores: Vec<f64>,
    pub log_losses: Vec<f64>,
    pub quantile_errors: Vec<f64>,
    pub rolling_brier: Vec<f64>,
    pub rolling_log: Vec<f64>,
    pub rolling_quantile: Vec<f64>,
    pub rolling_best: Vec<AccuracyMetric>,
    pub avg_brier: f64,
    pub avg_log_loss: f64,
    pub avg_quantile_error: f64,
    pub best_metric: AccuracyMetric,
}

pub struct Backtester {
    pub initial_equity: f64,
    pub max_drawdown_allowed: f64, // e.g. 0.20
    pub router: StrategyRouter,
    execution: ExecutionEngine,
    pub lookback_days: usize,
    pub forecast_horizon: usize,
    pub buckets: usize,
    pub paths: usize,
    pub accuracy_horizon: usize,
    pub cal_window: usize,
    pub cal_min_samples: usize,
    pub embargo_bars: usize,
}

impl Default for Backtester {
    fn default() -> Self {
        Self {
            initial_equity: 1.0,
            max_drawdown_allowed: 0.20,
            router: StrategyRouter::default(),
            execution: ExecutionEngine {
                fee_bps: 6.0,
                spread_bps: 6.0,
                slippage_bps: 4.0,
                impact_power: 0.6,
                participation_rate: 0.10,
                queue_penalty: 0.10,
                latency_bars: 0,
            },
            lookback_days: 30,
            forecast_horizon: 50,
            buckets: 80,
            paths: 3000,
            accuracy_horizon: 6,
            cal_window: 500,
            cal_min_samples: 50,
            embargo_bars: 2,
        }
    }
}

impl ExecutionEngine {
    fn apply_config(&mut self, cfg: &crate::config::ExecutionConfig) {
        self.fee_bps = cfg.fee_bps;
        self.spread_bps = cfg.spread_bps;
        self.slippage_bps = cfg.slippage_bps;
        self.impact_power = cfg.impact_power;
        self.participation_rate = cfg.participation_rate;
        self.queue_penalty = cfg.queue_penalty;
        self.latency_bars = cfg.latency_bars;
    }

    fn exec_price(&self, mid: f64, side: Side) -> f64 {
        let adj = (self.spread_bps * 0.5 + self.slippage_bps) / 10_000.0;
        match side {
            Side::Buy => mid * (1.0 + adj),
            Side::Sell => mid * (1.0 - adj),
        }
    }

    fn exec_fill(
        &self,
        mid: f64,
        side: Side,
        notional: f64,
        avg_vol: f64,
    ) -> (f64, f64) {
        let vol = avg_vol.max(1e-6);
        let impact_scale = (notional / vol).max(0.0).powf(self.impact_power);
        let slip_bps = self.slippage_bps * impact_scale;
        let adj = (self.spread_bps * 0.5 + slip_bps) / 10_000.0;

        let mut fill_ratio = 1.0;
        let capacity = vol * self.participation_rate;
        if capacity > 0.0 && notional > 0.0 {
            fill_ratio = (capacity / notional).clamp(0.0, 1.0);
        }
        fill_ratio *= (1.0 - self.queue_penalty).clamp(0.0, 1.0);

        let exec_price = match side {
            Side::Buy => mid * (1.0 + adj),
            Side::Sell => mid * (1.0 - adj),
        };

        (exec_price, fill_ratio.clamp(0.0, 1.0))
    }

    fn fee(&self, notional: f64) -> f64 {
        notional.abs() * (self.fee_bps / 10_000.0)
    }
}

/* ============================
   Helper structs for TF stats
   ============================ */

struct RetStats {
    rets: Vec<f64>,
    cum: Vec<f64>,
    cum2: Vec<f64>,
}

fn build_ret_stats(bars: &[MarketBar]) -> RetStats {
    let rets = compute_log_returns(bars);
    let mut cum = Vec::with_capacity(rets.len() + 1);
    let mut cum2 = Vec::with_capacity(rets.len() + 1);

    cum.push(0.0);
    cum2.push(0.0);

    let mut s = 0.0;
    let mut s2 = 0.0;

    for &r in &rets {
        s += r;
        s2 += r * r;
        cum.push(s);
        cum2.push(s2);
    }

    RetStats { rets, cum, cum2 }
}

/// Compute mean/std of returns in a trailing window up to `last_ret_idx`.
fn window_mean_std(stats: &RetStats, last_ret_idx: usize, window: usize) -> (f64, f64) {
    let nrets = stats.rets.len();
    if nrets == 0 {
        return (0.0, 0.0);
    }

    let end = (last_ret_idx + 1).min(nrets);
    if end == 0 {
        return (0.0, 0.0);
    }

    let start = end.saturating_sub(window);
    let len = end - start;
    if len == 0 {
        return (0.0, 0.0);
    }

    let sum = stats.cum[end] - stats.cum[start];
    let sum2 = stats.cum2[end] - stats.cum2[start];

    let mean = sum / (len as f64);
    let var = (sum2 / (len as f64)) - mean * mean;

    (mean, var.max(0.0).sqrt())
}

/// For each 1H bar, find the index of the most recent bar in `tf`
/// with `unix <= 1h_time`. If none, store None.
fn build_index_map(base_1h: &[MarketBar], tf: &[MarketBar]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(base_1h.len());

    if tf.is_empty() {
        out.resize(base_1h.len(), None);
        return out;
    }

    for b in base_1h {
        let t = b.unix;

        // upper_bound: first index with tf[idx].unix > t
        let mut lo = 0usize;
        let mut hi = tf.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if tf[mid].unix <= t {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo == 0 {
            out.push(None);
        } else {
            out.push(Some(lo - 1));
        }
    }

    out
}

/// Compute a simple (drift, sigma) for a TF aligned at 1H index `i`.
/// - Uses the mapped TF bar index from `map_tf`.
/// - Uses trailing `window` returns.
fn multi_tf_drift_sigma(
    stats: &RetStats,
    map_tf: &[Option<usize>],
    idx_1h: usize,
    window: usize,
) -> (f64, f64) {
    if stats.rets.is_empty() || idx_1h >= map_tf.len() {
        return (0.0, 0.01);
    }

    let bar_idx = match map_tf[idx_1h] {
        Some(j) => j,
        None => return (0.0, 0.01),
    };

    // Need at least 1 return: bar_idx >= 1 → last_ret_idx = bar_idx - 1.
    if bar_idx == 0 {
        return (0.0, 0.01);
    }

    let last_ret_idx = (bar_idx - 1).min(stats.rets.len().saturating_sub(1));
    let (mu, sig) = window_mean_std(stats, last_ret_idx, window);

    let sig = sig.max(0.005);
    (mu, sig)
}

impl Backtester {
    pub fn apply_config(&mut self, cfg: &crate::config::RunConfig) {
        self.lookback_days = cfg.lookback_days;
        self.forecast_horizon = cfg.forecast_horizon;
        self.buckets = cfg.buckets;
        self.paths = cfg.paths;
        self.accuracy_horizon = cfg.accuracy_horizon;
        self.cal_window = cfg.cal_window;
        self.cal_min_samples = cfg.cal_min_samples;
        self.embargo_bars = cfg.embargo_bars;
        self.execution.apply_config(&cfg.execution);
    }
    /// Multi-timeframe backtest:
    /// - Trading resolution: 1H bars
    /// - Context uses 1D, 4H, and 15m drift/vol summaries aligned by time.
    pub fn run(
        &mut self,
        all_1h: &[MarketBar],
        all_1d: &[MarketBar],
        all_4h: &[MarketBar],
        all_15m: &[MarketBar],
    ) -> BacktestResult {
        if all_1h.len() < 500 {
            panic!("Need at least 500 1H bars for a meaningful backtest.");
        }

        // === Only use the last N days of 1H data (tweak as you like) ===
        let lookback = 24 * self.lookback_days;
        let bars: &[MarketBar] = if all_1h.len() > lookback {
            &all_1h[all_1h.len() - lookback..]
        } else {
            all_1h
        };

        let n = bars.len();

        let mut cum_vol = Vec::with_capacity(n + 1);
        cum_vol.push(0.0);
        for b in bars {
            let v = b.volume.max(0.0);
            cum_vol.push(cum_vol.last().unwrap() + v);
        }

        let avg_vol_at = |idx: usize, window: usize| -> f64 {
            if n == 0 {
                return 0.0;
            }
            let end = idx + 1;
            let start = end.saturating_sub(window);
            let len = (end - start).max(1) as f64;
            (cum_vol[end] - cum_vol[start]) / len
        };

        let mut equity = self.initial_equity;
        let mut peak_equity = equity;
        let mut equity_curve = Vec::with_capacity(n);

        let mut current_pos: f64 = 0.0; // fraction of equity in asset, long (+) or short (-)
        let mut entry_price: f64 = 0.0;
        let mut entry_equity: f64 = equity;
        let mut entry_fee: f64 = 0.0;

        let mut trades: Vec<Trade> = Vec::new();
        let mut trading_enabled = true;

        let mut prev_forecast: Option<ForecastSnapshot> = None;

        // Accuracy tracking configuration.
        let acc_horizon = self.accuracy_horizon;
        let acc_window = 200usize;
        let acc_buckets = self.buckets;
        let acc_paths = self.paths;
        let forecast_horizon = self.forecast_horizon;
        let cal_window = self.cal_window;
        let cal_min_samples = self.cal_min_samples;

        let mut brier_scores = Vec::new();
        let mut log_losses = Vec::new();
        let mut quantile_errors = Vec::new();
        let mut rolling_brier = Vec::new();
        let mut rolling_log = Vec::new();
        let mut rolling_quantile = Vec::new();
        let mut rolling_best = Vec::new();

        let mut pit_history_1 = Vec::new();
        let mut pit_history_6 = Vec::new();

        // Warmup: need enough history for ML/GARCH/etc.
        let start_idx = 500;

        // Precompute TF stats + alignment maps (1D / 4H / 15m).
        let stats_1d = build_ret_stats(all_1d);
        let stats_4h = build_ret_stats(all_4h);
        let stats_15m = build_ret_stats(all_15m);

        let map_1d = build_index_map(bars, all_1d);
        let map_4h = build_index_map(bars, all_4h);
        let map_15m = build_index_map(bars, all_15m);

        for i in 0..n {
            let price = bars[i].close;
            let exec_idx = (i + self.execution.latency_bars).min(n - 1);
            let exec_price_mid = bars[exec_idx].close;
            let avg_vol = avg_vol_at(exec_idx, 50);

            // Mark-to-market equity if we hold a position.
            if current_pos != 0.0 && entry_price > 0.0 {
                let ret = price / entry_price - 1.0;
                let pos_value = current_pos * ret;
                equity = (entry_equity * (1.0 + pos_value)).max(0.0);
            }

            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd = 1.0 - equity / peak_equity;
            equity_curve.push(equity);

            // Enforce max drawdown: once breached, close and stop trading.
            if dd >= self.max_drawdown_allowed && trading_enabled {
                if current_pos != 0.0 {
                    let side_exit = if current_pos > 0.0 { Side::Sell } else { Side::Buy };
                    let notional = entry_equity * current_pos.abs();
                    let (exit_price, fill_ratio) =
                        self.execution.exec_fill(exec_price_mid, side_exit, notional, avg_vol);
                    let closed_frac = fill_ratio.clamp(0.0, 1.0);
                    let pnl =
                        current_pos * closed_frac * (exit_price / entry_price - 1.0) * entry_equity;
                    let fee = self.execution.fee(notional * closed_frac);
                    let entry_fee_used = entry_fee * closed_frac;
                    trades.push(Trade {
                        entry_price,
                        exit_price,
                        size: current_pos * closed_frac,
                        pnl: pnl - fee - entry_fee_used,
                    });
                    equity = (entry_equity + pnl - fee - entry_fee_used).max(0.0);
                    entry_equity = (entry_equity * (1.0 - closed_frac)).max(0.0);
                    entry_fee = (entry_fee * (1.0 - closed_frac)).max(0.0);
                    current_pos *= 1.0 - closed_frac;
                    if current_pos.abs() < 1e-6 {
                        current_pos = 0.0;
                        entry_fee = 0.0;
                    }
                }
                trading_enabled = false;
            }

            // If not enough history yet, skip signal generation.
            if i < start_idx || !trading_enabled {
                continue;
            }

            // === Build forecast from 1H history up to i ===
            let hist = &bars[..=i];
            let returns_full = compute_log_returns(hist);
            let garch_vol = compute_garch_like_vol(&returns_full);

            let train_end = hist.len().saturating_sub(self.embargo_bars);
            if train_end < 20 {
                continue;
            }
            let train_hist = &hist[..train_end];
            let returns_train = compute_log_returns(train_hist);

            let (features, targets) = build_ml_dataset(train_hist, &returns_train);
            if features.is_empty() {
                continue;
            }

            let (beta, ml_resid_std) = fit_linear_regression_ensemble(
                &features,
                &targets,
                1e-4,
                64,
                0.7,
            );

            let last_features = features.last().unwrap().clone();
            let lin_pred = dot(&last_features, &beta);
            let stump = fit_stump(&features, &targets);
            let stump_pred = stump
                .as_ref()
                .map(|s| predict_stump(s, &last_features))
                .unwrap_or(0.0);
            let mlp = fit_mlp(&features, &targets, 8, 3, 0.005);
            let mlp_pred = mlp
                .as_ref()
                .map(|m| predict_mlp(m, &last_features))
                .unwrap_or(0.0);

            let ml_pred_ret = 0.6 * lin_pred + 0.25 * stump_pred + 0.15 * mlp_pred;
            let ml_pred_vol = if ml_resid_std > 0.0 { ml_resid_std } else { 0.01 };

            let ml_regime = classify_regime_ml(ml_pred_ret, ml_pred_vol, &returns_full);

            let (drift_1h, base_sigma_1h) = compute_drift_and_vol_scale(
                hist,
                &returns_full,
                &garch_vol,
                ml_regime,
                ml_pred_ret,
                ml_pred_vol,
            );

            // === Multi-timeframe drift/sigma summaries at this 1H bar ===
            let (d1d, s1d) = multi_tf_drift_sigma(&stats_1d, &map_1d, i, 60);
            let (d4h, s4h) = multi_tf_drift_sigma(&stats_4h, &map_4h, i, 120);
            let (d15, s15) = multi_tf_drift_sigma(&stats_15m, &map_15m, i, 240);

            // Fuse drift/sigma like the final run.
            let w_1d = 0.35;
            let w_4h = 0.30;
            let w_1h = 0.25;
            let w_15m = 0.10;

            let drift_fused = w_1d * d1d + w_4h * d4h + w_1h * drift_1h + w_15m * d15;
            let sigma_fused_raw =
                w_1d * s1d + w_4h * s4h + w_1h * base_sigma_1h + w_15m * s15;
            let sigma_fused = sigma_fused_raw.clamp(0.0075, 0.05);

            let sabr = estimate_sabr_params(&returns_full, sigma_fused, price);
            let sim = simulate_heatmap(
                hist,
                drift_fused,
                sigma_fused,
                &sabr,
                forecast_horizon.max(acc_horizon),
                acc_buckets,
                acc_paths,
            );

            if sim.heatmap.is_empty() {
                continue;
            }

            let dist_raw = ForecastDist {
                row: sim.heatmap[0].clone(),
                min_price: sim.min_price,
                max_price: sim.max_price,
            };

            let row_1 = if pit_history_1.len() >= cal_min_samples {
                calibrate_row(&dist_raw.row, &pit_history_1)
            } else {
                dist_raw.row.clone()
            };
            let forecast = snapshot_from_row(&row_1, dist_raw.min_price, dist_raw.max_price);

            // === Per-step forecast report (same style as final run) ===
            println!(
                "Regime (1H, ML): {:?} | pred_ret {:.5} | pred_vol {:.5}",
                ml_regime, ml_pred_ret, ml_pred_vol
            );
            println!("1H drift = {:.6}", drift_1h);
            println!("1H base sigma = {:.6}", base_sigma_1h);
            println!(
                "TF summary 1D : drift = {:.6}, sigma = {:.6}",
                d1d, s1d
            );
            println!(
                "TF summary 4H : drift = {:.6}, sigma = {:.6}",
                d4h, s4h
            );
            println!(
                "TF summary 15m: drift = {:.6}, sigma = {:.6}",
                d15, s15
            );
            println!("\n===== MULTI-TF FUSED SIGNAL =====");
            println!("Fused drift = {:.6}", drift_fused);
            println!("Fused sigma = {:.6}", sigma_fused);
            println!("(1D, 4H, 1H, 15m all contributing)");
            println!("=================================\n");
            println!("SABR (fused) = {:?}\n", sabr);

            let horizon_end = (i + forecast_horizon).min(n - 1);
            let mut actual_low = f64::INFINITY;
            let mut actual_high = f64::NEG_INFINITY;
            for b in &bars[(i + 1)..=horizon_end] {
                if b.low < actual_low {
                    actual_low = b.low;
                }
                if b.high > actual_high {
                    actual_high = b.high;
                }
            }
            if actual_low.is_finite() && actual_high.is_finite() {
                println!(
                    "Actual range (next {} bars): {:.4} → {:.4}",
                    horizon_end - i,
                    actual_low,
                    actual_high
                );
            }

            print_bucket_summary(&sim.heatmap, sim.min_price, sim.max_price, hist);

            let ctx = Context {
                bar: &bars[i],
                idx: i,
                forecast,
                drift: drift_fused,
                base_sigma: sigma_fused,
                ml_regime,
                last_price: price,
                drift_1d: d1d,
                sigma_1d: s1d,
                drift_4h: d4h,
                sigma_4h: s4h,
                drift_15m: d15,
                sigma_15m: s15,
            };

            let target = self
                .router
                .on_bar(&ctx, prev_forecast.as_ref(), current_pos);

            let new_pos = target.target_fraction.clamp(-1.0, 1.0);

            // If position changes, treat it as closing previous and opening new.
            if (new_pos - current_pos).abs() > 1e-6 {
                if current_pos != 0.0 && entry_price > 0.0 {
                    let side_exit = if current_pos > 0.0 { Side::Sell } else { Side::Buy };
                    let notional = entry_equity * current_pos.abs();
                    let (exit_price, fill_ratio) =
                        self.execution.exec_fill(exec_price_mid, side_exit, notional, avg_vol);
                    let closed_frac = fill_ratio.clamp(0.0, 1.0);
                    let pnl =
                        current_pos * closed_frac * (exit_price / entry_price - 1.0) * entry_equity;
                    let fee = self.execution.fee(notional * closed_frac);
                    let entry_fee_used = entry_fee * closed_frac;
                    trades.push(Trade {
                        entry_price,
                        exit_price,
                        size: current_pos * closed_frac,
                        pnl: pnl - fee - entry_fee_used,
                    });
                    equity = (entry_equity + pnl - fee - entry_fee_used).max(0.0);
                    if equity > peak_equity {
                        peak_equity = equity;
                    }
                    entry_equity = (entry_equity * (1.0 - closed_frac)).max(0.0);
                    entry_fee = (entry_fee * (1.0 - closed_frac)).max(0.0);
                    current_pos *= 1.0 - closed_frac;
                    if current_pos.abs() < 1e-6 {
                        current_pos = 0.0;
                        entry_fee = 0.0;
                    }
                }

                if new_pos.abs() > 1e-6 {
                    let side_entry = if new_pos > 0.0 { Side::Buy } else { Side::Sell };
                    let desired_notional = equity * new_pos.abs();
                    let (entry_px, fill_ratio) =
                        self.execution.exec_fill(exec_price_mid, side_entry, desired_notional, avg_vol);
                    let fill_ratio = fill_ratio.clamp(0.0, 1.0);
                    let filled_pos = new_pos * fill_ratio;
                    if filled_pos.abs() > 1e-6 {
                        entry_price = entry_px;
                        let notional = equity * filled_pos.abs();
                        let fee = self.execution.fee(notional);
                        equity = (equity - fee).max(0.0);
                        entry_equity = equity;
                        entry_fee = fee;
                        current_pos = filled_pos;
                    }
                } else {
                    current_pos = 0.0;
                }
            }

            prev_forecast = Some(forecast);

            if i + 1 < n {
                let realized_1 = bars[i + 1].close;
                let pit = pit_value(
                    &dist_raw.row,
                    dist_raw.min_price,
                    dist_raw.max_price,
                    realized_1,
                );
                pit_history_1.push(pit);
                if pit_history_1.len() > cal_window {
                    pit_history_1.remove(0);
                }
            }

            // === Accuracy scoring on 6-bar projection ===
            if i + acc_horizon < n && acc_horizon <= sim.heatmap.len() {
                let realized = bars[i + acc_horizon].close;
                let row_raw = &sim.heatmap[acc_horizon - 1];
                if !row_raw.is_empty() {
                    let row = if pit_history_6.len() >= cal_min_samples {
                        calibrate_row(row_raw, &pit_history_6)
                    } else {
                        row_raw.clone()
                    };

                    let b = price_to_bucket(realized, sim.min_price, sim.max_price, acc_buckets);
                    let p_real = row[b].clamp(1e-12, 1.0);

                    // Brier score for multi-class (one-hot) outcome.
                    let mut sum_sq = 0.0;
                    for (idx, &p) in row.iter().enumerate() {
                        let o = if idx == b { 1.0 } else { 0.0 };
                        let d = p - o;
                        sum_sq += d * d;
                    }
                    brier_scores.push(sum_sq);

                    // Log loss for realized bucket.
                    log_losses.push(-p_real.ln());

                    // Quantile error: closeness of realized price to median.
                    let mut cdf = 0.0;
                    for (idx, &p) in row.iter().enumerate() {
                        cdf += p;
                        if idx == b {
                            break;
                        }
                    }
                    let q_err = (cdf - 0.5).abs();
                    quantile_errors.push(q_err);

                    let rb = rolling_avg(&brier_scores, acc_window);
                    let rl = rolling_avg(&log_losses, acc_window);
                    let rq = rolling_avg(&quantile_errors, acc_window);
                    rolling_brier.push(rb);
                    rolling_log.push(rl);
                    rolling_quantile.push(rq);

                    let best = best_metric(rb, rl, rq);
                    rolling_best.push(best);

                    let pit = pit_value(
                        row_raw,
                        sim.min_price,
                        sim.max_price,
                        realized,
                    );
                    pit_history_6.push(pit);
                    if pit_history_6.len() > cal_window {
                        pit_history_6.remove(0);
                    }
                }
            }
        }

        // Close any open position at final bar.
        if current_pos != 0.0 {
            let last_price = bars.last().unwrap().close;
            let side_exit = if current_pos > 0.0 { Side::Sell } else { Side::Buy };
            let notional = entry_equity * current_pos.abs();
            let avg_vol = avg_vol_at(n - 1, 50);
            let (exit_price, fill_ratio) =
                self.execution.exec_fill(last_price, side_exit, notional, avg_vol);
            let closed_frac = fill_ratio.clamp(0.0, 1.0);
            let pnl =
                current_pos * closed_frac * (exit_price / entry_price - 1.0) * entry_equity;
            let fee = self.execution.fee(notional * closed_frac);
            let entry_fee_used = entry_fee * closed_frac;
            trades.push(Trade {
                entry_price,
                exit_price,
                size: current_pos * closed_frac,
                pnl: pnl - fee - entry_fee_used,
            });
            equity = (entry_equity + pnl - fee - entry_fee_used).max(0.0);
            equity_curve.push(equity);
            entry_fee = 0.0;
        }

        let final_equity = equity;
        let total_return = final_equity / self.initial_equity - 1.0;

        // Assume 1H bars, ~24 * 365 bars per year.
        let years = (n as f64) / (24.0 * 365.0);
        let annualized_return = if years > 0.0 {
            (final_equity / self.initial_equity).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Compute max drawdown properly from equity_curve.
        let max_drawdown = compute_max_drawdown(&equity_curve);
        let stress_dd_5 = compute_stress_drawdown(&equity_curve, 0.05);
        let stress_dd_10 = compute_stress_drawdown(&equity_curve, 0.10);

        let (winrate, sharpe) = compute_stats(&equity_curve, &trades, years);

        let avg_brier = mean(&brier_scores);
        let avg_log_loss = mean(&log_losses);
        let avg_quantile_error = mean(&quantile_errors);
        let best_metric = best_metric(avg_brier, avg_log_loss, avg_quantile_error);

        let accuracy = AccuracyReport {
            horizon: acc_horizon,
            window: acc_window,
            brier_scores,
            log_losses,
            quantile_errors,
            rolling_brier,
            rolling_log,
            rolling_quantile,
            rolling_best,
            avg_brier,
            avg_log_loss,
            avg_quantile_error,
            best_metric,
        };

        BacktestResult {
            final_equity,
            equity_curve,
            trades,
            max_drawdown,
            annualized_return,
            total_return,
            winrate,
            sharpe,
            accuracy,
            stress_dd_5,
            stress_dd_10,
        }
    }
}

/// Summarize one discrete distribution over price grid.
fn snapshot_from_row(
    row: &[f64],
    min_p: f64,
    max_p: f64,
) -> ForecastSnapshot {
    let buckets = row.len();
    if buckets == 0 {
        return ForecastSnapshot {
            mean: 0.0,
            std: 0.0,
            skew: 0.0,
            kurtosis: 0.0,
            mode: 0.0,
            band_lower: 0.0,
            band_upper: 0.0,
            entropy: 0.0,
            confidence: 0.0,
        };
    }

    let width = (max_p - min_p) / buckets as f64;

    // mean
    let mut mean = 0.0;
    for (i, &p) in row.iter().enumerate() {
        let price = min_p + (i as f64 + 0.5) * width;
        mean += price * p;
    }

    // central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for (i, &p) in row.iter().enumerate() {
        let price = min_p + (i as f64 + 0.5) * width;
        let d = price - mean;
        m2 += p * d.powi(2);
        m3 += p * d.powi(3);
        m4 += p * d.powi(4);
    }

    let std = m2.sqrt();
    let skew = if std > 0.0 { m3 / std.powi(3) } else { 0.0 };
    let kurt = if std > 0.0 { m4 / std.powi(4) } else { 0.0 };

    // mode
    let (mode_idx, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let mode = min_p + (mode_idx as f64 + 0.5) * width;

    let entropy = distribution_entropy(row);
    let confidence = (1.0 - entropy).clamp(0.0, 1.0);

    let band_mult = 1.0;
    let band_lower = mean - band_mult * std;
    let band_upper = mean + band_mult * std;

    ForecastSnapshot {
        mean,
        std,
        skew,
        kurtosis: kurt,
        mode,
        band_lower,
        band_upper,
        entropy,
        confidence,
    }
}

fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::MIN;
    let mut max_dd = 0.0;
    for &e in equity_curve {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 {
            let dd = 1.0 - e / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

fn compute_stress_drawdown(equity_curve: &[f64], shock: f64) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    let mut shocked = Vec::with_capacity(equity_curve.len());
    for &e in equity_curve {
        shocked.push(e * (1.0 - shock).max(0.0));
    }
    compute_max_drawdown(&shocked)
}

fn distribution_entropy(row: &[f64]) -> f64 {
    if row.is_empty() {
        return 0.0;
    }
    let sum: f64 = row.iter().cloned().filter(|x| x.is_finite() && *x > 0.0).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0;
    for &p in row {
        if p.is_finite() && p > 0.0 {
            let q = p / sum;
            ent -= q * q.ln();
        }
    }
    let max_ent = (row.len() as f64).ln().max(1e-12);
    (ent / max_ent).clamp(0.0, 1.0)
}

fn pit_value(row: &[f64], min_p: f64, max_p: f64, price: f64) -> f64 {
    if row.is_empty() {
        return 0.5;
    }
    let sum: f64 = row.iter().cloned().filter(|x| x.is_finite() && *x >= 0.0).sum();
    if sum <= 0.0 {
        return 0.5;
    }
    let b = price_to_bucket(price, min_p, max_p, row.len());
    let mut cdf = 0.0;
    for (idx, &p) in row.iter().enumerate() {
        if p.is_finite() && p >= 0.0 {
            cdf += p / sum;
        }
        if idx == b {
            break;
        }
    }
    cdf.clamp(0.0, 1.0)
}

fn empirical_cdf(sorted: &[f64], x: f64) -> f64 {
    if sorted.is_empty() {
        return x.clamp(0.0, 1.0);
    }
    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    (lo as f64 / sorted.len() as f64).clamp(0.0, 1.0)
}

fn calibrate_row(row: &[f64], pit_history: &[f64]) -> Vec<f64> {
    if row.is_empty() || pit_history.is_empty() {
        return row.to_vec();
    }
    let sum: f64 = row.iter().cloned().filter(|x| x.is_finite() && *x >= 0.0).sum();
    if sum <= 0.0 {
        return row.to_vec();
    }

    let mut sorted = pit_history
        .iter()
        .cloned()
        .map(|x| x.clamp(0.0, 1.0))
        .collect::<Vec<f64>>();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut out = vec![0.0; row.len()];
    let mut cdf = 0.0;
    let mut prev = 0.0;

    for (i, &p) in row.iter().enumerate() {
        if p.is_finite() && p >= 0.0 {
            cdf += p / sum;
        }
        let cal = empirical_cdf(&sorted, cdf);
        let prob = (cal - prev).max(0.0);
        out[i] = prob;
        prev = cal;
    }

    let s: f64 = out.iter().cloned().sum();
    if s > 0.0 {
        for v in out.iter_mut() {
            *v /= s;
        }
    }

    out
}

fn rolling_avg(vals: &[f64], window: usize) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    if window == 0 || vals.len() < window {
        return mean(vals);
    }
    let start = vals.len() - window;
    mean(&vals[start..])
}

fn best_metric(brier: f64, log_loss: f64, quantile_err: f64) -> AccuracyMetric {
    // Lower is better for all three.
    if brier <= log_loss && brier <= quantile_err {
        AccuracyMetric::Brier
    } else if log_loss <= brier && log_loss <= quantile_err {
        AccuracyMetric::LogLoss
    } else {
        AccuracyMetric::Quantile
    }
}

fn compute_stats(
    equity_curve: &[f64],
    trades: &[Trade],
    _years: f64,
) -> (f64, f64) {
    let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    let total = trades.len().max(1);
    let winrate = wins as f64 / total as f64;

    // Daily returns from equity curve (approx: 24 bars per day).
    let mut daily_rets = Vec::new();
    let mut i = 24;
    while i < equity_curve.len() {
        let prev = equity_curve[i - 24];
        let cur = equity_curve[i];
        if prev > 0.0 {
            daily_rets.push(cur / prev - 1.0);
        }
        i += 24;
    }

    let (mean_ret, std_ret) = mean_std(&daily_rets);
    let sharpe = if std_ret > 0.0 {
        mean_ret / std_ret * (365.0_f64).sqrt()
    } else {
        0.0
    };

    (winrate, sharpe)
}
