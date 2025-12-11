
// src/backtest.rs

use crate::data::MarketBar;
use crate::features::{
    build_ml_dataset,
    compute_garch_like_vol,
    compute_log_returns,
    mean_std,
};
use crate::ml::{
    classify_regime_ml,
    fit_linear_regression_ensemble,
    dot,
    MarketRegime,
};
use crate::drift::compute_drift_and_vol_scale;
use crate::sabr::estimate_sabr_params;
use crate::simulate::simulate_heatmap;
use crate::strategy::{Context, ForecastSnapshot, StrategyRouter};

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
}

pub struct Backtester {
    pub initial_equity: f64,
    pub max_drawdown_allowed: f64, // e.g. 0.20
    pub router: StrategyRouter,
}

impl Default for Backtester {
    fn default() -> Self {
        Self {
            initial_equity: 1.0,
            max_drawdown_allowed: 0.20,
            router: StrategyRouter::default(),
        }
    }
}

impl Backtester {
    /// Run on 1H bars; uses only the last ~1.5 years of data for PnL
    /// and regime inference, to keep the environment recent.
    pub fn run(&mut self, all_bars: &[MarketBar]) -> BacktestResult {
        if all_bars.len() < 500 {
            panic!("Need at least 500 bars for a meaningful backtest.");
        }

        // === Only use the last ~1.5 years of 1H data ===
        let lookback = 24 * 36; // 36 days * 24h (you can stretch this if you want)
        let bars: &[MarketBar] = if all_bars.len() > lookback {
            &all_bars[all_bars.len() - lookback..]
        } else {
            all_bars
        };

        let n = bars.len();

        let mut equity = self.initial_equity;
        let mut peak_equity = equity;
        let mut equity_curve = Vec::with_capacity(n);

        let mut current_pos: f64 = 0.0; // fraction of equity in asset, long (+) or short (-)
        let mut entry_price: f64 = 0.0;

        let mut trades: Vec<Trade> = Vec::new();
        let mut trading_enabled = true;

        let mut prev_forecast: Option<ForecastSnapshot> = None;

        // For mark-to-market we need last price when a position is open.
        let mut last_price: f64 = bars[0].close;

        // Warmup: need enough history for ML/GARCH/etc.
        let start_idx = 500.min(n.saturating_sub(2)); // avoid out-of-range

        for i in 0..n {
            let price = bars[i].close;

            // === Mark-to-market equity if we hold a position ===
            if current_pos != 0.0 && i > 0 {
                let prev_price = last_price;
                if prev_price > 0.0 {
                    let ret = price / prev_price - 1.0;
                    equity *= 1.0 + current_pos * ret;
                    equity = equity.max(0.0);
                }
            }

            last_price = price.max(1e-9);

            // Track equity + max drawdown.
            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd = if peak_equity > 0.0 {
                1.0 - equity / peak_equity
            } else {
                0.0
            };
            equity_curve.push(equity);

            // Enforce max drawdown: once breached, close and stop trading.
            if dd >= self.max_drawdown_allowed && trading_enabled {
                if current_pos != 0.0 {
                    let pnl = equity - self.initial_equity;
                    trades.push(Trade {
                        entry_price,
                        exit_price: price,
                        size: current_pos,
                        pnl,
                    });
                    current_pos = 0.0;
                }
                trading_enabled = false;
            }

            // If not enough history yet, skip signal generation (but still
            // record equity for stats).
            if i < start_idx || !trading_enabled {
                continue;
            }

            // === Build forecast from history up to i ===
            let hist = &bars[..=i];
            let (forecast, drift, base_sigma, ml_regime) =
                build_forecast_for_slice(hist);

            let ctx = Context {
                bar: &bars[i],
                idx: i,
                forecast,
                drift,
                base_sigma,
                ml_regime,
                last_price: price,
            };

            let target = self
                .router
                .on_bar(&ctx, prev_forecast.as_ref(), current_pos);

            let new_pos = target.target_fraction.clamp(-1.0, 1.0);

            // === Position management ===
            //
            // If position changes, treat it as:
            // - closing previous position (if any)
            // - opening new position (if non-zero)
            if (new_pos - current_pos).abs() > 1e-6 {
                if current_pos != 0.0 {
                    // Realized PnL of this trade is the difference between
                    // current equity and an implicit "flat" reference.
                    // For logging, approximate trade PnL based on price move
                    // from entry to current.
                    let approx_pnl = current_pos
                        * (price / entry_price - 1.0)
                        * equity.max(1e-9);

                    trades.push(Trade {
                        entry_price,
                        exit_price: price,
                        size: current_pos,
                        pnl: approx_pnl,
                    });
                }

                // Open new position or go flat.
                if new_pos.abs() > 1e-6 {
                    entry_price = price;
                    current_pos = new_pos;
                } else {
                    current_pos = 0.0;
                }
            }

            prev_forecast = Some(forecast);
        }

        // Close any open position at final bar (for logging).
        if current_pos != 0.0 {
            let last_price = bars.last().unwrap().close;
            let approx_pnl = current_pos
                * (last_price / entry_price - 1.0)
                * equity.max(1e-9);

            trades.push(Trade {
                entry_price,
                exit_price: last_price,
                size: current_pos,
                pnl: approx_pnl,
            });
            // Equity is already mark-to-market; no extra adjustment.
            equity_curve.push(equity);
        }

        let final_equity = equity;
        let total_return = if self.initial_equity > 0.0 {
            final_equity / self.initial_equity - 1.0
        } else {
            0.0
        };

        // Assume 1H bars, ~24 * 365 bars per year.
        let years = (n as f64) / (24.0 * 365.0);
        let annualized_return = if years > 0.0 && final_equity > 0.0 && self.initial_equity > 0.0 {
            (final_equity / self.initial_equity).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Compute max drawdown properly from equity_curve.
        let max_drawdown = compute_max_drawdown(&equity_curve);

        let (winrate, sharpe) = compute_stats(&equity_curve, &trades, years);

        BacktestResult {
            final_equity,
            equity_curve,
            trades,
            max_drawdown,
            annualized_return,
            total_return,
            winrate,
            sharpe,
        }
    }
}

/// Build a one-step-ahead forecast from a prefix of bars using your existing
/// machinery. This is called in a walk-forward fashion inside the backtest.
fn build_forecast_for_slice(
    hist: &[MarketBar],
) -> (ForecastSnapshot, f64, f64, MarketRegime) {
    // Returns + GARCH vol
    let returns = compute_log_returns(hist);
    let garch_vol = compute_garch_like_vol(&returns);

    // ML features / targets
    let (features, targets) = build_ml_dataset(hist, &returns);
    if features.is_empty() {
        let last_close = hist.last().unwrap().close;
        let snap = ForecastSnapshot {
            mean: last_close,
            std: 0.0,
            skew: 0.0,
            kurtosis: 3.0,
            mode: last_close,
            band_lower: last_close,
            band_upper: last_close,
        };
        return (snap, 0.0, 0.01, MarketRegime::SidewaysLowVol);
    }

    let (beta, ml_resid_std) = fit_linear_regression_ensemble(
        &features,
        &targets,
        1e-4,
        64,
        0.7,
    );

    let last_features = features.last().unwrap().clone();
    let ml_pred_ret = dot(&last_features, &beta);
    let ml_pred_vol = if ml_resid_std > 0.0 { ml_resid_std } else { 0.01 };

    let ml_regime = classify_regime_ml(ml_pred_ret, ml_pred_vol, &returns);

    let (drift, base_sigma) = compute_drift_and_vol_scale(
        hist,
        &returns,
        &garch_vol,
        ml_regime,
        ml_pred_ret,
        ml_pred_vol,
    );

    let last_close = hist.last().unwrap().close;
    let sabr = estimate_sabr_params(&returns, base_sigma, last_close);

    // Small Monte Carlo to get distribution for next step.
    // NOTE: horizon = 1 here to keep grid tight for the router.
    let horizon = 1;
    let buckets = 80;
    let paths = 3000;

    let sim = simulate_heatmap(hist, drift, base_sigma, &sabr, horizon, buckets, paths);

    let min_p = sim.min_price;
    let max_p = sim.max_price;
    let row = &sim.heatmap[0];

    let (mean_price, std_price, skew, kurt, mode_price) =
        summarize_distribution_from_row(row, min_p, max_p);

    let band_lower = mean_price - 1.96 * std_price;
    let band_upper = mean_price + 1.96 * std_price;

    let snap = ForecastSnapshot {
        mean: mean_price,
        std: std_price,
        skew,
        kurtosis: kurt,
        mode: mode_price,
        band_lower,
        band_upper,
    };

    (snap, drift, base_sigma, ml_regime)
}

/// Summarize one discrete distribution over price grid.
fn summarize_distribution_from_row(
    row: &[f64],
    min_p: f64,
    max_p: f64,
) -> (f64, f64, f64, f64, f64) {
    let buckets = row.len();
    if buckets == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let width = (max_p - min_p) / buckets as f64;

    // Mean
    let mut mean = 0.0;
    for (i, &p) in row.iter().enumerate() {
        let price = min_p + (i as f64 + 0.5) * width;
        mean += price * p;
    }

    // Central moments
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

    // Mode
    let (mode_idx, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let mode = min_p + (mode_idx as f64 + 0.5) * width;

    (mean, std, skew, kurt, mode)
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
