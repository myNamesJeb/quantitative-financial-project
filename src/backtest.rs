
// src/backtest.rs

use crate::data::MarketBar;
use crate::features::{
    build_ml_dataset, classify_regime_ml, compute_garch_like_vol, compute_log_returns,
    fit_linear_regression_ensemble, mean_std, MarketRegime,
};
use crate::drift::compute_drift_and_vol_scale;
use crate::sabr::{estimate_sabr_params, SabrParams};
use crate::simulate::{compute_price_bounds, simulate_heatmap};
use crate::strategy::{Context, ForecastSnapshot, StrategyRouter};
use crate::viz::summarize_distribution_extended;

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
    /// Run on 1H bars; you can wire in other TFs if you want later.
    pub fn run(&mut self, bars: &[MarketBar]) -> BacktestResult {
        let n = bars.len();
        if n < 500 {
            panic!("Need at least 500 bars for a meaningful backtest.");
        }

        let mut equity = self.initial_equity;
        let mut peak_equity = equity;
        let mut equity_curve = Vec::with_capacity(n);

        let mut current_pos: f64 = 0.0; // fraction of equity in asset, long (+) or short (-)
        let mut entry_price: f64 = 0.0;

        let mut trades: Vec<Trade> = Vec::new();
        let mut trading_enabled = true;

        let mut prev_forecast: Option<ForecastSnapshot> = None;

        // Warmup: need enough history for ML/GARCH/etc.
        // We'll start at, say, idx = 500.
        let start_idx = 500;

        for i in 0..n {
            let price = bars[i].close;
            // Mark-to-market equity if we hold a position:
            if current_pos != 0.0 {
                let pos_value = current_pos * (price / entry_price - 1.0);
                equity = (self.initial_equity * (1.0 + pos_value)).max(0.0);
            }

            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd = 1.0 - equity / peak_equity;
            equity_curve.push(equity);

            // Enforce max drawdown: once breached, close and stop trading.
            if dd >= self.max_drawdown_allowed && trading_enabled {
                if current_pos != 0.0 {
                    let pnl = current_pos * (price / entry_price - 1.0) * self.initial_equity;
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

            // If not enough history yet, skip signal generation.
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

            // If position changes, treat it as closing previous and opening new.
            if (new_pos - current_pos).abs() > 1e-6 {
                if current_pos != 0.0 {
                    let pnl = current_pos * (price / entry_price - 1.0) * self.initial_equity;
                    trades.push(Trade {
                        entry_price,
                        exit_price: price,
                        size: current_pos,
                        pnl,
                    });
                    equity = equity + pnl; // adjust equity explicitly
                    if equity > peak_equity {
                        peak_equity = equity;
                    }
                }

                if new_pos.abs() > 1e-6 {
                    entry_price = price;
                    current_pos = new_pos;
                } else {
                    current_pos = 0.0;
                }
            }

            prev_forecast = Some(forecast);
        }

        // Close any open position at final bar.
        if current_pos != 0.0 {
            let last_price = bars.last().unwrap().close;
            let pnl = current_pos * (last_price / entry_price - 1.0) * self.initial_equity;
            trades.push(Trade {
                entry_price,
                exit_price: last_price,
                size: current_pos,
                pnl,
            });
            equity = equity + pnl;
            equity_curve.push(equity);
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

/// Build a one-step-ahead forecast from a prefix of bars using your existing machinery.
fn build_forecast_for_slice(
    hist: &[MarketBar],
) -> (ForecastSnapshot, f64, f64, MarketRegime) {
    let returns = compute_log_returns(hist);
    let garch_vol = compute_garch_like_vol(&returns);
    let (features, targets) = crate::features::build_ml_dataset(hist, &returns);
    let (beta, ml_resid_std) = crate::features::fit_linear_regression_ensemble(
        &features,
        &targets,
        1e-4,
        64,
        0.7,
    );

    let last_features = features.last().unwrap().clone();
    let ml_pred_ret = crate::features::dot(&beta, &last_features);
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
    let horizon = 4; // a few hours ahead
    let buckets = 80;
    let paths = 3000;

    let heatmap = simulate_heatmap(hist, drift, base_sigma, &sabr, horizon, buckets, paths);

    let (min_p, max_p) = compute_price_bounds(hist, last_close);
    let row = &heatmap[0];

    let (mean_price, std_price, skew, kurt, mode_price) =
        summarize_distribution_extended(row, min_p, max_p);

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
    years: f64,
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
