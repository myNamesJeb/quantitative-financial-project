
// src/strategy.rs

use crate::data::MarketBar;
use crate::ml::MarketRegime;

/// Snapshot of the one-step-ahead distribution
/// (built in backtest.rs via build_forecast_for_slice).
#[derive(Debug, Copy, Clone)]
pub struct ForecastSnapshot {
    pub mean: f64,
    pub std: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub mode: f64,
    pub band_lower: f64,
    pub band_upper: f64,
}

/// Per-bar context passed to the router.
#[derive(Debug, Copy, Clone)]
pub struct Context<'a> {
    pub bar: &'a MarketBar,
    pub idx: usize,
    pub forecast: ForecastSnapshot,
    pub drift: f64,
    pub base_sigma: f64,
    pub ml_regime: MarketRegime,
    pub last_price: f64,
}

/// Output of the router: position as fraction of equity
/// (long > 0, short < 0, flat = 0).
#[derive(Debug, Copy, Clone)]
pub struct Target {
    pub target_fraction: f64,
}

/// High-level router that turns forecasts into a position.
/// This is where the "signals" live.
#[derive(Debug, Clone)]
pub struct StrategyRouter {
    /// Max absolute fraction of equity to risk in the asset.
    pub max_pos: f64,

    /// Sensitivity of exposure to drift Z-score.
    pub drift_sensitivity: f64,

    /// Sensitivity of exposure to mean-vs-spot shift.
    pub band_sensitivity: f64,

    /// Regime directional bias added in trending regimes.
    pub regime_bias: f64,

    /// Cooldown in bars between major position flips.
    pub cooldown_bars: usize,

    /// Min change needed before we consider it a "new" trade.
    pub min_rebalance_step: f64,

    /// Internal: last bar index where we changed position meaningfully.
    last_trade_idx: Option<usize>,
}

impl Default for StrategyRouter {
    fn default() -> Self {
        Self {
            max_pos: 0.5,              // at most 50% of equity in SOL
            drift_sensitivity: 6.0,    // how strongly drift_z moves exposure
            band_sensitivity: 2.0,     // how strongly mean-spot moves exposure
            regime_bias: 0.25,         // extra directional push in trends
            cooldown_bars: 3,          // don't flip every bar
            min_rebalance_step: 0.05,  // ignore tiny target changes
            last_trade_idx: None,
        }
    }
}

impl StrategyRouter {
    /// Main decision function.
    ///
    /// Inputs:
    /// - `ctx`: current bar context
    /// - `prev_forecast`: previous forecast snapshot (if any)
    /// - `current_pos`: current position (fraction of equity)
    pub fn on_bar(
        &mut self,
        ctx: &Context,
        prev_forecast: Option<&ForecastSnapshot>,
        current_pos: f64,
    ) -> Target {
        // If volatility estimate is totally degenerate, stay flat.
        let sigma = ctx.forecast.std.max(1e-6);
        let base_sigma = ctx.base_sigma.max(1e-6);

        // 1) Drift signal: "is the process trending on average?"
        let drift_z = ctx.drift / base_sigma;

        // 2) Mean-reversion / trend from MC mean vs current price.
        let mean_shift = (ctx.forecast.mean - ctx.last_price) / sigma;

        // 3) Expected range width in units of sigma.
        let band_width_sigma =
            (ctx.forecast.band_upper - ctx.forecast.band_lower) / (2.0 * sigma);

        // 4) Skew (tail bias) from the MC distribution.
        let skew = ctx.forecast.skew;

        // 5) Forecast slope (is the mean moving up or down over time?)
        let slope_term = if let Some(prev) = prev_forecast {
            (ctx.forecast.mean - prev.mean) / sigma
        } else {
            0.0
        };

        // ===== Raw score construction =====
        //
        // Think of this as a composite alpha:
        //   score > 0  -> want to be long
        //   score < 0  -> want to be short
        let mut score = 0.0;

        // Drift pushes us toward trend following.
        score += self.drift_sensitivity * drift_z;

        // Mean-spot shift: if mean > spot, tilt long; if mean < spot, tilt short.
        score += self.band_sensitivity * mean_shift;

        // Positive skew -> heavier right tail -> slight long bias.
        score += 0.5 * skew;

        // If band-width is very small, distribution is tight -> reduce conviction.
        if band_width_sigma < 0.8 {
            score *= 0.5;
        }

        // Add slope term: if forecast mean is climbing vs previous bar, reward longs, etc.
        score += 1.0 * slope_term;

        // ===== Regime adjustments =====
        use MarketRegime::*;
        match ctx.ml_regime {
            TrendingUp => {
                score += self.regime_bias;
            }
            TrendingDown => {
                score -= self.regime_bias;
            }
            SidewaysLowVol => {
                // Chop zone: cut risk.
                score *= 0.5;
            }
            SidewaysHighVol => {
                // Volatile range: keep score but cap exposure later.
            }
        }

        // ===== Convert score → target position =====
        //
        // Squash via tanh to keep in [-1, 1], then scale by max_pos.
        let raw_signal = (score / 5.0).tanh();
        let mut desired = self.max_pos * raw_signal;

        // If volatility is extremely high, reduce exposure.
        if band_width_sigma > 3.0 {
            desired *= 0.7;
        }
        if sigma > 0.03 {
            desired *= 0.7;
        }

        // ===== Cooldown & anti-chop logic =====
        let idx = ctx.idx;
        let mut final_pos = current_pos;

        let change = (desired - current_pos).abs();

        let can_trade = match self.last_trade_idx {
            None => true,
            Some(last) => idx.saturating_sub(last) >= self.cooldown_bars,
        };

        // Only adjust if:
        // - we are outside a cooldown window, OR
        // - the change is small and in the same direction (soft rebalance)
        if change > self.min_rebalance_step {
            if can_trade {
                final_pos = desired;
                self.last_trade_idx = Some(idx);
            } else {
                // inside cooldown: allow gentle reduction of risk if we're way off.
                if (desired.signum() != current_pos.signum())
                    && change > 0.3 * self.max_pos
                {
                    final_pos = 0.0;
                    self.last_trade_idx = Some(idx);
                }
            }
        } else {
            // Tiny signals: keep current position as-is.
            final_pos = current_pos;
        }

        // Clamp to [-max_pos, max_pos] for safety.
        final_pos = final_pos.clamp(-self.max_pos, self.max_pos);

        Target {
            target_fraction: final_pos,
        }
    }
}
