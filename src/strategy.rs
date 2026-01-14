
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
    pub entropy: f64,
    pub confidence: f64,
}

/// Per-bar context passed to the router.
///
/// Now includes multi-timeframe drift/vol:
/// - 1D drift/sigma
/// - 4H drift/sigma
/// - 15m drift/sigma
#[derive(Debug, Copy, Clone)]
pub struct Context<'a> {
    pub bar: &'a MarketBar,
    pub idx: usize,
    pub forecast: ForecastSnapshot,
    pub drift: f64,
    pub base_sigma: f64,
    pub ml_regime: MarketRegime,
    pub last_price: f64,

    // Higher-timeframe summaries
    pub drift_1d: f64,
    pub sigma_1d: f64,
    pub drift_4h: f64,
    pub sigma_4h: f64,
    pub drift_15m: f64,
    pub sigma_15m: f64,
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

    /// EMA smoothing factor for the raw score.
    pub ema_alpha: f64,

    /// Minimum confidence needed to act on signals.
    pub min_confidence: f64,

    /// Target per-bar volatility for sizing (roughly 1H).
    pub target_vol: f64,

    /// Minimum absolute score to allow a position flip.
    pub flip_threshold: f64,

    /// Internal: last bar index where we changed position meaningfully.
    last_trade_idx: Option<usize>,

    /// Internal: smoothed score.
    score_ema: f64,
}

impl Default for StrategyRouter {
    fn default() -> Self {
        Self {
            max_pos: 0.75,             // allow up to 75% of equity in SOL
            drift_sensitivity: 6.0,    // how strongly drift_z moves exposure
            band_sensitivity: 2.5,     // how strongly mean-spot moves exposure
            regime_bias: 0.30,         // extra directional push in trends
            cooldown_bars: 1,          // trade more often
            min_rebalance_step: 0.02,  // smaller adjustments allowed
            ema_alpha: 0.30,
            min_confidence: 0.20,
            target_vol: 0.015,
            flip_threshold: 0.20,
            last_trade_idx: None,
            score_ema: 0.0,
        }
    }
}

impl StrategyRouter {
    /// Main decision function.
    ///
    /// Inputs:
    /// - `ctx`: current bar context (now multi-TF aware)
    /// - `prev_forecast`: previous forecast snapshot (if any)
    /// - `current_pos`: current position (fraction of equity)
    pub fn on_bar(
        &mut self,
        ctx: &Context,
        prev_forecast: Option<&ForecastSnapshot>,
        current_pos: f64,
    ) -> Target {
        // If volatility estimate is totally degenerate, stay flat-ish.
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

        // ===== Higher-timeframe trend fusion (1D, 4H, 15m) =====
        //
        // We build a combined HTF trend score from Z-scored drifts.
        let mut htf_score = 0.0;

        let mut add_tf = |drift: f64, sig: f64, weight: f64| {
            if sig > 0.0 {
                let z = (drift / sig).clamp(-5.0, 5.0);
                htf_score += weight * z;
            }
        };

        // 1D: macro anchor
        add_tf(ctx.drift_1d, ctx.sigma_1d, 0.4);
        // 4H: swing direction
        add_tf(ctx.drift_4h, ctx.sigma_4h, 0.35);
        // 15m: local flow bias
        add_tf(ctx.drift_15m, ctx.sigma_15m, 0.25);

        // Squash the HTF score to avoid nuking everything.
        let htf_bias = 0.6 * htf_score.tanh();

        // ===== Regime-aware weighting =====
        let mut drift_w = self.drift_sensitivity;
        let mut band_w = self.band_sensitivity;
        let mut score_mult = 1.0;

        use MarketRegime::*;
        match ctx.ml_regime {
            TrendingUp | TrendingDown => {
                drift_w *= 1.2;
                band_w *= 0.8;
            }
            SidewaysLowVol => {
                drift_w *= 0.5;
                band_w *= 1.3;
                score_mult *= 0.8;
            }
            SidewaysHighVol => {
                drift_w *= 0.6;
                band_w *= 0.9;
                score_mult *= 0.7;
            }
        }

        // ===== Raw score construction =====
        //
        // Think of this as a composite alpha:
        //   score > 0  -> want to be long
        //   score < 0  -> want to be short
        let mut score = 0.0;

        // Drift pushes us toward trend following.
        score += drift_w * drift_z;

        // Mean-spot shift: if mean > spot, tilt long; if mean < spot, tilt short.
        score += band_w * mean_shift;

        // Positive skew -> heavier right tail -> slight long bias.
        score += 0.5 * skew;

        // If band-width is very small, distribution is tight -> reduce conviction.
        if band_width_sigma < 0.8 {
            score *= 0.5;
        }

        // Add slope term: if forecast mean is climbing vs previous bar, reward longs, etc.
        score += 1.0 * slope_term;

        // Add higher-timeframe directional bias
        score += htf_bias;

        score *= score_mult;

        // ===== Regime adjustments =====
        match ctx.ml_regime {
            TrendingUp => {
                score += self.regime_bias;
            }
            TrendingDown => {
                score -= self.regime_bias;
            }
            SidewaysLowVol => {
                // Chop zone: cut some risk, but if HTF trend is strong,
                // keep a bit more exposure.
                score *= 0.6;
            }
            SidewaysHighVol => {
                // Volatile range: keep score but cap exposure later.
            }
        }

        // ===== Signal timing and confidence gating =====
        self.score_ema = self.ema_alpha * score + (1.0 - self.ema_alpha) * self.score_ema;
        let mut timed_score = self.score_ema;

        if ctx.forecast.confidence < self.min_confidence {
            let scale = (ctx.forecast.confidence / self.min_confidence).clamp(0.0, 1.0);
            timed_score *= scale;
        }

        // ===== Convert score → target position =====
        //
        // Squash via tanh to keep in [-1, 1], then scale by max_pos.
        let raw_signal = (timed_score / 5.0).tanh();
        let mut desired = self.max_pos * raw_signal;

        // Vol targeting: scale exposure by inverse of estimated volatility.
        let vol_scale = (self.target_vol / base_sigma).clamp(0.3, 1.5);
        desired *= vol_scale;

        // If volatility is extremely high, reduce exposure.
        if band_width_sigma > 3.0 {
            desired *= 0.7;
        }
        if sigma > 0.03 {
            desired *= 0.7;
        }

        // Avoid weak flips.
        if current_pos != 0.0 && desired.signum() != current_pos.signum() {
            if score.abs() < self.flip_threshold {
                desired = 0.0;
            }
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
                // inside cooldown: allow gentle reduction of risk if we're way off
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
