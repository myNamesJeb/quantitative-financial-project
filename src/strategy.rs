
// src/strategy.rs

use crate::data::MarketBar;

/// Which high-level behavior we’re in.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Regime {
    MeanReversion,
    Trend,
    Breakout,
    Tail,
    Flat,
}

/// Condensed info from your MC distribution for the next step.
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

/// Info the strategy sees for each bar.
#[derive(Debug)]
pub struct Context<'a> {
    pub bar: &'a MarketBar,
    pub idx: usize,
    pub forecast: ForecastSnapshot,
    pub drift: f64,
    pub base_sigma: f64,
    pub ml_regime: crate::features::MarketRegime, // from your ML classifier
    pub last_price: f64,
}

/// What the strategy wants the position to be.
/// -1.0 = fully short, 0 = flat, +1.0 = fully long.
#[derive(Debug, Copy, Clone)]
pub struct TargetPosition {
    pub target_fraction: f64,
}

impl TargetPosition {
    pub fn flat() -> Self {
        TargetPosition { target_fraction: 0.0 }
    }

    pub fn long(f: f64) -> Self {
        TargetPosition { target_fraction: f.clamp(0.0, 1.0) }
    }

    pub fn short(f: f64) -> Self {
        TargetPosition { target_fraction: (-f).clamp(-1.0, 0.0) }
    }
}

/// Trait all strategies implement.
pub trait Strategy {
    fn on_bar(&mut self, ctx: &Context, current_pos: f64) -> TargetPosition;
}

//////////////////////////////
// Mean Reversion Strategy
//////////////////////////////

pub struct MeanReversionStrategy {
    /// How strong the signal must be in σ units to enter.
    pub band_sigma_enter: f64,
    /// How close to mean to exit.
    pub band_sigma_exit: f64,
    /// Max fraction of capital to deploy.
    pub max_leverage: f64,
}

impl Default for MeanReversionStrategy {
    fn default() -> Self {
        Self {
            band_sigma_enter: 1.2,  // enter near ~1.2σ away
            band_sigma_exit: 0.4,   // take profits near ±0.4σ region
            max_leverage: 0.6,
        }
    }
}

impl Strategy for MeanReversionStrategy {
    fn on_bar(&mut self, ctx: &Context, current_pos: f64) -> TargetPosition {
        let p = ctx.last_price;
        let f = &ctx.forecast;

        if f.std <= 0.0 {
            return TargetPosition::flat();
        }

        let z = (p - f.mean) / f.std;

        // If far below mean (z < -enter), go long toward reversion.
        if z < -self.band_sigma_enter {
            return TargetPosition::long(self.max_leverage);
        }

        // If far above mean (z > +enter), go short.
        if z > self.band_sigma_enter {
            return TargetPosition::short(self.max_leverage);
        }

        // If we are already long/short but have reverted close enough to mean, flatten.
        if current_pos > 0.0 && z.abs() < self.band_sigma_exit {
            return TargetPosition::flat();
        }

        if current_pos < 0.0 && z.abs() < self.band_sigma_exit {
            return TargetPosition::flat();
        }

        // Otherwise keep current position.
        TargetPosition {
            target_fraction: current_pos,
        }
    }
}

//////////////////////////////
// Trend Following Strategy
//////////////////////////////

pub struct TrendStrategy {
    pub drift_to_vol_ratio_enter: f64,
    pub max_leverage: f64,
}

impl Default for TrendStrategy {
    fn default() -> Self {
        Self {
            drift_to_vol_ratio_enter: 0.4, // |drift| > 0.4 * sigma
            max_leverage: 0.8,
        }
    }
}

impl Strategy for TrendStrategy {
    fn on_bar(&mut self, ctx: &Context, current_pos: f64) -> TargetPosition {
        let sigma = ctx.base_sigma.max(1e-6);
        let ratio = ctx.drift / sigma;

        // Strong positive drift → long.
        if ratio > self.drift_to_vol_ratio_enter {
            return TargetPosition::long(self.max_leverage);
        }

        // Strong negative drift → short.
        if ratio < -self.drift_to_vol_ratio_enter {
            return TargetPosition::short(self.max_leverage);
        }

        // If drift weakens, slowly mean-revert position back to flat.
        TargetPosition {
            target_fraction: current_pos * 0.7,
        }
    }
}

//////////////////////////////
// Breakout Strategy
//////////////////////////////

pub struct BreakoutStrategy {
    pub sigma_expand_threshold: f64,
    pub max_leverage: f64,
}

impl Default for BreakoutStrategy {
    fn default() -> Self {
        Self {
            sigma_expand_threshold: 1.15, // std_t > 1.15 * std_{t-1}
            max_leverage: 1.0,
        }
    }
}

impl BreakoutStrategy {
    pub fn on_bar_with_prev(
        &mut self,
        ctx: &Context,
        prev_forecast: &ForecastSnapshot,
        current_pos: f64,
    ) -> TargetPosition {
        let f = &ctx.forecast;
        if prev_forecast.std <= 0.0 || f.std <= 0.0 {
            return TargetPosition::flat();
        }

        let vol_ratio = f.std / prev_forecast.std;

        // Only care when vol is expanding meaningfully.
        if vol_ratio < self.sigma_expand_threshold {
            return TargetPosition {
                target_fraction: current_pos * 0.5,
            };
        }

        let p = ctx.last_price;
        let upper = f.band_upper;
        let lower = f.band_lower;

        // Breakout up: price pushes above upper band.
        if p > upper {
            return TargetPosition::long(self.max_leverage);
        }

        // Breakout down: price pushes below lower band.
        if p < lower {
            return TargetPosition::short(self.max_leverage);
        }

        // No clear breakout yet.
        TargetPosition {
            target_fraction: current_pos * 0.8,
        }
    }
}

//////////////////////////////
// Tail / Skew Strategy
//////////////////////////////

pub struct TailStrategy {
    pub skew_threshold: f64,
    pub max_leverage: f64,
}

impl Default for TailStrategy {
    fn default() -> Self {
        Self {
            skew_threshold: 0.6,
            max_leverage: 0.5,
        }
    }
}

impl Strategy for TailStrategy {
    fn on_bar(&mut self, ctx: &Context, current_pos: f64) -> TargetPosition {
        let f = &ctx.forecast;

        if f.kurtosis < 3.5 {
            // Not really in a fat-tail regime; don’t do anything special.
            return TargetPosition {
                target_fraction: current_pos * 0.8,
            };
        }

        // Positive skew: upside tail → bias long.
        if f.skew > self.skew_threshold {
            return TargetPosition::long(self.max_leverage);
        }

        // Negative skew: downside tail → bias short.
        if f.skew < -self.skew_threshold {
            return TargetPosition::short(self.max_leverage);
        }

        TargetPosition {
            target_fraction: current_pos * 0.8,
        }
    }
}

//////////////////////////////
// Strategy Router
//////////////////////////////

pub struct StrategyRouter {
    pub mr: MeanReversionStrategy,
    pub tf: TrendStrategy,
    pub bo: BreakoutStrategy,
    pub tail: TailStrategy,
}

impl Default for StrategyRouter {
    fn default() -> Self {
        Self {
            mr: MeanReversionStrategy::default(),
            tf: TrendStrategy::default(),
            bo: BreakoutStrategy::default(),
            tail: TailStrategy::default(),
        }
    }
}

impl StrategyRouter {
    /// Decide which regime we’re in based on forecast / ML / vol.
    pub fn detect_regime(
        &self,
        ctx: &Context,
        prev_forecast: Option<&ForecastSnapshot>,
    ) -> Regime {
        let f = &ctx.forecast;

        // 1) Mean Reversion regime: low vol, near-zero skew, moderate kurtosis.
        if f.std / ctx.last_price < 0.012 && f.skew.abs() < 0.3 && (2.0..=4.0).contains(&f.kurtosis) {
            return Regime::MeanReversion;
        }

        // 2) Trend regime: drift strong relative to vol and ML regime trending.
        let sigma = ctx.base_sigma.max(1e-6);
        let ratio = ctx.drift / sigma;
        if ratio.abs() > 0.5 {
            if matches!(
                ctx.ml_regime,
                crate::features::MarketRegime::TrendingUp | crate::features::MarketRegime::TrendingDown
            ) {
                return Regime::Trend;
            }
        }

        // 3) Breakout regime: vol expansion and kurtosis rising.
        if let Some(prev) = prev_forecast {
            if prev.std > 0.0 {
                let vol_ratio = f.std / prev.std;
                if vol_ratio > self.bo.sigma_expand_threshold && f.kurtosis > 3.3 {
                    return Regime::Breakout;
                }
            }
        }

        // 4) Tail regime: large skew & fat tails.
        if f.kurtosis > 3.5 && f.skew.abs() > self.tail.skew_threshold {
            return Regime::Tail;
        }

        // Otherwise: flat/defensive.
        Regime::Flat
    }

    /// Main call for backtester.
    pub fn on_bar(
        &mut self,
        ctx: &Context,
        prev_forecast: Option<&ForecastSnapshot>,
        current_pos: f64,
    ) -> TargetPosition {
        let regime = self.detect_regime(ctx, prev_forecast);

        match regime {
            Regime::MeanReversion => self.mr.on_bar(ctx, current_pos),
            Regime::Trend => self.tf.on_bar(ctx, current_pos),
            Regime::Breakout => {
                if let Some(prev) = prev_forecast {
                    self.bo.on_bar_with_prev(ctx, prev, current_pos)
                } else {
                    TargetPosition::flat()
                }
            }
            Regime::Tail => self.tail.on_bar(ctx, current_pos),
            Regime::Flat => TargetPosition {
                target_fraction: current_pos * 0.5,
            },
        }
    }
}
