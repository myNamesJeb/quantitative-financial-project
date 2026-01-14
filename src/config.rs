// src/config.rs

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub fee_bps: f64,
    pub spread_bps: f64,
    pub slippage_bps: f64,
    pub impact_power: f64,
    pub participation_rate: f64,
    pub queue_penalty: f64,
    pub latency_bars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub lookback_days: usize,
    pub forecast_horizon: usize,
    pub buckets: usize,
    pub paths: usize,
    pub accuracy_horizon: usize,
    pub cal_window: usize,
    pub cal_min_samples: usize,
    pub embargo_bars: usize,
    pub execution: ExecutionConfig,
    pub log_path: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            lookback_days: 30,
            forecast_horizon: 50,
            buckets: 80,
            paths: 3000,
            accuracy_horizon: 6,
            cal_window: 500,
            cal_min_samples: 50,
            embargo_bars: 2,
            execution: ExecutionConfig {
                fee_bps: 6.0,
                spread_bps: 6.0,
                slippage_bps: 4.0,
                impact_power: 0.6,
                participation_rate: 0.10,
                queue_penalty: 0.10,
                latency_bars: 0,
            },
            log_path: "runs/summary.csv".to_string(),
        }
    }
}

impl RunConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if let Ok(raw) = fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<RunConfig>(&raw) {
                return cfg;
            }
        }
        RunConfig::default()
    }
}
