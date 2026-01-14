// src/logger.rs

use std::fs::{create_dir_all, OpenOptions};
use std::io::{Result, Write};

use crate::backtest::BacktestResult;

pub fn append_run_summary(path: &str, result: &BacktestResult) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    if file.metadata()?.len() == 0 {
        writeln!(
            file,
            "final_equity,total_return,annualized_return,sharpe,max_drawdown,stress_dd_5,stress_dd_10,winrate,brier,log_loss,quantile_error,best_metric"
        )?;
    }

    writeln!(
        file,
        "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
        result.final_equity,
        result.total_return,
        result.annualized_return,
        result.sharpe,
        result.max_drawdown,
        result.stress_dd_5,
        result.stress_dd_10,
        result.winrate,
        result.accuracy.avg_brier,
        result.accuracy.avg_log_loss,
        result.accuracy.avg_quantile_error,
        format!("{:?}", result.accuracy.best_metric)
    )?;

    Ok(())
}
