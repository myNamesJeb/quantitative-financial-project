// src/research.rs

use crate::features::build_ml_dataset_indexed;
use crate::ml::{
    fit_linear_regression_ensemble,
    fit_mlp,
    fit_stump,
    predict_mlp,
    predict_stump,
    dot,
};
use crate::data::MarketBar;

pub struct WalkForwardResult {
    pub mse: Vec<f64>,
}

pub fn walk_forward_cv(
    bars: &[MarketBar],
    train_size: usize,
    test_size: usize,
    purge: usize,
    embargo: usize,
) -> WalkForwardResult {
    let returns = crate::features::compute_log_returns(bars);
    let (features_idx, targets) = build_ml_dataset_indexed(bars, &returns);
    if features_idx.is_empty() {
        return WalkForwardResult { mse: vec![] };
    }

    let mut mse = Vec::new();
    let mut start = 0usize;
    let end = features_idx.len();

    while start + train_size + purge + test_size + embargo <= end {
        let train_end = start + train_size;
        let test_start = train_end + purge;
        let test_end = test_start + test_size;

        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        for i in start..train_end.saturating_sub(embargo) {
            train_x.push(features_idx[i].1.clone());
            train_y.push(targets[i]);
        }

        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        for i in test_start..test_end {
            test_x.push(features_idx[i].1.clone());
            test_y.push(targets[i]);
        }

        if train_x.len() < 50 || test_x.is_empty() {
            break;
        }

        let (beta, _resid) = fit_linear_regression_ensemble(&train_x, &train_y, 1e-4, 64, 0.7);
        let stump = fit_stump(&train_x, &train_y);
        let mlp = fit_mlp(&train_x, &train_y, 8, 3, 0.005);

        let mut sum = 0.0;
        for (x, y) in test_x.iter().zip(test_y.iter()) {
            let lin = dot(x, &beta);
            let st = stump.as_ref().map(|s| predict_stump(s, x)).unwrap_or(0.0);
            let nn = mlp.as_ref().map(|m| predict_mlp(m, x)).unwrap_or(0.0);
            let pred = 0.6 * lin + 0.25 * st + 0.15 * nn;
            let err = y - pred;
            sum += err * err;
        }
        mse.push(sum / test_y.len() as f64);

        start = test_end;
    }

    WalkForwardResult { mse }
}
