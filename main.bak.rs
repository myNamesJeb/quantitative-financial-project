
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;
use image::{ImageBuffer, Rgb};
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
struct MarketBar {
    unix: i64,
    time: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    trades: f64,
}

#[derive(Debug, Copy, Clone)]
enum MarketRegime {
    TrendingUp,
    TrendingDown,
    SidewaysLowVol,
    SidewaysHighVol,
}

#[derive(Debug, Copy, Clone)]
struct SabrParams {
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    f0: f64,
    atm_vol: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.csv> <output.png>", args[0]);
        std::process::exit(1);
    }

    let input_csv = &args[1];
    let output_png = &args[2];

    let mut bars = read_bars(input_csv)?;
    if bars.len() < 100 {
        eprintln!("Need at least 100 rows, got {}", bars.len());
        std::process::exit(1);
    }

    // Sort by time ascending, just in case.
    bars.sort_by(|a, b| a.unix.cmp(&b.unix));

    println!("Parsed {} bars.", bars.len());
    println!("Last row = {:?}", bars.last().unwrap());
    println!("Last close = {}", bars.last().unwrap().close);

    let returns = compute_log_returns(&bars);
    let garch_vol = compute_garch_like_vol(&returns);

    let (features, targets) = build_ml_dataset(&bars, &returns);
    if features.is_empty() {
        eprintln!("Not enough data to build ML dataset.");
        std::process::exit(1);
    }

    // Ensemble ridge regression (bootstrapped)
    // Crank n_models up for more training time if you want.
    let n_models = 128;
    let sample_frac = 0.7;
    let (beta, ml_resid_std) = fit_linear_regression_ensemble(
        &features,
        &targets,
        1e-4,
        n_models,
        sample_frac,
    );

    let last_features = features.last().unwrap().clone();
    let ml_pred_ret = dot(&beta, &last_features);
    let ml_pred_vol = if ml_resid_std > 0.0 { ml_resid_std } else { 0.01 };

    let regime = classify_regime_ml(ml_pred_ret, ml_pred_vol, &returns);
    println!(
        "Regime (ML): {:?} | pred_ret = {:.5} | pred_vol = {:.5}",
        regime, ml_pred_ret, ml_pred_vol
    );

    let last_close = bars.last().unwrap().close;
    let (drift, base_sigma) = compute_drift_and_vol_scale(
        &bars,
        &returns,
        &garch_vol,
        regime,
        ml_pred_ret,
        ml_pred_vol,
    );

    println!("Drift = {}", drift);
    println!("Base sigma = {}", base_sigma);

    let sabr_params = estimate_sabr_params(&returns, base_sigma, last_close);
    println!("SABR params = {:?}", sabr_params);

    let horizon = 50;
    let buckets = 80;
    let paths = 10_000;

    let heatmap = simulate_heatmap(
        &bars,
        drift,
        base_sigma,
        &sabr_params,
        horizon,
        buckets,
        paths,
    );

    print_bucket_summary(&heatmap, &bars);

    save_heatmap_png(&heatmap, output_png)?;
    println!("Saved heatmap to {}", output_png);

    Ok(())
}

//////////////////////////////////////////////////////////////
// CSV PARSER
//////////////////////////////////////////////////////////////

fn read_bars<P: AsRef<Path>>(path: P) -> Result<Vec<MarketBar>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(file);

    let mut bars = Vec::new();

    fn clean(s: &str) -> String {
        s.trim()
            .trim_matches('\u{feff}')
            .trim_matches('"')
            .trim_matches('\'')
            .to_string()
    }

    for (i, result) in rdr.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Row {} parse error: {}", i, e);
                continue;
            }
        };

        if record.len() < 10 {
            continue;
        }

        let cols: Vec<String> = record.iter().map(|s| clean(s)).collect();

        let unix = parse_i64(i, 0, cols.get(0));
        let time = parse_str(cols.get(1));
        let open = parse_f64(i, 3, cols.get(3));
        let high = parse_f64(i, 4, cols.get(4));
        let low = parse_f64(i, 5, cols.get(5));
        let close = parse_f64(i, 6, cols.get(6));
        let volume = parse_f64(i, 7, cols.get(7));
        let trades = parse_f64(i, 9, cols.get(9));

        bars.push(MarketBar {
            unix,
            time,
            open,
            high,
            low,
            close,
            volume,
            trades,
        });
    }

    Ok(bars)
}

fn parse_str(x: Option<&String>) -> String {
    x.cloned().unwrap_or_else(|| "0".into())
}

fn parse_i64(row: usize, col: usize, x: Option<&String>) -> i64 {
    match x {
        Some(v) => v.parse::<i64>().unwrap_or_else(|_| {
            eprintln!("Row {} col {}: failed to parse '{}' as i64 → 0", row, col, v);
            0
        }),
        None => 0,
    }
}

fn parse_f64(row: usize, col: usize, x: Option<&String>) -> f64 {
    match x {
        Some(v) => v.parse::<f64>().unwrap_or_else(|_| {
            eprintln!("Row {} col {}: failed to parse '{}' as f64 → 0.0", row, col, v);
            0.0
        }),
        None => 0.0,
    }
}

//////////////////////////////////////////////////////////////
// CORE RETURN / VOL
//////////////////////////////////////////////////////////////

fn compute_log_returns(bars: &[MarketBar]) -> Vec<f64> {
    let mut v = Vec::with_capacity(bars.len().saturating_sub(1));
    for i in 1..bars.len() {
        let p0 = bars[i - 1].close;
        let p1 = bars[i].close;
        if p0 > 0.0 && p1 > 0.0 {
            v.push((p1 / p0).ln());
        } else {
            v.push(0.0);
        }
    }
    v
}

fn compute_garch_like_vol(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![];
    }

    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let var = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / n as f64;

    let mut sigma2 = var.max(1e-8);

    let omega = 1e-7;
    let alpha = 0.1;
    let beta = 0.85;

    let mut out = Vec::with_capacity(n);

    for r in returns {
        sigma2 = omega + alpha * (r * r) + beta * sigma2;
        out.push(sigma2.sqrt());
    }

    out
}

//////////////////////////////////////////////////////////////
// STATS UTILS
//////////////////////////////////////////////////////////////

fn mean_std(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.max(0.0).sqrt())
}

//////////////////////////////////////////////////////////////
// SIMPLE ML: ENSEMBLE LINEAR REGRESSION
//////////////////////////////////////////////////////////////

fn build_ml_dataset(bars: &[MarketBar], returns: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = returns.len();
    if n < 40 {
        return (Vec::new(), Vec::new());
    }

    let vol_window_short = 5;
    let vol_window_long = 20;

    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let trades_vec: Vec<f64> = bars.iter().map(|b| b.trades).collect();

    let (_v_mean, v_std) = mean_std(&volumes);
    let (_t_mean, t_std) = mean_std(&trades_vec);

    let mut features = Vec::new();
    let mut targets = Vec::new();

    let start = vol_window_long;
    for t in start..(n - 1) {
        let end = t + 1;

        let s_short = end.saturating_sub(vol_window_short);
        let s_long = end.saturating_sub(vol_window_long);

        let r_slice_short = &returns[s_short..end];
        let r_slice_long = &returns[s_long..end];

        let r_t = returns[t];
        let r5 = r_slice_short.iter().sum::<f64>();
        let r20 = r_slice_long.iter().sum::<f64>();

        let (_, vol5) = mean_std(r_slice_short);
        let (_, vol20) = mean_std(r_slice_long);

        let vol_t = bars[end].volume;
        let vol_z = if v_std > 0.0 {
            (vol_t - volumes.iter().sum::<f64>() / volumes.len() as f64) / v_std
        } else {
            0.0
        };

        let trades_t = bars[end].trades;
        let trades_z = if t_std > 0.0 {
            (trades_t - trades_vec.iter().sum::<f64>() / trades_vec.len() as f64) / t_std
        } else {
            0.0
        };

        let range = if bars[end].close > 0.0 {
            (bars[end].high - bars[end].low).abs() / bars[end].close
        } else {
            0.0
        };

        // Feature vector: [bias, r_t, r5, r20, vol5, vol20, vol_z, trades_z, range]
        let x = vec![1.0, r_t, r5, r20, vol5, vol20, vol_z, trades_z, range];
        let y = returns[t + 1];

        features.push(x);
        targets.push(y);
    }

    (features, targets)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = a.len();
    let mut x = vec![0.0; n];

    for i in 0..n {
        // Pivot
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            continue;
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back-substitution
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-12 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    x
}


fn fit_linear_regression_ensemble(
    features: &[Vec<f64>],
    targets: &[f64],
    lambda: f64,
    n_models: usize,
    sample_frac: f64,
) -> (Vec<f64>, f64) {
    let n = features.len();
    if n == 0 {
        return (vec![0.0], 0.0);
    }

    let d = features[0].len();
    let mut rng = rand::thread_rng();

    let mut beta_acc = vec![0.0; d];
    let mut rss_acc = 0.0;
    let mut model_count = 0usize;

    for _ in 0..n_models {
        let m = ((n as f64 * sample_frac).round() as usize).clamp(d + 1, n);

        let mut xtx = vec![vec![0.0; d]; d];
        let mut xty = vec![0.0; d];

        // bootstrap sample
        for _ in 0..m {
            let idx = rng.gen_range(0..n);
            let x = &features[idx];
            let y = targets[idx];

            for i in 0..d {
                xty[i] += x[i] * y;
                for j in 0..d {
                    xtx[i][j] += x[i] * x[j];
                }
            }
        }

        // ridge
        for i in 0..d {
            xtx[i][i] += lambda;
        }

        let mut xtx_clone = xtx.clone();
        let mut xty_clone = xty.clone();
        let beta_k = solve_linear_system(&mut xtx_clone, &mut xty_clone);

        // RSS
        let mut rss_k = 0.0;
        for (x, &y) in features.iter().zip(targets.iter()) {
            let y_hat = dot(&beta_k, x);
            let e = y - y_hat;
            rss_k += e * e;
        }

        // accumulate
        for j in 0..d {
            beta_acc[j] += beta_k[j];
        }
        rss_acc += rss_k;
        model_count += 1;
    }

    for j in 0..d {
        beta_acc[j] /= model_count as f64;
    }

    // FIX: type-inference ambiguity → explicitly f64
    let resid_std = ((rss_acc / (n as f64 * model_count as f64)) as f64)
        .max(0.0_f64)
        .sqrt();

    (beta_acc, resid_std)
}
//////////////////////////////////////////////////////////////
// REGIME (ML-DRIVEN)
//////////////////////////////////////////////////////////////

fn classify_regime_ml(pred_ret: f64, pred_vol: f64, returns: &[f64]) -> MarketRegime {
    let n = returns.len();
    let look = n.min(80);
    let slice = &returns[n - look..];
    let (_, vol_realized) = mean_std(slice);

    let pv = pred_vol.max(1e-4);
    if pred_ret > 1.5 * pv {
        MarketRegime::TrendingUp
    } else if pred_ret < -1.5 * pv {
        MarketRegime::TrendingDown
    } else if vol_realized > 0.015 {
        MarketRegime::SidewaysHighVol
    } else {
        MarketRegime::SidewaysLowVol
    }
}

//////////////////////////////////////////////////////////////
// DRIFT + VOL SCALE (FLOW + MICROSTRUCTURE)
//////////////////////////////////////////////////////////////

fn compute_drift_and_vol_scale(
    bars: &[MarketBar],
    returns: &[f64],
    garch_vol: &[f64],
    regime: MarketRegime,
    ml_pred_ret: f64,
    ml_pred_vol: f64,
) -> (f64, f64) {
    let last_ret = *returns.last().unwrap_or(&0.0);
    let last_bar = bars.last().unwrap();

    let look = bars.len().min(100);
    let slice = &bars[bars.len() - look..];

    let avg_vol = slice.iter().map(|b| b.volume).sum::<f64>() / look as f64;
    let avg_trades = slice.iter().map(|b| b.trades).sum::<f64>() / look as f64;
    let avg_range = slice
        .iter()
        .map(|b| {
            if b.close > 0.0 {
                (b.high - b.low).abs() / b.close
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / look as f64;

    let last_range = if last_bar.close > 0.0 {
        (last_bar.high - last_bar.low).abs() / last_bar.close
    } else {
        0.0
    };

    let vol_z = if avg_vol > 0.0 {
        (last_bar.volume - avg_vol) / avg_vol
    } else {
        0.0
    };

    let trades_z = if avg_trades > 0.0 {
        (last_bar.trades - avg_trades) / avg_trades
    } else {
        0.0
    };

    let range_z = if avg_range > 0.0 {
        (last_range - avg_range) / avg_range
    } else {
        0.0
    };

    // base drift from ML, but damped
    let mut drift = ml_pred_ret * 0.5;

    // Flow: big vol + same-direction return pushes drift
    let flow_signal = last_ret.signum() * vol_z.signum();
    let flow_drift_adj = 0.25 * flow_signal * ml_pred_vol.abs();
    drift += flow_drift_adj;

    // base vol: combine GARCH + ML vol
    let last_garch = garch_vol.last().cloned().unwrap_or(0.02);
    let mut base_vol = 0.5 * last_garch + 0.5 * ml_pred_vol.abs();

    let (dr_mult, vol_mult) = match regime {
        MarketRegime::TrendingUp => (1.2, 1.15),
        MarketRegime::TrendingDown => (1.2, 1.15),
        MarketRegime::SidewaysLowVol => (0.5, 0.8),
        MarketRegime::SidewaysHighVol => (0.8, 1.4),
    };

    drift *= dr_mult;

    // microstructure scaling
    let micro_mult = (1.0 + 0.3 * vol_z.abs() + 0.2 * range_z.abs() + 0.2 * trades_z.abs())
        .clamp(0.5, 3.0);

    base_vol *= vol_mult * micro_mult;

    let base_vol = base_vol.max(0.01);

    // clamp drift per step so MC doesn't go ballistic
    let drift = drift.clamp(-0.05, 0.05);

    (drift, base_vol)
}

//////////////////////////////////////////////////////////////
// SABR PARAMS
//////////////////////////////////////////////////////////////

fn estimate_sabr_params(returns: &[f64], base_sigma: f64, f0: f64) -> SabrParams {
    let n = returns.len();
    let look = n.min(200);
    let slice = &returns[n - look..];

    let (mean_r, std_r) = mean_std(slice);
    let std_r = std_r.max(1e-6);

    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &r in slice {
        let x = r - mean_r;
        m3 += x.powi(3);
        m4 += x.powi(4);
    }
    m3 /= look as f64;
    m4 /= look as f64;

    let skew = m3 / std_r.powi(3);
    let kurt = m4 / std_r.powi(4);
    let kurt_excess = (kurt - 3.0).max(0.0);

    let beta = 0.7;
    let rho = (skew / 3.0).tanh().clamp(-0.7, 0.7);
    let nu = (0.5 + 0.25 * kurt_excess).clamp(0.3, 3.0);

    let alpha = if f0 > 0.0 {
        base_sigma * f0.powf(1.0 - beta)
    } else {
        base_sigma
    };

    SabrParams {
        alpha,
        beta,
        rho,
        nu,
        f0,
        atm_vol: base_sigma,
    }
}

fn sabr_implied_vol(F: f64, K: f64, p: &SabrParams) -> f64 {
    if F <= 0.0 || K <= 0.0 {
        return p.atm_vol;
    }

    let alpha = p.alpha;
    let beta = p.beta;
    let rho = p.rho;
    let nu = p.nu;

    let one_minus_beta = 1.0 - beta;
    let fk = (F * K).sqrt();
    let fk_beta = fk.powf(one_minus_beta);
    let log_fk = (F / K).ln();

    let z = if alpha > 0.0 {
        (nu / alpha) * fk_beta * log_fk
    } else {
        0.0
    };

    let eps = 1e-8;
    let term1 = alpha / fk_beta;

    let x_z = if z.abs() < eps {
        1.0
    } else {
        let arg = (1.0 - 2.0 * rho * z + z * z).max(1e-12).sqrt();
        let num = arg + z - rho;
        let den = 1.0 - rho;
        if num <= 0.0 || den <= 0.0 {
            1.0
        } else {
            (num / den).ln()
        }
    };

    let z_over_x = if x_z.abs() < eps { 1.0 } else { z / x_z };

    let log_fk2 = log_fk * log_fk;

    let alpha2 = alpha * alpha;
    let fk2_1beta = fk.powf(2.0 * one_minus_beta);

    let alpha_term = (1.0 / 24.0) * (alpha2 / fk2_1beta);
    let rho_term = 0.25 * rho * beta * nu * alpha / fk_beta;
    let nu_term = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
    let beta_term = one_minus_beta * one_minus_beta * log_fk2 / 24.0;

    let correction = 1.0 + alpha_term + rho_term + nu_term + beta_term;

    let vol = term1 * z_over_x * correction;

    if !vol.is_finite() || vol <= 0.0 {
        p.atm_vol
    } else {
        vol
    }
}

//////////////////////////////////////////////////////////////
// MONTE CARLO + HEATMAP (SABR + STOCH VOL + JUMPS)
//////////////////////////////////////////////////////////////

fn compute_price_bounds(_bars: &[MarketBar], last: f64) -> (f64, f64) {
    let min_p = last * 0.85;
    let max_p = last * 1.15;
    (min_p, max_p)
}

fn simulate_heatmap(
    bars: &[MarketBar],
    drift: f64,
    base_sigma: f64,
    sabr: &SabrParams,
    horizon: usize,
    buckets: usize,
    paths: usize,
) -> Vec<Vec<f64>> {
    let last_close = bars.last().unwrap().close;
    let (min_p, max_p) = compute_price_bounds(bars, last_close);

    println!("Price grid: {} → {}", min_p, max_p);

    let mut heatmap = vec![vec![0.0; buckets]; horizon];

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0f64, 1.0f64).unwrap();

    let atm_vol = sabr.atm_vol.max(0.01);

    let drift_step = (drift * 0.25).clamp(-0.03, 0.03);

    for _ in 0..paths {
        let mut price = last_close;

        for t in 0..horizon {
            // SABR local vol
            let sabr_vol = sabr_implied_vol(sabr.f0, price, sabr);
            let smile_scale = (sabr_vol / atm_vol).clamp(0.5, 2.0);

            // stochastic vol factor
            let vol_shock: f64 = normal.sample(&mut rng) * 0.5f64;
            let local_sigma = (base_sigma * smile_scale * (1.0f64 + vol_shock.abs())).max(0.01);

            let z: f64 = normal.sample(&mut rng);
            price *= (drift_step + local_sigma * z).exp();

            // rare jump
            if rng.gen_bool(0.02) {
                let jump_z: f64 = normal.sample(&mut rng);
                price *= (local_sigma * 4.0 * jump_z).exp();
            }

            if price.is_finite() && price > 0.0 {
                let b = price_to_bucket(price, min_p, max_p, buckets);
                heatmap[t][b] += 1.0;
            }
        }
    }

    for row in heatmap.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }

    heatmap
}

fn price_to_bucket(price: f64, min: f64, max: f64, buckets: usize) -> usize {
    if price <= min {
        0
    } else if price >= max {
        buckets - 1
    } else {
        let r = (price - min) / (max - min);
        let idx = (r * buckets as f64) as isize;
        idx.clamp(0, buckets as isize - 1) as usize
    }
}

//////////////////////////////////////////////////////////////
// VISUALIZATION
//////////////////////////////////////////////////////////////

fn save_heatmap_png(heatmap: &[Vec<f64>], path: &str) -> Result<(), Box<dyn Error>> {
    let h = heatmap.len();
    let w = heatmap[0].len();

    let mut img = ImageBuffer::new(h as u32, w as u32);

    let mut maxv = 0.0;
    for row in heatmap {
        for &v in row {
            if v > maxv {
                maxv = v;
            }
        }
    }
    if maxv <= 0.0 {
        maxv = 1e-6;
    }

    for t in 0..h {
        for b in 0..w {
            let p = heatmap[t][b] / maxv;
            let (r, g, bb) = probability_to_color(p);
            img.put_pixel(t as u32, (w - 1 - b) as u32, Rgb([r, g, bb]));
        }
    }

    img.save(path)?;
    Ok(())
}

fn probability_to_color(p: f64) -> (u8, u8, u8) {
    let x = p.sqrt();
    let r = (255.0 * x) as u8;
    let g = (140.0 * x.powf(1.5)) as u8;
    let b = (40.0 * (1.0 - x)) as u8;
    (r, g, b)
}

//////////////////////////////////////////////////////////////
// DISTRIBUTION STATS + SUMMARY
//////////////////////////////////////////////////////////////

fn bucket_to_price(bucket: usize, min: f64, max: f64, buckets: usize) -> f64 {
    let r = (bucket as f64 + 0.5) / buckets as f64;
    min + r * (max - min)
}

/// Optional smoothing of a discrete probability row.
fn smooth_distribution(row: &[f64]) -> Vec<f64> {
    if row.len() < 3 {
        return row.to_vec();
    }

    let kernel: [f64; 5] = [0.06, 0.24, 0.40, 0.24, 0.06];
    let k = kernel.len() as isize;
    let half = k / 2;

    let n = row.len();
    let mut out = vec![0.0; n];

    for i in 0..n {
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for (offset, &w) in kernel.iter().enumerate() {
            let j = i as isize + offset as isize - half;
            if j >= 0 && j < n as isize {
                acc += w * row[j as usize];
                wsum += w;
            }
        }
        if wsum > 0.0 {
            out[i] = acc / wsum;
        }
    }

    let s: f64 = out.iter().sum();
    if s > 0.0 {
        for v in out.iter_mut() {
            *v /= s;
        }
    }

    out
}

/// mean, std, skew, kurtosis, mode
fn summarize_distribution_extended(
    row: &[f64],
    min_p: f64,
    max_p: f64,
) -> (f64, f64, f64, f64, f64) {
    let buckets = row.len();
    if buckets == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let width = (max_p - min_p) / buckets as f64;

    let mut mean = 0.0;
    for (i, &prob) in row.iter().enumerate() {
        let price = min_p + (i as f64 + 0.5) * width;
        mean += price * prob;
    }

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for (i, &prob) in row.iter().enumerate() {
        let price = min_p + (i as f64 + 0.5) * width;
        let diff = price - mean;
        let diff2 = diff * diff;
        m2 += prob * diff2;
        m3 += prob * diff2 * diff;
        m4 += prob * diff2 * diff2;
    }

    let std = m2.max(0.0).sqrt();
    let skew = if std > 0.0 { m3 / std.powi(3) } else { 0.0 };
    let kurt = if std > 0.0 { m4 / std.powi(4) } else { 0.0 };

    let (mode_idx, _) = row
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let mode_price = min_p + (mode_idx as f64 + 0.5) * width;

    (mean, std, skew, kurt, mode_price)
}

fn print_bucket_summary(heatmap: &[Vec<f64>], bars: &[MarketBar]) {
    let last = bars.last().unwrap().close;
    let (min_p, max_p) = compute_price_bounds(bars, last);
    let buckets = heatmap[0].len();

    println!("\n========= FUTURE PRICE PROJECTIONS =========");
    println!("Current price: {:.4}", last);
    println!("Grid range: {:.4} → {:.4}", min_p, max_p);

    for t in 0..20.min(heatmap.len()) {
        let raw_row = &heatmap[t];
        let row = smooth_distribution(raw_row);

        let (mean_price, std_price, skew, kurt, mode_price) =
            summarize_distribution_extended(&row, min_p, max_p);

        let var95_down = mean_price - 1.96 * std_price;
        let var95_up = mean_price + 1.96 * std_price;
        let band2_down = mean_price - 2.0 * std_price;
        let band2_up = mean_price + 2.0 * std_price;

        println!("\nt+{}:", t);
        println!("  mean     = {:.4}", mean_price);
        println!("  std      = {:.4}", std_price);
        println!("  mode     = {:.4}", mode_price);
        println!("  skew     = {:.4}", skew);
        println!("  kurtosis = {:.4}", kurt);
        println!("  95% band (≈1.96σ)  = [{:.4}, {:.4}]", var95_down, var95_up);
        println!("  2σ band (approx)   = [{:.4}, {:.4}]", band2_down, band2_up);

        let mut v: Vec<(usize, f64)> = row.iter().cloned().enumerate().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (bucket, prob) in v.into_iter().take(3) {
            let price = bucket_to_price(bucket, min_p, max_p, buckets);
            println!("  {:>6.2}% → {:.4}", prob * 100.0, price);
        }
    }

    println!("============================================\n");
}
