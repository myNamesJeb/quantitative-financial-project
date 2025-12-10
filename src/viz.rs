
use image::{ImageBuffer, Rgb};
use std::error::Error;

use crate::data::MarketBar;

/// Map bucket index → price
fn bucket_to_price(bucket: usize, min: f64, max: f64, buckets: usize) -> f64 {
    let r = (bucket as f64 + 0.5) / buckets as f64;
    min + r * (max - min)
}

/// Simple smoothing kernel over discrete distribution
fn smooth_row(row: &[f64]) -> Vec<f64> {
    if row.len() < 5 {
        return row.to_vec();
    }

    let kernel = [0.06, 0.24, 0.40, 0.24, 0.06];
    let mut out = vec![0.0; row.len()];

    for i in 0..row.len() {
        let mut acc = 0.0;
        let mut wsum = 0.0;

        for (k, &w) in kernel.iter().enumerate() {
            let j = i as isize + k as isize - 2;
            if j >= 0 && (j as usize) < row.len() {
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

/// Summarize one discrete distribution over price grid.
fn summarize_distribution(
    row: &[f64],
    min_p: f64,
    max_p: f64,
) -> (f64, f64, f64, f64, f64) {
    let buckets = row.len();
    if buckets == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
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

    (mean, std, skew, kurt, mode)
}

/// Print the 20-step summary using the **same grid** used in simulation.
pub fn print_bucket_summary(
    heatmap: &[Vec<f64>],
    min_p: f64,
    max_p: f64,
    bars: &[MarketBar],
) {
    let last = bars.last().unwrap().close;
    let buckets = heatmap[0].len();

    println!("\n========= FUTURE PRICE PROJECTIONS =========");
    println!("Current price: {:.4}", last);
    println!("Grid range: {:.4} → {:.4}", min_p, max_p);

    for t in 0..20.min(heatmap.len()) {
        let raw = &heatmap[t];
        let row = smooth_row(raw);

        let (mean, std, skew, kurt, mode) =
            summarize_distribution(&row, min_p, max_p);

        println!("\nt+{}:", t);
        println!("  mean     = {:.4}", mean);
        println!("  std      = {:.4}", std);
        println!("  mode     = {:.4}", mode);
        println!("  skew     = {:.4}", skew);
        println!("  kurtosis = {:.4}", kurt);

        let lo95 = mean - 1.96 * std;
        let hi95 = mean + 1.96 * std;
        let lo2 = mean - 2.0 * std;
        let hi2 = mean + 2.0 * std;

        println!("  95% band (≈1.96σ)  = [{:.4}, {:.4}]", lo95, hi95);
        println!("  2σ band (approx)   = [{:.4}, {:.4}]", lo2, hi2);

        // top 3 buckets
        let mut sorted: Vec<(usize, f64)> =
            row.iter().cloned().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, prob) in sorted.into_iter().take(3) {
            let price = bucket_to_price(idx, min_p, max_p, buckets);
            println!("  {:>6.2}% → {:.4}", prob * 100.0, price);
        }
    }

    println!("============================================\n");
}

/// Save heatmap PNG using the **same grid** as simulation.
pub fn save_heatmap_png(
    heatmap: &[Vec<f64>],
    min_p: f64,
    max_p: f64,
    path: &str,
) -> Result<(), Box<dyn Error>> {
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
