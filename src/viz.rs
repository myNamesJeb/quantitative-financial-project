
// src/viz.rs

use image::{ImageBuffer, Rgb};
use std::error::Error;

use crate::data::MarketBar;

/// Map bucket index → price on the [min, max] grid.
fn bucket_to_price(bucket: usize, min: f64, max: f64, buckets: usize) -> f64 {
    if buckets == 0 {
        return min;
    }
    let r = (bucket as f64 + 0.5) / buckets as f64;
    min + r * (max - min)
}

/// Simple smoothing kernel over a discrete distribution.
/// Keeps the original shape but reduces noise.
///
/// - Applies a symmetric 5-tap kernel
/// - Preserves total mass (row sums to 1 if input did)
/// - Ensures non-negative entries after smoothing
fn smooth_row(row: &[f64]) -> Vec<f64> {
    if row.len() < 5 {
        // For very short rows, just renormalize and return.
        let mut out = row.to_vec();
        let s: f64 = out.iter().cloned().filter(|x| x.is_finite()).sum();
        if s > 0.0 {
            for v in out.iter_mut() {
                *v = (*v).max(0.0);
                *v /= s;
            }
        }
        return out;
    }

    let kernel = [0.06, 0.24, 0.40, 0.24, 0.06];
    let mut out = vec![0.0; row.len()];

    for i in 0..row.len() {
        let mut acc = 0.0;
        let mut wsum = 0.0;

        for (k, &w) in kernel.iter().enumerate() {
            let j = i as isize + k as isize - 2;
            if j >= 0 && (j as usize) < row.len() {
                let val = row[j as usize];
                if val.is_finite() && val >= 0.0 {
                    acc += w * val;
                    wsum += w;
                }
            }
        }

        if wsum > 0.0 {
            out[i] = (acc / wsum).max(0.0);
        }
    }

    // Normalize to sum to 1 (if possible).
    let s: f64 = out.iter().cloned().filter(|x| x.is_finite()).sum();
    if s > 0.0 {
        for v in out.iter_mut() {
            *v = (*v).max(0.0);
            *v /= s;
        }
    }

    out
}

/// Summarize one discrete distribution over price grid.
///
/// Returns:
/// (mean, std, skew, kurtosis, mode_price)
fn summarize_distribution(
    row: &[f64],
    min_p: f64,
    max_p: f64,
) -> (f64, f64, f64, f64, f64) {
    let buckets = row.len();
    if buckets == 0 || !min_p.is_finite() || !max_p.is_finite() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let width = (max_p - min_p) / buckets as f64;

    // Mean
    let mut mean = 0.0;
    for (i, &p) in row.iter().enumerate() {
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        let price = min_p + (i as f64 + 0.5) * width;
        mean += price * p;
    }

    // Central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for (i, &p) in row.iter().enumerate() {
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        let price = min_p + (i as f64 + 0.5) * width;
        let d = price - mean;
        m2 += p * d.powi(2);
        m3 += p * d.powi(3);
        m4 += p * d.powi(4);
    }

    let std = m2.sqrt();
    let skew = if std > 0.0 { m3 / std.powi(3) } else { 0.0 };
    let kurt = if std > 0.0 { m4 / std.powi(4) } else { 0.0 };

    // Mode (bucket with max probability)
    let mut mode_idx = 0usize;
    let mut mode_val = f64::NEG_INFINITY;
    for (i, &p) in row.iter().enumerate() {
        if p.is_finite() && p > mode_val {
            mode_val = p;
            mode_idx = i;
        }
    }
    let mode = min_p + (mode_idx as f64 + 0.5) * width;

    (
        mean,
        if std.is_finite() { std } else { 0.0 },
        if skew.is_finite() { skew } else { 0.0 },
        if kurt.is_finite() { kurt } else { 0.0 },
        mode,
    )
}

/// Print the 20-step summary using the same grid used in simulation.
pub fn print_bucket_summary(
    heatmap: &[Vec<f64>],
    min_p: f64,
    max_p: f64,
    bars: &[MarketBar],
) {
    if heatmap.is_empty() || heatmap[0].is_empty() || bars.is_empty() {
        eprintln!("print_bucket_summary: empty heatmap or bars");
        return;
    }

    let last = bars.last().unwrap().close;
    let buckets = heatmap[0].len();

    println!("\n========= FUTURE PRICE PROJECTIONS =========");
    println!("Current price: {:.4}", last);
    println!("Grid range: {:.4} → {:.4}", min_p, max_p);

    let steps = 20.min(heatmap.len());

    for t in 0..steps {
        let raw = &heatmap[t];
        let row = smooth_row(raw);

        let (mut mean, mut std, mut skew, mut kurt, mut mode) =
            summarize_distribution(&row, min_p, max_p);

        // Clean up any remaining NaN/inf artifacts.
        if !mean.is_finite() {
            mean = last;
        }
        if !std.is_finite() || std < 0.0 {
            std = 0.0;
        }
        if !skew.is_finite() {
            skew = 0.0;
        }
        if !kurt.is_finite() {
            kurt = 0.0;
        }
        if !mode.is_finite() {
            mode = mean;
        }

        println!("\nt+{}:", t);
        println!("  mean     = {:.4}", mean);
        println!("  std      = {:.4}", std);
        println!("  mode     = {:.4}", mode);
        println!("  skew     = {:.4}", skew);
        println!("  kurtosis = {:.4}", kurt);

        // Confidence bands: if std is tiny, bands collapse to mean.
        let lo95 = mean - 1.96 * std;
        let hi95 = mean + 1.96 * std;
        let lo2 = mean - 2.0 * std;
        let hi2 = mean + 2.0 * std;

        println!("  95% band (≈1.96σ)  = [{:.4}, {:.4}]", lo95, hi95);
        println!("  2σ band (approx)   = [{:.4}, {:.4}]", lo2, hi2);

        // Top 3 most probable buckets.
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

/// Save heatmap PNG using the same grid as simulation.
///
/// - x-axis: time step
/// - y-axis: price bucket (top = high, bottom = low)
pub fn save_heatmap_png(
    heatmap: &[Vec<f64>],
    min_p: f64,
    max_p: f64,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    if heatmap.is_empty() || heatmap[0].is_empty() {
        return Err("save_heatmap_png: empty heatmap".into());
    }

    let h = heatmap.len();
    let w = heatmap[0].len();

    let mut img = ImageBuffer::new(h as u32, w as u32);

    // Find max probability to normalize intensity.
    let mut maxv = 0.0;
    for row in heatmap {
        for &v in row {
            if v.is_finite() && v > maxv {
                maxv = v;
            }
        }
    }
    if maxv <= 0.0 || !maxv.is_finite() {
        maxv = 1e-6;
    }

    for t in 0..h {
        for b in 0..w {
            let raw_p = heatmap[t][b];
            let p = if raw_p.is_finite() && raw_p >= 0.0 {
                (raw_p / maxv).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (r, g, bb) = probability_to_color(p);
            // Flip y so higher buckets are visually at the top.
            img.put_pixel(t as u32, (w - 1 - b) as u32, Rgb([r, g, bb]));
        }
    }

    img.save(path)?;
    Ok(())
}

/// Map probability ∈ [0, 1] to a color.
///
/// We use a simple "warm" colormap:
/// - low p: dark, slightly bluish
/// - medium p: orange
/// - high p: bright yellow / white-ish
fn probability_to_color(p: f64) -> (u8, u8, u8) {
    let x = p.clamp(0.0, 1.0).sqrt(); // gamma for more contrast in low values

    // Base intensities in [0, 255].
    let r = (255.0 * x) as u8;
    let g = (140.0 * x.powf(1.3)) as u8;
    let b = (40.0 * (1.0 - x)).max(0.0) as u8;

    (r, g, b)
}
