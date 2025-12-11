
// src/data.rs

use chrono::TimeZone;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;

/// Single OHLCV bar with basic microstructure info.
#[derive(Debug, Clone)]
pub struct MarketBar {
    pub unix: i64,
    pub time: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trades: f64,
}

/// Clean a string cell:
/// - trim whitespace
/// - strip BOM
/// - strip surrounding quotes
fn clean(s: &str) -> String {
    s.trim()
        .trim_matches('\u{feff}')
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

/// Parse an f64 from an optional string cell, logging failures but not panicking.
fn parse_f64(row: usize, col: usize, x: Option<&String>) -> f64 {
    match x {
        Some(v) => {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return 0.0;
            }
            match trimmed.parse::<f64>() {
                Ok(val) => val,
                Err(_) => {
                    eprintln!("Row {} col {}: '{}' invalid f64 → 0", row, col, v);
                    0.0
                }
            }
        }
        None => 0.0,
    }
}

/// Try to parse a datetime string into a UNIX timestamp (UTC).
///
/// Supports:
/// - "%Y-%m-%d %H:%M:%S%.f"
/// - "%Y-%m-%d %H:%M:%S"
fn parse_unix_ts(row: usize, raw: &str) -> i64 {
    use chrono::{DateTime, NaiveDateTime, Utc};

    let s = raw.trim();
    if s.is_empty() {
        eprintln!("Row {} datetime empty", row);
        return 0;
    }

    // First try with fractional seconds.
    let try_formats = [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
    ];

    for fmt in &try_formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            
            let dt_utc = Utc.from_utc_datetime(&dt);
            return dt_utc.timestamp();
        }
    }

    eprintln!("Row {} datetime invalid '{}'", row, raw);
    0
}

/// Read OHLCV bars from a CSV with header.
///
/// Expected column layout (by index):
/// 0: datetime string (e.g. "2024-01-01 00:00:00")
/// 1: open
/// 2: high
/// 3: low
/// 4: close
/// 5: volume
/// 8: trades (some datasets have this at index 8; others might be 0)
///
/// Rows with too few columns are skipped.
pub fn read_bars<P: AsRef<Path>>(path: P) -> Result<Vec<MarketBar>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let mut bars = Vec::new();

    for (i, row) in rdr.records().enumerate() {
        let record = match row {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Row {} skipped, error = {}", i, e);
                continue;
            }
        };

        // Need at least enough columns for datetime + OHLCV + trades.
        if record.len() < 9 {
            eprintln!("Row {} skipped ({} cols < 9)", i, record.len());
            continue;
        }

        let cols: Vec<String> = record.iter().map(|s| clean(s)).collect();

        let datetime_str = cols[0].clone();
        let unix = parse_unix_ts(i, &datetime_str);

        let open = parse_f64(i, 1, cols.get(1));
        let high = parse_f64(i, 2, cols.get(2));
        let low = parse_f64(i, 3, cols.get(3));
        let close = parse_f64(i, 4, cols.get(4));
        let volume = parse_f64(i, 5, cols.get(5));
        let trades = parse_f64(i, 8, cols.get(8));

        // Basic sanity: skip bars with totally broken price fields.
        if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() {
            eprintln!("Row {} skipped (non-finite OHLC)", i);
            continue;
        }

        bars.push(MarketBar {
            unix,
            time: datetime_str,
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
