
use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;

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

fn clean(s: &str) -> String {
    s.trim()
        .trim_matches('\u{feff}')
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn parse_f64(row: usize, col: usize, x: Option<&String>) -> f64 {
    match x {
        Some(v) => v.parse::<f64>().unwrap_or_else(|_| {
            eprintln!("Row {} col {}: '{}' invalid f64 → 0", row, col, v);
            0.0
        }),
        None => 0.0,
    }
}

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

        if record.len() < 12 {
            eprintln!("Row {} skipped ({} cols)", i, record.len());
            continue;
        }

        let cols: Vec<String> = record.iter().map(|s| clean(s)).collect();

        let datetime_str = cols[0].clone();
        let unix = match chrono::NaiveDateTime::parse_from_str(
            &datetime_str,
            "%Y-%m-%d %H:%M:%S%.f",
        ) {
            Ok(dt) => chrono::DateTime::<chrono::Utc>::from_utc(dt, chrono::Utc).timestamp(),
            Err(_) => {
                eprintln!("Row {} datetime invalid '{}'", i, cols[0]);
                0
            }
        };

        bars.push(MarketBar {
            unix,
            time: datetime_str,
            open: parse_f64(i, 1, cols.get(1)),
            high: parse_f64(i, 2, cols.get(2)),
            low: parse_f64(i, 3, cols.get(3)),
            close: parse_f64(i, 4, cols.get(4)),
            volume: parse_f64(i, 5, cols.get(5)),
            trades: parse_f64(i, 8, cols.get(8)),
        });
    }

    Ok(bars)
}
