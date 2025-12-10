
use crate::features::mean_std;

#[derive(Debug, Copy, Clone)]
pub struct SabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
    pub f0: f64,
    pub atm_vol: f64,
}

/// Estimate SABR params from returns — OK logic
pub fn estimate_sabr_params(returns: &[f64], base_sigma: f64, f0: f64) -> SabrParams {
    let n = returns.len().min(200);
    let slice = &returns[returns.len() - n..];

    let (mean_r, std_r) = mean_std(slice);
    let std_r = std_r.max(1e-6);

    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &r in slice {
        let x = r - mean_r;
        m3 += x.powi(3);
        m4 += x.powi(4);
    }

    m3 /= n as f64;
    m4 /= n as f64;

    let skew = m3 / std_r.powi(3);
    let kurt = m4 / std_r.powi(4);
    let kurt_excess = (kurt - 3.0).max(0.0);

    let beta = 0.7;
    let rho = (skew / 3.0).tanh().clamp(-0.7, 0.7);
    let nu = (0.4 + 0.25 * kurt_excess).clamp(0.2, 3.0);

    let alpha = base_sigma * f0.powf(1.0 - beta);

    SabrParams {
        alpha,
        beta,
        rho,
        nu,
        f0,
        atm_vol: base_sigma,
    }
}

/// Correct SABR: vol(F,K)
pub fn sabr_implied_vol(F: f64, K: f64, p: &SabrParams) -> f64 {
    if F <= 0.0 || K <= 0.0 {
        return p.atm_vol;
    }

    let alpha = p.alpha;
    let beta = p.beta;
    let rho = p.rho;
    let nu = p.nu;

    let one_minus_beta = 1.0 - beta;
    let fk_beta = (F * K).powf(0.5 * one_minus_beta);

    let log_fk = (F / K).ln();

    let z = (nu / alpha) * fk_beta * log_fk;
    let eps = 1e-12;

    let xz = if z.abs() < eps {
        1.0
    } else {
        let arg = (1.0 - 2.0 * rho * z + z * z).sqrt();
        let num = (arg + z - rho).max(eps);
        let den = (1.0 - rho).max(eps);
        (num / den).ln()
    };

    let z_over_x = if xz.abs() < eps { 1.0 } else { z / xz };

    let vol0 = alpha / fk_beta;
    let corr = 1.0
        + (one_minus_beta * one_minus_beta * log_fk.powi(2) / 24.0)
        + (rho * beta * nu * alpha / (4.0 * fk_beta))
        + ((2.0 - 3.0 * rho * rho) * nu * nu / 24.0);

    let vol = vol0 * z_over_x * corr;

    if !vol.is_finite() || vol <= 0.0 {
        p.atm_vol
    } else {
        vol
    }
}
