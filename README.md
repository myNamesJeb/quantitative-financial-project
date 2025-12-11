# Overview

This project is a full forecasting and simulation pipeline for financial time series built in Rust. It ingests historical price data, constructs machine learning features, estimates volatility through several statistical models, classifies market regimes, produces forward price distributions using Monte Carlo simulation with SABR local volatility, and finally renders a heatmap that visualizes probable future price paths.

The system also includes a one hour backtester that evaluates a strategy which uses the forecasts in real time. The strategy dynamically adjusts exposure based on estimated drift, volatility, the shape of the future price distribution, and the inferred market regime. Everything is written to be deterministic, debuggable, and easy to extend.

# Design Goals

1. **Full pipeline transparency**  
   Every stage of computation is explicit. There is no black box model such as a neural network that hides its internal reasoning. Each statistical component can be inspected, logged, and replaced with alternative models.

2. **Time consistent forecasting**  
   The forecasting engine uses only data available up to the current bar. No future information leaks into the model. This creates correct out of sample behavior.

3. **Distributional forecasting instead of single point estimates**  
   The system produces a probability distribution for future prices rather than a single prediction. This allows the strategy to evaluate uncertainty, skew, and tail risks.

4. **Modular architecture**  
   Each major component lives in its own module:
   - `data` loads and cleans CSVs  
   - `features` builds ML inputs  
   - `ml` handles regression and regime classification  
   - `advanced` holds custom statistical tools such as Kalman drift filtering and GARCH  
   - `drift` combines signals into a single drift and volatility estimate  
   - `sabr` estimates local volatility curvature  
   - `simulate` runs Monte Carlo using SABR dynamics  
   - `viz` prints statistical summaries and creates PNG heatmaps  
   - `strategy` defines the trading rules  
   - `backtest` evaluates everything in a realistic loop  

   This separation makes it possible to test components individually and reason about the outputs at each stage.

5. **Math and modeling over heuristics**  
   Instead of ad hoc rules, the project consistently uses statistical definitions:
   - Drift is smoothed with a one dimensional Kalman filter.  
   - Volatility combines GARCH estimation, ML residuals, an RNN like smoother, and clustering diagnostics based on autocorrelation.  
   - Market regime classification uses distributional signals and a simple three state hidden Markov style process.  
   - The Monte Carlo and SABR solver implement well defined continuous time models.

6. **Interpretability**  
   For teaching and debugging purposes, every forecast is summarized numerically (mean, standard deviation, skewness, kurtosis, and mode) and visually through the heatmap.

# High Level Architecture

### 1. Data ingestion  
The system loads multiple timeframes (1D, 4H, 1H, and 15M) from CSV files, cleans each row, parses timestamps, and stores everything as `MarketBar` objects.

### 2. Feature construction  
The project computes:
- Log returns  
- Short and long horizon volatility estimates  
- Volume and trade statistics standardized to z scores  
- Rolling ranges and microstructure patterns  

These become input features for the regression model.

### 3. Machine learning prediction  
A simple ridge regression ensemble (bagged linear models) predicts one step ahead returns and the residual standard deviation becomes an ML based volatility estimate. This approach is chosen for transparency and stability. The weights are directly interpretable.

### 4. Regime classification  
The system quantizes the volatility of returns into low, medium, and high states, similar to a simplified hidden Markov model. It combines this with the ML directional signal to decide whether the environment is likely trending up, trending down, low volatility sideways, or high volatility sideways.

### 5. Drift and volatility synthesis  
The project does not trust any single model. It merges:
- Kalman filtered drift  
- ML prediction  
- Flow signals based on microstructure  
- GARCH predicted volatility  
- RNN like volatility smoothing  
- Autocorrelation of squared returns for clustering  

The result is a stable drift and volatility estimate that reacts to changing market structure.

### 6. SABR curvature  
SABR is a local volatility model used to capture skew and kurtosis in continuous time diffusion processes. The implementation here estimates approximate parameters from return moments. These parameters are then used to compute a volatility value for each simulated step.

### 7. Monte Carlo simulation  
The simulator generates thousands of price paths ahead using:
- Drift per step  
- Local volatility based on SABR  
- Stochastic volatility shocks  
- Rare but mild price jumps  

The output is a matrix where each row is a future time and each column is a price bucket. Values represent probabilities.

### 8. Heatmap and statistical summaries  
The heatmap is saved as a PNG. A textual summary reports mean, standard deviation, skewness, kurtosis, confidence bands, and the most probable prices.

### 9. Strategy and backtester  
The backtester uses a strategy that converts forecast statistics into a position:
- Drift z score  
- Mean versus current price shift  
- Distribution skew  
- Distribution slope  
- Regime bias  

It applies cooldown, risk limits, and a drawdown stop. This produces a realistic trades list, equity curve, Sharpe ratio, and drawdown numbers.

# Project Journal  
Newest entries first.

## 12/11/2025  
refactored the code so the simulation and backtesting pipelines share the same statistical inputs and SABR configuration. The forecasts now behave consistently which fixed an earlier mismatch between Monte Carlo outputs and strategy inputs.  
---

## 12/09/2025  
improved the visualization logic and made the bucket summary easier to interpret. This helped confirm that the log return distribution from the model matches the expected heavy tail behavior.  
---

## 12/07/2025  
tightened the SABR parameter clamps to prevent unrealistic wings in the forecast distribution. This fixed a rare failure where a few simulated paths exploded upward due to extreme local variance.  
---

## 12/05/2025  
added a volatility clustering diagnostic based on autocorrelation of squared returns. This made the volatility estimate respond more realistically to regime transitions.  
---

## 12/03/2025  
implemented the RNN style volatility smoother which stabilizes the short term volatility estimate. This reduces noise and improves the Monte Carlo step sizes.  
---

## 12/01/2025  
The Kalman drift filter was added to smooth the directional signal. This improved the stability of the strategy since raw returns were too noisy to use directly.  
---

## 11/29/2025  
finished the ridge regression ensemble which is now responsible for generating the baseline drift and ML volatility estimate. This allowed the system to detect directional changes earlier.  
---

## 11/27/2025  
Initial project setup. structured the repository into separate modules for data handling, feature building, ML, simulation, and strategy logic so each part could be tested and replaced independently.  
---

# How to Run

```
cargo run out.png
```

The program loads the datasets, runs the backtester, generates a forecast for the most recent bar, prints a statistical summary, and saves a heatmap named `out.png`.

# Final Notes

This project focuses on interpretability and modular statistical modeling rather than predictive performance alone. Every part of the pipeline expresses a mathematical idea directly. The goal is to create a system that is explainable, testable, and extensible for future experimentation.
