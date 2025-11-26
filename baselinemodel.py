import numpy as np
import pandas as pd
from pathlib import Path

# ===============================
# CONFIG
# ===============================
CSV_PATH = "sp500_features.csv"   # <-- your merged file
TICKER_COL_CANDIDATES = [
    "Close", "Adj Close",
    "('Close','^GSPC')", "('Adj Close','^GSPC')",
    "Close_^GSPC", "Adj Close_^GSPC"
]
RISK_AVERSION_GAMMA = 4.0        # higher = more risk-averse (try 2–10)
ANNUALIZATION = 252              # daily data
TC_BPS = 5                       # round-trip transaction cost in basis points (e.g., 5 = 0.05%)
BAND_MULT = 1.5                  # scales the no-trade band vs costs/vol
MAX_LEVERAGE = 1.0               # cap weight to [0,1] (long-only)
MIN_WEIGHT, MAX_WEIGHT = 0.0, 1.0

# Short vs long horizon blending
ALPHA_SHORT = 0.5                # weight on short-term mu (0..1)
LOOKBACK_ST = 20                 # short-term vol/mu windows
LOOKBACK_LT = 126                # long-term vol/mu windows (~6 months)

# ===============================
# HELPERS
# ===============================
def pick_price_column(df):
    # Try common variants
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    # Try to find a "Close" in MultiIndex->stringified names
    for c in df.columns:
        cs = str(c).lower()
        if "close" in cs:
            return c
    raise KeyError("Could not find an S&P 500 close column. Inspect df.columns")

def to_daily_rate(fed_funds):
    """Convert annualized % rate to daily decimal; handles NaNs."""
    # If given in percent (e.g., 5.25), convert to 0.0525
    r_ann = fed_funds.copy().astype(float) / (100.0 if fed_funds.max() > 1.0 else 1.0)
    # Continuous daily rate approximation
    return r_ann / ANNUALIZATION

def rolling_ewm_mean(x, span):
    return x.ewm(span=span, adjust=False).mean()

def sharpe_ratio(rets):
    mu = rets.mean() * ANNUALIZATION
    sd = rets.std() * np.sqrt(ANNUALIZATION)
    return np.nan if sd == 0 else mu / sd

def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return dd.min()

# ===============================
# LOAD
# ===============================
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0)
df = df.sort_index()

# Normalize column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

price_col = pick_price_column(df)
px = df[price_col].astype(float).copy()

# Basic daily log returns
logret = np.log(px).diff()
ret = px.pct_change()  # simple return for PnL

# Macros/Trends if present (optional; the script auto-detects)
macro_cols = [c for c in ["CPI","Unemployment","FedFunds"] if c in df.columns]
trend_cols = [c for c in ["recession","stock market crash","inflation"] if c in df.columns]

# Risk-free daily rate from FedFunds if available, else 0
if "FedFunds" in df.columns:
    r_daily = to_daily_rate(df["FedFunds"]).reindex(df.index).ffill().fillna(0.0)
else:
    r_daily = pd.Series(0.0, index=df.index)

# ===============================
# FEATURE ENGINEERING (ST + LT)
# ===============================
# Short-term features (OHLCV-driven)
mu_st = rolling_ewm_mean(logret, span=LOOKBACK_ST) * ANNUALIZATION  # annualized drift estimate (short)
sigma_st = logret.ewm(span=LOOKBACK_ST, adjust=False).std() * np.sqrt(ANNUALIZATION)

# Long-term features (macro/trends + slow price)
mu_lt_price = rolling_ewm_mean(logret, span=LOOKBACK_LT) * ANNUALIZATION
sigma_lt = logret.ewm(span=LOOKBACK_LT, adjust=False).std() * np.sqrt(ANNUALIZATION)

# Macro signal: simple z-scored blend to tilt expected excess return
macro_signal = pd.Series(0.0, index=df.index)
if macro_cols:
    zmac = pd.DataFrame({c:(df[c]-df[c].rolling(LOOKBACK_LT).mean())/df[c].rolling(LOOKBACK_LT).std()
                         for c in macro_cols}).clip(-3,3)
    # Heuristic weights: higher CPI -> lower expected ER; higher Unemployment -> lower ER; higher FedFunds -> lower ER
    w = {
        "CPI": -0.4,
        "Unemployment": -0.3,
        "FedFunds": -0.3
    }
    w = {k:v for k,v in w.items() if k in zmac.columns}
    macro_signal = (zmac.mul(pd.Series(w))).sum(axis=1)

trend_signal = pd.Series(0.0, index=df.index)
if trend_cols:
    ztr = pd.DataFrame({c:(df[c]-df[c].rolling(LOOKBACK_LT).mean())/df[c].rolling(LOOKBACK_LT).std()
                        for c in trend_cols}).clip(-3,3)
    # Heuristic: higher search interest in recession/crash/inflation -> lower ER
    wt = {c:-1/len(ztr.columns) for c in ztr.columns}
    trend_signal = (ztr.mul(pd.Series(wt))).sum(axis=1)

# Combine into a long-term tilt to mu (scale to bps)
mu_tilt = (macro_signal + trend_signal).fillna(0.0) * 0.01  # 1% per z-unit total tilt (tunable)

# Final expected annual drift (blend ST/LT + macro/trend tilt)
mu_ann = (ALPHA_SHORT * mu_st + (1-ALPHA_SHORT) * mu_lt_price) + mu_tilt
# Use a conservative volatility estimate (LT)
sigma_ann = sigma_lt.clip(lower=1e-6)

# ===============================
# HJB/MERTON TARGET WEIGHT
# u* = (mu - r) / (gamma * sigma^2)
# ===============================
excess_mu = (mu_ann - r_daily * ANNUALIZATION)  # both annualized
u_star_cont = excess_mu / (RISK_AVERSION_GAMMA * (sigma_ann**2))
u_star = u_star_cont.clip(MIN_WEIGHT, MAX_WEIGHT)

# ===============================
# TRANSACTION COSTS & NO-TRADE BAND (DP/MPC flavor)
# Band ~ f(cost, vol): wider band when costs higher or vol higher
# ===============================
tc = TC_BPS / 1e4  # convert bps to decimal
band = BAND_MULT * np.sqrt(tc + 1e-8) * (sigma_ann / sigma_ann.median()).clip(lower=0.5, upper=2.0) * 0.02
band = band.fillna(band.median())

# ===============================
# BACKTEST LOOP
# ===============================
dates = df.index
w = 0.0                       # current portfolio weight
equity = 1.0                  # start wealth
equity_curve = []
weights, targets, turnovers = [], [], []

for t in range(1, len(dates)):
    date = dates[t]
    prev = dates[t-1]

    # Desired target weight today (from HJB/Merton)
    u_tgt = float(u_star.loc[date]) if date in u_star.index else w
    targets.append(u_tgt)

    # No-trade band: only rebalance if |target - current| > band
    b = float(band.loc[date]) if date in band.index else 0.0
    if abs(u_tgt - w) > b:
        w_new = np.clip(u_tgt, MIN_WEIGHT, MAX_WEIGHT)
        # Apply transaction cost on traded notional
        traded = abs(w_new - w)
        turnovers.append(traded)
        w = w_new
        # deduct cost immediately from equity
        equity *= (1 - tc * traded)
    else:
        turnovers.append(0.0)

    weights.append(w)

    # Realized portfolio return over (prev,date)
    r_eq = ret.loc[date] if date in ret.index else 0.0
    r_rf = r_daily.loc[date] if date in r_daily.index else 0.0
    # Discrete daily return combining risky and risk-free sleeves
    port_ret = w * r_eq + (1 - w) * r_rf
    equity *= (1 + port_ret)
    equity_curve.append(equity)

# ===============================
# METRICS
# ===============================
equity_series = pd.Series(equity_curve, index=dates[1:])
port_rets = equity_series.pct_change().fillna(0.0)
cagr = equity_series.iloc[-1]**(ANNUALIZATION/len(equity_series)) - 1
sr = sharpe_ratio(port_rets)
mdd = max_drawdown(equity_series)
turnover_annual = (pd.Series(turnovers, index=dates[1:]).sum() / len(equity_series)) * ANNUALIZATION

print("=== HJB-inspired Dynamic Allocation Backtest ===")
print(f"Start: {dates[0].date()}  End: {dates[-1].date()}  N={len(dates)}")
print(f"CAGR:   {cagr:6.2%}")
print(f"Sharpe: {sr:6.2f}")
print(f"MaxDD:  {mdd:6.2%}")
print(f"Turnover (ann.): {turnover_annual:6.2f}")
print(f"Avg target weight: {pd.Series(targets, index=dates[1:]).mean():.2f}")
print(f"Avg actual weight: {pd.Series(weights, index=dates[1:]).mean():.2f}")

# Optional: save time series
out = pd.DataFrame({
    "price": px.reindex(equity_series.index),
    "equity": equity_series,
    "port_ret": port_rets,
    "w_actual": pd.Series(weights, index=equity_series.index),
    "w_target": pd.Series(targets, index=equity_series.index),
    "mu_ann": mu_ann.reindex(equity_series.index),
    "sigma_ann": sigma_ann.reindex(equity_series.index),
    "band": band.reindex(equity_series.index)
}).dropna(how='all')
out.to_csv("hjb_dynamic_allocation_results.csv")
print("Saved: hjb_dynamic_allocation_results.csv")
