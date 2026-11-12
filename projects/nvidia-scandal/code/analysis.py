import json
import datetime
import numpy as np
import pandas as pd

pd.set_option("display.width", 140)

SHARES_OUT = {
    "NVDA": 24.19e9,
    "AMD": 1.632e9,
    "AVGO": 4.77e9,
}

def load(sym):
    d = json.load(open(f"{sym}.json"))["chart"]["result"][0]
    ts = d["timestamp"]
    closes = d["indicators"]["quote"][0]["close"]
    dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date() for t in ts]
    s = pd.Series(closes, index=pd.to_datetime(dates), name=sym).dropna()
    return s

nvda = load("NVDA")
amd = load("AMD")
avgo = load("AVGO")
ixic = load("IXIC")

prices = pd.concat([nvda, amd, avgo, ixic], axis=1).dropna()
rets = prices.pct_change().dropna()

# ---------------------------------------------------------------
# 1. Market model (beta vs Nasdaq Composite), estimated on a CLEAN
#    window well before the Oct/Nov 2026 turmoil, so the event itself
#    doesn't contaminate the beta estimate.
# ---------------------------------------------------------------
est = rets.loc["2025-10-15":"2026-10-15"]

def fit_market_model(stock_ret, mkt_ret):
    X = np.column_stack([np.ones(len(mkt_ret)), mkt_ret.values])
    beta_hat, *_ = np.linalg.lstsq(X, stock_ret.values, rcond=None)
    alpha, beta = beta_hat
    resid = stock_ret.values - X @ beta_hat
    resid_std = resid.std(ddof=2)
    return alpha, beta, resid_std

models = {}
for sym in ["NVDA", "AMD", "AVGO"]:
    alpha, beta, resid_std = fit_market_model(est[sym], est["IXIC"])
    models[sym] = dict(alpha=alpha, beta=beta, resid_std=resid_std)

print("=== Market model (estimation window 2025-10-15 to 2026-10-15) ===")
for sym, m in models.items():
    print(f"{sym}: alpha={m['alpha']*100:.4f}%/day  beta={m['beta']:.2f}  resid_std={m['resid_std']*100:.2f}%")

# ---------------------------------------------------------------
# 2. Event window: the two-session scandal-specific crash, Nov 9-10 2026
#    (board/FBI/subpoena escalation). Nov 5 is kept separate: it was a
#    broad AI/semiconductor selloff (AMD and AVGO fell even harder that day).
# ---------------------------------------------------------------
sector_selloff_window = ["2026-11-05"]
scandal_window = ["2026-11-09", "2026-11-10"]

def abnormal_returns(sym, dates):
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        actual = rets.loc[d, sym]
        mkt = rets.loc[d, "IXIC"]
        expected = models[sym]["alpha"] + models[sym]["beta"] * mkt
        ar = actual - expected
        rows.append(dict(date=d.date(), actual=actual, market=mkt, expected=expected, abnormal=ar))
    return pd.DataFrame(rows)

print("\n=== Sector-wide selloff, Nov 5 2026 (all three chip stocks fall together) ===")
for sym in ["NVDA", "AMD", "AVGO"]:
    df = abnormal_returns(sym, sector_selloff_window)
    print(sym, df.round(4).to_dict("records"))

print("\n=== Scandal-specific window, Nov 9-10 2026 ===")
car = {}
for sym in ["NVDA", "AMD", "AVGO"]:
    df = abnormal_returns(sym, scandal_window)
    print(f"\n{sym}")
    print(df.round(4))
    car[sym] = df["abnormal"].sum()
    print(f"{sym} cumulative abnormal return (CAR) over 2 sessions: {car[sym]*100:.2f}%")

# ---------------------------------------------------------------
# 3. Dollar impact: total vs abnormal (scandal-specific) vs
#    market/sector-explained, using actual market caps.
# ---------------------------------------------------------------
print("\n=== Dollar market-value impact, Nov 9-10 2026 ===")
summary_rows = []
for sym in ["NVDA", "AMD", "AVGO"]:
    p_before = prices.loc["2026-11-06", sym]
    p_after = prices.loc["2026-11-10", sym]
    shares = SHARES_OUT[sym]
    mcap_before = p_before * shares
    mcap_after = p_after * shares
    total_dollar = mcap_after - mcap_before
    total_pct = p_after / p_before - 1
    abnormal_pct = car[sym]
    abnormal_dollar = abnormal_pct * mcap_before
    market_explained_dollar = total_dollar - abnormal_dollar
    summary_rows.append(dict(
        symbol=sym,
        price_before=p_before, price_after=p_after,
        mcap_before_B=mcap_before/1e9, mcap_after_B=mcap_after/1e9,
        total_pct=total_pct*100, total_dollar_B=total_dollar/1e9,
        abnormal_pct=abnormal_pct*100, abnormal_dollar_B=abnormal_dollar/1e9,
        market_explained_dollar_B=market_explained_dollar/1e9,
    ))

summary = pd.DataFrame(summary_rows).set_index("symbol")
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(summary.T)

summary.to_csv("summary.csv")
rets.to_csv("daily_returns.csv")
prices.to_csv("daily_prices.csv")
print("\nSaved summary.csv, daily_returns.csv, daily_prices.csv")
