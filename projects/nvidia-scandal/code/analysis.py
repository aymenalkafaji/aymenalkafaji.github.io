import json
import datetime
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 160)

SHARES_OUT = {"NVDA": 24.19e9, "AMD": 1.632e9, "AVGO": 4.77e9}
BENCHMARKS = ["IXIC", "GSPC", "SOXX", "SMH"]
SUBJECTS = ["NVDA", "AMD", "AVGO"]

def load(sym):
    d = json.load(open(f"{sym}.json"))["chart"]["result"][0]
    ts = d["timestamp"]
    closes = d["indicators"]["quote"][0]["close"]
    dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date() for t in ts]
    return pd.Series(closes, index=pd.to_datetime(dates), name=sym).dropna()

prices = pd.concat([load(s) for s in SUBJECTS + BENCHMARKS], axis=1).dropna()
rets = prices.pct_change().dropna()
est = rets.loc["2025-10-15":"2026-10-15"]

def fit_market_model(y, x):
    """OLS market model with the full statistics needed for the refined
    (parameter-estimation-uncertainty-aware) event-window standard error."""
    X = np.column_stack([np.ones(len(x)), x.values])
    beta_hat, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    alpha, beta = beta_hat
    resid = y.values - X @ beta_hat
    n = len(y)
    dof = n - 2
    sigma2 = (resid ** 2).sum() / dof
    xbar = x.values.mean()
    Sxx = ((x.values - xbar) ** 2).sum()
    ss_tot = ((y.values - y.values.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot
    return dict(alpha=alpha, beta=beta, sigma2=sigma2, n=n, dof=dof, xbar=xbar, Sxx=Sxx, r2=r2)

models = {(s, b): fit_market_model(est[s], est[b]) for s in SUBJECTS for b in BENCHMARKS}

def event_study(subj, bench, dates):
    """CAR and its standard error over an event window, using the refined
    standard error that accounts for estimation uncertainty in alpha and
    beta (a prediction-interval-style correction), not just residual
    variance. See the methodology note in the article for the derivation."""
    m = models[(subj, bench)]
    ars, mkts = [], []
    for d in dates:
        d = pd.Timestamp(d)
        actual = rets.loc[d, subj]
        mkt = rets.loc[d, bench]
        ars.append(actual - (m["alpha"] + m["beta"] * mkt))
        mkts.append(mkt)
    L = len(dates)
    car = sum(ars)
    M = sum(mkts)
    var_refined = m["sigma2"] * (L + L ** 2 / m["n"] + L * ((M / L - m["xbar"]) ** 2) / m["Sxx"])
    se = np.sqrt(var_refined)
    t_stat = car / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=m["dof"]))
    t_crit = stats.t.ppf(0.975, df=m["dof"])
    return dict(car=car, se=se, t=t_stat, p=p_value, ci_low=car - t_crit * se, ci_high=car + t_crit * se, L=L)

PRIMARY_WINDOW = ["2026-11-09", "2026-11-10"]
SENSITIVITY_WINDOWS = {
    "Nov 9-10 (primary, 2 days)": ["2026-11-09", "2026-11-10"],
    "Nov 9-11 (+1 day, includes rebound)": ["2026-11-09", "2026-11-10", "2026-11-11"],
    "Nov 6-10 (from pre-event close)": ["2026-11-06", "2026-11-09", "2026-11-10"],
}
SECTOR_SELLOFF_DAY = ["2026-11-05"]

print("=== Market models ===")
for (s, b), m in models.items():
    print(f"{s} ~ {b}: alpha={m['alpha']*100:.4f}%/day beta={m['beta']:.3f} sigma={np.sqrt(m['sigma2'])*100:.3f}% R2={m['r2']:.3f} n={m['n']}")

print("\n=== Robustness, primary window (refined SE) ===")
rob_rows = []
for s in SUBJECTS:
    for b in BENCHMARKS:
        r = event_study(s, b, PRIMARY_WINDOW)
        rob_rows.append(dict(subject=s, benchmark=b, beta=models[(s,b)]["beta"], car=r["car"], se=r["se"],
                              t=r["t"], p=r["p"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
robustness = pd.DataFrame(rob_rows)
print(robustness.round(5).to_string(index=False))

print("\n=== Sensitivity (NVDA vs IXIC, refined SE) ===")
sens_rows = []
for label, dates in SENSITIVITY_WINDOWS.items():
    r = event_study("NVDA", "IXIC", dates)
    sens_rows.append(dict(window=label, L=r["L"], car=r["car"], se=r["se"], t=r["t"], p=r["p"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
sensitivity = pd.DataFrame(sens_rows)
print(sensitivity.round(5).to_string(index=False))

print("\n=== Nov 5 context (refined SE) ===")
sell_rows = []
for s in SUBJECTS:
    r = event_study(s, "IXIC", SECTOR_SELLOFF_DAY)
    sell_rows.append(dict(subject=s, car=r["car"], se=r["se"], t=r["t"], p=r["p"]))
print(pd.DataFrame(sell_rows).round(5).to_string(index=False))

print("\n=== Dollar reconciliation (NVDA, IXIC benchmark, refined SE) ===")
p_before = prices.loc["2026-11-06", "NVDA"]
p_after = prices.loc["2026-11-10", "NVDA"]
mcap_before = p_before * SHARES_OUT["NVDA"]
mcap_after = p_after * SHARES_OUT["NVDA"]
total_dollar = mcap_after - mcap_before
total_pct = p_after / p_before - 1
r = event_study("NVDA", "IXIC", PRIMARY_WINDOW)
abnormal_dollar = r["car"] * mcap_before
ci_low_dollar = r["ci_low"] * mcap_before
ci_high_dollar = r["ci_high"] * mcap_before
market_explained = total_dollar - abnormal_dollar
print(f"mcap_before=${mcap_before/1e9:.2f}B mcap_after=${mcap_after/1e9:.2f}B")
print(f"total_pct={total_pct*100:.2f}% total_dollar=${total_dollar/1e9:.2f}B")
print(f"abnormal_pct={r['car']*100:.2f}% t={r['t']:.2f} p={r['p']:.2e}")
print(f"abnormal_dollar=${abnormal_dollar/1e9:.2f}B  95% CI [${ci_low_dollar/1e9:.2f}B, ${ci_high_dollar/1e9:.2f}B]")
print(f"market_explained=${market_explained/1e9:.2f}B")

robustness.to_csv("robustness_benchmarks.csv", index=False)
sensitivity.to_csv("sensitivity_windows.csv", index=False)
pd.DataFrame([{
    "mcap_before_B": mcap_before/1e9, "mcap_after_B": mcap_after/1e9,
    "total_pct": total_pct*100, "total_dollar_B": total_dollar/1e9,
    "abnormal_pct": r["car"]*100, "abnormal_dollar_B": abnormal_dollar/1e9,
    "abnormal_dollar_low_B": ci_low_dollar/1e9, "abnormal_dollar_high_B": ci_high_dollar/1e9,
    "t_stat": r["t"], "p_value": r["p"], "market_explained_dollar_B": market_explained/1e9,
}]).to_csv("summary.csv", index=False)
rets.to_csv("daily_returns.csv")
prices.to_csv("daily_prices.csv")
print("\nSaved.")
