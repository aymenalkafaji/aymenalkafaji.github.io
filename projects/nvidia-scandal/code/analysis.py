import json
import datetime
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

SHARES_OUT = {
    "NVDA": 24.19e9,
    "AMD": 1.632e9,
    "AVGO": 4.77e9,
}

BENCHMARKS = ["IXIC", "GSPC", "SOXX", "SMH"]
BENCH_LABEL = {"IXIC": "Nasdaq Composite", "GSPC": "S&P 500", "SOXX": "iShares Semiconductor ETF (SOXX)", "SMH": "VanEck Semiconductor ETF (SMH)"}
SUBJECTS = ["NVDA", "AMD", "AVGO"]

def load(sym):
    d = json.load(open(f"{sym}.json"))["chart"]["result"][0]
    ts = d["timestamp"]
    closes = d["indicators"]["quote"][0]["close"]
    dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date() for t in ts]
    s = pd.Series(closes, index=pd.to_datetime(dates), name=sym).dropna()
    return s

prices = pd.concat([load(s) for s in SUBJECTS + BENCHMARKS], axis=1).dropna()
rets = prices.pct_change().dropna()

# ---------------------------------------------------------------
# 1. Estimation window (clean, pre-turmoil) market models, one per
#    (subject, benchmark) pair.
# ---------------------------------------------------------------
est = rets.loc["2025-10-15":"2026-10-15"]

def fit_market_model(y, x):
    X = np.column_stack([np.ones(len(x)), x.values])
    beta_hat, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    alpha, beta = beta_hat
    resid = y.values - X @ beta_hat
    n = len(y)
    dof = n - 2
    resid_std = np.sqrt((resid ** 2).sum() / dof)
    ss_tot = ((y.values - y.values.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return dict(alpha=alpha, beta=beta, resid_std=resid_std, n=n, dof=dof, r2=r2)

models = {}
for subj in SUBJECTS:
    for bench in BENCHMARKS:
        models[(subj, bench)] = fit_market_model(est[subj], est[bench])

print("=== Market models (estimation window 2025-10-15 to 2026-10-15) ===")
for (subj, bench), m in models.items():
    print(f"{subj} ~ {bench}: alpha={m['alpha']*100:.4f}%/day  beta={m['beta']:.3f}  "
          f"resid_std={m['resid_std']*100:.3f}%  R2={m['r2']:.3f}  n={m['n']}")

# ---------------------------------------------------------------
# 2. Event-study inference: CAR, SE, t-stat, p-value, 95% CI
#    Standard MacKinlay (1997) event-study test statistic.
# ---------------------------------------------------------------
def event_study(subj, bench, dates):
    m = models[(subj, bench)]
    ars = []
    for d in dates:
        d = pd.Timestamp(d)
        actual = rets.loc[d, subj]
        mkt = rets.loc[d, bench]
        expected = m["alpha"] + m["beta"] * mkt
        ars.append(actual - expected)
    car = sum(ars)
    L = len(dates)
    se = m["resid_std"] * np.sqrt(L)
    t_stat = car / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=m["dof"]))
    t_crit = stats.t.ppf(0.975, df=m["dof"])
    ci_low, ci_high = car - t_crit * se, car + t_crit * se
    return dict(car=car, se=se, t=t_stat, p=p_value, ci_low=ci_low, ci_high=ci_high, L=L)

PRIMARY_WINDOW = ["2026-11-09", "2026-11-10"]
SENSITIVITY_WINDOWS = {
    "Nov 9-10 (primary, 2 days)": ["2026-11-09", "2026-11-10"],
    "Nov 9-11 (+1 day, includes rebound)": ["2026-11-09", "2026-11-10", "2026-11-11"],
    "Nov 6-10 (from pre-event close)": ["2026-11-06", "2026-11-09", "2026-11-10"],
}
SECTOR_SELLOFF_DAY = ["2026-11-05"]

print("\n=== Robustness across benchmarks, primary window (Nov 9-10) ===")
robustness_rows = []
for subj in SUBJECTS:
    for bench in BENCHMARKS:
        r = event_study(subj, bench, PRIMARY_WINDOW)
        robustness_rows.append(dict(subject=subj, benchmark=bench, beta=models[(subj,bench)]["beta"],
                                     car=r["car"], se=r["se"], t=r["t"], p=r["p"],
                                     ci_low=r["ci_low"], ci_high=r["ci_high"]))
robustness = pd.DataFrame(robustness_rows)
print(robustness.to_string(index=False))

print("\n=== Event-window sensitivity, NVDA vs Nasdaq Composite ===")
sensitivity_rows = []
for label, dates in SENSITIVITY_WINDOWS.items():
    r = event_study("NVDA", "IXIC", dates)
    sensitivity_rows.append(dict(window=label, L=r["L"], car=r["car"], se=r["se"], t=r["t"], p=r["p"],
                                  ci_low=r["ci_low"], ci_high=r["ci_high"]))
sensitivity = pd.DataFrame(sensitivity_rows)
print(sensitivity.to_string(index=False))

print("\n=== Nov 5 broad selloff (context, not part of primary event window) ===")
selloff_rows = []
for subj in SUBJECTS:
    r = event_study(subj, "IXIC", SECTOR_SELLOFF_DAY)
    selloff_rows.append(dict(subject=subj, car=r["car"], se=r["se"], t=r["t"], p=r["p"]))
print(pd.DataFrame(selloff_rows).to_string(index=False))

# ---------------------------------------------------------------
# 3. Dollar-value reconciliation (primary benchmark: Nasdaq Composite)
# ---------------------------------------------------------------
print("\n=== Dollar reconciliation, Nov 9-10, NVDA (primary benchmark: Nasdaq Composite) ===")
p_before = prices.loc["2026-11-06", "NVDA"]
p_after = prices.loc["2026-11-10", "NVDA"]
shares = SHARES_OUT["NVDA"]
mcap_before = p_before * shares
mcap_after = p_after * shares
total_dollar = mcap_after - mcap_before
total_pct = p_after / p_before - 1

r_primary = event_study("NVDA", "IXIC", PRIMARY_WINDOW)
abnormal_pct = r_primary["car"]
abnormal_dollar = abnormal_pct * mcap_before
abnormal_dollar_low = r_primary["ci_low"] * mcap_before
abnormal_dollar_high = r_primary["ci_high"] * mcap_before
market_explained_dollar = total_dollar - abnormal_dollar

print(f"Market cap Nov 6 close: ${mcap_before/1e9:,.1f}B")
print(f"Market cap Nov 10 close: ${mcap_after/1e9:,.1f}B")
print(f"Total observed change: {total_pct*100:.2f}% = ${total_dollar/1e9:,.1f}B")
print(f"Abnormal (unexplained) return: {abnormal_pct*100:.2f}%  t={r_primary['t']:.2f}  p={r_primary['p']:.4f}")
print(f"Abnormal dollar estimate: ${abnormal_dollar/1e9:,.1f}B  95% CI [${abnormal_dollar_low/1e9:,.1f}B, ${abnormal_dollar_high/1e9:,.1f}B]")
print(f"Market/sector-explained portion: ${market_explained_dollar/1e9:,.1f}B")

# save everything needed for the article
robustness.to_csv("robustness_benchmarks.csv", index=False)
sensitivity.to_csv("sensitivity_windows.csv", index=False)
summary2 = pd.DataFrame([{
    "mcap_before_B": mcap_before/1e9, "mcap_after_B": mcap_after/1e9,
    "total_pct": total_pct*100, "total_dollar_B": total_dollar/1e9,
    "abnormal_pct": abnormal_pct*100, "abnormal_dollar_B": abnormal_dollar/1e9,
    "abnormal_dollar_low_B": abnormal_dollar_low/1e9, "abnormal_dollar_high_B": abnormal_dollar_high/1e9,
    "t_stat": r_primary["t"], "p_value": r_primary["p"],
    "market_explained_dollar_B": market_explained_dollar/1e9,
}])
summary2.to_csv("summary2.csv", index=False)
rets.to_csv("daily_returns2.csv")
prices.to_csv("daily_prices2.csv")
print("\nSaved robustness_benchmarks.csv, sensitivity_windows.csv, summary2.csv")
