"""
Recompute the headline results from the PUBLISHED data only and check them
against the published result files. Needs no API access and no raw data.

Checks:
  1. shared_ip and template_reuse flags, rebuilt from page_ip and
     template_family, match the flags in domains_labeled.csv
  2. every primary row of group_metrics.csv (counts, Wilson CIs, Newcombe
     difference CIs, Fisher p-values)
  3. dedup_template_reuse.csv (domain / mirrors-collapsed / families-collapsed)
  4. sample_flow.csv counts

Usage: python verify.py <data_dir>      (exits non-zero on any mismatch)
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

from analysis import DECEPTIVE, MIRRORS, defang, newcombe_diff, wilson

fails = 0


def check(name, got, want, tol=0.051):
    global fails
    ok = abs(float(got) - float(want)) <= tol if isinstance(want, (int, float)) else str(got) == str(want)
    if not ok:
        fails += 1
    print(f"{'ok  ' if ok else 'FAIL'} {name}: computed {got}, published {want}")


def main(data):
    data = Path(data)
    d = pd.read_csv(data / "domains_labeled.csv")
    rel = d[d.label != "out_of_scope"].copy()

    # 1. rebuild flags
    ipn = rel.groupby("page_ip").reg_domain.nunique()
    shared = rel.page_ip.map(ipn).fillna(0) >= 2
    fam = rel.groupby("template_family").reg_domain.nunique()
    reuse = rel.template_family.map(fam).fillna(0) >= 2
    check("shared_ip flags identical", int((shared != rel.shared_ip.astype(bool)).sum()), 0)
    check("template_reuse flags identical", int((reuse != rel.template_reuse.astype(bool)).sum()), 0)

    # 2. group metrics (primary definition)
    a, b = rel[rel.label.isin(DECEPTIVE)], rel[rel.label == "fan_news"]
    gm = pd.read_csv(data / "group_metrics.csv").set_index("metric")
    for col in ["shared_ip", "shared_ip_noncdn", "shared_title", "template_reuse", "cdn_paas",
                "platform_subdomain", "strong_tag", "new_domain"]:
        aa, bb = a, b
        if col == "new_domain":
            aa = a[(~a.platform_subdomain.astype(bool)) & a.domain_age_days.notna()]
            bb = b[(~b.platform_subdomain.astype(bool)) & b.domain_age_days.notna()]
            k1, k2 = int((aa.domain_age_days < 30).sum()), int((bb.domain_age_days < 30).sum())
        else:
            k1, k2 = int(aa[col].astype(bool).sum()), int(bb[col].astype(bool).sum())
        n1, n2 = len(aa), len(bb)
        r = gm.loc[col]
        _, l1, u1 = wilson(k1, n1)
        d_, dl, du = newcombe_diff(k1, n1, k2, n2)
        p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
        check(f"{col} deceptive k/n", f"{k1}/{n1}", f"{int(r.deceptive_k)}/{int(r.deceptive_n)}")
        check(f"{col} fan k/n", f"{k2}/{n2}", f"{int(r.fan_k)}/{int(r.fan_n)}")
        check(f"{col} deceptive CI", f"{100*l1:.1f}-{100*u1:.1f}", r.deceptive_ci)
        check(f"{col} difference CI", f"{100*dl:.1f} to {100*du:.1f}", r.diff_ci)
        check(f"{col} Fisher p", round(p, 4), float(r.fisher_p), tol=0.00011)

    # 3. deduplicated template reuse
    two = pd.concat([a.assign(g="deceptive"), b.assign(g="fan_news")])
    mir = {defang(k): defang(v) for k, v in MIRRORS.items()}
    two["mirror_unit"] = two.reg_domain.map(lambda x: mir.get(x, x))
    fam_units = two.groupby("template_family").mirror_unit.nunique()
    two["family_unit"] = two.template_family.fillna(two.reg_domain)
    dd = pd.read_csv(data / "dedup_template_reuse.csv").set_index("level")
    for level, unit in [("domain", "reg_domain"), ("mirrors_collapsed", "mirror_unit"),
                        ("families_collapsed", "family_unit")]:
        u = two.drop_duplicates(unit).copy()
        u["reuse"] = u.template_reuse.astype(bool) if level == "domain" else u.template_family.map(fam_units).fillna(0) >= 2
        ga, gb = u[u.g == "deceptive"], u[u.g == "fan_news"]
        k1, n1, k2, n2 = int(ga.reuse.sum()), len(ga), int(gb.reuse.sum()), len(gb)
        r = dd.loc[level]
        check(f"dedup {level}", f"{k1}/{n1} vs {k2}/{n2}",
              f"{int(r.deceptive_k)}/{int(r.deceptive_n)} vs {int(r.fan_k)}/{int(r.fan_n)}")
        check(f"dedup {level} Fisher p", round(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1], 4),
              float(r.fisher_p), tol=0.00011)

    # 4. sample flow
    fl = dict(pd.read_csv(data / "sample_flow.csv").values)
    check("registrable domains", len(d), fl["registrable domains"])
    check("relevant domains", len(rel), fl["relevant domains"])
    check("deceptive group", len(a), fl["comparison: deceptive (suspicious + tag-corroborated)"])
    # scans.csv omits the scan(s) of the one redacted personal page
    red = int(d.loc[d.reg_domain == "[redacted]", "n_scans"].sum())
    check("unique scans (published + redacted)", len(pd.read_csv(data / "scans.csv")) + red, fl["unique scans"])

    print(f"\n{fails} mismatches")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv[1])
