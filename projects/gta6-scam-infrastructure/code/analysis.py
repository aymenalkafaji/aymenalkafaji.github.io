"""
Compare infrastructure signals between deceptive-offer sites and fan/news sites.

Inputs (work dir):
  gta6.duckdb      - scans / scans_dedup / domains tables (collect.py, prepare.py)
  labels_raw.csv   - one row per registrable domain: label, offer_type,
                     confidence, evidence, second_pass_label

Outputs (out dir, all domains defanged, no URL paths or query strings):
  domains_labeled.csv, scans.csv, query_log.csv, group_metrics.csv,
  sensitivity.csv, offer_types.csv, label_counts.csv, agreement.csv,
  shared_infra_clusters.csv, title_clusters.csv

Usage: python analysis.py <work_dir> <out_dir>
"""
import math
import random
import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

# Major CDN / PaaS networks (including free static-site platforms). An IP shared on these is usually an anycast edge
# shared by thousands of unrelated customers, so it says little about common
# operation. Reported separately from shared IPs on other networks.
CDN_PAAS_ASNS = {
    "AS13335": "Cloudflare",
    "AS16509": "Amazon (incl. Vercel/Netlify edges)",
    "AS14618": "Amazon",
    "AS54113": "Fastly (incl. GitHub Pages)",
    "AS15169": "Google",
    "AS16625": "Akamai",
    "AS20940": "Akamai",
    "AS8075": "Microsoft",
    "AS139341": "Tencent EdgeOne Pages",
    "AS13238": "Yandex Cloud",
    "AS208398": "Yandex Cloud",
}
# Strong third-party tags. 'possiblethreat' is an explicitly weak automated
# flag, so 'phishing' only counts as strong when it is not paired with it.
STRONG_TAGS = {"malicious", "scam", "drainer", "crypto-scam", "cryptoscam", "threat"}
REDACT = {"jade-sprite-a92f3c.netlify.app"}  # personal fundraiser, not analysed

DECEPTIVE = {"suspicious_offer", "corroborated_malicious"}

# Free site-hosting platforms: each subdomain is a separate deployment, and
# urlscan's domain-age field reports the platform's age, not the site's.
PLATFORM_SUFFIXES = ("pages.dev", "netlify.app", "vercel.app", "github.io", "edgeone.dev",
                     "onrender.com", "website.yandexcloud.net", "blogspot.com", "duckdns.org")

OFFER_GROUPS = {
    "fake_download": "Fake download, early access or port",
    "fake_mobile": "Fake download, early access or port",
    "fake_pc_mobile": "Fake download, early access or port",
    "early_access": "Fake download, early access or port",
    "giveaway": "Giveaway, free key or free copy",
    "free_key": "Giveaway, free key or free copy",
    "free_game": "Giveaway, free key or free copy",
    "fake_preorder": "Fake pre-order or key store",
    "fake_key_store": "Fake pre-order or key store",
    "branded_gate": "Branded gate, content hidden",
    "token_presale": "Crypto presale or wallet connect",
    "wallet_connect": "Crypto presale or wallet connect",
    "gambling_lure": "Gambling or money generator",
    "money_generator": "Gambling or money generator",
}


def defang(d):
    return None if d is None or (isinstance(d, float) and math.isnan(d)) else str(d).replace(".", "[.]")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def newcombe_diff(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def strong_tag(tags):
    if not isinstance(tags, str) or not tags:
        return False
    t = set(tags.split("|"))
    if t & STRONG_TAGS:
        return True
    return "phishing" in t and "possiblethreat" not in t


def hhi(series):
    s = series.value_counts(normalize=True)
    return float((s ** 2).sum())


def main(work, out):
    work, out = Path(work), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(work / "gta6.duckdb"), read_only=True)
    dom = con.execute("SELECT * FROM domains").df()
    scans = con.execute("SELECT * FROM scans_dedup").df()
    qlog = con.execute("SELECT * FROM query_log").df()
    lab = pd.read_csv(work / "labels_raw.csv")
    df = dom.merge(lab, on="reg_domain", how="left")
    assert df.label.notna().all(), "every domain needs a label"

    # ---- derived per-domain signals ------------------------------------
    df["norm_title"] = df.page_title.fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["cdn_paas"] = df.page_asn.isin(CDN_PAAS_ASNS.keys())
    df["strong_tag"] = df.all_tags.map(strong_tag)
    df["platform_subdomain"] = df.reg_domain.map(lambda d: any(d == x or d.endswith("." + x) for x in PLATFORM_SUFFIXES))
    # age only meaningful for independently registered domains
    df["age_known"] = df.domain_age_days.notna() & ~df.platform_subdomain
    df["new_domain"] = df.age_known & (df.domain_age_days < 30)
    df["offer_group"] = df.offer_type.map(OFFER_GROUPS)

    rel = df[df.label != "out_of_scope"].copy()
    ip_counts = rel.groupby("page_ip").reg_domain.nunique()
    rel["shared_ip"] = rel.page_ip.map(ip_counts).fillna(0) >= 2
    rel["shared_ip_noncdn"] = rel.shared_ip & ~rel.cdn_paas
    generic = {"", "404: not_found", "nan", "not found", "404 not found", "coming soon",
               "application placeholder", "nur einen moment…", "just a moment..."}
    tcounts = rel[~rel.norm_title.isin(generic)].groupby("norm_title").reg_domain.nunique()
    rel["shared_title"] = rel.norm_title.map(tcounts).fillna(0) >= 2
    # visual template family identified during screenshot review (manual)
    fam_size = rel.groupby("template_family").reg_domain.nunique()
    rel["template_reuse"] = rel.template_family.map(fam_size).fillna(0) >= 2
    df = df.merge(rel[["reg_domain", "shared_ip", "shared_ip_noncdn", "shared_title", "template_reuse"]],
                  on="reg_domain", how="left")

    # ---- group metrics ---------------------------------------------------
    def group(defn):
        if defn == "primary":
            return rel[rel.label.isin(DECEPTIVE)], rel[rel.label == "fan_news"]
        if defn == "incl_uncertain":
            return rel[rel.label.isin(DECEPTIVE | {"uncertain"})], rel[rel.label == "fan_news"]
        if defn == "high_conf_only":
            hc = rel[rel.confidence == "high"]
            return hc[hc.label.isin(DECEPTIVE)], hc[hc.label == "fan_news"]
        raise ValueError(defn)

    metrics = [
        ("shared_ip", "Shares an IP with another relevant domain (any network)"),
        ("shared_ip_noncdn", "Shares an IP on a non-CDN/PaaS network"),
        ("shared_title", "Shares an exact page title with another domain"),
        ("template_reuse", "Belongs to a visually identical template family (2+ domains)"),
        ("cdn_paas", "Hosted on a major CDN/PaaS network"),
        ("platform_subdomain", "Deployed on a free site-hosting platform subdomain"),
        ("strong_tag", "Carries a strong third-party threat tag"),
    ]
    rows, sens = [], []
    for defn in ["primary", "incl_uncertain", "high_conf_only"]:
        a, b = group(defn)
        for col, desc in metrics:
            k1, n1, k2, n2 = int(a[col].sum()), len(a), int(b[col].sum()), len(b)
            p1, l1, u1 = wilson(k1, n1)
            p2, l2, u2 = wilson(k2, n2)
            d, dl, du = newcombe_diff(k1, n1, k2, n2)
            p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
            r = dict(definition=defn, metric=col, description=desc,
                     deceptive_k=k1, deceptive_n=n1, deceptive_pct=round(100 * p1, 1),
                     deceptive_ci=f"{100*l1:.1f}-{100*u1:.1f}",
                     fan_k=k2, fan_n=n2, fan_pct=round(100 * p2, 1), fan_ci=f"{100*l2:.1f}-{100*u2:.1f}",
                     diff_pp=round(100 * d, 1), diff_ci=f"{100*dl:.1f} to {100*du:.1f}",
                     fisher_p=round(p, 4))
            (rows if defn == "primary" else sens).append(r)
            if defn == "primary":
                sens.append(r)
        # new domain share, among independently registered domains only
        a2, b2 = a[a.age_known], b[b.age_known]
        k1, n1, k2, n2 = int(a2.new_domain.sum()), len(a2), int(b2.new_domain.sum()), len(b2)
        p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2); d, dl, du = newcombe_diff(k1, n1, k2, n2)
        r = dict(definition=defn, metric="new_domain",
                 description="Registered domain under 30 days old at first scan (platform subdomains excluded)",
                 deceptive_k=k1, deceptive_n=n1, deceptive_pct=round(100 * p1, 1), deceptive_ci=f"{100*l1:.1f}-{100*u1:.1f}",
                 fan_k=k2, fan_n=n2, fan_pct=round(100 * p2, 1), fan_ci=f"{100*l2:.1f}-{100*u2:.1f}",
                 diff_pp=round(100 * d, 1), diff_ci=f"{100*dl:.1f} to {100*du:.1f}",
                 fisher_p=round(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1], 4))
        sens.append(r)
        if defn == "primary":
            rows.append(r)
        # domain age (continuous)
        aa, bb = a2.domain_age_days, b2.domain_age_days
        u = mannwhitneyu(aa, bb, alternative="two-sided")
        r = dict(definition=defn, metric="domain_age_median_days",
                 description="Median domain age at first scan (days)",
                 deceptive_k=None, deceptive_n=len(aa), deceptive_pct=float(aa.median()), deceptive_ci="",
                 fan_k=None, fan_n=len(bb), fan_pct=float(bb.median()), fan_ci="",
                 diff_pp=float(aa.median() - bb.median()), diff_ci="", fisher_p=round(u.pvalue, 4))
        sens.append(r)
        if defn == "primary":
            rows.append(r)
        # ASN concentration
        for name, g in [("deceptive", a), ("fan", b)]:
            top = g.page_asn.value_counts()
            sens.append(dict(definition=defn, metric=f"asn_hhi_{name}",
                             description=f"ASN concentration (HHI) - {name}",
                             deceptive_pct=round(hhi(g.page_asn.dropna()), 3), deceptive_n=len(g),
                             fan_pct=None, diff_ci=f"top ASN {top.index[0]} {100*top.iloc[0]/top.sum():.0f}%"))

    pd.DataFrame(rows).to_csv(out / "group_metrics.csv", index=False)
    pd.DataFrame(sens).to_csv(out / "sensitivity.csv", index=False)

    # ---- descriptive tables ---------------------------------------------
    df.label.value_counts().rename_axis("label").reset_index(name="domains").to_csv(out / "label_counts.csv", index=False)
    dec = rel[rel.label.isin(DECEPTIVE)]
    dec.groupby(["offer_group", "offer_type", "label"]).size().unstack(fill_value=0).reset_index().to_csv(out / "offer_types.csv", index=False)

    # ASN share by group (top networks)
    asn = rel.assign(group=rel.label.map(lambda x: "deceptive" if x in DECEPTIVE else x))
    asn = asn[asn.group.isin(["deceptive", "fan_news", "token"])]
    tab = (asn.groupby(["group", "page_asn"]).size().rename("domains").reset_index())
    names = rel.drop_duplicates("page_asn").set_index("page_asn").page_asnname
    tab["asn_name"] = tab.page_asn.map(names)
    tab.to_csv(out / "asn_by_group.csv", index=False)

    # clusters: shared non-CDN IPs and shared titles across 2+ domains
    cl = rel[rel.shared_ip].groupby(["page_ip", "page_asn"]).agg(
        domains=("reg_domain", lambda s: "; ".join(defang(x) for x in sorted(s))),
        labels=("label", lambda s: "; ".join(sorted(set(s)))), n=("reg_domain", "size")).reset_index()
    cl["cdn_paas"] = cl.page_asn.isin(CDN_PAAS_ASNS.keys())
    cl.sort_values("n", ascending=False).to_csv(out / "shared_infra_clusters.csv", index=False)
    tc = rel[rel.shared_title].groupby("norm_title").agg(
        domains=("reg_domain", lambda s: "; ".join(defang(x) for x in sorted(s))),
        labels=("label", lambda s: "; ".join(sorted(set(s)))), n=("reg_domain", "size")).reset_index()
    tc.sort_values("n", ascending=False).to_csv(out / "title_clusters.csv", index=False)

    # ---- intra-rater agreement (second pass) ------------------------------
    rr = df[df.second_pass_label.notna() & (df.second_pass_label != "")]
    agree = (rr.label == rr.second_pass_label)
    cats = sorted(set(rr.label) | set(rr.second_pass_label))
    po = agree.mean()
    pe = sum((rr.label == c).mean() * (rr.second_pass_label == c).mean() for c in cats)
    kappa = (po - pe) / (1 - pe)
    dis = rr[~agree][["reg_domain", "label", "second_pass_label"]].copy()
    dis["reg_domain"] = dis.reg_domain.map(defang)
    pd.DataFrame([dict(n=len(rr), agreements=int(agree.sum()), percent_agreement=round(100 * po, 1),
                       cohens_kappa=round(kappa, 3),
                       disagreements="; ".join(f"{r.reg_domain}: {r.label} -> {r.second_pass_label}" for r in dis.itertuples()))]
                 ).to_csv(out / "agreement.csv", index=False)

    # ---- published row-level data (defanged, redacted) --------------------
    pub = df.copy()
    red = pub.reg_domain.isin(REDACT)
    pub.loc[red, ["page_title", "page_domain", "evidence"]] = ["[redacted]", "[redacted]", "personal page, not analysed"]
    pub.loc[red, "reg_domain"] = "[redacted]"
    for c in ["reg_domain", "page_domain"]:
        pub[c] = pub[c].map(lambda x: x if x == "[redacted]" else defang(x))
    pub[["reg_domain", "page_domain", "label", "offer_type", "confidence", "evidence", "second_pass_label",
         "first_seen", "n_scans", "page_title", "page_ip", "page_asn", "page_asnname", "page_country",
         "page_status", "domain_age_days", "platform_subdomain", "offer_group", "all_tags", "strong_tag", "cdn_paas", "shared_ip",
         "shared_ip_noncdn", "shared_title", "template_family", "template_reuse", "first_uuid"]].to_csv(out / "domains_labeled.csv", index=False)

    s = scans.copy()
    s = s[~s.reg_domain.isin(REDACT)]
    s["reg_domain"] = s.reg_domain.map(defang)
    s["page_domain"] = s.page_domain.map(defang)
    s[["uuid", "task_time", "reg_domain", "page_domain", "page_title", "page_ip", "page_asn",
       "page_country", "page_status", "domain_age_days", "task_tags", "matched_queries"]].to_csv(out / "scans.csv", index=False)
    qlog.to_csv(out / "query_log.csv", index=False)

    # ---- console summary --------------------------------------------------
    print("labels:", df.label.value_counts().to_dict())
    print(f"relevant domains: {len(rel)}; deceptive {len(dec)}; fan/news {int((rel.label=='fan_news').sum())}")
    print(pd.DataFrame(rows)[["metric", "deceptive_k", "deceptive_n", "deceptive_pct", "deceptive_ci",
                              "fan_k", "fan_n", "fan_pct", "fan_ci", "diff_pp", "diff_ci", "fisher_p"]].to_string())
    print(pd.DataFrame(sens)[["definition", "metric", "deceptive_pct", "deceptive_n", "fan_pct", "fan_n", "diff_pp", "diff_ci", "fisher_p"]].to_string())
    print(f"agreement {po:.3f}, kappa {kappa:.3f}, n={len(rr)}")
    print(cl.sort_values("n", ascending=False).head(15).to_string())
    print(tc.sort_values("n", ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
