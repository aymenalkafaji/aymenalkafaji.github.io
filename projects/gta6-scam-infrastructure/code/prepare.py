"""
Deduplicate scans and build the domain-level table.

1. Deduplicate scan UUIDs (the same scan can match several queries).
2. Map each page.domain to its registrable domain using the Public Suffix List
   (including the private section, so foo.pages.dev and bar.pages.dev are
   separate sites). EdgeOne Pages (edgeone.dev) and Yandex Cloud website
   hosting are not in the PSL snapshot used, so they are added as extra
   suffixes: each subdomain there is a separately deployed site.
3. Keep the earliest observed scan per registrable domain (domains table)
   and the full scan-level table (scans_dedup).
4. Strip query strings / fragments from URLs before export (they can carry
   tokens or personal data).

Usage: python prepare.py <work_dir>
"""
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import duckdb
import tldextract

EXTRA_SUFFIXES = ["edgeone.dev", "website.yandexcloud.net"]
ext = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True,
                            extra_suffixes=EXTRA_SUFFIXES)


def registrable(host):
    if not host:
        return None
    r = ext(host)
    return r.top_domain_under_public_suffix or host


def strip_url(u):
    if not u:
        return u
    s = urlsplit(u)
    return urlunsplit((s.scheme, s.netloc, s.path, "", ""))


def main(work):
    con = duckdb.connect(str(Path(work) / "gta6.duckdb"))
    con.create_function("registrable", registrable, ["VARCHAR"], "VARCHAR")
    con.create_function("strip_url", strip_url, ["VARCHAR"], "VARCHAR")
    con.execute("""
        CREATE OR REPLACE TABLE scans_dedup AS
        SELECT uuid,
               string_agg(DISTINCT query, ' || ') AS matched_queries,
               any_value(task_time) AS task_time,
               strip_url(any_value(task_url)) AS task_url,
               any_value(page_domain) AS page_domain,
               registrable(coalesce(any_value(page_domain), any_value(task_domain))) AS reg_domain,
               strip_url(any_value(page_url)) AS page_url,
               any_value(page_title) AS page_title,
               any_value(page_ip) AS page_ip,
               any_value(page_asn) AS page_asn,
               any_value(page_asnname) AS page_asnname,
               any_value(page_country) AS page_country,
               any_value(page_status) AS page_status,
               TRY_CAST(any_value(domain_age_days) AS INT) AS domain_age_days,
               any_value(task_tags) AS task_tags,
               any_value(screenshot_url) AS screenshot_url
        FROM scans GROUP BY uuid
    """)
    con.execute("""
        CREATE OR REPLACE TABLE domains AS
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY reg_domain ORDER BY task_time) AS rn,
                   count(*) OVER (PARTITION BY reg_domain) AS n_scans,
                   string_agg(DISTINCT nullif(task_tags, ''), '|') OVER (PARTITION BY reg_domain) AS all_tags
            FROM scans_dedup WHERE reg_domain IS NOT NULL
        )
        SELECT reg_domain, n_scans, task_time AS first_seen, page_domain, page_url, page_title,
               page_ip, page_asn, page_asnname, page_country, page_status, domain_age_days,
               all_tags, uuid AS first_uuid, screenshot_url
        FROM ranked WHERE rn = 1 ORDER BY reg_domain
    """)
    n_raw = con.execute("SELECT count(*) FROM scans").fetchone()[0]
    n_scan = con.execute("SELECT count(*) FROM scans_dedup").fetchone()[0]
    n_dom = con.execute("SELECT count(*) FROM domains").fetchone()[0]
    rng = con.execute("SELECT min(task_time), max(task_time) FROM scans_dedup").fetchone()
    print(f"raw rows {n_raw}, unique scans {n_scan}, registrable domains {n_dom}, scan range {rng}")
    con.execute(f"COPY domains TO '{Path(work) / 'domains_to_label.csv'}' (HEADER)")
    con.close()


if __name__ == "__main__":
    main(sys.argv[1])
