"""
Collect GTA VI-themed scans from the urlscan.io public search API.

- Unauthenticated search API (free tier), 30-day history limit.
- Window: completed UTC days 2026-10-18 .. 2026-11-15.
- Paginates with search_after; obeys rate-limit headers and HTTP 429 backoff.
- Logs every query with retrieval timestamp and the returned history limit.
- Writes scan-level rows into DuckDB (scans table) plus a query_log table.

Usage: python collect.py <out_dir>
"""
import json
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import duckdb

API = "https://urlscan.io/api/v1/search/"
DATE_FILTER = "date:[2026-10-18 TO 2026-11-15]"  # inclusive completed UTC days
PAGE_SIZE = 100
MAX_PAGES = 5

# Several narrow variants; a domain-prefix query alone misses many sites.
QUERIES = [
    "page.domain:gta6*",
    "page.domain:gtavi*",
    "page.domain:gta-6*",
    "page.domain:gta-vi*",
    "page.domain:gtavi6*",
    "page.domain:grandtheftauto6*",
    "page.domain:gta6-*",
    'page.title:"GTA VI"',
    'page.title:"GTA 6"',
    'page.title:"Grand Theft Auto VI"',
    'page.title:"Grand Theft Auto 6"',
    "task.domain:gta6*",
]

FIELDS = [
    "uuid", "query", "task_time", "task_url", "task_domain", "page_domain",
    "page_apex", "page_url", "page_title", "page_ip", "page_asn",
    "page_asnname", "page_country", "page_server", "page_status",
    "domain_age_days", "tls_issuer", "result_url", "screenshot_url", "task_tags", "task_method",
]


def get(url):
    """GET with 429 backoff; returns (json, headers)."""
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"User-Agent": "portfolio-research-script/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                hdr = dict(r.headers)
                body = json.load(r)
            remaining = int(hdr.get("x-rate-limit-remaining", "10"))
            if remaining <= 1:
                time.sleep(float(hdr.get("x-rate-limit-reset-after", "60")) + 1)
            return body, hdr
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = float(e.headers.get("x-rate-limit-reset-after") or e.headers.get("Retry-After") or 30)
                print(f"  429, sleeping {wait:.0f}s", file=sys.stderr)
                time.sleep(wait + 1)
                continue
            raise
    raise RuntimeError("too many 429s")


def row(q, r):
    t, p = r.get("task", {}), r.get("page", {})
    return [
        r.get("_id"), q, t.get("time"), t.get("url"), t.get("domain"),
        p.get("domain"), p.get("apexDomain"), p.get("url"), p.get("title"),
        p.get("ip"), p.get("asn"), p.get("asnname"), p.get("country"),
        p.get("server"), p.get("status"), p.get("apexDomainAgeDays"),
        p.get("tlsIssuer"), r.get("result"), r.get("screenshot"),
        "|".join(t.get("tags") or []), t.get("method"),
    ]


def main(out_dir):
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out / "gta6.duckdb"))
    con.execute("DROP TABLE IF EXISTS scans; DROP TABLE IF EXISTS query_log")
    con.execute("CREATE TABLE scans (" + ", ".join(f"{f} VARCHAR" for f in FIELDS) + ")")
    con.execute("""CREATE TABLE query_log (query VARCHAR, page INT, retrieved_utc VARCHAR,
                   returned INT, total INT, has_more BOOLEAN, search_date_limit_days INT)""")

    for qi, q in enumerate(QUERIES):
        full_q = f"({q}) AND {DATE_FILTER}"
        search_after = None
        for page in range(MAX_PAGES):
            params = {"q": full_q, "size": PAGE_SIZE}
            if search_after:
                params["search_after"] = search_after
            try:
                body, _ = get(API + "?" + urllib.parse.urlencode(params))
            except urllib.error.HTTPError as e:
                print(f"{q!r}: HTTP {e.code}, skipped", file=sys.stderr)
                con.execute("INSERT INTO query_log VALUES (?,?,?,?,?,?,?)", [
                    q, page, datetime.now(timezone.utc).isoformat(), 0, None, None, None])
                break
            results = body.get("results", [])
            (out / "raw" / f"q{qi:02d}_p{page}.json").write_text(json.dumps(body))
            con.execute("INSERT INTO query_log VALUES (?,?,?,?,?,?,?)", [
                q, page, datetime.now(timezone.utc).isoformat(), len(results),
                body.get("total"), body.get("has_more"), body.get("search_date_limit_days")])
            for r in results:
                con.execute(f"INSERT INTO scans VALUES ({','.join('?' * len(FIELDS))})", row(q, r))
            print(f"{q!r} page {page}: {len(results)} (total={body.get('total')}, has_more={body.get('has_more')})")
            # has_more is unreliable on the free tier; page whenever a full page returns
            if len(results) < PAGE_SIZE:
                break
            search_after = ",".join(str(x) for x in results[-1]["sort"])
            time.sleep(2.5)
        time.sleep(2.5)
    con.close()


if __name__ == "__main__":
    main(sys.argv[1])
