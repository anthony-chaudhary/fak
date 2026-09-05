#!/usr/bin/env python3
"""Audit every panel query in tools/grafana/dashboards/*.json against live Prometheus."""

import glob
import json
import os
import re
import sys
import urllib.parse
import urllib.request

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9091")

def query_prometheus(expr):
    url = f"{PROMETHEUS_URL}/api/v1/query?query=" + urllib.parse.quote(expr)
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") != "success":
                return "ERROR", data.get("error", "unknown error")
            result = data.get("data", {}).get("result", [])
            if not result:
                return "NO_DATA", None
            # Inspect values
            all_zero = True
            samples = []
            for item in result:
                val = item.get("value")
                if val and len(val) >= 2:
                    try:
                        fval = float(val[1])
                        samples.append(fval)
                        if fval != 0.0:
                            all_zero = False
                    except (ValueError, TypeError):
                        all_zero = False
                        samples.append(val[1])
            if all_zero and samples:
                return "ZERO", samples
            return "NON_ZERO", samples
    except Exception as e:
        return "ERROR", str(e)

def extract_panels(panels_list):
    res = []
    for p in panels_list:
        if p.get("type") == "row":
            if "panels" in p and p["panels"]:
                res.extend(extract_panels(p["panels"]))
            continue
        res.append(p)
    return res

def audit_dashboard(path, default_vars=None):
    if default_vars is None:
        default_vars = {}
    with open(path) as f:
        d = json.load(fp=f)
    uid = d.get("uid", "")
    title = d.get("title", "")
    panels = extract_panels(d.get("panels", []))
    results = []

    # Identify variables and default values
    var_defaults = dict(default_vars)
    for v in d.get("templating", {}).get("list", []):
        vname = v.get("name")
        curr = v.get("current", {})
        val = curr.get("value")
        if isinstance(val, list) and val:
            var_defaults[vname] = val[0]
        elif isinstance(val, str) and val and val != "$__all":
            var_defaults[vname] = val
        elif vname not in var_defaults:
            var_defaults[vname] = ".*"

    for p in panels:
        p_title = p.get("title", "Untitled")
        p_type = p.get("type", "unknown")
        targets = p.get("targets", [])
        for t in targets:
            expr = t.get("expr", "").strip()
            if not expr:
                continue
            # substitute variables
            clean_expr = expr
            for var_name, var_val in var_defaults.items():
                clean_expr = re.sub(rf"\${var_name}\b", var_val, clean_expr)
                clean_expr = re.sub(rf"\${{{var_name}}}", var_val, clean_expr)
            # handle time intervals like $__rate_interval, $__range, $__interval
            clean_expr = re.sub(r"\$(?:__)?rate_interval\b", "5m", clean_expr)
            clean_expr = re.sub(r"\$(?:__)?range\b", "1h", clean_expr)
            clean_expr = re.sub(r"\$(?:__)?interval\b", "1m", clean_expr)

            status, detail = query_prometheus(clean_expr)
            results.append({
                "panel_title": p_title,
                "panel_type": p_type,
                "ref_id": t.get("refId", "A"),
                "expr": expr,
                "clean_expr": clean_expr,
                "status": status,
                "detail": detail
            })
    return uid, title, results

def main():
    dashboards = sorted(glob.glob("tools/grafana/dashboards/*.json"))
    summary = {}
    total_queries = 0
    total_errors = 0
    total_no_data = 0
    total_zero = 0
    total_non_zero = 0

    print(f"{'Dashboard UID':32} | {'Queries':7} | {'Non-Zero':8} | {'Zero':6} | {'No-Data':7} | {'Error':5}")
    print("-" * 75)

    all_details = {}
    for path in dashboards:
        uid, title, results = audit_dashboard(path)
        q_count = len(results)
        c_nonzero = sum(1 for r in results if r["status"] == "NON_ZERO")
        c_zero = sum(1 for r in results if r["status"] == "ZERO")
        c_nodata = sum(1 for r in results if r["status"] == "NO_DATA")
        c_error = sum(1 for r in results if r["status"] == "ERROR")

        total_queries += q_count
        total_non_zero += c_nonzero
        total_zero += c_zero
        total_no_data += c_nodata
        total_errors += c_error

        print(f"{uid:32} | {q_count:7} | {c_nonzero:8} | {c_zero:6} | {c_nodata:7} | {c_error:5}")
        all_details[uid] = results

    print("-" * 75)
    print(f"{'TOTAL':32} | {total_queries:7} | {total_non_zero:8} | {total_zero:6} | {total_no_data:7} | {total_errors:5}")

    if "--details" in sys.argv:
        for uid, res in all_details.items():
            zero_or_nodata = [r for r in res if r["status"] in ("ZERO", "NO_DATA", "ERROR")]
            if zero_or_nodata:
                print(f"\n=== {uid} (Issues: {len(zero_or_nodata)}) ===")
                for r in zero_or_nodata:
                    print(f"[{r['status']}] {r['panel_title']} ({r['ref_id']}): {r['clean_expr']}")
                    if r["status"] == "ERROR":
                        print(f"    Error: {r['detail']}")

    if "--verify-all" in sys.argv:
        # Exit non-zero if there are unexpected zeroes/no-data
        if total_errors > 0 or total_no_data > 0:
            sys.exit(1)

if __name__ == "__main__":
    main()
