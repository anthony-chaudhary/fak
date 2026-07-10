#!/usr/bin/env python3
"""Doc-numbers gate: every headline number in a cache/benchmark doc traces to a
committed snapshot field, renders to the right rounding, and stays internally
consistent — the checking layer for operational cachevalue docs.

`BENCHMARK-AUTHORITY.md` is the single source of truth for *committed* benchmark
claims, and `tools/readme_freshness_audit.py` keeps the README's headline numbers
honest. But a doc like `docs/integrations/fable5-more-usage-for-free.md` reports
RECENT OPERATIONAL TELEMETRY — this week's fleet `fak cachevalue report`, the last
few days of this repo's dev sessions — that is too live-moving to earn an
authority row, yet is exactly where a number silently rots or an arithmetic
mistake slips in (the real one this gate was born from: "60 discovered — 15
priced; 36 held out" read as 60 = 15 + 36, which is 51). This is that class of
doc's checking layer.

Each guarded doc has a manifest under `tools/docnumbers/<stem>.json` that binds,
per claim: the literal string the doc renders (`appears_as`), the trimmed frozen
snapshot the number was captured from (`source` + `field` + `expected`), and the
window the number lives in. A directory of frozen, trimmed snapshots sits beside
it under `tools/docnumbers/snapshots/<stem>/`. The manifest also carries the
arithmetic INVARIANTS the doc asserts (sums that must close, formulas that must
hold). See `tools/docnumbers/README.md` for the schema and how to add a doc.

The checks, all HERMETIC (no network, no `fak`, no live data) unless noted:

  binding      every claim's `appears_as` string is present in the doc.      FAIL
  snapshot     `expected` equals the committed snapshot `field` — the         FAIL
               manifest cannot fabricate a number the snapshot disproves.
  rounding     the doc's rendered digits are a correct rounding of            FAIL
               `expected * scale` (approx "≈" values allowed 3% slack).
  invariants   declared sums close and formulas hold (the 60≠15+36 guard).    FAIL
  provenance   each table-row claim's line carries its WITNESSED/OBSERVED/     WARN
               ESTIMATED label (same discipline as check_cache_headlines).
  staleness    the snapshot is younger than `stale_after_days` — a nudge to   WARN
               re-run the refresh pass, never a hard fail.

  --live       ADDITIONALLY re-derive `since_floored` claims from a live
               `fak cachevalue report` and drift-check them. SKIPs (advisory,
               exit 0) when `fak` is not on PATH or the feed is absent — the
               same availability contract as the `salience` / `dos-lint` gates.
               live_snapshot and all_time_last_measured windows are NEVER
               live-equality-checked: they grow between captures, so equality
               would false-flag drift.

  --refresh    regenerate the trimmed snapshots for a doc from live `fak`
               output and print the derived numbers, so a refresh pass starts
               from reproducible data instead of ad-hoc shell. Writes only
               inside the doc's `snapshot_dir`. SKIPs sources with no `cmd`
               (machine-specific transcript paths) or when `fak` is absent.

Modes are mutually exclusive except that --live composes with the default audit.
Exit: 0 = clean (WARNs allowed), 1 = a FAIL fired, 2 = could not run (setup).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


MANIFEST_GLOB = "tools/docnumbers/*.json"
SCHEMA = "fak.docnumbers.v1"
PROVENANCE_TOKENS = ("WITNESSED", "OBSERVED", "ESTIMATED")

# Window taxonomy — what --live may assert about a claim:
#   bounded    doubly-bounded and settled (--since X --until Y, Y in the past):
#              the value is REPRODUCIBLE, so live may equality-check it.
#   since_open open-ended (--since X, no end): the window GROWS with wall-clock,
#              so today's larger number is elapsed time, not drift — live may
#              only probe that the cited fields still exist (schema liveness).
#   all_time_last_measured  last-measured bucket; value moves as new buckets land
#              — schema-probe only.
#   live_snapshot  dev transcripts / per-session status that grow between
#              captures — NEVER touched by live; the frozen snapshot is the gate.
LIVE_EQUALITY_WINDOWS = ("bounded",)
LIVE_NO_TOUCH_WINDOWS = ("live_snapshot",)


# --------------------------------------------------------------------------- #
# small pure helpers                                                          #
# --------------------------------------------------------------------------- #
def dotted(obj: dict, path: str):
    """Fetch a dotted path; keys may themselves contain no '.' (e.g. <synthetic>)."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


_NUM_CORE = re.compile(r"-?[\d,]*\.?\d+")


def parse_display(s: str) -> dict:
    """Parse a rendered number ("≈$1,181.37", "62.6M", "89.3%") to a comparable value.

    Returns {value, place, is_pct, approx}. `place` is the magnitude of the last
    shown digit (0.01 for "1,181.37", 1e5 for "62.6M") — half of it is the
    correct-rounding tolerance for a non-approx render.
    """
    raw = s.strip()
    approx = bool(raw) and (raw[0] in "≈~" or raw.lower().startswith("about"))
    t = raw.lstrip("≈~ ").strip()
    is_pct = t.endswith("%")
    t = t.rstrip("%").strip().replace("$", "").strip()
    mult = 1.0
    if t and t[-1] in "MmKkBb":
        mult = {"k": 1e3, "m": 1e6, "b": 1e9}[t[-1].lower()]
        t = t[:-1]
    core = t.replace(",", "").strip()
    m = _NUM_CORE.search(core)
    if not m:
        raise ValueError(f"cannot parse a number from {s!r}")
    core = m.group(0).replace(",", "")
    value = float(core) * mult
    decimals = len(core.split(".")[1]) if "." in core else 0
    place = (10.0 ** (-decimals)) * mult
    return {"value": value, "place": place, "is_pct": is_pct, "approx": approx}


def rounding_ok(display: str, true_value: float) -> tuple[bool, str]:
    """Is `display` a correct rendering of `true_value`?"""
    d = parse_display(display)
    if d["approx"]:
        denom = abs(true_value) if abs(true_value) > 1e-9 else 1.0
        rel = abs(true_value - d["value"]) / denom
        ok = rel <= 0.03
        return ok, f"approx |{true_value:.6g}-{d['value']:.6g}| rel={rel:.3%} (≤3%)"
    tol = 0.5 * d["place"] + 1e-9 * max(1.0, abs(true_value))
    diff = abs(true_value - d["value"])
    return diff <= tol, f"|{true_value:.6g}-{d['value']:.6g}|={diff:.4g} (≤{tol:.4g})"


_SAFE_EXPR = re.compile(r"^[\d\s.+\-*/()]+$")


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression over numbers and + - * / ( ) only."""
    if not _SAFE_EXPR.match(expr):
        raise ValueError(f"unsafe expression: {expr!r}")
    # nosemgrep: python.lang.security.audit.eval-detected — input gated by _SAFE_EXPR
    return float(eval(expr, {"__builtins__": {}}, {}))


# --------------------------------------------------------------------------- #
# loading                                                                     #
# --------------------------------------------------------------------------- #
def load_manifests(root: str, only: str | None) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, MANIFEST_GLOB))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema") != SCHEMA:
            continue
        if only and os.path.basename(path) != os.path.basename(only):
            continue
        data["_manifest_path"] = path
        out.append(data)
    return out


def read_doc(root: str, rel: str) -> str | None:
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def read_snapshot(root: str, snap_dir: str, source: str) -> dict | None:
    try:
        with open(os.path.join(root, snap_dir, source), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# hermetic audit                                                              #
# --------------------------------------------------------------------------- #
def audit_manifest(root: str, m: dict, today: dt.date) -> tuple[list[dict], list[dict]]:
    """Return (fails, warns) for one manifest."""
    fails: list[dict] = []
    warns: list[dict] = []
    mid = os.path.basename(m["_manifest_path"])

    def fail(check, msg):
        fails.append({"manifest": mid, "check": check, "msg": msg})

    def warn(check, msg):
        warns.append({"manifest": mid, "check": check, "msg": msg})

    doc = read_doc(root, m["doc"])
    if doc is None:
        fail("binding", f"doc not found: {m['doc']}")
        return fails, warns
    doc_lines = doc.splitlines()
    snap_dir = m["snapshot_dir"]
    sources = m.get("sources", {})

    for c in m.get("claims", []):
        cid = c["id"]
        appears = c["appears_as"]
        # 1. binding: the rendered string is present in the doc
        if appears not in doc:
            fail("binding", f"[{cid}] appears_as not found in doc: {appears!r}")
            continue
        snap = read_snapshot(root, snap_dir, c["source"])
        if snap is None:
            fail("snapshot", f"[{cid}] snapshot unreadable: {c['source']}")
            continue
        for n in c.get("numbers", []):
            disp = n["display"]
            scale = n.get("scale", 1)
            expected = n["expected"]
            # 2. snapshot binding: expected == committed snapshot field
            if "field" in n:
                try:
                    got = dotted(snap, n["field"])
                except KeyError:
                    fail("snapshot", f"[{cid}] field {n['field']} missing in {c['source']}")
                    continue
                if isinstance(got, bool) or not isinstance(got, (int, float)):
                    fail("snapshot", f"[{cid}] field {n['field']} not numeric: {got!r}")
                    continue
                if abs(float(got) - float(expected)) > 1e-6:
                    fail("snapshot",
                         f"[{cid}] expected {expected} != snapshot {c['source']}::{n['field']}={got}")
                    continue
            # 3. rounding: the doc render is a correct rounding of expected*scale
            ok, detail = rounding_ok(disp, float(expected) * scale)
            if not ok:
                fail("rounding", f"[{cid}] {disp!r} does not render {expected}×{scale}: {detail}")
        # 5. provenance co-location (table-row claims only)
        prov = c.get("provenance")
        if prov and c.get("provenance_inline", True):
            hosts = [ln for ln in doc_lines if appears in ln]
            if hosts and not any(prov in ln for ln in hosts):
                warn("provenance",
                     f"[{cid}] line with {appears[:32]!r} lacks its {prov} label")

    # 4. invariants
    for inv in m.get("invariants", []):
        label = inv.get("label", inv.get("kind"))
        if inv["kind"] == "sum":
            total, parts = inv["total"], inv["parts"]
            if abs(sum(parts) - total) > inv.get("tol", 0):
                fail("invariants",
                     f"sum '{label}': {' + '.join(map(str, parts))} = {sum(parts)} != {total}")
        elif inv["kind"] == "formula":
            try:
                got = safe_eval(inv["expr"])
            except ValueError as e:
                fail("invariants", f"formula '{label}': {e}")
                continue
            if abs(got - inv["expect"]) > inv.get("tol", 1e-6):
                fail("invariants",
                     f"formula '{label}': {inv['expr']} = {got:.6g} != {inv['expect']}")
        else:
            warn("invariants", f"unknown invariant kind: {inv['kind']!r}")

    # 6. staleness (advisory)
    sd = m.get("snapshot_date")
    limit = m.get("stale_after_days")
    if sd and limit:
        try:
            age = (today - dt.date.fromisoformat(sd)).days
            if age > limit:
                warn("staleness",
                     f"snapshot {sd} is {age}d old (> {limit}d) — run the refresh pass")
        except ValueError:
            warn("staleness", f"bad snapshot_date: {sd!r}")

    return fails, warns


# --------------------------------------------------------------------------- #
# live re-derivation (opt-in, SKIP-if-absent)                                 #
# --------------------------------------------------------------------------- #
def _fak_bin() -> str | None:
    return shutil.which("fak") or shutil.which("fak.exe")


def run_live(root: str, m: dict) -> tuple[list[dict], list[str]]:
    """Re-run the cited commands against live `fak`. Returns (fails, notes).

    Two rungs, by window:
      * `bounded` (reproducible) claims are equality-checked — a >1% move is
        real drift and FAILs.
      * every other cmd-bearing, non-live_snapshot source is SCHEMA-PROBED — the
        cited field must still resolve to a number. A field that has vanished
        FAILs (the doc's provenance path is stale); a field whose VALUE has
        merely moved does not (open-ended windows grow — that is elapsed time,
        not rot). SKIPs (advisory) when `fak` or the field's command is absent.
    """
    fails: list[dict] = []
    notes: list[str] = []
    mid = os.path.basename(m["_manifest_path"])
    fak = _fak_bin()
    if not fak:
        return fails, [f"{mid}: SKIP live (fak not on PATH)"]
    probed = equality = 0
    for src_name, src in m.get("sources", {}).items():
        window = src.get("window", "")
        if not src.get("cmd") or window in LIVE_NO_TOUCH_WINDOWS:
            continue
        try:
            proc = subprocess.run([fak, *src["cmd"]], cwd=root, capture_output=True,
                                  text=True, timeout=120, creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired) as e:
            notes.append(f"{mid}:{src_name}: SKIP live ({e})")
            continue
        if proc.returncode != 0:
            notes.append(f"{mid}:{src_name}: SKIP live (fak exit {proc.returncode})")
            continue
        try:
            live = json.loads(proc.stdout)
        except json.JSONDecodeError:
            notes.append(f"{mid}:{src_name}: SKIP live (non-JSON output)")
            continue
        equality_window = window in LIVE_EQUALITY_WINDOWS
        for c in m.get("claims", []):
            if c["source"] != src_name:
                continue
            for n in c.get("numbers", []):
                if "field" not in n:
                    continue
                try:
                    got = float(dotted(live, n["field"]))
                except (KeyError, TypeError, ValueError):
                    fails.append({"manifest": mid, "check": "live",
                                  "msg": f"[{c['id']}] cited field {n['field']} no "
                                         f"longer emitted by `fak {' '.join(src['cmd'])}`"})
                    continue
                if equality_window:
                    equality += 1
                    exp = float(n["expected"])
                    denom = abs(exp) if abs(exp) > 1e-9 else 1.0
                    if abs(got - exp) / denom > 0.01:
                        fails.append({"manifest": mid, "check": "live",
                                      "msg": f"[{c['id']}] {n['field']} live={got:.6g} "
                                             f"vs frozen {exp:.6g} (>1% drift, bounded window)"})
                else:
                    probed += 1
    if probed or equality:
        notes.append(f"{mid}: live probed {probed} open-window field(s), "
                     f"equality-checked {equality} bounded field(s)")
    return fails, notes


# --------------------------------------------------------------------------- #
# refresh (regenerate trimmed snapshots)                                       #
# --------------------------------------------------------------------------- #
def _nest(fields: list[str], src: dict) -> dict:
    out: dict = {}
    for path in fields:
        parts = path.split(".")
        cur_src, cur_out = src, out
        ok = True
        for p in parts[:-1]:
            if not isinstance(cur_src, dict) or p not in cur_src:
                ok = False
                break
            cur_src = cur_src[p]
            cur_out = cur_out.setdefault(p, {})
        if ok and isinstance(cur_src, dict) and parts[-1] in cur_src:
            cur_out[parts[-1]] = cur_src[parts[-1]]
    return out


def run_refresh(root: str, m: dict) -> int:
    mid = os.path.basename(m["_manifest_path"])
    fak = _fak_bin()
    if not fak:
        print(f"refresh {mid}: cannot run — fak not on PATH", file=sys.stderr)
        return 2
    snap_dir = os.path.join(root, m["snapshot_dir"])
    os.makedirs(snap_dir, exist_ok=True)
    sources = m.get("sources", {})
    by_source: dict[str, list[str]] = {}
    for c in m.get("claims", []):
        for n in c.get("numbers", []):
            if "field" in n:
                by_source.setdefault(c["source"], []).append(n["field"])
    wrote = 0
    for src_name, src in sources.items():
        if not src.get("cmd"):
            print(f"  {src_name}: SKIP (no cmd — machine-specific path)")
            continue
        proc = subprocess.run([fak, *src["cmd"]], cwd=root, capture_output=True,
                              text=True, timeout=120, creationflags=_win_creationflags())
        if proc.returncode != 0:
            print(f"  {src_name}: SKIP (fak exit {proc.returncode})", file=sys.stderr)
            continue
        try:
            live = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(f"  {src_name}: SKIP (non-JSON output)", file=sys.stderr)
            continue
        trimmed = {"_source": "fak " + " ".join(src["cmd"]),
                   "_window": src.get("window", ""),
                   "_captured": live.get("generated_at", "")}
        trimmed.update(_nest(by_source.get(src_name, []), live))
        with open(os.path.join(snap_dir, src_name), "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, sort_keys=True)
            f.write("\n")
        wrote += 1
        print(f"  {src_name}: refreshed ({len(by_source.get(src_name, []))} fields)")
    print(f"refresh {mid}: wrote {wrote} snapshot(s) to {m['snapshot_dir']}")
    print("Next: update the doc's numbers, bump snapshot_date, re-run the audit.")
    return 0


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--manifest", help="limit to one manifest (basename)")
    ap.add_argument("--live", action="store_true",
                    help="additionally re-derive since_floored claims from live fak")
    ap.add_argument("--refresh", action="store_true",
                    help="regenerate trimmed snapshots from live fak (mutates snapshot_dir)")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable output")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)

    manifests = load_manifests(root, args.manifest)
    if not manifests:
        print("doc-numbers: no manifests found under tools/docnumbers/*.json",
              file=sys.stderr)
        return 2

    if args.refresh:
        rc = 0
        for m in manifests:
            rc = run_refresh(root, m) or rc
        return rc

    today = dt.date.today()
    all_fails: list[dict] = []
    all_warns: list[dict] = []
    for m in manifests:
        f, w = audit_manifest(root, m, today)
        all_fails += f
        all_warns += w

    live_skips: list[str] = []
    if args.live:
        for m in manifests:
            lf, ls = run_live(root, m)
            all_fails += lf
            live_skips += ls

    if args.as_json:
        print(json.dumps({"fails": all_fails, "warns": all_warns,
                          "live": live_skips, "ok": not all_fails}, indent=2))
        return 1 if all_fails else 0

    n_claims = sum(len(m.get("claims", [])) for m in manifests)
    for w in all_warns:
        print(f"  WARN [{w['check']}] {w['manifest']}: {w['msg']}")
    for s in live_skips:
        print(f"  live: {s}")
    if not all_fails:
        print(f"doc-numbers: clean — {n_claims} claims across {len(manifests)} "
              f"doc(s) trace to their snapshots ({len(all_warns)} warn).")
        return 0
    print(f"DOC_NUMBERS: {len(all_fails)} FAIL(s):", file=sys.stderr)
    for fl in all_fails:
        print(f"  FAIL [{fl['check']}] {fl['manifest']}: {fl['msg']}", file=sys.stderr)
    print("  fix: correct the doc render, the manifest expected, or re-run "
          "--refresh; see tools/docnumbers/README.md.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
