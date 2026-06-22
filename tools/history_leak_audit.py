#!/usr/bin/env python3
"""Full-HISTORY secret-leak audit — the durable form of the one-off go-public scan.

`scrub_public_copy.py --audit-tree` scans the current tree; the pre-commit/CI gates
scan added lines. None of them see HISTORY. But making a repo public publishes every
reachable commit and tag, not just the tree — so a value redacted in a *follow-up*
commit still ships in the pre-redaction blob. This tool closes that gap: it sweeps the
content of every blob reachable from the refs and reports any that carry a real
operator needle or a secret SHAPE, attributing each to whether it publishes from the
trunk (`main`) or only via another ref (e.g. a local backup tag).

It REUSES `scrub_public_copy.py`'s own needle logic so it can't drift:
  * the REAL needles come from the gitignored sidecar (`tools/pull_scan_needles.py`);
    absent the sidecar the literal tier is SKIPPED and the run is reported DEGRADED
    (shape-only) — never silently "clean".
  * the secret SHAPES are `AUDIT_REGEXES` (live Slack token, …).
  * `SELF_REFERENTIAL` files (the denylist/policy that must name needles) are exempt.

Output is REDACTED: it never prints a needle's value, only its tier + a masked snippet,
so the report itself is safe. Exit 0 = clean, 1 = leak in history, 2 = could not run.

Usage:
  python tools/history_leak_audit.py [--root .] [--ref main] [--json]

NOTE (Windows Git Bash): this does ALL matching in Python, not shell `grep -F`/`-f`,
which aborts (rc 134) on this platform and silently produces a FALSE-CLEAN.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_scrubber():
    path = os.path.join(HERE, "scrub_public_copy.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("scrub_public_copy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(root, *args, binary=False):
    return subprocess.run(["git", "-C", root, *args], capture_output=True,
                          check=False, **({} if binary else dict(text=True, encoding="utf-8", errors="replace")))


def _reachable_blobs(root, rev):
    """sha -> set(paths) for every blob reachable from `rev` (a ref or --all)."""
    out = _git(root, "rev-list", "--objects", rev)
    blobs = {}
    for line in out.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue  # commits/trees/tags have no path
        sha, path = parts[0], parts[1]
        blobs.setdefault(sha, set()).add(path)
    return blobs


def _blob_shas_reachable(root, rev):
    out = _git(root, "rev-list", "--objects", rev)
    shas = set()
    for line in out.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            shas.add(parts[0])
    return shas


def _batch_read(root, shas):
    """Yield (sha, text) for each blob sha via one `git cat-file --batch` call."""
    if not shas:
        return
    proc = subprocess.run(["git", "-C", root, "cat-file", "--batch"],
                          input="\n".join(shas).encode(), capture_output=True, check=False)
    data = proc.stdout
    i, n = 0, len(data)
    while i < n:
        nl = data.find(b"\n", i)
        if nl < 0:
            break
        header = data[i:nl].decode("utf-8", "replace").split(" ")
        i = nl + 1
        if len(header) != 3 or header[1] != "blob":
            # missing/other; skip its (no) payload
            continue
        sha, size = header[0], int(header[2])
        content = data[i:i + size]
        i += size + 1  # skip trailing newline
        yield sha, content.decode("utf-8", "replace")


def _mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 4:
        return "****"
    return v[:2] + "…" + v[-1]


def audit_history(root, ref="main", as_json=False):
    scr = _load_scrubber()
    if scr is None:
        print("history-leak: scrub_public_copy.py not found (the secret-leak gate); cannot run", file=sys.stderr)
        return 2

    priv = scr.load_private_needles(root)
    real_needles = []
    if priv:
        real_needles = [n for n in (list(priv.get("audit_needles") or [])
                                    + list(priv.get("export_audit_needles") or [])) if n]
    mode = "full" if real_needles else "degraded(shape-only)"
    regexes = scr.AUDIT_REGEXES
    self_ref = scr.SELF_REFERENTIAL
    is_text = scr.is_text

    all_blobs = _reachable_blobs(root, "--all")
    if not all_blobs:
        print("history-leak: no blobs reachable (not a git repo / empty)", file=sys.stderr)
        return 2
    publishes = _blob_shas_reachable(root, ref)

    needle_low = [(n, n.lower()) for n in real_needles]
    hits = []
    for sha, text in _batch_read(root, list(all_blobs.keys())):
        paths = all_blobs.get(sha, set())
        # exempt only if EVERY path this blob ever had is self-referential
        norm = {p.replace("\\", "/") for p in paths}
        if norm and norm <= self_ref:
            continue
        if paths and all(not is_text(p) for p in paths):
            continue
        low = text.lower()
        found = []
        for n, nl in needle_low:
            if nl in low:
                found.append(("needle", n))
        for rx, label in regexes:
            m = rx.search(text)
            if m:
                found.append(("shape:" + label, m.group(0)))
        if found:
            example_path = sorted(paths)[0] if paths else "(unknown)"
            for kind, val in found:
                hits.append({
                    "sha": sha,
                    "path": example_path,
                    "paths": sorted(paths),
                    "tier": "shape" if kind.startswith("shape:") else "needle",
                    "label": kind if kind.startswith("shape:") else "operator needle",
                    "masked": _mask(val),
                    "publishes_from_ref": sha in publishes,
                })

    pub = [h for h in hits if h["publishes_from_ref"]]
    other = [h for h in hits if not h["publishes_from_ref"]]

    if as_json:
        print(json.dumps({
            "mode": mode, "ref": ref,
            "blobs_scanned": len(all_blobs),
            "verdict": "HISTORY-DIRTY" if hits else ("DEGRADED-CLEAN" if mode != "full" else "HISTORY-CLEAN"),
            "publishes": pub, "other_refs_only": other,
        }, indent=2))
    else:
        print(f"history-leak: mode={mode}  ref={ref}  blobs={len(all_blobs)}")
        if not hits:
            verdict = "HISTORY-CLEAN" if mode == "full" else "DEGRADED-CLEAN (literal tier skipped — pull the sidecar for a full verdict)"
            print(f"  {verdict}")
        else:
            if pub:
                print(f"  PUBLISHES FROM {ref} ({len(pub)} hit(s) — these go public):")
                for h in pub:
                    print(f"    {h['sha'][:12]} {h['path']}  [{h['label']}]  ~{h['masked']}")
            if other:
                print(f"  OTHER REFS ONLY ({len(other)} hit(s) — e.g. local backup tags; contained unless tags are pushed):")
                for h in other:
                    print(f"    {h['sha'][:12]} {h['path']}  [{h['label']}]  ~{h['masked']}")
            print("  VERDICT: HISTORY-DIRTY — rewrite history before publishing (see issue #74).")
        if mode != "full":
            print("  NOTE: sidecar absent — real-needle (literal) tier NOT run; run tools/pull_scan_needles.py for a full scan.", file=sys.stderr)

    return 1 if hits else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--ref", default="main", help="the publishable ref (default: main)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    return audit_history(root, ref=args.ref, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
