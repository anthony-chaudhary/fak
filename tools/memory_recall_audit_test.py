#!/usr/bin/env python3
"""Tests for the memory-recall freshness auditor.

Drives the PURE grader (build_payload / collect with an injected verifier) so the
tests need no `dos` binary, then a tolerant live smoke check that `collect` folds
the real per-project store when `dos memory verify` is available.

Run: `python tools/memory_recall_audit_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_recall_audit as mra  # noqa: E402

FRESH, STALE, UNVER = mra.FRESH, mra.STALE, mra.UNVERIFIABLE


def _rec(name, verdict, **kw):
    r = {"memory": name, "verdict": verdict}
    r.update(kw)
    return r


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(name)

    ws = "/repo"

    # 1) all fresh + unverifiable, no stale -> ok, rate ignores unverifiable.
    p = mra.build_payload(workspace=ws, records=[
        _rec("a", FRESH), _rec("b", FRESH),
        _rec("c", UNVER), _rec("d", UNVER), _rec("e", UNVER),
        _rec("MEMORY", UNVER),  # index file -> excluded from counts entirely
    ])
    check("clean store is ok", p["ok"] is True, str(p))
    check("verdict OK when no stale", p["verdict"] == "OK")
    check("freshness_rate ignores unverifiable", p["memory_freshness_rate"] == 1.0,
          str(p["memory_freshness_rate"]))
    check("index file excluded from audit", p["totals"]["memories_audited"] == 5,
          str(p["totals"]))
    check("counts exclude index", p["counts"][FRESH] == 2 and p["counts"][UNVER] == 3,
          str(p["counts"]))

    # 2) a STALE memory -> not ok, ACTION, rate = fresh/(fresh+stale), culprit set.
    p = mra.build_payload(workspace=ws, records=[
        _rec("fresh1", FRESH), _rec("fresh2", FRESH), _rec("fresh3", FRESH),
        _rec("gone", STALE,
             culprit={"claim": {"raw": "deadbeef"}, "ground_truth": "not an ancestor of HEAD"},
             reason="a load-bearing SHA fell out of history"),
        _rec("opinion", UNVER),
    ])
    check("stale store is not ok", p["ok"] is False)
    check("verdict ACTION on stale", p["verdict"] == "ACTION", p["verdict"])
    check("freshness_rate = 3/4", p["memory_freshness_rate"] == 0.75,
          str(p["memory_freshness_rate"]))
    stale_rows = [m for m in p["memories"] if m["verdict"] == STALE]
    check("stale row carries culprit", bool(stale_rows) and "deadbeef" in stale_rows[0]["culprit"],
          str(stale_rows))
    check("stale sorts first", p["memories"][0]["verdict"] == STALE, str(p["memories"][0]))

    # 3) all unverifiable (the common fleet case) -> ok, rate None, NOT a failure.
    p = mra.build_payload(workspace=ws, records=[_rec("x", UNVER), _rec("y", UNVER)])
    check("all-unverifiable is ok", p["ok"] is True)
    check("rate is None when no fresh/stale", p["memory_freshness_rate"] is None,
          str(p["memory_freshness_rate"]))

    # 4) tooling error -> not ok, AUDIT_ERROR (a missing verifier is not a clean pass).
    p = mra.collect(Path(ws), verifier=lambda w, s: {"error": "dos not on PATH"})
    check("tooling error is not ok", p["ok"] is False)
    check("tooling error verdict", p["verdict"] == "AUDIT_ERROR", p["verdict"])

    # 5) collect with an injected record list folds through cleanly.
    p = mra.collect(Path(ws), verifier=lambda w, s: {"records": [_rec("m", FRESH)]})
    check("collect folds injected records", p["counts"][FRESH] == 1 and p["ok"] is True, str(p["counts"]))

    # 6) exit code mirrors ok: 0 on clean, 1 on stale (via main with a fake store).
    #    Use a tiny monkeypatch of verify_store so main() needs no dos binary.
    orig = mra.verify_store
    try:
        mra.verify_store = lambda w, s: {"records": [_rec("m", STALE,
            culprit={"claim": {"raw": "x"}, "ground_truth": "gone"})]}
        rc = mra.main(["--workspace", ws, "--json"])
        check("main exits 1 on stale", rc == 1, str(rc))
        mra.verify_store = lambda w, s: {"records": [_rec("m", FRESH)]}
        rc = mra.main(["--workspace", ws, "--json"])
        check("main exits 0 on clean", rc == 0, str(rc))
    finally:
        mra.verify_store = orig

    # 6b) the default store resolves to the real per-project location, NOT a
    #     repo-relative ".claude/memory" mirror (regression guard for #1141).
    #     project_namespace resolves its arg, so the encoding is asserted on the
    #     transform itself: every drive-colon / separator collapses to "-", and
    #     no separator survives in the namespace.
    ns = mra.project_namespace(Path("/work/fak"))
    check("namespace has no path separators", "/" not in ns and "\\" not in ns and ":" not in ns, ns)
    check("namespace collapses separators to dash", ns.endswith("-work-fak"), ns)
    store = mra.default_store(Path("/work/fak"))
    check("default store is under ~/.claude/projects", "projects" in store.parts, str(store))
    check("default store ends in <ns>/memory",
          store.parts[-1] == "memory" and store.parts[-2] == ns, str(store))
    check("default store is NOT repo-relative .claude/memory",
          ".claude/memory" not in str(store).replace("\\", "/").rsplit("projects", 1)[0],
          str(store))
    old_mem = os.environ.get("CLAUDE_MEMORY_DIR")
    old_cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    try:
        os.environ.pop("CLAUDE_MEMORY_DIR", None)
        os.environ["CLAUDE_CONFIG_DIR"] = "/tmp/claude-config"
        cfg_store = mra.default_store(Path("/work/fak"))
        check("CLAUDE_CONFIG_DIR relocates default store",
              str(cfg_store).replace("\\", "/").endswith("/tmp/claude-config/projects/" + ns + "/memory"),
              str(cfg_store))
        os.environ["CLAUDE_MEMORY_DIR"] = "/tmp/explicit-memory"
        mem_store = mra.default_store(Path("/work/fak"))
        check("CLAUDE_MEMORY_DIR overrides CLAUDE_CONFIG_DIR",
              str(mem_store).replace("\\", "/") == "/tmp/explicit-memory",
              str(mem_store))
    finally:
        if old_mem is None:
            os.environ.pop("CLAUDE_MEMORY_DIR", None)
        else:
            os.environ["CLAUDE_MEMORY_DIR"] = old_mem
        if old_cfg is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = old_cfg

    # 7) LIVE smoke (tolerant): if dos is available and the real per-project
    #    store exists, collect must fold it without crashing and return a
    #    well-formed payload.
    root = mra.repo_root()
    store = mra.default_store(root)
    if store.is_dir():
        live = mra.collect(root)
        ok_shape = (
            live.get("schema") == mra.SCHEMA
            and "memory_freshness_rate" in live
            and isinstance(live.get("counts"), dict)
        )
        # AUDIT_ERROR is acceptable here (dos may be absent in CI before the pip
        # install step); we only require a well-formed payload, never a crash.
        check("live collect returns well-formed payload", ok_shape, str(live)[:200])
        print(f"    (live verdict={live.get('verdict')} rate={live.get('memory_freshness_rate')} "
              f"counts={live.get('counts')})")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("memory_recall_audit_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
