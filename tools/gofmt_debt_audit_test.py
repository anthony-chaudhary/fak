#!/usr/bin/env python3
"""Tests for the gofmt format-debt auditor.

Drives the PURE grader (build_payload) so the core needs no git, plus a gofmt-gated
check of is_clean on known-clean/known-unclean source, plus a tolerant live smoke
that `collect` folds the real committed tree when git + gofmt are available.

Run: `python tools/gofmt_debt_audit_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gofmt_debt_audit as gda  # noqa: E402


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(name)

    ws = "/repo"

    # 1) clean scan -> ok True, verdict OK.
    p = gda.build_payload(workspace=ws, scan_result={"scanned": 12, "unclean": [], "unparseable": []})
    check("clean tree is ok", p["ok"] is True, str(p))
    check("verdict OK when no debt", p["verdict"] == "OK")
    check("finding gofmt_clean", p["finding"] == "gofmt_clean")
    check("counts.scanned passthrough", p["counts"]["scanned"] == 12, str(p["counts"]))
    check("schema stamped", p["schema"] == gda.SCHEMA)

    # 2) gofmt debt -> not ok, ACTION, names the files, points at /gofmt-sweep.
    p = gda.build_payload(workspace=ws, scan_result={
        "scanned": 9, "unclean": ["fak/internal/model/awq.go", "fak/cmd/x/main.go"],
        "unparseable": [],
    })
    check("debt is not ok", p["ok"] is False)
    check("verdict ACTION on debt", p["verdict"] == "ACTION")
    check("finding gofmt_debt", p["finding"] == "gofmt_debt")
    check("reason names a file", "awq.go" in p["reason"], p["reason"])
    check("next_action points at /gofmt-sweep", "/gofmt-sweep" in p["next_action"])
    check("next_action warns off gofmt -w .", "gofmt -w ." in p["next_action"])
    check("counts.unclean == 2", p["counts"]["unclean"] == 2, str(p["counts"]))

    # 3) tooling error -> not ok, AUDIT_ERROR (so the pane surfaces it, not silent OK).
    p = gda.build_payload(workspace=ws, scan_result={
        "scanned": 0, "unclean": [], "unparseable": [], "error": "gofmt not found on PATH",
    })
    check("error is not ok", p["ok"] is False)
    check("verdict AUDIT_ERROR", p["verdict"] == "AUDIT_ERROR")
    check("error reason surfaced", "gofmt not found" in p["reason"])

    # 4) committed .go that won't parse -> ACTION (rare, but never a silent pass).
    p = gda.build_payload(workspace=ws, scan_result={
        "scanned": 3, "unclean": [], "unparseable": [{"path": "fak/x.go", "error": "expected ';'"}],
    })
    check("unparseable is not ok", p["ok"] is False)
    check("verdict ACTION on unparseable", p["verdict"] == "ACTION")
    check("finding unparseable_go", p["finding"] == "unparseable_go")

    # 5) is_clean against the real gofmt (skip if gofmt absent — tolerant like the smoke).
    if shutil.which("gofmt"):
        clean_src = b"package x\n"
        unclean_src = b"package x\nfunc  F() {}\n"  # double space after func -> gofmt rewrites
        c1, e1 = gda.is_clean(clean_src)
        check("is_clean True on canonical source", c1 is True, f"err={e1}")
        c2, _ = gda.is_clean(unclean_src)
        check("is_clean False on unformatted source", c2 is False)
        bad, err = gda.is_clean(b"package x\nfunc (\n")  # parse error
        check("is_clean None on unparseable source", bad is None and bool(err))
    else:
        print("  [skip] gofmt not on PATH — is_clean checks skipped")

    # 6) live smoke: collect over the real repo (skip if git/gofmt absent). Tolerant
    #    on ok (trunk state varies) but the structure + invariants must hold, and any
    #    file it reports unclean must actually exist on disk.
    if shutil.which("git") and shutil.which("gofmt"):
        root = gda.repo_root()
        live = gda.collect(root)
        check("live payload has ok bool", isinstance(live.get("ok"), bool), str(live.get("verdict")))
        check("live payload stamps schema", live.get("schema") == gda.SCHEMA)
        check("live counts present", "scanned" in (live.get("counts") or {}))
        if not live.get("error"):
            check("live scanned > 0 (found the module)", live["counts"]["scanned"] > 0,
                  str(live["counts"]))
            for f in (live.get("unclean") or [])[:5]:
                check(f"reported unclean file exists: {f}", (root / f).exists())
    else:
        print("  [skip] git/gofmt not on PATH — live smoke skipped")

    print(f"\ngofmt_debt_audit_test: {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
