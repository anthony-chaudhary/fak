#!/usr/bin/env python3
"""Self-running tests for stale_work_watchdog.py.

Pure stdlib, no process spawn, no network: every test builds a throwaway fak-shaped
clone in a tempdir, stamps mtimes via os.utime, and injects a fake `now` and a fake
git-porcelain so the WIP scan needs no real repo. Runnable two ways:

    python tools/stale_work_watchdog_test.py        # self-running (CI idiom)
    python -m pytest tools/stale_work_watchdog_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stale_work_watchdog as wd  # noqa: E402

NOW = 1_000_000_000.0  # a fixed, injected clock


def _mkfile(p: Path, age_days: float, body: str = "{}") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    mtime = NOW - age_days * 86400.0
    os.utime(p, (mtime, mtime))
    return p


def _fixture_repo(tmp: Path) -> Path:
    repo = tmp / "fak"
    # over-age ephemera (should be GC candidates at max_age_days=7)
    _mkfile(repo / ".dos" / "markers" / "old1.jsonl", 10)
    _mkfile(repo / ".dos" / "streams" / "old2.jsonl", 8)
    _mkfile(repo / ".dos" / "stop-failures" / "old3.json", 30, '{"consecutive":1,"total":1}')
    _mkfile(repo / "tools" / "_watchdog" / "old4.log", 9)
    # fresh ephemera (must be kept)
    _mkfile(repo / ".dos" / "markers" / "fresh1.jsonl", 1)
    _mkfile(repo / "tools" / "_watchdog" / "fresh2.log", 0.5)
    return repo


def test_age_scan_picks_only_overage():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        cands = wd.scan_ephemera(repo, NOW, max_age_days=7)
        rels = {c["rel"] for c in cands}
        assert ".dos/markers/old1.jsonl" in rels
        assert ".dos/streams/old2.jsonl" in rels
        assert ".dos/stop-failures/old3.json" in rels
        assert "tools/_watchdog/old4.log" in rels
        assert ".dos/markers/fresh1.jsonl" not in rels
        assert "tools/_watchdog/fresh2.log" not in rels
        # sorted worst (oldest) first
        assert cands[0]["age_days"] >= cands[-1]["age_days"]


def test_dry_run_deletes_nothing():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        cands = wd.scan_ephemera(repo, NOW, 7)
        res = wd.sweep(repo, cands, live=False)
        assert res["files"] == len(cands)
        # nothing actually removed
        assert (repo / ".dos" / "markers" / "old1.jsonl").exists()


def test_live_sweep_deletes_only_overage():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        cands = wd.scan_ephemera(repo, NOW, 7)
        res = wd.sweep(repo, cands, live=True)
        assert res["files"] == len(cands)
        assert not (repo / ".dos" / "markers" / "old1.jsonl").exists()
        assert not (repo / "tools" / "_watchdog" / "old4.log").exists()
        # fresh ones survive
        assert (repo / ".dos" / "markers" / "fresh1.jsonl").exists()
        assert (repo / "tools" / "_watchdog" / "fresh2.log").exists()


def test_sweep_refuses_outside_ephemeral_dirs():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        victim = _mkfile(repo / "internal" / "kernel" / "kernel.go", 99, "package kernel")
        forged = [{"path": str(victim), "rel": "internal/kernel/kernel.go",
                   "label": "markers", "age_days": 99.0, "size": victim.stat().st_size}]
        res = wd.sweep(repo, forged, live=True)
        assert res["files"] == 0, "must refuse to delete outside the ephemeral dirs"
        assert victim.exists(), "a tracked source file must never be touched"


def test_stuck_detection():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "fak"
        _mkfile(repo / ".dos" / "stop-failures" / "a.json", 1, '{"consecutive":5,"total":9}')
        _mkfile(repo / ".dos" / "stop-failures" / "b.json", 1, '{"consecutive":1,"total":1}')
        _mkfile(repo / ".dos" / "stop-failures" / "c.json", 1, '{"consecutive":3,"total":3}')
        stuck = wd.scan_stuck(repo, threshold=3)
        sessions = {s["session"]: s["consecutive"] for s in stuck}
        assert sessions == {"a": 5, "c": 3}
        assert stuck[0]["session"] == "a"  # worst first


def test_wip_scan_with_injected_porcelain():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "fak"
        old = _mkfile(repo / "Makefile", 3, "all:")          # 72h old -> stale
        _mkfile(repo / "INDEX.md", 0.1, "# idx")             # fresh
        lines = [" M Makefile", " M INDEX.md", "?? new.txt"]
        wipinfo = wd.scan_wip(repo, NOW, wip_stale_hours=24, porcelain=lambda _r: lines)
        assert wipinfo["count"] == 3
        assert wipinfo["stale"] is True
        assert wipinfo["oldest_path"] == "Makefile"
        assert wipinfo["oldest_age_hours"] >= 71.0
        _ = old


def test_wip_not_stale_when_recent():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "fak"
        _mkfile(repo / "INDEX.md", 0.1, "# idx")
        wipinfo = wd.scan_wip(repo, NOW, 24, porcelain=lambda _r: [" M INDEX.md"])
        assert wipinfo["stale"] is False


def test_report_aggregation_and_has_stale():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        rep = wd.build_report(repo, NOW, 7, 3, 24, live=False, porcelain=lambda _r: [])
        assert rep.has_stale is True
        assert rep.age_files == 4
        assert rep.age_bytes > 0
        # dry-run leaves files in place
        assert (repo / ".dos" / "streams" / "old2.jsonl").exists()


def test_main_fail_on_stale_exit_code():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        rc = wd.main(["--repo", str(repo), "--json", "--fail-on-stale"])
        assert rc == 2, "stale work + --fail-on-stale must exit 2"


def test_main_clean_repo_exit_zero():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "fak"
        (repo / ".dos" / "markers").mkdir(parents=True)
        rc = wd.main(["--repo", str(repo), "--json", "--fail-on-stale"])
        assert rc == 0, "a clean repo must exit 0 even with --fail-on-stale"


def test_render_human_smoke():
    with tempfile.TemporaryDirectory() as d:
        repo = _fixture_repo(Path(d))
        rep = wd.build_report(repo, NOW, 7, 3, 24, live=False, porcelain=lambda _r: [])
        text = wd.render_human(rep, live=False)
        assert "stale-work watchdog" in text
        assert "AGE-GC" in text
        # JSON path stays parseable
        json.loads(wd.render_json(rep, live=False))


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
