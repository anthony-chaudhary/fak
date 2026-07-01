#!/usr/bin/env python3
"""Dogfood-coverage scorecard: how much of the REAL dev workflow eats the kernel.

The fak thesis is that the kernel (`fak serve` / `fak guard`) belongs in front of
every tool call an agent makes. "Dogfooding" only counts when *our own* dev work —
the dispatch fleet, the interactive Claude Code session — actually crosses that
boundary. This scorecard measures it, and (like every other fak scorecard) it
cross-checks REALITY, not config: it imports `dispatch_worker` and calls the live
`guarded_launch_command` on THIS host, and it counts decision rows in the durable
audit journals — so the number cannot be gamed by editing a flag.

KPIs fold into one `coverage` percent + a `dogfood_debt` integer (count of unmet
HARD affordances) + an A–F grade, and emit a control-pane JSON payload. Drive it on
a /loop cadence to keep the dev loop kernel-adjudicated.

    python tools/dogfood_coverage.py            # human report
    python tools/dogfood_coverage.py --json      # control-pane payload
    python tools/dogfood_coverage.py --check      # exit 1 if any HARD KPI is unmet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _load_module(name: str, path: Path) -> Any:
    """Import a sibling tool by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if not (spec and spec.loader):
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _grep(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def count_audit_rows(root: Path) -> tuple[int, int]:
    """Count decision rows the kernel recorded across the fleet's per-lane decision
    journals (the WITNESS that real dogfood happened, not just got configured).

    Returns (rows, journals). Each non-empty JSONL line is one kernel verdict. The
    per-lane journals live under the gitignored .dispatch-runs/guard-audit/; the
    interactive `fak guard` default writes one under the user config dir, which we
    add when present.
    """
    rows = 0
    journals = 0
    candidates: list[Path] = []
    fleet_dir = root / ".dispatch-runs" / "guard-audit"
    if fleet_dir.is_dir():
        candidates.extend(sorted(fleet_dir.glob("*.jsonl")))
    # The interactive front door's per-user default journal is host-global, so
    # include it only for real repo roots. Temp fixture roots should count only
    # the journals they create, otherwise a developer's live audit history makes
    # hermetic tests nondeterministic.
    if (root / ".git").exists():
        cfg = os.environ.get("XDG_CONFIG_HOME") or os.environ.get("APPDATA")
        if not cfg:
            home = os.environ.get("HOME") or os.path.expanduser("~")
            cfg = str(Path(home) / ".config")
        user_journal = Path(cfg) / "fak" / "guard-audit.jsonl"
        if user_journal.is_file():
            candidates.append(user_journal)
    for jp in candidates:
        try:
            text = jp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n = sum(1 for ln in text.splitlines() if ln.strip())
        if n:
            rows += n
            journals += 1
    return rows, journals


def diagnose_audit_gap(root: Path) -> str:
    """Explain WHY the audit journal is empty, so a zero row count is self-diagnosing
    instead of an undifferentiated 0 (the default-observability point: a blank witness
    must say which blank it is). Returns "" when rows exist (no gap to explain).

    The three blanks a `0` can hide are distinct actions:
      * no journal DIR at all      -> no guarded worker has ever run here
      * journal dir, only empty/blank files -> guard booted but the wrapped agent
        proposed no tool call the kernel adjudicated (e.g. it exited before its first
        tool use — the silent auth-failure / empty-turn case)
      * dir present, files present, rows>0 -> not a gap (handled by the caller)
    """
    fleet_dir = root / ".dispatch-runs" / "guard-audit"
    if not fleet_dir.is_dir():
        return ("no guard-audit journal directory yet — no guarded worker has run on "
                "this host (arm `fak guard -- <agent>` so the kernel records verdicts)")
    jsonls = sorted(fleet_dir.glob("*.jsonl"))
    if not jsonls:
        return ("guard-audit directory exists but holds no journal files — the guard "
                "wire is configured but never exercised by a launched worker")
    # Dir + files present but every file is blank: the worker booted under guard but
    # never reached a tool call the kernel adjudicated (the silent empty-turn signature
    # — e.g. the wrapped agent exited on an auth/login failure before its first tool use).
    return (f"{len(jsonls)} journal file(s) present but all blank — a guarded worker "
            "booted but proposed no adjudicated tool call (check the agent reached a "
            "tool use; an auth/login failure exits before the first verdict)")


def evaluate(root: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Compute the dogfood-coverage KPIs. `env` defaults to the process env so the
    LIVE opt-out (FLEET_DOGFOOD_GUARD=0) shows up in the score."""
    e = env if env is not None else dict(os.environ)

    kpis: list[dict[str, Any]] = []

    def add(key: str, ok: bool, hard: bool, detail: str, evidence: str = "") -> None:
        kpis.append({"key": key, "ok": bool(ok), "hard": hard,
                     "detail": detail, "evidence": evidence})

    # KPIs 1-3 cross-check the LIVE dispatch_worker surface (a real call, not a grep).
    # If that module moves or a symbol it calls gets renamed, this must degrade the
    # three dependent KPIs to an honest unmet-with-reason -- not raise ImportError/
    # AttributeError/OSError and crash the whole scorecard (#1941).
    try:
        dw = _load_module("dispatch_worker", root / "tools" / "dispatch_worker.py")
        raw = dw.build_command("probe", "claude")
        launch_cmd, guarded = dw.guarded_launch_command(raw, "probe", "claude", root, env=e)
        fak_bin = dw.resolve_fak_bin(root, e)
        live_on = dw.guard_enabled(e)
    except (ImportError, AttributeError, OSError) as exc:
        reason = f"dispatch surface moved: {exc}"
        add("fleet_leaf_guarded", False, True,
            "dispatch_worker.guarded_launch_command fronts a claude worker with `fak guard`",
            evidence=reason)
        add("fak_bin_resolvable", False, True,
            "a `fak` binary resolves (FAK_BIN / tools/.bin / PATH) so guard-wrapping engages",
            evidence=reason)
        add("guard_default_on", False, True,
            "FLEET_DOGFOOD_GUARD is not disabled in the live environment (default ON)",
            evidence=reason)
    else:
        # 1. The supervised leaf launcher fronts a claude worker with the kernel, on
        #    THIS host, RIGHT NOW. This is the real behavior cross-check — not a grep.
        add("fleet_leaf_guarded", guarded, True,
            "dispatch_worker.guarded_launch_command fronts a claude worker with `fak guard`",
            evidence=" ".join(launch_cmd[:3]) if guarded else "UNWRAPPED (coverage=0 on this host)")

        # 2. A `fak` binary resolves, so the fail-open path is not silently dropping the
        #    fleet to 0% coverage.
        add("fak_bin_resolvable", bool(fak_bin), True,
            "a `fak` binary resolves (FAK_BIN / tools/.bin / PATH) so guard-wrapping engages",
            evidence=fak_bin or "no fak binary found — workers run UNGUARDED")

        # 3. Dogfood mode is ON in the live env (default-on, not opted out here).
        add("guard_default_on", live_on, True,
            "FLEET_DOGFOOD_GUARD is not disabled in the live environment (default ON)",
            evidence=f"FLEET_DOGFOOD_GUARD={e.get('FLEET_DOGFOOD_GUARD', '<unset=ON>')}")

    # 4. The scheduled-task lane (issue_dispatch) is wired to the same guard path.
    wired = _grep(root / "tools" / "issue_dispatch.py", "guarded_launch_command")
    add("issue_dispatch_wired", wired, True,
        "issue_dispatch.evaluate routes its detached spawn through guarded_launch_command",
        evidence="tools/issue_dispatch.py calls guarded_launch_command" if wired else "MISSING")

    # 5. The interactive front door exists as a productized verb.
    guard_go = (root / "cmd" / "fak" / "guard.go").is_file()
    add("guard_verb_present", guard_go, True,
        "`fak guard -- claude` exists (cmd/fak/guard.go) as the one-command kernel front door",
        evidence="cmd/fak/guard.go" if guard_go else "MISSING")

    # 6. The front door is documented for a human/agent to find.
    documented = _grep(root / "DOGFOOD-CLAUDE.md", "fak guard")
    add("guard_documented", documented, False,
        "DOGFOOD-CLAUDE.md documents the `fak guard` front door", evidence="DOGFOOD-CLAUDE.md")

    # 7. Durable witness: kernel decisions actually recorded by guarded workers. Soft
    #    because a freshly-configured host has 0 until the fleet runs — but it is the
    #    proof the wire is exercised, not merely wired.
    rows, journals = count_audit_rows(root)
    audit_evidence = f"{rows} decision row(s) across {journals} journal(s)"
    if rows == 0:
        # Make the blank witness self-diagnosing: say WHICH zero this is so the
        # operator gets the unblock action, not an undifferentiated 0.
        audit_evidence += f" — {diagnose_audit_gap(root)}"
    add("audit_journal_evidence", rows > 0, False,
        "guarded workers have recorded kernel decisions in a durable audit journal",
        evidence=audit_evidence)

    # 8/9. Always-on compute so the dogfood loop runs 24/7 (the goal's other half).
    doc = (root / "docs" / "fak" / "always-on-dogfood-server.md").is_file()
    add("always_on_server_doc", doc, False,
        "an always-on dogfood server design exists (Mac/GCP tiers)",
        evidence="docs/fak/always-on-dogfood-server.md" if doc else "MISSING")
    plist = (root / "tools" / "com.fak.serve-gateway.plist").is_file()
    add("always_on_serve_plist", plist, False,
        "a launchd unit keeps a shared `fak serve` gateway alive 24/7",
        evidence="tools/com.fak.serve-gateway.plist" if plist else "MISSING")

    total = len(kpis)
    met = sum(1 for k in kpis if k["ok"])
    hard_unmet = [k for k in kpis if k["hard"] and not k["ok"]]
    coverage = round(100.0 * met / total, 1) if total else 0.0
    dogfood_debt = len(hard_unmet)
    grade = _grade(coverage, dogfood_debt)

    return {
        "schema": "dogfood-coverage/1",
        "coverage": coverage,
        "met": met,
        "total": total,
        "dogfood_debt": dogfood_debt,
        "grade": grade,
        "audit_rows": rows,
        "kpis": kpis,
        "worst_first": [k["key"] for k in hard_unmet] + [k["key"] for k in kpis if not k["hard"] and not k["ok"]],
    }


def _grade(coverage: float, debt: int) -> str:
    if debt == 0 and coverage >= 95:
        return "A"
    if debt == 0:
        return "B"
    if debt <= 1:
        return "C"
    if debt <= 2:
        return "D"
    return "F"


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"dogfood-coverage: {payload['coverage']}% "
        f"({payload['met']}/{payload['total']} KPIs)  "
        f"grade {payload['grade']}  dogfood_debt {payload['dogfood_debt']}  "
        f"audit_rows {payload['audit_rows']}",
    ]
    for k in payload["kpis"]:
        mark = "OK " if k["ok"] else ("XX " if k["hard"] else ".. ")
        tag = "HARD" if k["hard"] else "soft"
        lines.append(f"  [{mark}] {k['key']:<24} ({tag})  {k['detail']}")
        if k.get("evidence"):
            lines.append(f"          -> {k['evidence']}")
    if payload["worst_first"]:
        lines.append("  next: " + ", ".join(payload["worst_first"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure how much of the dev workflow eats the kernel.")
    ap.add_argument("--json", action="store_true", help="emit the control-pane JSON payload")
    ap.add_argument("--check", action="store_true", help="exit 1 if any HARD KPI is unmet")
    ap.add_argument("--workspace", default="", help="repo root (default: auto)")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    if args.check and payload["dogfood_debt"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
