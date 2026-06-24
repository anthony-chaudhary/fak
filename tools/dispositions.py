#!/usr/bin/env python3
"""Build a dispatch-dispositions sidecar from real dos verbs.

No "dos next-up" verb exists. This shim drives the three real verbs in
sequence so a dispatch loop can make a typed routing decision:

  dos enumerate <PLAN_DOC> --json           -> declared unit list
  dos pickable <UNIT> --state <JSON> --json -> OFFERABLE / HELD per unit
  dos gate <sidecar> --json                 -> typed verdict

The sidecar (``.dispositions-<tag>.json``) is a JSON list; each row is::

  {"phase": <unit-id>, "live": <bool>, "drop_reason": "<reason or ''>"}

``dos gate`` classifies the sidecar into one of: LIVE / DRAIN /
STALE-STAMP / BLOCKED / RACE.

All subprocess calls accept a ``runner`` injectable so tests never need a
live dos installation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "fleet-dispatch-dispositions/1"
SIDECAR_PREFIX = ".dispositions-"

Runner = Callable[[Sequence[str]], dict[str, Any]]


def _run(argv: Sequence[str], runner: Runner | None) -> dict[str, Any]:
    """Execute argv, returning {returncode, stdout, stderr}."""
    if runner is not None:
        return runner(list(argv))
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}


def enumerate_units(
    plan_doc: Path,
    *,
    workspace: Path | None = None,
    runner: Runner | None = None,
) -> list[str]:
    """Run ``dos enumerate <PLAN_DOC> --json`` and return the declared unit list.

    Exit 3 (DriftNote) is accepted — the doc disagrees with itself but still
    yields a unit list.  Exit 4 (empty) returns an empty list.
    """
    argv: list[str] = ["dos", "enumerate", str(plan_doc), "--json"]
    if workspace:
        argv += ["--workspace", str(workspace)]
    result = _run(argv, runner)
    rc = result["returncode"]
    if rc not in (0, 3, 4):
        raise RuntimeError(
            f"dos enumerate failed (rc={rc}): {result['stderr'].strip()}"
        )
    raw = (result["stdout"] or "").strip()
    data: dict[str, Any] = json.loads(raw) if raw else {}
    return list(data.get("units") or [])


def pickable_row(
    unit: str,
    state: dict[str, Any],
    *,
    workspace: Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run ``dos pickable <UNIT> --state <JSON> --json`` for one unit.

    Returns the parsed JSON (keys: held, reason, evidence, …).
    Exit 0 = OFFERABLE; exits 10–25 = typed HELD — both are valid responses.
    """
    argv: list[str] = ["dos", "pickable", unit, "--state", json.dumps(state), "--json"]
    if workspace:
        argv += ["--workspace", str(workspace)]
    result = _run(argv, runner)
    rc = result["returncode"]
    valid = {0, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25}
    if rc not in valid:
        raise RuntimeError(
            f"dos pickable {unit!r} failed (rc={rc}): {result['stderr'].strip()}"
        )
    return json.loads(result["stdout"])


def build_dispositions(
    units: list[str],
    state_map: dict[str, dict[str, Any]],
    *,
    workspace: Path | None = None,
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    """Build a disposition row for each unit.

    Each row: ``{"phase": unit, "live": bool, "drop_reason": "..."}``.
    ``state_map`` maps unit id -> state dict for ``dos pickable --state``;
    missing entries default to ``{}``.
    """
    rows: list[dict[str, Any]] = []
    for unit in units:
        state = state_map.get(unit, {})
        pick = pickable_row(unit, state, workspace=workspace, runner=runner)
        held: bool = bool(pick.get("held"))
        drop_reason: str = pick.get("reason") or ""
        rows.append({"phase": unit, "live": not held, "drop_reason": drop_reason})
    return rows


def write_sidecar(dispositions: list[dict[str, Any]], path: Path) -> None:
    """Atomically write the dispositions list to a sidecar file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dispositions, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def classify(
    sidecar_path: Path,
    *,
    workspace: Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run ``dos gate <sidecar> --json`` and return the parsed verdict dict.

    Gate exit codes: 0=LIVE, 3=DRAIN, 4=STALE-STAMP, 5=BLOCKED, 6=RACE.
    Exit 2 is a contract error and always raises.
    """
    argv: list[str] = ["dos", "gate", str(sidecar_path), "--json"]
    if workspace:
        argv += ["--workspace", str(workspace)]
    result = _run(argv, runner)
    rc = result["returncode"]
    if rc == 2:
        raise RuntimeError(
            f"dos gate contract error (rc=2): {result['stderr'].strip()}"
        )
    if rc == 127:
        raise RuntimeError(
            f"dos gate not found (rc=127): {result['stderr'].strip()}"
        )
    return json.loads(result["stdout"])


def build_payload(
    *,
    plan_doc: Path,
    tag: str,
    dispositions: list[dict[str, Any]],
    sidecar_path: Path,
    gate: dict[str, Any],
) -> dict[str, Any]:
    verdict = gate.get("verdict") or ""
    return {
        "schema": SCHEMA,
        "plan_doc": str(plan_doc),
        "tag": tag,
        "sidecar_path": str(sidecar_path),
        "dispositions": dispositions,
        "gate": gate,
        "verdict": verdict,
        "ok": verdict == "LIVE",
    }


def render(payload: dict[str, Any]) -> str:
    gate = payload.get("gate") or {}
    rows = payload.get("dispositions") or []
    live_count = sum(1 for r in rows if r.get("live"))
    held_count = len(rows) - live_count
    lines = [
        f"dispositions: plan={payload.get('plan_doc')} tag={payload.get('tag')}",
        f"units: {len(rows)} live={live_count} held={held_count}",
        f"verdict: {payload.get('verdict')} ({gate.get('reason', '')})",
        f"sidecar: {payload.get('sidecar_path')}",
    ]
    return "\n".join(lines)


def sidecar_path(tag: str, directory: Path) -> Path:
    return directory / f"{SIDECAR_PREFIX}{tag}.json"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a dispatch-dispositions sidecar from real dos verbs."
    )
    ap.add_argument("plan_doc", help="path to the plan-doc to enumerate")
    ap.add_argument(
        "--tag",
        required=True,
        help="sidecar tag (written to .dispositions-<tag>.json)",
    )
    ap.add_argument(
        "--state",
        default="{}",
        help="JSON object mapping unit id -> state dict for dos pickable",
    )
    ap.add_argument(
        "--sidecar-dir",
        default="",
        help="directory to write the sidecar (default: plan_doc directory)",
    )
    ap.add_argument(
        "--workspace",
        default="",
        help="workspace root the substrate serves (default: repo root)",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    plan = Path(args.plan_doc).resolve()
    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    out_dir = Path(args.sidecar_dir).resolve() if args.sidecar_dir else plan.parent
    state_map: dict[str, Any] = json.loads(args.state)

    units = enumerate_units(plan, workspace=workspace)
    dispositions = build_dispositions(units, state_map, workspace=workspace)
    out_path = sidecar_path(args.tag, out_dir)
    write_sidecar(dispositions, out_path)
    gate = classify(out_path, workspace=workspace)
    payload = build_payload(
        plan_doc=plan,
        tag=args.tag,
        dispositions=dispositions,
        sidecar_path=out_path,
        gate=gate,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
