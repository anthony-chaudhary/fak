#!/usr/bin/env python3
"""Tier-2 launcher for the GitHub-issue *gardener* — the model-driven pass.

The backlog gardener is two layers, mirroring the supervisor/dispatch split:

* a **deterministic spine** — ``tools/issue_triage.py --status`` — rides the
  existing ``FleetControlPaneTick`` heartbeat as the tracked ``issue-gardener``
  control-pane loop. It is read-only and needs no model; it only decides *when*
  the backlog has drifted past a gardening threshold (ACTION).
* a **model-driven pass** — THIS launcher — does the *judgment* the spine
  cannot: assign priority/area/kind to unlabeled issues, propose **expansion**
  (split an oversized ``needs-split`` issue into focused sub-issues) and
  **contraction** (confirm duplicate/stale/dormant clusters for merge/close).

Per the operator policy, the gardener runs on a **tier-2 model by default**
(Sonnet — the mid-tier Claude), not the tier-1 model the interactive fleet
uses: gardening is a high-volume, low-stakes cadence task where a cheaper model
is the right default. Model selection precedence (highest first):

1. ``--model`` CLI flag (an explicit model id/alias, e.g. ``opus``/``sonnet``).
2. ``--tier {1,2,3}`` → ``opus`` / ``sonnet`` / ``haiku``.
3. ``FLEET_GARDENER_MODEL`` env var.
4. ``fak route`` over the gardening routing preset (opt-in via
   ``FLEET_GARDENER_ROUTE``; fail-safe — a routing failure falls through to 5).
   This rung makes the gardener the first DEPLOYED consumer of fak's
   model-routing spine, dogfooding routing on the lowest-risk workload.
5. ``sonnet`` (tier 2, the default).

Like the supervisor/watchdog/canary layer (and UNLIKE the leaf
``dispatch_worker.py``), this launcher is **dry-run by default** — it prints the
``claude`` command it *would* run. Pass ``--live`` to actually launch it. The
gardening prompt is **propose-only**: it writes a dated proposal under
``docs/_audits/`` and NEVER edits, labels, comments on, or closes an issue.
Applying the proposed labels/closes stays the operator-gated ``/issue-triage``
step. This is the same read-only-decides / operator-acts discipline the helper
and skill already enforce.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-issue-gardener-worker/1"

# Tier -> Claude model alias. Tier 2 (Sonnet) is the gardener default.
TIER_MODELS = {"1": "opus", "2": "sonnet", "3": "haiku"}
MODEL_TIERS = {v: k for k, v in TIER_MODELS.items()}
DEFAULT_TIER = "2"
DEFAULT_MODEL = TIER_MODELS[DEFAULT_TIER]  # "sonnet"

# The model-routing bridge (opt-in via FLEET_GARDENER_ROUTE). The gardening pass
# is the lowest-risk first workload to dogfood fak's routing spine: high-volume,
# low-stakes, batch-latency, propose-only. When enabled, resolve_model asks
# `fak route` to classify the gardening subject against the committed gardening
# routing preset and maps the chosen ABSTRACT id (small/medium/large) onto the
# Claude alias ladder below. The whole bridge is FAIL-SAFE: any failure (no fak
# binary, a nonzero exit, unparseable JSON, an unknown id) falls straight through
# to the existing tier ladder, so a routing hiccup never blocks a gardening tick.
ROUTE_ENABLE_ENV = "FLEET_GARDENER_ROUTE"
# The routing manifest the gardening subject is classified against (overridable).
ROUTE_MANIFEST_ENV = "FLEET_GARDENER_ROUTE_MANIFEST"
DEFAULT_ROUTE_MANIFEST = "examples/routing-presets/gardening.json"
# Abstract routed id -> Claude alias. The gardener launches a `claude -p` backend,
# so a routed id is mapped onto the same alias ladder the tier path uses.
ROUTED_ID_MODELS = {"small": "haiku", "medium": "sonnet", "large": "opus"}

BACKENDS = ("claude",)
DEFAULT_BACKEND = "claude"

# Default permission posture: read + propose, no execution of writes. The
# gardener is autonomous and on a cadence, so the safe default is plan mode;
# the operator opts into a writing mode explicitly.
DEFAULT_PERMISSION_MODE = "plan"

# Default wall-clock cap on the spawned gardener session (seconds). The gardener
# runs UNATTENDED on a cadence, so an unbounded run (the old default=None) let a
# wedged session burn tokens unbounded. Plan-mode triage is light, so 30 min is
# generous; opt out with `--timeout-s 0` (normalized to None below).
DEFAULT_TIMEOUT_S = 1800


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


RouteRunner = Callable[[Sequence[str]], dict[str, Any]]


def _truthy(value: str | None) -> bool:
    """A permissive on-switch: anything but the explicit off tokens is ON."""
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "off", "false", "no", "disable", "disabled"}


def _default_route_runner(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    """Run `fak route` and capture its stdout (the real, non-test runner)."""
    exe = resolve_exe(command[0]) if command else ""
    argv = [exe, *command[1:]]
    try:
        proc = subprocess.run(argv, cwd=str(cwd) if cwd else None,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"returncode": 127, "error": str(exc)}
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def route_model(
    env: dict[str, str] | None,
    *,
    workspace: Path | None = None,
    runner: RouteRunner | None = None,
) -> str | None:
    """Ask `fak route` which model the gardening subject routes to, or None.

    FAIL-SAFE by construction: returns None on ANY problem (routing disabled, no
    fak binary, a nonzero exit, unparseable JSON, an unknown routed id) so the
    caller falls through to the tier ladder. Only returns a Claude alias when the
    route cleanly resolves to a known abstract id (small/medium/large).
    """
    e = env if env is not None else os.environ
    if not _truthy(e.get(ROUTE_ENABLE_ENV)):
        return None
    manifest = (e.get(ROUTE_MANIFEST_ENV) or DEFAULT_ROUTE_MANIFEST).strip()
    fak = (e.get("FAK_BIN") or "fak").strip()
    command = [
        fak, "route",
        "--manifest", manifest,
        "--aspect", "request",
        "--latency", "batch",
        "--complexity", "low",
        "--labels", "work_kind=gardening,stakes=low",
        "--json",
    ]
    try:
        if runner is not None:
            result = runner(command)
        else:
            result = _default_route_runner(command, cwd=workspace)
    except Exception:  # a custom runner blowing up must not break a gardening tick
        return None
    if not isinstance(result, dict) or result.get("returncode") != 0:
        return None
    try:
        decision = json.loads(result.get("stdout") or "")
        routed_id = str(decision.get("primary") or "").strip()
    except (ValueError, AttributeError):
        return None
    return ROUTED_ID_MODELS.get(routed_id)


def resolve_model(
    explicit: str | None,
    tier: str | None,
    env: dict[str, str] | None,
    *,
    workspace: Path | None = None,
    route_runner: RouteRunner | None = None,
) -> str:
    """Pick the gardener model. Precedence: --model > --tier > env > route > sonnet.

    The route rung (opt-in via FLEET_GARDENER_ROUTE) makes the gardener the first
    deployed consumer of fak's model-routing spine; it is fail-safe, so a routing
    failure leaves the historical --model > --tier > env > sonnet behavior intact.
    """
    if explicit:
        return explicit.strip()
    if tier:
        t = str(tier).strip()
        if t not in TIER_MODELS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIER_MODELS)}")
        return TIER_MODELS[t]
    env_model = (env if env is not None else os.environ).get("FLEET_GARDENER_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    routed = route_model(env, workspace=workspace, runner=route_runner)
    if routed:
        return routed
    return DEFAULT_MODEL


def model_tier(model: str) -> str:
    """Best-effort reverse lookup (a bare model id maps to '?')."""
    return MODEL_TIERS.get(model, "?")


def build_prompt(*, as_of: str, scope: str | None, apply_mechanical: bool) -> str:
    """The gardening instruction handed to ``claude -p``.

    Concrete and propose-only: it runs the read-only helper, then asks the model
    for the three judgment moves (priorities/tags, expansion, contraction) and
    to write the enriched proposal to a dated audit file. The model is told NOT
    to write to GitHub; applying is the operator-gated /issue-triage step.
    """
    scope_clause = f" Limit to the `{scope}` bucket." if scope else ""
    mech = (
        " You MAY additionally run the calendar-defensible mechanical batch "
        "(mark-stale / close-dormant-question) from the actions manifest, one "
        "batch at a time."
        if apply_mechanical
        else " Do NOT edit, label, comment on, or close any issue - propose only."
    )
    return (
        "Garden the open GitHub issue backlog (tier-2 pass).{scope}\n"
        "1. Run `python tools/issue_triage.py --detect-split --markdown "
        "--out docs/_audits/issue-triage-{date}.md` and "
        "`python tools/issue_triage.py --detect-split --actions "
        "--out docs/_audits/issue-actions-{date}.json` for the ranked classification.\n"
        "2. TRIAGE / priorities / tags: for each needs-priority / needs-kind / "
        "needs-area issue, propose a concrete label from fleet's taxonomy.\n"
        "3. EXPANSION: for each `needs-split` (oversized) issue, propose a "
        "concrete breakdown into focused sub-issues.\n"
        "4. CONTRACTION: for each likely-dup cluster and each stale / "
        "dormant-question row, propose the merge/close with the issue's own evidence.\n"
        "5. Write the enriched gardening proposal to "
        "docs/_audits/issue-gardener-{date}.md.\n"
        "{mech}"
    ).format(scope=scope_clause, date=as_of, mech=mech)


def build_command(
    model: str,
    *,
    as_of: str,
    scope: str | None = None,
    apply_mechanical: bool = False,
    permission_mode: str = DEFAULT_PERMISSION_MODE,
    backend: str = DEFAULT_BACKEND,
) -> list[str]:
    """Pure: the logical argv for one gardener launch (no path resolution)."""
    if backend != "claude":
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if not model:
        raise ValueError("model must be a non-empty string")
    prompt = build_prompt(as_of=as_of, scope=scope, apply_mechanical=apply_mechanical)
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--permission-mode",
        permission_mode,
        prompt,
    ]


def child_env(
    model: str,
    workspace: Path,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """The env the gardener child runs under (self-describing, like dispatch)."""
    env = dict(base if base is not None else os.environ)
    env["DISPATCH_WORKSPACE"] = str(workspace)
    env["GARDENER_MODEL"] = model
    env["GARDENER_TIER"] = model_tier(model)
    return env


def resolve_exe(name: str) -> str:
    found = shutil.which(name)
    return found or name


def normalize_timeout(value: int | None) -> int | None:
    """Map a CLI ``--timeout-s`` value to the launch timeout: a positive value is
    the wall-clock cap; ``0``/negative/``None`` is the explicit unbounded opt-out."""
    if value and value > 0:
        return value
    return None


Runner = Callable[[Sequence[str], Path, dict[str, str]], dict[str, Any]]


def launch(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    *,
    runner: Runner | None = None,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Exec the gardener command. ``runner`` is injectable for hermetic tests."""
    if runner is not None:
        return runner(command, cwd, env)
    resolved = list(command)
    if resolved:
        resolved[0] = resolve_exe(resolved[0])
    try:
        proc = subprocess.run(resolved, cwd=cwd, env=env, timeout=timeout_s)
    except FileNotFoundError as exc:
        return {"returncode": 127, "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "timeout": True}
    return {"returncode": proc.returncode}


def build_payload(
    *,
    model: str,
    backend: str,
    workspace: Path,
    dry_run: bool,
    as_of: str,
    scope: str | None,
    apply_mechanical: bool,
    permission_mode: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    command = (
        build_command(
            model,
            as_of=as_of,
            scope=scope,
            apply_mechanical=apply_mechanical,
            permission_mode=permission_mode,
            backend=backend,
        )
        if not error
        else []
    )
    ok = error is None and (result is None or result.get("returncode") == 0)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "backend": backend,
        "model": model,
        "tier": model_tier(model),
        "workspace": str(workspace),
        "dry_run": dry_run,
        "propose_only": not apply_mechanical,
        "permission_mode": permission_mode,
        "scope": scope,
        "as_of": as_of,
        "command": command,
        "env": {
            "DISPATCH_WORKSPACE": str(workspace),
            "GARDENER_MODEL": model,
            "GARDENER_TIER": model_tier(model),
        },
        "result": result,
        "error": error,
    }


def render(payload: dict[str, Any]) -> str:
    cmd = " ".join(payload.get("command") or []) or "-"
    lines = [
        f"issue-gardener: model={payload.get('model')} (tier {payload.get('tier')}) "
        f"backend={payload.get('backend')} dry_run={payload.get('dry_run')} "
        f"propose_only={payload.get('propose_only')}",
        f"command: {cmd}",
    ]
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    result = payload.get("result")
    if isinstance(result, dict):
        lines.append(f"returncode: {result.get('returncode')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Launch the tier-2 GitHub-issue gardener (dry-run by default)."
    )
    ap.add_argument("--model", default=None,
                    help="explicit model id/alias (overrides --tier/env/default)")
    ap.add_argument("--tier", default=None, choices=sorted(TIER_MODELS),
                    help="model tier: 1=opus, 2=sonnet (default), 3=haiku")
    ap.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                    help="worker backend (default: claude)")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--scope", default=None,
                    help="limit to one bucket: priority|kind|area|orphans|stale|dup|question")
    ap.add_argument("--permission-mode", default=DEFAULT_PERMISSION_MODE,
                    help="claude permission mode (default: plan — read + propose, no writes)")
    ap.add_argument("--apply-mechanical", action="store_true",
                    help="also allow the calendar-defensible mechanical batch "
                         "(mark-stale / close-dormant); priorities/splits stay proposal-only")
    ap.add_argument("--as-of", default=None, help="date stamp (default: today UTC)")
    ap.add_argument("--live", action="store_true",
                    help="actually launch (default: dry-run; print the command)")
    ap.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S,
                    help=f"child wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT_S}; "
                         "use 0 for unbounded)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    as_of = args.as_of or dt.datetime.now(dt.timezone.utc).date().isoformat()

    error: str | None = None
    model = DEFAULT_MODEL
    try:
        model = resolve_model(args.model, args.tier, None, workspace=workspace)
    except ValueError as exc:
        error = str(exc)

    dry_run = not args.live or bool(error)

    if dry_run:
        payload = build_payload(
            model=model, backend=args.backend, workspace=workspace, dry_run=True,
            as_of=as_of, scope=args.scope, apply_mechanical=args.apply_mechanical,
            permission_mode=args.permission_mode, error=error,
        )
        print(json.dumps(payload, indent=2) if args.json else render(payload))
        return 0 if not error else 2

    command = build_command(
        model, as_of=as_of, scope=args.scope, apply_mechanical=args.apply_mechanical,
        permission_mode=args.permission_mode, backend=args.backend,
    )
    env = child_env(model, workspace)
    result = launch(command, workspace, env, timeout_s=normalize_timeout(args.timeout_s))
    payload = build_payload(
        model=model, backend=args.backend, workspace=workspace, dry_run=False,
        as_of=as_of, scope=args.scope, apply_mechanical=args.apply_mechanical,
        permission_mode=args.permission_mode, result=result,
    )
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return int(result.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
