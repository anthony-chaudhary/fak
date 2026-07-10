#!/usr/bin/env python3
"""meta_superloop_night.py — the META super-loop ABOVE the night runs.

One operator intent: **progress ~200 productive issues overnight**, keeping every
capacity axis genuinely full — account session slots, the local host cap, and the
lab/mac/gpu node resources — while never crossing the honesty boundary the
`/super-loop` skill defines (a launch is not a ship; only a witnessed close counts).

It is a META loop: it does not resolve issues itself and does not re-implement the
launchers. Each TICK it WALKS the live capacity/closure surfaces first, then DRIVES
the proven, self-sizing primitives that already own account/host/seat accounting:

  orient   -> fak dispatch progress --json         (witnessed close-N odometer, #2639)
              tools/dispatch_preflight.py --json    (SPAWN_OK / REFUSE_* — the no-DoS gate)
  reclaim  -> tools/issue_dispatch.py --no-refresh  (rung-1: self-heal dead inflight markers)
  refill   -> fak dispatch auto --live --goal G     (self-sizes to Target-live across seats)
  nodes    -> fak nightrun run --apply --loop        (mac/gpu data-collection, feasible-only)
  witness  -> fak dispatch progress --json          (closed_by_loop_total delta => toward 200)

The cadence is tight (default 5m) so headroom is re-checked often, and a pure
skip-if-inflight gate (`wave_decision`) makes that safe: a tick whose preflight
shows no free slot — a prior wave still settling, or a REFUSE_AT_CAP — no-ops with a
logged reason instead of piling a second wave onto full slots. Fast iteration without
double-dispatch.

The target is measured on WITNESSED closes (`closed_by_loop_total`, which ignores
backlog drift), not on launches. The loop stops on the honest signals from the
super-loop marathon contract: target met, backlog drained, preflight refuses for a
seat/host reason that waiting (not spawning) fixes, or a hard wall-clock deadline.

DRY-RUN by default: it prints the plan for every tick and drives NOTHING. Only
--apply issues real `--live` waves. This is the witnessable artifact the operator
approves before the fleet grows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "docs" / "nightrun" / "meta-superloop-history.jsonl"

# The two goal-scoped drain intents the account-wave already knows (see
# `fak dispatch auto --goal`). Alternating keeps priority AND raw throughput fed.
# The Python wave launcher is goal-AGNOSTIC: it spreads across the busiest pairwise
# tree-disjoint lanes automatically (priced + arbitrated), so one wave call per tick
# feeds both priority and raw-throughput work without our alternating goals. The
# goal-scoped intents live on `fak dispatch auto` (agent-path; needs a green tree).
WAVE_MAX = 8  # per-tick cap; the preflight re-check still binds the live population


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], timeout: int = 240) -> tuple[int, str, str]:
    """Run a command in the repo; never raise — a failing fold must not kill the loop."""
    try:
        p = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 — orient must be crash-proof
        return 1, "", str(e)


def run_json(cmd: list[str], timeout: int = 240) -> dict | None:
    rc, out, _ = run(cmd, timeout=timeout)
    if rc != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


# ---- orient folds (pure reads) -------------------------------------------------

def preflight() -> dict:
    j = run_json([sys.executable, "tools/dispatch_preflight.py", "--json"]) or {}
    return {
        "verdict": j.get("verdict", "UNKNOWN"),
        "ok": bool(j.get("ok")),
        "headroom": j.get("headroom"),
        "live": j.get("live"),
        "reason": j.get("reason", ""),
    }


def progress() -> dict:
    """The witnessed close-N odometer (#2639) — the only honest progress signal."""
    j = run_json(["go", "run", "./cmd/fak", "dispatch", "progress", "--json"], timeout=300) or {}
    return {
        "closed_by_loop_total": j.get("closed_by_loop_total"),
        "baseline_open": j.get("baseline_open"),
        "current_issues_per_hour": j.get("current_issues_per_hour"),
    }


def cron_state() -> dict:
    """Is the standing FleetIssueDispatch cron live? If it is, the meta-loop must NOT
    hand-launch waves beside it (double-dispatch on the same slots/lanes is the named
    super-loop anti-pattern). The meta-loop's job then is to SUPERVISE the cron —
    verify it fires and ships, self-heal residue, feed the nodes it doesn't cover —
    and only take the wave itself when the cron has gone dark."""
    rc, out, _ = run([
        "powershell.exe", "-NoProfile", "-Command",
        "Get-ScheduledTask -TaskName 'FleetIssueDispatch' -ErrorAction SilentlyContinue "
        "| Select-Object -ExpandProperty State"
    ], timeout=60)
    state = (out or "").strip().splitlines()[-1].strip() if out.strip() else "ABSENT"
    return {"name": "FleetIssueDispatch", "state": state, "live": state in ("Running", "Ready")}


def nightrun_feasible() -> bool:
    """True when THIS box has a feasible next() data-collection datum."""
    j = run_json(["go", "run", "./cmd/fak", "nightrun", "next", "--json"], timeout=180)
    return bool(j and j.get("has_next"))


# ---- reclaim (rung-1: free, no kills) -----------------------------------------

def reclaim_rung1(apply: bool) -> str:
    """Any dry-run dispatch tick self-heals dead/expired inflight markers.

    This is the ONLY reclaim the meta-loop does unattended — rungs 2/3 (reads that
    inform kills, and the kills themselves) carry the same operator bar as -Launch
    and are never automated (super-loop skill, Step 0.5)."""
    if not apply:
        return "reclaim: (dry-run) would self-heal inflight markers via issue_dispatch --no-refresh"
    rc, _, err = run([sys.executable, "tools/issue_dispatch.py", "--no-refresh"], timeout=180)
    return f"reclaim: rung-1 self-heal rc={rc}{(' ' + err.strip()) if err.strip() else ''}"


# ---- build-tree health (a broken working tree caps the whole night) -----------

def tree_builds() -> tuple[bool, str]:
    """`go build ./cmd/fak` over the live working tree. A broken tree means every
    worker in this checkout that recompiles the agent path fails to launch — the
    single most important thing the meta-loop must surface before it drives anything.
    Prefer the Python launcher path when this is red; it does not recompile the
    agent package and so survives a mid-flight edit on the shared trunk.

    Compile to os.devnull, never into the checkout: a plain `go build ./cmd/fak`
    writes `fak.exe` into the cwd, and on Windows a LIVE fak process (i.e. any time
    the fleet is up) holds that file locked, so the build fails with a file-lock
    error that is NOT a compile error — a false BROKEN that mis-routes the refill
    onto the legacy path even on a green tree. Discarding the output tests
    compilation only."""
    rc, _, err = run(["go", "build", "-o", os.devnull, "./cmd/fak"], timeout=300)
    if rc == 0:
        return True, "go build ./cmd/fak: OK"
    first = (err.strip().splitlines() or ["build failed"])[0][:200]
    return False, f"go build ./cmd/fak: BROKEN — {first}"


# ---- tree-break diagnosis (classify the poison; recommend, never auto-mutate) --

# A red working tree is the single largest throughput cap: every worker in this
# checkout that runs a full-tree `go build`/`go test` trips a break that is not
# theirs. `tree_builds` already ROUTES the refill around it (prefers the Python wave
# path, which does not recompile the agent package); this rung goes one step further
# and CLASSIFIES the break so the operator knows which fix it needs.
#
# Why classify-and-SURFACE, never auto-quarantine: a single snapshot cannot prove a
# red file is not under a peer's ACTIVE edit. Witnessed 2026-07-09: a red untracked
# file (an in-flight #2601 change-set) flapped present->absent->present within ~2
# minutes as a concurrent worker did surgery on it — an unattended `mv`/`git checkout`
# heal would have corrupted that peer's operation. So the auto-mutate carries the same
# operator bar as a kill (super-loop skill, Step 0.5): the rung names the exact
# reversible action; a human runs it.

TREE_OK = "TREE_OK"
TREE_QUARANTINE_SAFE = "QUARANTINE_SAFE"   # untracked-only poison → operator can mv it aside (reversible)
TREE_COUPLED_TRACKED = "COUPLED_TRACKED"   # break lives in a MODIFIED/tracked file → its owner fixes/reverts
TREE_UNLOCALIZED = "UNLOCALIZED"           # red, but no single file:line to point at

# `go build` prints "pkg\file.go:line:col: msg"; grab the first such file path
# (Windows backslashes included — normalized to forward slashes below).
_BUILD_OFFENDER_RE = re.compile(r"([\w./\\-]+\.go):\d+:\d+:")


def parse_build_offender(full_err: str) -> str | None:
    """PURE: first `file.go:line:col:` path in a go build stderr, normalized to
    forward slashes. None when the failure has no file locus (link/toolchain error)."""
    m = _BUILD_OFFENDER_RE.search(full_err or "")
    return m.group(1).replace("\\", "/") if m else None


def classify_tree_break(offender: str | None, *, tracked: bool | None,
                        age_min: float | None) -> dict:
    """PURE: from (offending file, is-it-tracked, mtime-staleness) → a typed
    recommendation. Never mutates: the meta-loop prints/logs this and leaves the fix
    to the operator, because a snapshot can't rule out a peer's active edit."""
    if not offender:
        return {"code": TREE_UNLOCALIZED, "offender": None,
                "action": "no single-file locus in `go build ./...` — operator to inspect"}
    if tracked:
        return {"code": TREE_COUPLED_TRACKED, "offender": offender,
                "action": (f"{offender} is a TRACKED file — its owning worker must fix or revert "
                           f"its commit; do not edit beside them")}
    age = f"{age_min:.0f}m" if isinstance(age_min, (int, float)) else "?"
    return {"code": TREE_QUARANTINE_SAFE, "offender": offender,
            "action": (f"untracked {offender} (mtime {age}) poisons the build — operator: confirm no "
                       f"tracked file references its symbols, then `mv` it to a hold dir + breadcrumb (reversible)")}


def diagnose_tree(tree_ok: bool) -> dict:
    """IMPURE wrapper: runs ONLY when the tree is red (no extra build on green ticks).
    Localizes the offending file over the FULL tree (`./...`, so a break in any package
    is caught, not just ./cmd/fak), reads its git-tracked state + mtime, and returns the
    pure classification."""
    if tree_ok:
        return {"code": TREE_OK, "offender": None, "action": ""}
    _, _, err = run(["go", "build", "-o", os.devnull, "./..."], timeout=300)
    offender = parse_build_offender(err)
    tracked: bool | None = None
    age_min: float | None = None
    if offender:
        rc, out, _ = run(["git", "status", "--porcelain", "--", offender], timeout=60)
        # porcelain: "?? path" = untracked; " M path"/"M  path" = tracked+modified;
        # empty = tracked+clean. Untracked iff the first line starts with "??".
        lines = (out or "").strip().splitlines()
        tracked = (not (lines and lines[0].lstrip().startswith("??"))) if rc == 0 else None
        try:
            age_min = max(0.0, (time.time() - (REPO / offender).stat().st_mtime) / 60.0)
        except OSError:
            age_min = None
    return classify_tree_break(offender, tracked=tracked, age_min=age_min)


# ---- wave decision (pure, observable, unit-testable) --------------------------

# Decision codes the tick emits so the ledger reads WHY a wave did or didn't fire.
WAVE_SPAWN = "SPAWN"            # headroom exists → drive a refill this tick
WAVE_SKIP_INFLIGHT = "SKIP_INFLIGHT"   # prior wave still settling (no headroom) → no-op
WAVE_SKIP_REFUSED = "SKIP_REFUSED"     # preflight refused for a capacity/host reason
WAVE_SKIP_CRON = "SKIP_CRON"           # a live cron owns the waves → supervise, don't double-dispatch


def wave_decision(pf: dict, cron_owns_waves: bool) -> dict:
    """Decide whether THIS tick should drive a wave — the single skip-if-inflight gate.

    Pure: it reads only the preflight fold and the cron-ownership flag and returns a
    typed decision, so a 5-minute cadence cannot double-dispatch. At a tight cadence a
    tick can re-fire before the previous wave's workers have settled; when they have
    NOT (headroom<=0, or preflight is REFUSE_AT_CAP), we skip rather than pile a second
    wave onto slots that are already full. The decision carries an observable `reason`
    that lands in the ledger, so every skip is witnessable, not silent.

    Kept side-effect-free on purpose: the loop owns the I/O (spawning the refill); this
    only owns the yes/no and the why. That split is what makes it unit-testable.
    """
    if cron_owns_waves:
        return {"code": WAVE_SKIP_CRON,
                "reason": "FleetIssueDispatch cron live — supervising, not double-dispatching"}
    if not pf.get("ok"):
        return {"code": WAVE_SKIP_REFUSED,
                "reason": f"preflight {pf.get('verdict')} — {pf.get('reason', '')}".strip(" —")}
    headroom = pf.get("headroom")
    if isinstance(headroom, int) and headroom <= 0:
        return {"code": WAVE_SKIP_INFLIGHT,
                "reason": f"prior wave still settling — live={pf.get('live')} headroom={headroom} (no free slot)"}
    return {"code": WAVE_SPAWN,
            "reason": f"headroom={headroom} free — driving a refill"}


# ---- refill (drive the priced tree-disjoint wave; build-break-immune) ----------

def refill(apply: bool, tree_ok: bool) -> dict:
    """Refill the fleet to headroom via the Python tree-disjoint wave
    (`tools/issue_dispatch.py --wave`) — the super-loop skill's stated DEFAULT:
    collision-priced by `dos arbitrate`, preflight re-checked per spawn, spread across
    the busiest pairwise-disjoint lanes, and it does NOT recompile `internal/agent`,
    so it survives a mid-flight edit that reddens the shared tree. Dry-run reads the
    plan (verdict/size/free_seats); --apply adds --live to spawn the real wave."""
    cmd = [sys.executable, "tools/issue_dispatch.py", "--wave",
           "--max-workers", str(WAVE_MAX), "--work-kind", "engineering", "--json"]
    if apply:
        cmd.append("--live")
    plan = run_json(cmd, timeout=420) or {}
    return {
        "path": "issue_dispatch --wave" + ("" if tree_ok else " (tree red → this path required)"),
        "verdict": plan.get("verdict", "UNKNOWN"),
        "size": plan.get("size"),
        "free_seats": plan.get("free_seats"),
        "lanes": [l.get("lane") for l in (plan.get("lanes") or []) if isinstance(l, dict)][:8],
        "launched": bool(apply and plan.get("live") and plan.get("verdict") in ("WAVE", "WAVED", "SPAWNED")),
    }


# ---- nodes (lab/mac/gpu data collection) --------------------------------------

def drive_nodes(apply: bool) -> str:
    """Kick the local box's nightrun collection loop when it has a feasible datum.

    The mac/gpu NODES pull-and-run tools/fak_node_bench.sh on their OWN box (they
    already have the repo); this meta-loop cannot ssh-drive them unattended without
    the gitignored node registry, so here it drives the LOCAL box's feasible night
    datum and REPORTS the node fan-out as an operator follow-on rather than faking it."""
    if not nightrun_feasible():
        return "nodes: local box has no feasible nightrun datum (skip); mac/gpu run tools/fak_node_bench.sh on-box"
    if not apply:
        return "nodes: (dry-run) would `fak nightrun run --apply` one local datum"
    rc, _, _ = run(["go", "run", "./cmd/fak", "nightrun", "run", "--apply", "--max", "1"], timeout=420)
    return f"nodes: local nightrun datum collected rc={rc}"


# ---- worker effectiveness (surface the ships-per-worker leaks; never auto-kill) -

# 10x issues/hour is a ships-PER-WORKER problem, not a spawn-count problem: with
# seats free (headroom>0) the cap is workers that burn a slot and ship NOTHING — a
# silent exit (sub-floor log, dead pid; dispatch_status #2062) or a majority-stub
# backend. `reclaim_rung1` already self-heals DEAD inflight markers; this rung
# surfaces the LIVE leaks it cannot (a still-listed silent worker, a backend stubbing
# out) so the operator can rescope/cool them. Per Step 0.5 it is a READ THAT INFORMS
# A KILL — surfaced every tick, never automated (a kill carries the -Launch bar).

EFFECTIVE = "EFFECTIVE"
LEAKING = "LEAKING"


def effectiveness_summary(status: dict | None) -> dict:
    """PURE: fold a `dispatch_status.py --json` payload into the ships-per-worker leak
    signal — silent (empty-exit) workers + majority-stub backends. None/partial
    payloads degrade to UNKNOWN, never crash."""
    if not status:
        return {"verdict": "UNKNOWN", "silent_count": None, "silent_issues": [],
                "majority_stub": [], "ships_per_worker": None,
                "action": "dispatch_status unavailable this tick — no effectiveness read"}
    workers = status.get("workers") or {}
    silent = [w for w in (workers.get("silent") or []) if isinstance(w, dict)]
    silent_count = workers.get("silent_count")
    if not isinstance(silent_count, int):
        silent_count = len(silent)
    stub_rows = ((status.get("backend_health") or {}).get("stub_rate")) or []
    # A row can be majority-stub yet carry no backend name (no .backend sidecar on its
    # logs) — that is still a real leak; surface it as "unattributed", never drop it.
    majority_stub = [(r.get("backend") or "unattributed") for r in stub_rows
                     if isinstance(r, dict) and r.get("majority_stub")]
    bits = []
    if silent_count:
        issues = ", ".join("#" + str(w.get("issue")) for w in silent[:6])
        bits.append(f"{silent_count} silent worker(s) exited empty ({issues}) — "
                    f"operator: rescope/close, free the slot")
    if majority_stub:
        bits.append(f"backend(s) majority-stub [{', '.join(str(b) for b in majority_stub)}] — "
                    f"operator: cool the backend/account so refills route productive")
    return {"verdict": LEAKING if (silent_count or majority_stub) else EFFECTIVE,
            "silent_count": silent_count,
            "silent_issues": [w.get("issue") for w in silent[:12]],
            "majority_stub": majority_stub,
            "ships_per_worker": (status.get("ships_per_worker") or {}) or None,
            "action": " | ".join(bits) if bits else "no silent/stub leak this tick"}


# Compute ONLY the two cheap, pure-local folds we need (silent workers + backend
# stub rate) in an isolated child — NOT the whole `dispatch_status --json` payload.
# That payload's gh / run-status / utilization folds push it PAST TWO MINUTES on a
# busy box (measured on a full fleet), but these two are a runs-dir glob + one psutil
# sweep — seconds. Shelling out (vs importing dispatch_status into the loop) keeps a
# fold crash from ever killing the marathon, matching every other fold here.
_EFFECTIVENESS_PROBE = (
    "import json,sys; sys.path.insert(0,'tools'); import dispatch_status as d; "
    "from pathlib import Path; runs=Path('.')/d.RUNS_DIRNAME; "
    "print(json.dumps({'workers':{'silent':d.silent_workers(runs)},"
    "'backend_health':{'stub_rate':d.backend_stub_rates(runs)}}))"
)


def read_effectiveness() -> dict:
    """IMPURE: the ships-per-worker leak signal from the two cheap pure-local folds in
    dispatch_status (silent workers + backend stub rate), computed in an isolated child.
    Deliberately NOT `dispatch_status --json` — its gh/run-status folds run for minutes
    on a busy box; these two are seconds. effectiveness_summary derives silent_count
    from the list, so the trimmed payload is enough."""
    return effectiveness_summary(
        run_json([sys.executable, "-c", _EFFECTIVENESS_PROBE], timeout=90))


# ---- the marathon loop --------------------------------------------------------

@dataclass
class TickReport:
    ts: str
    tick: int
    verdict: str
    closed_total: int | None
    closed_delta: int
    toward_target: int
    target: int
    per_hour: float | None
    cron: str = ""
    tree: str = ""
    tree_class: dict = field(default_factory=dict)
    reclaim: str = ""
    refills: list = field(default_factory=list)
    nodes: str = ""
    effectiveness: dict = field(default_factory=dict)
    stop: str | None = None


def append_ledger(rep: TickReport) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rep)) + "\n")


def main() -> int:
    # Windows consoles default to cp1252; the plan uses box glyphs/em-dashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — best-effort; non-UTF terminals still get ASCII fallbacks below
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=200, help="witnessed closes to progress tonight")
    ap.add_argument("--cadence-min", type=float, default=5.0,
                    help="minutes between ticks (skip-if-inflight makes a tight cadence safe: "
                         "a tick with no headroom no-ops instead of double-dispatching)")
    ap.add_argument("--deadline-hours", type=float, default=12.0, help="hard wall-clock stop")
    ap.add_argument("--max-ticks", type=int, default=64, help="runaway backstop")
    ap.add_argument("--apply", action="store_true", help="run the marathon for real (default: single dry-run tick)")
    ap.add_argument("--force-wave", action="store_true",
                    help="hand-launch waves even while the FleetIssueDispatch cron is live "
                         "(default: SUPERVISE the cron, never double-dispatch beside it)")
    args = ap.parse_args()

    start = time.monotonic()
    baseline = progress().get("closed_by_loop_total")
    if baseline is None:
        print("orient: cannot read witnessed close counter — refusing to run blind", file=sys.stderr)
        return 2

    mode = "APPLY (real --live waves)" if args.apply else "DRY-RUN (plans only, no spawns)"
    print(f"═ META super-loop above the night runs — {mode}")
    print(f"  intent: progress {args.target} WITNESSED closes | baseline closed_by_loop_total={baseline}")
    print(f"  cadence={args.cadence_min}m deadline={args.deadline_hours}h target_total={baseline + args.target}")
    print(f"  ledger: {LEDGER.relative_to(REPO)}\n")

    tick = 0
    while True:
        tick += 1
        elapsed_h = (time.monotonic() - start) / 3600.0

        pf = preflight()
        prog = progress()
        closed = prog.get("closed_by_loop_total")
        delta = (closed - baseline) if (closed is not None and baseline is not None) else 0

        rep = TickReport(
            ts=_now(), tick=tick, verdict=pf["verdict"],
            closed_total=closed, closed_delta=delta,
            toward_target=delta, target=args.target,
            per_hour=prog.get("current_issues_per_hour"),
        )

        # --- stop conditions (honest marathon signals) ---
        if delta >= args.target:
            rep.stop = f"TARGET_MET: {delta}/{args.target} witnessed closes"
        elif elapsed_h >= args.deadline_hours:
            rep.stop = f"DEADLINE: {elapsed_h:.1f}h >= {args.deadline_hours}h ({delta}/{args.target} closed)"
        elif tick > args.max_ticks:
            rep.stop = f"MAX_TICKS backstop ({delta}/{args.target} closed)"
        elif pf["verdict"] in ("REFUSE_NO_SEAT", "REFUSE_NO_ACCOUNT"):
            # waiting fixes these (seat returns on a window, not a retry) — pause, don't spin
            rep.stop = f"SEAT_EXHAUSTED: {pf['verdict']} — {pf['reason']}"

        if rep.stop:
            print(f"[tick {tick}] STOP — {rep.stop}")
            append_ledger(rep)
            break

        # --- drive the fleet (cron-check -> build-check -> reclaim -> refill -> nodes) ---
        cron = cron_state()
        rep.cron = f"{cron['name']}={cron['state']}"
        tree_ok, tree_msg = tree_builds()
        rep.tree = tree_msg
        rep.tree_class = diagnose_tree(tree_ok)
        rep.reclaim = reclaim_rung1(args.apply)
        rep.effectiveness = read_effectiveness()

        # One gate decides whether this tick drives a wave — skip-if-inflight, so a
        # tight cadence re-checks headroom sooner without ever double-dispatching onto
        # slots a prior wave still holds. The decision is pure and its reason is logged.
        cron_owns_waves = cron["live"] and not args.force_wave
        wd = wave_decision(pf, cron_owns_waves)
        if wd["code"] == WAVE_SPAWN:
            rep.refills.append(refill(args.apply, tree_ok))
        else:
            rep.refills.append({"skipped": wd["code"], "reason": wd["reason"]})
        rep.nodes = drive_nodes(args.apply)

        # --- report the tick ---
        print(f"[tick {tick}] {pf['verdict']} | closed {closed} (Δ{delta:+d}/{args.target}) "
              f"| {prog.get('current_issues_per_hour')}/h | {elapsed_h:.1f}h")
        print(f"    cron: {rep.cron}  ({'supervising — cron owns the waves' if cron['live'] and not args.force_wave else 'meta-loop owns the waves'})")
        tree_flag = "OK" if tree_ok else "!! BROKEN — caps every in-checkout worker"
        print(f"    tree: {rep.tree}  [{tree_flag}]")
        if not tree_ok and rep.tree_class.get("code") not in (None, "", TREE_OK):
            print(f"        break: {rep.tree_class.get('code')} — {rep.tree_class.get('action')}")
        print(f"    {rep.reclaim}")
        for r in rep.refills:
            if "skipped" in r:
                print(f"    refill: SKIPPED ({r['skipped']}) — {r['reason']}")
            else:
                print(f"    refill: {r['path']} | {r['verdict']} size={r['size']} "
                      f"free_seats={r['free_seats']} launched={r['launched']}")
                if r.get("lanes"):
                    print(f"        lanes: {', '.join(str(x) for x in r['lanes'])}")
        print(f"    {rep.nodes}")
        eff = rep.effectiveness or {}
        print(f"    effectiveness: {eff.get('verdict', 'UNKNOWN')} — {eff.get('action', '')}")
        append_ledger(rep)

        if not args.apply:
            print("\n(dry-run: single planning tick only — pass --apply to run the marathon)")
            break

        time.sleep(args.cadence_min * 60.0)

    print("\n═ marathon done — the honest result is WITNESSED closes, not launches:")
    fin = progress()
    fclosed = fin.get("closed_by_loop_total")
    fdelta = (fclosed - baseline) if (fclosed is not None) else 0
    print(f"  closed_by_loop_total {baseline} -> {fclosed}  (Δ{fdelta:+d} toward {args.target})")
    print(f"  verify: dos commit-audit --json ; fak dispatch progress --json ; ledger {LEDGER.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
