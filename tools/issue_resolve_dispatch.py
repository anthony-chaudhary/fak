#!/usr/bin/env python3
r"""One guarded, switcher-routed dispatch TICK that spawns an ISSUE-RESOLUTION
worker — the arm that moves the open-issue counter on a plan-empty repo.

``issue_dispatch.py`` spawns the generic ``/dos-kernel:dos-dispatch-loop``
worker, which resolves units from the *plan portfolio*. This public repo ships
no ``PLAN-*.md`` (``PLAN_SURFACE_EMPTY``), so that worker has no work surface and
closes nothing — the backlog lives in GitHub *issues*. This tick is the
issue-driven sibling: it picks ONE concrete open issue on the busiest lane,
renders a scoped resolution prompt (``issue_worker_prompt.py`` — with the
``#N``-in-subject rule that lets the closure auditor witness the resulting
commit), and launches one detached ``claude -p "<prompt>"`` worker to land it.

It shares every safety primitive with ``issue_dispatch.py`` (imported, not
re-implemented): the per-tick registry refresh (route off current account
evidence), the ``dispatch_preflight`` DoS gate (host clean ∧ account free ∧ live
< cap), the switcher-pinned account env, and the detached spawn. DRY-RUN BY
DEFAULT — prints the issue, account, command, and prompt path. ``--live`` spawns.

In-flight de-dup: it skips an issue that already has a live resolution worker (a
dispatch log naming ``#N`` whose process is still alive) so two ticks never storm
the same issue.

Pre-spawn lane-lease gate (#1310): it also HOLDS the target lane before launching
rather than trusting the worker to self-arbitrate. fak's taxonomy is one worker
per leaf tree, so a second worker on a lane that already has a live one co-edits
the same files. The auto-pick reroutes around held lanes (busiest FREE lane); an
explicitly-named lane that is held is refused ``LANE_BUSY`` (COLLISION_RISK)
instead of raced. This is the upstream half of the verified loop — deny the
collision by structure, *before* the spawn — paired with the downstream
commit-time closure audit (the ``#N``-in-subject witness, still the close arm).

Contract-ready pick, next()-queue style: the issue-contract gate holds THIN
issues, never the whole tick. One tick contract-reviews a bounded window
(``--contract-scan``) of candidates drawn round-robin ACROSS eligible lanes
(oldest first within a lane), held verdicts persist in a TTL'd skip ledger so
the window advances across ticks, and an issue whose GitHub ``updatedAt`` is
newer than its held verdict re-enters early. When the WHOLE window fails the
gate, the tick spawns one contract-REPAIR worker (``--repair-batch``) that
brings the held issues up to contract themselves via ``gh issue edit`` — the
self-serve arm that keeps a pre-schema backlog from wedging dispatch forever.

Loop ledger: the CLI path appends this dispatcher tick to fak's durable loop
ledger by default (``fak loop append``): every tick records ``fire`` and
admitted/refused ``admit`` rows, live spawns record ``start``, and successful ticks
record ``end``. Disable with ``--no-loop-ledger`` for hermetic/manual probes.

    python tools/issue_resolve_dispatch.py                 # plan one tick (dry-run)
    python tools/issue_resolve_dispatch.py --live          # spawn one issue worker
    python tools/issue_resolve_dispatch.py --lane gateway --live
"""
from __future__ import annotations

import argparse
import account_probe
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_dispatch  # noqa: E402  (refresh_registry/preflight/worker_env/spawn_detached)
import worker_worktree  # noqa: E402  (#3181: per-worker worktree isolation, shared with the Go spine)
import issue_worker_prompt  # noqa: E402  (render the per-issue resolution prompt)
import dispatch_worker  # noqa: E402  (child_env for the opencode backend)
import dispatch_preflight  # noqa: E402  (pid-sidecar identity probe)
import lane_yield  # noqa: E402  (shared low-yield lane fold, #2062)
import issue_lane_router  # noqa: E402  (lane_taxonomy: dos.toml lane→tree map, #2062)
import fleet_trend  # noqa: E402  (per-tick fleet-status-history append, #4594)
import lane_core  # noqa: E402  (core-source / trust-critical lane predicates, mirror of internal/dispatchtick/selfmodify.go)
import tier_launch  # noqa: E402  (per-issue tier launch profile, mirror of internal/dispatchtick/launchprofile.go)
import issue_closure_audit  # noqa: E402  (shared resolving-commit witness grammar, #5071)

# Re-export the shared console-window suppressor so the account-topup / glm-docs
# entry scripts (which import THIS module as `ird`) route every helper subprocess
# through the one canonical guard. See dispatch_worker.no_window_creationflags.
no_window_creationflags = dispatch_worker.no_window_creationflags

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
# Per-worker membership sidecar (rank/wave/size/shortfall), written next to the
# .pid/.backend sidecars so an auditor enumerates a wave straight from disk.
WAVE_SIDECAR_SUFFIX = ".wave"
ACCOUNT_SIDECAR_SUFFIX = ".account"
LEASE_SIDECAR_SUFFIX = ".lease"
# Commit-time diff-witness binding (residual of #1310/#1324, proposal #2). The
# dispatcher stamps repo HEAD into a .basesha sidecar at launch (per-worker
# commit-sha tracking) so a later tick can re-audit the commit THIS worker landed,
# and records the slot's witness verdict in a .witness sidecar so a bare `exit 0`
# never SILENTLY counts as productive. See the witness sweep below.
BASE_SHA_SIDECAR_SUFFIX = ".basesha"
WITNESS_SIDECAR_SUFFIX = ".witness"
# Fenced lane-lease (residual of #1310): the dispatcher ATOMICALLY acquires
# refs/fak/locks/resolve-<lane> via `fak leaseref acquire` before launching, so a
# tick on ANOTHER machine (and a same-host TOCTOU race the log-scan can't close)
# refuses the collision instead of co-editing the leaf tree. The same-host log
# scan (live_resolution_lanes) stays the fast path UNDER this lease. The lease id
# is one safe ref segment; fak's lane names are already that shape. The acquired
# holder/generation are stamped beside the worker log, then a dead-worker witness
# sweep releases the lease with `fak leaseref release --id --holder --generation`;
# TTL+reap stays the crash/failure backstop.
LEASE_ID_PREFIX = "resolve-"
# A held lease outlives its worker by this margin past the wall-clock worker
# timeout, so a crashed worker that never releases is reaped by `fak leaseref
# reap` on a later tick rather than wedging the lane forever. Generous: the TTL is
# a backstop, not the primary release (that is the witnessed release on exit).
LEASE_TTL_MARGIN_S = 600
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")
# Contract-repair workers get their OWN log prefix so the resolve-only scans
# (issue cooldown, live-issue/lane de-dup, the commit witness sweep) never see
# them -- a repair run must not cool an issue, hold a lane, or grade as a
# no-commit resolution slot. The reap/prune sweeps DO cover both prefixes.
_REPAIR_LOG_RE = re.compile(r"repair-(\d+)-")
_ANY_WORKER_LOG_RE = re.compile(r"(?:resolve|repair)-(\d+)-")
# The `lane=<L>` field of the `# fak-spawn` header `spawn_issue_worker` flushes as
# the first line of every worker log (used by the pre-spawn lane-lease gate, #1310).
_SPAWN_LANE_RE = re.compile(r"\blane=(\S+)")
_RID_RE = re.compile(r"^RID-[A-Z0-9]+$")
DEFAULT_WORKER_TIMEOUT_S = dispatch_worker.DEFAULT_TIMEOUT_S
DEFAULT_SPAWN_PROBE_S = 5.0
# Hard ceiling on how many extra worker slots a healthy backend may claim from dead
# siblings in one tick — bounds the blast radius so a transient mass-death can't blow
# the healthy backend's cap past what its account can actually serve.
DEFAULT_REALLOC_CEILING = 2
# Seat-adaptive tick sizing (#3246). The standing crons' fixed --max-workers (3/4/2)
# was a throttle BELOW the preflight's real DoS cap (min(host_cap, seats)): adding an
# account raised seat_free but never the live worker count, so new capacity idled.
# When the preflight probe carries a seat signal, the tick re-sizes its effective cap
# to  min(live + seat_free, host_cap, hard ceiling, live + ramp delta)  and re-runs
# the preflight at that cap. The preflight stays the AUTHORITATIVE floor — a REFUSE_*
# at the resized cap still stops growth; only the redundant configured_max term moves.
# The configured --max-workers remains the fail-safe cap whenever no seat signal is
# available (fail-open to exactly the pre-#3246 behavior), and it RAISES the hard
# ceiling when set above it (seat sizing never shrinks an explicit operator cap).
# The ramp delta keeps growth canary-safe: one tick may lift the cap at most this far
# above the live count, so new seats convert to workers over several 5-min ticks
# rather than in one burst (0 disables the ramp bound).
DEFAULT_SEAT_ADAPTIVE = True
DEFAULT_SEAT_CEILING = 20
DEFAULT_SEAT_RAMP_DELTA = 2
# Per-tick spawn FAN-OUT. The cap was never the binding constraint on steady-state
# concurrency — ARRIVAL RATE was. `evaluate()` spawns AT MOST ONE worker per call, and
# the armed FleetIssueDispatch task fires on a PT5M repeat, so with a ~5-min worker
# lifetime the fleet converges to  lifetime / tick_interval ~= 1 live worker  no matter
# how large --max-workers is (witnessed: cap=18, live=1, headroom=17). Raising the cap
# cannot fix that; only spawning more than once per tick can.
#
# `evaluate_burst()` is that arm: it calls `evaluate()` up to `spawn_burst` times in one
# tick, and each call re-runs the ENTIRE admission chain from scratch (registry-routed
# account, dispatch_preflight host/seat/cap gate, weekly-cap + backend-health gates,
# live-issue/live-lane de-dup, the contract + collision + multi-lane scans, and the
# fenced lane lease). Nothing is hoisted across spawns: sub-tick N+1 sees the worker
# sub-tick N just launched (its .pid sidecar makes it `live`, its log holds the lane,
# its issue enters the cooldown/skip set) and re-decides at the NEW headroom, so the
# fan-out fills capacity without ever bypassing a gate.
#
# THIS HOST LOCKS UP ON SPAWN BURSTS (#3153: soft-fault + process-spawn churn saturating
# the kernel scheduler/MM locks — invisible to every CPU/RAM/disk meter). So the fan-out
# is deliberately double-bounded and OFF by default:
#   * DEFAULT_SPAWN_BURST = 1  -> byte-identical to the pre-fan-out tick until an
#     operator opts in (--spawn-burst / FAK_DISPATCH_SPAWN_BURST).
#   * SPAWN_BURST_HARD_CEILING clamps ANY flag/env value, so no configuration — not a
#     typo'd env var, not a bad cron edit — can turn one tick into a spawn storm. This
#     limit is SEPARATE from --max-workers: --max-workers bounds the total live
#     population, the burst bounds NEW processes per tick (the churn axis that freezes
#     the box).
#   * a stagger between spawns spreads the process-creation transient instead of
#     issuing it as one burst; each sub-tick's own preflight then re-reads host health
#     before the next launch.
DEFAULT_SPAWN_BURST = 1
SPAWN_BURST_HARD_CEILING = 4
SPAWN_BURST_ENV = "FAK_DISPATCH_SPAWN_BURST"
DEFAULT_SPAWN_BURST_STAGGER_S = 15.0
SPAWN_BURST_STAGGER_ENV = "FAK_DISPATCH_SPAWN_BURST_STAGGER_S"
SPAWN_BURST_STAGGER_MAX_S = 120.0
LOOP_ID_PREFIX = "issue-resolve-dispatch"
DEFAULT_ISSUE_CONTRACT_MIN_SCORE = 100
# Contract-ready pick: how many lane issues (oldest first) one tick may contract-
# review while scanning past held THIN heads for a spawnable target. Each review
# shells one bounded `fak issue contract`, so this caps tick latency, not safety --
# the floor itself is never relaxed by the scan.
DEFAULT_CONTRACT_SCAN = 8
# How long a contract-HELD issue stays skipped before it is re-reviewed. The TTL is
# what lets a later backfill (contract fields added to the issue body) re-enter the
# pool without a manual reset; the skip is what makes the bounded scan CUMULATIVE
# across ticks instead of re-reviewing the same thin heads forever.
DEFAULT_CONTRACT_HOLD_TTL_H = 24
_CONTRACT_HOLD_LEDGER = "contract-holds.jsonl"
# Multi-lane scope holds (#2971): an issue that names file families outside the
# chosen lane lease needs a split/reroute, not another identical launch tick.
DEFAULT_MULTI_LANE_HOLD_TTL_H = 24
_MULTI_LANE_HOLD_LEDGER = "multi-lane-holds.jsonl"
# Dirty-path / same-issue-WIP collision holds (#2977/#2975 durability): a chosen
# issue that names a path already dirty in this shared checkout — or that stacks
# onto a prior resolver's uncommitted same-issue WIP — will collide IDENTICALLY on
# the next tick, because the dirty tree does not shrink between 5-minute ticks.
# Unlike the contract and multi-lane gates, the collision gates had NO durable hold,
# so the picker re-selected the same colliding head every tick, re-refused it, and
# burned the bounded scan budget (contract_scan) on the same 2-3 heads instead of
# reaching disjoint solvable work. This ledger gives a collision the same durable,
# TTL-bounded skip the other gates have: hold the colliding issue for the window so
# the cumulative scan advances past it. The TTL is short (a collision clears the
# moment a peer commits/reverts the dirty path — a LOCAL event that no gh updatedAt
# reflects), so the hold re-checks the live tree every few hours rather than pinning
# the issue for a day. Set to 0 to disable (readers return empty, matching the
# contract/multi-lane TTL<=0 short-circuit).
DEFAULT_COLLISION_HOLD_TTL_H = 3
_COLLISION_HOLD_LEDGER = "collision-holds.jsonl"
# Low-yield lane soft-exclude (#2062): a lane whose recent resolve sessions burned
# turns yet closed nothing is a poison-pill sink (e.g. a GPU-less host re-grabbing a
# P1 GPU epic it structurally cannot run). The shared lane_yield fold flags it; the
# picker SOFT-excludes it from the busiest-pick so a healthier lane runs instead.
# Guards keep this from ever freezing the fleet: at most LOW_YIELD_MAX_EXCLUDES lanes
# are demoted (worst-by-turns first), a lane needs LOW_YIELD_MIN_SESSIONS finished
# sessions before it is trusted as a signal, and lane_issue_numbers re-seats a
# demoted lane (reporting it under low_yield_relief) if it is the last eligible one.
# The fold's 180-min sliding lookback IS the self-healing TTL — no sticky state.
# Cap sized to the trust floor, not to 2: on a busy fleet 5+ lanes routinely clear
# LOW_YIELD_MIN_SESSIONS at once (measured: 5 proven low-yield lanes in one 180-min
# window, all internal/** core-source). At the old cap of 2 the picker demoted only
# the 2 worst and the value-weighted pick (which ranks core lanes first) then landed
# on the 3rd/4th 0-close core lane instead of a lane that is actually closing — the
# cap, not the starvation floor, was the binding constraint. 5 demotes the whole
# proven-sink set; still freeze-safe because MIN_SESSIONS bounds the candidate set
# upstream and lane_issue_numbers' batch starvation floor re-seats them ALL (under
# low_yield_relief) whenever excluding would leave zero eligible lanes.
LOW_YIELD_MAX_EXCLUDES = 5
LOW_YIELD_MIN_SESSIONS = 2
# Operator-forced contract-gate bypass ledger (#2637): every time --force downgrades
# a FAILED issue-contract readiness gate to advisory and spawns anyway, one row is
# appended here with the operator's structured --force-reason. This is the durable,
# operator-visible audit trail — the count folds into each tick's run artifact so an
# operator can see WHEN and HOW OFTEN the guard is being bypassed, not just that a
# single spawn happened to carry `bypassed=true` as telemetry.
_CONTRACT_FORCED_LEDGER = "contract-forced-bypasses.jsonl"
# Contract-repair dispatch: when the WHOLE scan window fails the gate, the tick
# spawns one worker to bring the held issues up to contract themselves (gh issue
# edit) instead of idling until a human grooms the backlog. The batch bounds one
# worker's grooming load; 0 disables repair dispatch entirely.
DEFAULT_REPAIR_BATCH = 5
# An issue a repair worker already ATTEMPTED is not re-groomed inside this window
# (anti-churn for un-repairables). A SUCCESSFUL repair needs no cooldown escape:
# its edit bumps the issue's updatedAt, which re-admits it past the hold ledger.
DEFAULT_REPAIR_COOLDOWN_MIN = 360
# Same-issue WIP hold (#2975): a finished resolver can leave useful local
# working-tree changes without a commit. If the same issue is immediately picked
# again, the second worker stacks onto uncommitted WIP that no live lease owns.
SAME_ISSUE_WIP_LOOKBACK_MIN = 24 * 60
# Orientation docs are NOT same-issue WIP evidence (#4321). The SAME_ISSUE_WIP scan
# infers "the prior resolver left this path dirty" from the path being NAMED in that
# resolver's log. Every worker prompt orders the worker to read the repo's
# orientation docs by name ("orient with AGENTS.md", "the doc map llms.txt"), so
# those names appear in EVERY resolve log regardless of what the worker touched.
# They are also chronically dirty in the one shared trunk checkout — a co-edit
# magnet. The two facts multiply into a false positive on essentially every issue:
# the ledger investigation behind #4321 measured ~175/180 SAME_ISSUE_WIP hits on
# AGENTS.md alone, and this repo's own .dispatch-runs/collision-holds.jsonl shows
# the same defect wearing different names (README.md 78, llms.txt 63, INDEX.md 42 —
# 183 of 208 same_issue_wip path-hits — vs 25 hits on a real code path). So the
# concentration is NOT an AGENTS.md property and splitting AGENTS.md into fragments
# would not have fixed it; it would relocate the magnet. Mention of an orientation
# doc is evidence the worker was told to READ it, never that it left it dirty.
# Narrow by construction: this suppresses the log-mention INFERENCE only. The
# dirty-path guard, which keys on the ISSUE text, still refuses when an issue's own
# body names one of these files as its work site, so a genuine AGENTS.md issue is
# still protected from landing on a peer's uncommitted edit.
SAME_ISSUE_WIP_ORIENTATION_DOCS = frozenset({
    "AGENTS.md", "CLAUDE.md", "README.md", "INDEX.md",
    "CONTRIBUTING.md", "llms.txt", "llms-full.txt",
})


def is_orientation_doc(path: str) -> bool:
    """True for a repo-root orientation doc every worker prompt names by rote."""
    return normalize_repo_path(path) in SAME_ISSUE_WIP_ORIENTATION_DOCS
# Active guard-livelock spawn hold: once a live guarded worker is repeatedly
# receiving the same quarantined tool_result, adding more resolver workers just
# consumes seats while the guard bug is already witnessed. This launcher-side
# fuse is deliberately conservative and fail-open: it only fires when a live
# resolve log names a guard-audit journal and that journal proves a repeated
# quarantined result.
ACTIVE_GUARD_LIVELOCK_MIN_COUNT = 10
_AUDIT_LOG_RE = re.compile(r"audit log\s*:\s*(.+?\.jsonl)")
# Active compact-runaway spawn hold: a worker that is already far past the
# compact threshold while still taking tool turns is not healthy headroom. This
# catches the guard/control-plane loop before a new worker compounds the same
# context burn.
ACTIVE_COMPACT_RUNAWAY_MIN_COUNT = 3
ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K = 20.0
# A turn whose context sits this many thousand tokens BELOW the previous turn's
# is a compaction shed: positive proof compact control just did its job. Context
# only grows within a compaction cycle, so a drop this large has no other cause.
# Hits recorded before the most recent shed describe an overshoot the harness has
# ALREADY recovered from, so they are forgotten (#5858). Without this the scan
# counts all-time hits and the hold LATCHES for the rest of a recovered worker's
# life -- the same "hold outlives the condition that caused it" shape as a lane
# goal-park that outlived its 429. The hold must describe compact control failing
# NOW, not once.
ACTIVE_COMPACT_RUNAWAY_SHED_K = 5.0
_CTX_BUDGET_RE = re.compile(r"\bctx:(\d+(?:\.\d+)?)k/(\d+(?:\.\d+)?)k")
_DIST_PAST_COMPACT_RE = re.compile(r"\bdist:(\d+(?:\.\d+)?)k-past-compact")
_COMPACT_FIELD_RE = re.compile(r"\bcompact=(\S+)")
# The pseudo-lane a repair worker runs under (guard-audit naming, child env). It
# takes NO lane lease -- it edits GitHub issues, not repo files; admission is
# serialized by the live repair-sidecar scan instead (max one in flight).
REPAIR_LANE = "contract-repair"
# Batch sidecar next to each repair-<N>-<stamp>.log naming EVERY issue in the
# worker's batch (the log name carries only the first), so the repair cooldown
# covers the whole batch.
REPAIR_ISSUES_SIDECAR_SUFFIX = ".issues"
# Where a repair worker LANDS its backfill: one local overlay file per issue
# (issue-<N>.md), merged into the issue record at contract-review time. GitHub
# issue mutations are operator-gated on this host (there is no sanctioned
# automated issue-edit verb; the egress floor blocks a worker's direct edit),
# so the overlay is the write path a worker may actually complete: dispatch
# admission reads body+overlay, while the REAL issue body edit stays a manual,
# operator-approved step driven by the repair manifest.
CONTRACT_OVERLAY_DIRNAME = "contract-overlays"

# Worker backends this tick can launch:
#   claude   = opus (t1) -- the reference path, the established quota pool.
#   opencode = glm-5.2 via the zai-coding-plan accounts (t2) -- a SEPARATE quota
#              pool that sits idle. Routing a lane (e.g. docs, where glm is proven)
#              to opencode relieves the opus weekly-quota throughput ceiling.
BACKENDS = ("claude", "opencode", "codex")
_BACKEND_PRODUCT = {"claude": "claude", "opencode": "opencode", "codex": "codex"}
OPENCODE_PROMPT_NOTICE = "Resolve GitHub issue # from the attached dispatch prompt."
OPENCODE_PROMPT_FILE_SUFFIX = ".prompt.txt"
CLAUDE_PROMPT_FILE_SUFFIX = ".prompt.txt"

# Operator directive: claude issue-resolution workers run Opus 5 at xhigh
# reasoning effort. Pinned here (not read from the per-account settings.json
# `model`, which the switcher/backup tooling live-rotates) so the choice STICKS.
# xhigh has no settings.json field — it is only settable via the `--effort` flag.
# Kept in step with the Go launch default (cmd/fak's defaultLaunchModel and
# internal/dispatchtick's WorkerModelOpus) so a worker spawned from this path and
# one spawned from `fak dispatch` are the same model, not two silent regimes.
# NOTE: unlike the Go path this launcher passes no --fallback-model, so a seat that
# cannot start this model fails rather than degrading; add the chain here if that
# becomes the observed wall.
CLAUDE_WORKER_MODEL = "claude-opus-5"
CLAUDE_WORKER_EFFORT = "xhigh"


def worker_model_effort(backend: str, acct: dict) -> tuple[str | None, str | None]:
    """The (model, reasoning-effort) to pin on one worker's CLI, per backend.

    - opencode keeps its per-account model and takes no effort knob.
    - claude is pinned to :data:`CLAUDE_WORKER_MODEL` at
      :data:`CLAUDE_WORKER_EFFORT` (operator directive) — EXCEPT a ``local``
      account, which routes to the local gateway model whose upstream ignores
      Claude's effort knob (mirrors ``claude_agent_chat``'s glm-mode skip).
    - codex uses its ambient login default (no pin).
    """
    if backend == "opencode":
        return acct.get("model"), None
    if backend == "claude":
        if str(acct.get("model") or "") == "local":
            return "local", None
        return CLAUDE_WORKER_MODEL, str(acct.get("model_effort") or CLAUDE_WORKER_EFFORT)
    return None, None


def launch_gate_for_guard(guarded: bool, backend: str) -> dict[str, Any]:
    """Machine-readable launch readiness for an already-priced worker command.

    ``WOULD_SPAWN`` means capacity/lane/contract gates passed. It does not, by
    itself, mean a Codex super-loop should approve ``--live``: the worker may have
    failed open around ``fak guard``. Surface that as data so operators and scripts
    gate on one field instead of remembering to interpret ``guarded`` separately.
    """
    if guarded:
        return {"ready": True, "guarded": True, "blockers": []}
    next_action = "make-fak-guard-resolvable"
    if backend != "claude":
        next_action = "make-fak-guard-resolvable-and-set-guard-base-url"
    return {
        "ready": False,
        "guarded": False,
        "blockers": [{
            "code": "UNGUARDED_WORKER",
            "reason": "worker command is spawnable but would run without fak guard",
            "next_action": next_action,
        }],
    }


# Refusal CLASS, distinct from the refusal CODE (#4321). Two mechanisms wear the
# word "collision" in this launcher and were therefore folded together in ledger
# analysis: the fenced lane-lease refusal (LANE_LEASE_HELD, whose reason text even
# says "refusing COLLISION_RISK") and the two working-tree co-tenancy refusals
# (SAME_ISSUE_WIP / DIRTY_PATH_COLLISION). They are NOT the same failure and do not
# share a fix. A lane-lease refusal means a live peer holds the lane's fenced lease:
# it clears on its own when that peer finishes or the TTL lapses, and the right
# response is to wait or pick a disjoint lane. A working-tree co-tenancy refusal
# means uncommitted WIP sits in the one shared checkout with no live lease owning
# it: nothing clears it but a human/peer commit or revert, and the right response is
# commit-by-path landing or the sanctioned detached per-worker worktree (#1334 /
# epic #3165). Stamping the class as its own FIELD lets a ledger reader separate the
# two by data instead of pattern-matching verdict names that both read "collision".
REFUSAL_CLASS_LANE_LEASE = "lane_lease"
REFUSAL_CLASS_WORKTREE_COTENANCY = "worktree_cotenancy"
_REFUSAL_CLASSES = {
    "LANE_LEASE_HELD": REFUSAL_CLASS_LANE_LEASE,
    "SAME_ISSUE_WIP": REFUSAL_CLASS_WORKTREE_COTENANCY,
    "DIRTY_PATH_COLLISION": REFUSAL_CLASS_WORKTREE_COTENANCY,
}


def refusal_class(verdict: str) -> str:
    """Contention class for a refusal verdict; "" when the verdict is neither.

    Deliberately a lookup over an explicit table rather than a substring rule: a
    future refusal code must be classified on purpose, not inherit a class because
    its name happens to contain "COLLISION" or "LEASE".
    """
    return _REFUSAL_CLASSES.get(str(verdict or "").strip().upper(), "")


def launch_gate_blocked(code: str, reason: str, next_action: str) -> dict[str, Any]:
    """Machine-readable launch hold for a tick that never reached command pricing."""
    blocker: dict[str, Any] = {
        "code": code,
        "reason": reason,
        "next_action": next_action,
    }
    contention = refusal_class(code)
    if contention:
        blocker["refusal_class"] = contention
    return {"ready": False, "blockers": [blocker]}


def _audit_path_from_log(log: Path, root: Path) -> Path | None:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found: Path | None = None
    for match in _AUDIT_LOG_RE.finditer(text):
        raw = match.group(1).strip().strip("`'\"")
        if not raw:
            continue
        cand = Path(raw)
        if not cand.is_absolute():
            cand = root / cand
        found = cand
    return found


def _guard_result_livelock(journal: Path, *,
                           min_count: int = ACTIVE_GUARD_LIVELOCK_MIN_COUNT) -> dict[str, Any]:
    """Return positive evidence for repeated quarantined tool_result rows.

    This is the result-side analogue of the status-card livelock fold, scoped to
    active worker journals. The key includes reason + digest so unrelated
    quarantines do not trip the fuse.
    """
    counts: dict[tuple[str, str, str], int] = {}
    try:
        lines = journal.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"livelock": False, "unavailable": True, "path": str(journal)}
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        verdict = str(row.get("verdict") or "").upper()
        tool = str(row.get("tool") or row.get("name") or "")
        if verdict != "QUARANTINE" or tool != "tool_result":
            continue
        reason = str(row.get("reason") or "")
        digest = str(row.get("args_digest") or row.get("digest") or "")
        if not digest:
            continue
        key = (tool, reason, digest)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {"livelock": False, "path": str(journal)}
    (tool, reason, digest), count = max(counts.items(), key=lambda item: item[1])
    if count < min_count:
        return {"livelock": False, "path": str(journal), "count": count}
    return {
        "livelock": True,
        "path": str(journal),
        "tool": tool,
        "reason": reason,
        "digest": digest,
        "count": count,
    }


def active_guard_livelock_hold(root: Path, runs_dir: Path, *,
                               min_count: int = ACTIVE_GUARD_LIVELOCK_MIN_COUNT) -> dict[str, Any]:
    """Positive live-worker evidence that should pause new resolver spawns."""
    candidates: list[dict[str, Any]] = []
    for log in sorted(runs_dir.glob("resolve-*.log")):
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(pid_file):
            continue
        audit = _audit_path_from_log(log, root)
        if not audit:
            continue
        hit = _guard_result_livelock(audit, min_count=min_count)
        if not hit.get("livelock"):
            continue
        issue = None
        m = _LOG_ISSUE_RE.search(log.name)
        if m:
            try:
                issue = int(m.group(1))
            except ValueError:
                issue = None
        hit.update({"log": str(log), "issue": issue})
        candidates.append(hit)
    if not candidates:
        return {"active": False}
    candidates.sort(key=lambda h: int(h.get("count") or 0), reverse=True)
    top = candidates[0]
    return {
        "active": True,
        "candidates": candidates[:5],
        "reason": (
            f"live worker #{top.get('issue') or '?'} is already in a guard "
            f"result livelock: {top.get('tool')} {top.get('reason')} "
            f"digest={str(top.get('digest') or '')[:12]} count={top.get('count')}"
        ),
    }


def _compact_runaway_from_log(
    log: Path,
    *,
    min_count: int = ACTIVE_COMPACT_RUNAWAY_MIN_COUNT,
    min_past_k: float = ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K,
    shed_k: float = ACTIVE_COMPACT_RUNAWAY_SHED_K,
) -> dict[str, Any]:
    """Return positive evidence that one live worker is past compact and looping.

    The guard's debug line is intentionally enough evidence here: once several
    consecutive tool turns are tens of thousands of tokens past the compact
    threshold, another resolver spawn is more likely to multiply the failure
    mode than add useful throughput.

    "Consecutive" is load-bearing and is enforced against the CURRENT compaction
    cycle only: a compaction shed (:data:`ACTIVE_COMPACT_RUNAWAY_SHED_K`) clears
    the accumulated hits, because a worker the harness already pulled back under
    budget is not a runaway no matter how far it once overshot.
    """
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"runaway": False, "unavailable": True, "log": str(log)}
    hits: list[dict[str, Any]] = []
    prev_ctx_k: float | None = None
    sheds = 0
    for line in lines:
        if "fak-turn" not in line or "finish=tool_use" not in line:
            continue
        ctx_m = _CTX_BUDGET_RE.search(line)
        if not ctx_m:
            continue
        try:
            ctx_k = float(ctx_m.group(1))
            limit_k = float(ctx_m.group(2))
        except ValueError:
            continue
        # Compact control demonstrably fired: forget the overshoot it just cured,
        # so the hold clears on recovery instead of latching forever.
        if prev_ctx_k is not None and ctx_k <= prev_ctx_k - shed_k:
            hits.clear()
            sheds += 1
        prev_ctx_k = ctx_k
        past_k = max(0.0, ctx_k - limit_k)
        dist_m = _DIST_PAST_COMPACT_RE.search(line)
        if dist_m:
            try:
                past_k = max(past_k, float(dist_m.group(1)))
            except ValueError:
                pass
        if past_k < min_past_k:
            continue
        compact_m = _COMPACT_FIELD_RE.search(line)
        hits.append({
            "ctx_k": ctx_k,
            "limit_k": limit_k,
            "past_k": past_k,
            "compact": compact_m.group(1) if compact_m else "",
            "line": line[-320:],
        })
    if len(hits) < min_count:
        return {
            "runaway": False,
            "log": str(log),
            "count": len(hits),
            "sheds": sheds,
            "max_past_k": round(max((h["past_k"] for h in hits), default=0.0), 1),
        }
    last = hits[-1]
    return {
        "runaway": True,
        "log": str(log),
        "count": len(hits),
        "sheds": sheds,
        "max_past_k": round(max(h["past_k"] for h in hits), 1),
        "ctx_k": last["ctx_k"],
        "limit_k": last["limit_k"],
        "past_k": round(last["past_k"], 1),
        "compact": last["compact"],
        "sample": last["line"],
    }


def active_compact_runaway_hold(
    root: Path,
    runs_dir: Path,
    *,
    min_count: int = ACTIVE_COMPACT_RUNAWAY_MIN_COUNT,
    min_past_k: float = ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K,
) -> dict[str, Any]:
    """Positive live-worker evidence that compact control is already failing."""
    del root
    candidates: list[dict[str, Any]] = []
    for log in sorted(runs_dir.glob("resolve-*.log")):
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(pid_file):
            continue
        hit = _compact_runaway_from_log(
            log,
            min_count=min_count,
            min_past_k=min_past_k,
        )
        if not hit.get("runaway"):
            continue
        issue = None
        m = _LOG_ISSUE_RE.search(log.name)
        if m:
            try:
                issue = int(m.group(1))
            except ValueError:
                issue = None
        hit["issue"] = issue
        # Which lane the runaway actually burns (#5858). A compact runaway is a
        # property of ONE worker's context, not of the fleet: stamp the lane so
        # the caller can scope the hold to the colliding lane instead of idling
        # every disjoint lane behind one worker's local condition. `None` when
        # the spawn header is unreadable -- the caller fails CLOSED on unknown.
        hit["lane"] = _spawn_header_lane(log)
        candidates.append(hit)
    if not candidates:
        return {"active": False}
    candidates.sort(
        key=lambda h: (float(h.get("max_past_k") or 0.0), int(h.get("count") or 0)),
        reverse=True,
    )
    top = candidates[0]
    lane_bit = f" on lane '{top.get('lane')}'" if top.get("lane") else ""
    return {
        "active": True,
        "candidates": candidates[:5],
        "lanes": sorted({str(c.get("lane")) for c in candidates if c.get("lane")}),
        "lane_unknown": any(not c.get("lane") for c in candidates),
        "reason": (
            f"live worker #{top.get('issue') or '?'}{lane_bit} is already past "
            f"compact by {top.get('max_past_k')}k tokens across "
            f"{top.get('count')} tool-use turns"
        ),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable or "python"


def _fak_command_prefix(root: Path) -> list[str]:
    explicit = os.environ.get("FAK_BIN")
    if explicit:
        return [explicit]
    for name in ("fak.exe", "fak"):
        cand = root / name
        if cand.is_file():
            return [str(cand)]
    return ["go", "run", "./cmd/fak"]


def contract_overlay_path(runs_dir: Path, number: int) -> Path:
    return runs_dir / CONTRACT_OVERLAY_DIRNAME / f"issue-{int(number)}.md"


def read_contract_overlay(runs_dir: Path, number: int | None) -> str:
    """The local contract backfill for one issue, '' when absent/unreadable
    (fail-open: no overlay simply reviews the bare body, exactly as before)."""
    if not number:
        return ""
    try:
        return contract_overlay_path(runs_dir, int(number)).read_text(
            encoding="utf-8", errors="replace").strip()
    except (OSError, TypeError, ValueError):
        return ""


def contract_overlay_times(runs_dir: Path) -> dict[int, float]:
    """``{issue: overlay mtime}`` for every local backfill on disk — folded into
    the hold ledger's re-admission timestamps so a fresh overlay re-enters its
    issue on the NEXT tick (the local analogue of a GitHub updatedAt bump)."""
    out: dict[int, float] = {}
    overlay_dir = runs_dir / CONTRACT_OVERLAY_DIRNAME
    if not overlay_dir.is_dir():
        return out
    for f in overlay_dir.glob("issue-*.md"):
        try:
            out[int(f.stem.split("-", 1)[1])] = f.stat().st_mtime
        except (OSError, ValueError, IndexError):
            continue
    return out


def _issue_record_for_contract(issue: dict[str, Any] | None,
                               number: int | None,
                               overlay: str = "") -> dict[str, Any]:
    issue = issue if isinstance(issue, dict) else {}
    labels = []
    for lab in issue.get("labels") or []:
        if isinstance(lab, dict):
            name = str(lab.get("name") or "").strip()
        else:
            name = str(lab or "").strip()
        if name:
            labels.append({"name": name})
    try:
        num = int(issue.get("number") or number or 0)
    except (TypeError, ValueError):
        num = int(number or 0)
    body = str(issue.get("body") or "")
    if overlay:
        # The dispatch-admission view of the issue is body + local backfill; the
        # marker keeps the merge legible in any dumped record.
        body = (body + "\n\n<!-- local contract overlay "
                f"({RUNS_DIRNAME}/{CONTRACT_OVERLAY_DIRNAME}) -->\n" + overlay)
    return {
        "number": num,
        "title": str(issue.get("title") or f"issue #{num}").strip(),
        "body": body,
        "labels": labels,
    }


# The typed hold token an explicitly-unresolved acceptance gate earns (#5070).
ACCEPTANCE_GATE_HOLD_REASON = "acceptance-gate-unresolved"

# The "## Acceptance gate" section of an issue contract, up to the next heading.
# ONLY this section binds the hold: prose elsewhere in the body that merely
# mentions an operator is never a hold. The heading terminator is `[ \t\r]*\n`
# rather than `$\n` on purpose -- a GitHub-authored body arrives CRLF, and `$`
# does not match before the `\r`, which would silently disable the whole gate.
_ACCEPTANCE_GATE_SECTION = re.compile(
    r"^[ \t]*#{1,6}[ \t]*acceptance[ \t]+gate[ \t\r]*\n"
    r"(.*?)(?=^[ \t]*#{1,6}[ \t]|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL)
# Explicit unresolved vocabulary that LEADS the gate -- a bare token line
# ("TBD") or a token heading the sentence ("unknown -- needs operator input").
# Anchored at the start so a concrete gate that merely uses the word later
# ("... covering unresolved and concrete controls") stays dispatchable.
_GATE_UNRESOLVED_TOKEN = re.compile(
    r"^(?:unknown|unresolved|tbd|to be determined|\?+)(?:\s*(?:$|[-–—:,;.]))")
# An unambiguous operator-input phrase anywhere in the gate section.
_GATE_OPERATOR_INPUT = re.compile(
    r"\b(?:needs?|requires?|awaiting|pending|blocked on)\s+(?:an?\s+)?"
    r"operator\s+(?:input|decision|answer|review|sign-?off)\b")


def unresolved_acceptance_gate(body: Any) -> str:
    """The verbatim acceptance-gate line that marks a contract as explicitly
    NOT executable yet, or ``""`` when the gate names a concrete witness (#5070).

    The always-on loop must spend a leased seat only on an executable contract. A
    score-only pass admitted #2806 whose gate read ``unknown -- needs operator
    input``: the aggregate score cleared the floor while the issue's own contract
    said the acceptance witness was still open, so a worker burned a `cmd` slot
    triaging instead of shipping. The explicit body contract is authoritative over
    the numeric score when it says operator input is still required.

    Binds the ``## Acceptance gate`` section ONLY, and within it only the first
    meaningful line, matched against explicit unresolved vocabulary that LEADS the
    gate or an unambiguous operator-input phrase. Prose that merely mentions an
    operator, or a concrete gate that happens to contain the word "unresolved",
    never matches -- fail-closed on the declared vocabulary, not on keywords."""
    text = body if isinstance(body, str) else ""
    section = _ACCEPTANCE_GATE_SECTION.search(text)
    if not section:
        return ""
    line = next((s for s in (ln.strip() for ln in section.group(1).splitlines()) if s), "")
    if not line:
        return ""
    probe = line.lstrip("-*+> \t").strip().strip("`*_").strip().lower()
    if _GATE_UNRESOLVED_TOKEN.match(probe) or _GATE_OPERATOR_INPUT.search(probe):
        return line[:200]
    return ""


def issue_contract_review(root: Path, issue: dict[str, Any] | None,
                          number: int | None,
                          runner: Any = subprocess.run) -> dict[str, Any]:
    tmp: Path | None = None
    # Merge any local backfill before scoring: the dispatch-admission view of an
    # issue is body + overlay, so a repair worker's landed overlay flips the very
    # next review without any GitHub write.
    overlay = read_contract_overlay(root / RUNS_DIRNAME, number)
    record = _issue_record_for_contract(issue, number, overlay)
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json",
                                         delete=False) as f:
            tmp = Path(f.name)
            json.dump([record], f, ensure_ascii=False)
            f.write("\n")
        cmd = [
            *_fak_command_prefix(root),
            "issue", "contract",
            "--from-issues", str(tmp),
            "--live", "--dedupe-checked", "--dedupe-cap", "300",
            "--json",
        ]
        proc = runner(
            cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, creationflags=no_window_creationflags())
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "unavailable": True, "score": 0, "reason": str(exc)}
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass

    try:
        doc = json.loads(proc.stdout or "{}")
    except (TypeError, ValueError):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = tail[-1] if tail else f"fak issue contract failed ({proc.returncode})"
        return {"ok": False, "unavailable": True, "score": 0, "reason": reason[:300]}
    reviews = doc.get("reviews") if isinstance(doc, dict) else None
    review = reviews[0] if isinstance(reviews, list) and reviews else {}
    score = ((review.get("score") or {}).get("total") if isinstance(review, dict) else 0) or 0
    spine = ((review.get("spine_priority") or {}).get("total")
             if isinstance(review, dict) else 0) or 0
    # Fail-closed on an explicitly-unresolved acceptance gate even when the
    # aggregate score clears the floor: the body contract is authoritative when
    # it says the acceptance witness still needs operator input (#5070).
    gate_hold = unresolved_acceptance_gate(record.get("body"))
    return {
        "ok": bool(proc.returncode == 0 and doc.get("ok") and review.get("ok")
                   and not gate_hold),
        "unavailable": False,
        "acceptance_gate_hold": gate_hold,
        "score": int(score),
        "spine_priority": int(spine),
        "review": review,
        "returncode": proc.returncode,
    }


def issue_contract_hold_reason(contract: dict[str, Any]) -> str:
    if contract.get("unavailable"):
        return str(contract.get("reason") or "issue contract unavailable")
    # A typed hold, reported ahead of the score terms: this contract can score
    # arbitrarily well and still be un-dispatchable (#5070).
    gate_hold = str(contract.get("acceptance_gate_hold") or "")
    if gate_hold:
        return f"{ACCEPTANCE_GATE_HOLD_REASON}: {gate_hold}"
    review = contract.get("review") if isinstance(contract.get("review"), dict) else {}
    parts = [str(r) for r in (review.get("reasons") or []) if r]
    parts.extend(f"missing:{m}" for m in (review.get("missing_fields") or []) if m)
    score = int(contract.get("score") or 0)
    if contract.get("ok") and score >= DEFAULT_ISSUE_CONTRACT_MIN_SCORE:
        return "issue contract passed"
    if score < DEFAULT_ISSUE_CONTRACT_MIN_SCORE:
        parts.append(f"score:{score}<floor:{DEFAULT_ISSUE_CONTRACT_MIN_SCORE}")
    return ", ".join(parts) if parts else "issue contract below spawn floor"


def _git_capture(root: Path, args: list[str], *, timeout: int = 30) -> tuple[int, str]:
    """Run one ``git`` subcommand under ``root``; return ``(rc, stdout)``. Never
    raises: an exec failure is reported as rc 127 so every caller fails open. Used
    by the per-worker commit-witness binding (#1324 proposal #2)."""
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(["git", *args], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, proc.stdout or ""


def low_yield_soft_excludes(
    root: Path,
    runs_dir: Path,
    *,
    lane_trees: dict[str, list[str]] | None = None,
    now_ts: float | None = None,
    max_excludes: int = LOW_YIELD_MAX_EXCLUDES,
    min_sessions: int = LOW_YIELD_MIN_SESSIONS,
) -> dict[str, Any]:
    """Fold the recent resolve corpus into the set of lanes to SOFT-exclude (#2062).

    A lane is a soft-exclude candidate when the shared ``lane_yield`` fold flags it
    ``LOW_YIELD`` — its recent FINISHED sessions burned ``>= turns_floor`` turns yet
    landed 0 ancestry-closes on its tree — AND it has ``>= min_sessions`` finished
    sessions (a trust floor, so a single unlucky session never demotes a lane). At
    most ``max_excludes`` lanes are returned, worst-by-turns first, so even a fully
    poisoned window cannot starve the picker (``lane_issue_numbers`` additionally
    re-seats the last eligible lane under ``low_yield_relief``).

    Only FINISHED sessions count: a still-live worker's mid-flight turns are dropped
    (its ``.pid`` sidecar reads live) so a lane is never demoted before its own
    worker has had the chance to close something. Fail-open throughout — any error
    (or a missing runs dir) yields an empty exclude set, and the ``git log`` join
    lives entirely inside the shared fold's default counter, run only for candidate
    lanes (``>= turns_floor`` turns). Returns
    ``{"exclude": set[str], "lanes": [evidence rows], "flagged": [lane names]}``.
    """
    import time
    result: dict[str, Any] = {"exclude": set(), "lanes": [], "flagged": []}
    try:
        if not runs_dir.is_dir():
            return result
        now = time.time() if now_ts is None else now_ts
        since_iso = time.strftime(
            "%Y-%m-%dT%H:%M:%S +0000",
            time.gmtime(now - lane_yield._LOW_YIELD_LOOKBACK_MIN * 60))

        def _closes(_lane: str, tree: list[str]) -> int | None:
            return lane_yield.count_lane_ancestry_closes(
                root, tree, since_iso=since_iso)

        def _finished(log: Path) -> bool:
            # Drop still-LIVE sessions: a live worker's mid-flight turns must not
            # flag its own lane before it has had the chance to close anything.
            return not dispatch_preflight.resolve_sidecar_pid_is_live(
                log.with_suffix(".pid"))

        fold = lane_yield.low_yield_lanes(
            runs_dir, closes_counter=_closes, lane_trees=lane_trees or {},
            now_ts=now, include_log=_finished)
        flagged = [
            r for r in fold.get("lanes", [])
            if r.get("verdict") == "LOW_YIELD"
            and int(r.get("sessions", 0)) >= min_sessions
            and r.get("tree_known")
        ]
        # Worst-by-turns first, then capped — never demote more than max_excludes.
        flagged.sort(key=lambda r: (-int(r.get("turns", 0)), str(r.get("lane"))))
        chosen = flagged[: max(0, max_excludes)]
        result["exclude"] = {str(r["lane"]) for r in chosen}
        result["lanes"] = [
            {"lane": str(r["lane"]), "turns": int(r.get("turns", 0)),
             "sessions": int(r.get("sessions", 0)), "closes": r.get("closes"),
             "evidence_logs": list(r.get("evidence_logs", []))}
            for r in chosen
        ]
        result["flagged"] = [str(r["lane"]) for r in flagged]
    except Exception:
        return {"exclude": set(), "lanes": [], "flagged": []}
    return result


def lane_issue_numbers(root: Path, explicit_lane: str | None,
                       exclude: set[str] | None = None,
                       guarded: bool | None = None,
                       soft_exclude: set[str] | None = None) -> dict[str, Any]:
    """Pick the lane (busiest, or explicit) and return its OPEN issue numbers,
    OLDEST first (ascending issue number -- GitHub issue numbers are assigned
    monotonically at creation, so the lowest number is the oldest open issue).
    ``gh issue list`` itself returns newest-first, and issue_lane_router.py's
    per-lane "issues" list inherits that native order; this fold explicitly
    reverses it rather than keeping the router's order, so the age-of-backlog
    priority is a decision this module makes on purpose, not an accident
    inherited from an unrelated API default. Reuses the same router fold
    issue_dispatch.pick_lane uses, but keeps the per-issue numbers (which
    pick_lane discards). ``exclude`` drops lanes from the busiest-pick (e.g.
    an opus task excludes 'docs' so the glm task owns it) -- ignored when an
    explicit lane is named.

    ``guarded`` (default: ``dispatch_worker.guard_enabled()``) HARD-EXCLUDES
    only the TRUST-CRITICAL lanes (``lane_core.lane_dispatchable_under_guard``,
    the Python mirror of the native Go path's LaneDispatchableUnderGuard) from
    the pick pool -- proactively, before any worker is spawned. The hold is the
    referee's own trees only: internal/{abi,kernel,adjudicator,policy,
    registrations,architest,shipgate} plus dos.toml/.dos/policy.json/VERSION,
    the witness machinery a self-improving worker must never grade its own
    homework by editing (#1397). Everything ELSE under cmd/**/internal/**
    (gateway, agent, compute, engine, model, ...) is guard-shippable and now
    DISPATCHABLE -- concurrent core work on the shared trunk is kept build-safe
    by the push-seam TRUNK_WOULD_NOT_COMPILE gate (cmd/fak/prepush_build.go via
    ``fak hooks pre-push``), not by holding the whole self-source tree. This
    replaces the historical BROAD self-source hold that left only the coarse
    docs/tools buckets pickable -- so the guarded loop ground docs while
    fragmented per-leaf core work starved behind the big buckets; the narrowed
    hold is the model the native Go dispatch path already validated. If every
    lane with open issues is trust-critical-held, ``lane`` comes back ``None``
    and the held lanes are named in ``self_modify_held``. An explicit lane is
    still honored verbatim regardless of guard state (operator intent overrides
    the guard, same as the Go path and issue_dispatch.pick_lane).

    Within the eligible (non-held, non-excluded) pool the lane is chosen by a
    VALUE weight, not raw open-issue count: a CORE self-source lane
    (``lane_core.is_core_source_lane_tree``) outranks the coarse docs/tools
    buckets, then the lane's strongest ``issue_triage`` priority score
    (P0=1000/P1=400/P2=150/default=60 -- ceremony notes/follow-ups carry the
    default), then open-issue count, then lane name -- a mirror of the Go
    corebias throughput ladder. ``core_lanes`` and ``lane_priority`` in the
    result expose the ranking inputs.

    ``soft_exclude`` (Part C, #2062) demotes lanes the low-yield detector
    flagged (>=40 turns / 0 closes) from the busiest-pick -- but SOFT, with a
    STARVATION FLOOR: if excluding them would leave zero eligible lanes they
    are seated back and reported in ``low_yield_relief`` instead, so a
    low-yield fold can never freeze the fleet. Orthogonal to the HARD
    ``exclude``/self-source skips and, like ``exclude``, ignored when an
    explicit lane is named (a deliberate operator override).

    The returned ``caps_by_issue`` maps each open issue number to the hardware
    capabilities it declares (from the router's flat per-issue ``required_caps``);
    ``evaluate`` uses it for the per-node capability gate (Part B)."""
    router = issue_dispatch.run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
        root, timeout=130)
    lanes = router.get("lanes") or {}
    guarded = dispatch_worker.guard_enabled() if guarded is None else guarded
    nums_by_lane: dict[str, list[int]] = {}
    trees: dict[str, Any] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        trees[ln] = info.get("tree") if isinstance(info, dict) else None
        nums: list[int] = []
        for it in (iss or []):
            n = it.get("number") if isinstance(it, dict) else it
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
        nums_by_lane[ln] = sorted(nums)  # oldest (lowest issue #) first
    # Part B: per-issue hardware capability, from the router's FLAT issue list
    # (the per-lane "issues" carry only numbers; required_caps rides the flat one).
    caps_by_issue: dict[int, list[str]] = {}
    for it in (router.get("issues") or []):
        if not isinstance(it, dict):
            continue
        try:
            num = int(it.get("number"))
        except (TypeError, ValueError):
            continue
        caps_by_issue[num] = [str(c).lower() for c in (it.get("required_caps") or []) if c]
    exclude = exclude or set()
    soft_exclude = soft_exclude or set()
    low_yield_excluded: list[str] = []
    low_yield_relief: list[str] = []
    core_lanes_out: list[str] = []
    lane_priority: dict[str, int] = {}
    by_lane_count = {k: len(v) for k, v in nums_by_lane.items()}
    if explicit_lane:
        chosen = explicit_lane
        held: list[str] = []
        # An explicit lane bounds the contract scan to that lane (operator intent).
        eligible_ranked = [[chosen, nums_by_lane.get(chosen, [])]]
    else:
        # HELD set narrowed from the BROAD self-source hold (all cmd/**/internal/**)
        # to the TRUST-CRITICAL referee only (mirror internal/dispatchtick/
        # selfmodify.go LaneDispatchableUnderGuard, ported via tools/lane_core.py).
        # A guarded worker may now ship the ~85% of self-source that is NOT the
        # referee's own trees (gateway, agent, compute, ...), kept build-safe on the
        # shared trunk by the push-seam TRUNK_WOULD_NOT_COMPILE gate; only the
        # trust-critical set stays HARD-held. See the docstring above.
        held_set = ({ln for ln in nums_by_lane
                     if not lane_core.lane_dispatchable_under_guard(True, trees.get(ln))}
                    if guarded else set())
        held = sorted(ln for ln in nums_by_lane if ln in held_set and nums_by_lane[ln])
        safe_open = {k: v for k, v in nums_by_lane.items() if k not in held_set and v}
        safe_lanes_busy = sorted(k for k in safe_open if k in exclude)
        eligible = {k: v for k, v in nums_by_lane.items()
                    if k not in exclude and k not in held_set}
        # Part C: apply the low-yield SOFT exclude with a STARVATION FLOOR --
        # demote flagged lanes only while >=1 eligible lane survives; otherwise
        # seat them back (relief) so the picker can never be starved to None by a
        # low-yield fold.
        soft_hits = sorted(k for k in soft_exclude if k in eligible)
        if soft_hits:
            survivors = {k: v for k, v in eligible.items() if k not in soft_exclude}
            if survivors:
                eligible = survivors
                low_yield_excluded = soft_hits
            else:
                low_yield_relief = soft_hits  # floor: keep the only lanes we have
        # VALUE-WEIGHTED pick, replacing the old busiest-by-count max. Mirror the Go
        # corebias throughput ladder (dispatch_tick_route.go dispatchLaneBetterForGoal):
        # a CORE self-source lane outranks the coarse docs/tools buckets, then the
        # lane's strongest issue_triage priority, then open-issue count, then lane
        # name. This is the "real work over ceremony" order -- core forward progress
        # leads volume. FAIL-OPEN: a gh/triage hiccup yields {} priorities and the key
        # degrades to (core, 0, count, name), never worse than the old count pick;
        # core-ness is a pure-local tree read that always holds.
        core_lanes = {ln for ln in eligible
                      if lane_core.is_core_source_lane_tree(trees.get(ln))}
        lane_priority = issue_dispatch.lane_priority_scores(
            root, {ln: list(v) for ln, v in eligible.items()})
        core_lanes_out = sorted(core_lanes)

        def _value_rank(kv: tuple[str, list[int]]) -> tuple[int, int, int, str]:
            ln, v = kv
            # DESCENDING on (core, priority, count) via negation; lane name ASCENDING
            # as the final deterministic tiebreak (matches the historical name break).
            return (-(1 if ln in core_lanes else 0),
                    -int(lane_priority.get(ln, 0)),
                    -len(v),
                    ln)

        eligible_ranked = [[k, v] for k, v in
                           sorted(eligible.items(), key=_value_rank)
                           if v]
        # chosen is the head of the value-ranked list, so the single pick and the
        # cross-lane contract-scan order agree (both name-deterministic on ties).
        chosen = eligible_ranked[0][0] if eligible_ranked else None
    return {"lane": chosen, "numbers": nums_by_lane.get(chosen or "", []),
            "by_lane_count": by_lane_count,
            "eligible_by_lane": eligible_ranked,
            "excluded_lanes": sorted(exclude),
            "safe_lanes_busy": safe_lanes_busy if not explicit_lane else [],
            "self_modify_held": held,
            "core_lanes": core_lanes_out,
            "lane_priority": lane_priority,
            "caps_by_issue": caps_by_issue,
            "low_yield_excluded": low_yield_excluded,
            "low_yield_relief": low_yield_relief,
            "router_error": router.get("_error")}


def live_resolution_issues(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> set[int]:
    """Issue numbers that already have a LIVE resolution worker — read from the
    dispatch logs (``resolve-<N>-<stamp>.log``) whose pid is still alive. Best
    effort: a log without a recoverable pid is treated as not-live."""
    live: set[int] = set()
    if not runs_dir.is_dir():
        return live
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore
            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        pid_file = log.with_suffix(".pid")
        if pid_file.exists():
            if dispatch_preflight.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
                live.add(int(m.group(1)))
    return live


def _spawn_header_lane(log: Path) -> str | None:
    """Parse ``lane=<L>`` from a worker log's ``# fak-spawn`` header — its first
    line, flushed before exec by :func:`spawn_issue_worker`. Returns ``None`` when
    the log is unreadable or has no recoverable lane field (best effort)."""
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
    except OSError:
        return None
    if not head.startswith("# fak-spawn"):
        return None
    m = _SPAWN_LANE_RE.search(head)
    return m.group(1) if m else None


def live_resolution_lanes(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> set[str]:
    """Lanes that already hold a LIVE resolution worker — the pre-spawn collision
    set (#1310). fak's lane taxonomy is ONE worker per leaf tree (``dos.toml``: a
    named lane serializes even on disjoint sub-trees), so a second worker on a
    held lane co-edits the same files. Read the ``lane=`` field the spawn header
    stamps into each ``resolve-<N>-<stamp>.log`` and keep only the lanes whose
    worker pid is still alive — the SAME identity gate as
    :func:`live_resolution_issues`. Best effort: a log we cannot parse, or whose
    pid is gone, contributes no lane. A lane whose worker log is a terminal
    banner no-op (#1275/#1398) is also dropped — see the inline note below."""
    lanes: set[str] = set()
    if not runs_dir.is_dir():
        return lanes
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore
            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None
    for log in runs_dir.glob("resolve-*.log"):
        if not _LOG_ISSUE_RE.search(log.name):
            continue
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
            continue
        lane = _spawn_header_lane(log)
        if not lane:
            continue
        # A worker whose log is a terminal banner no-op (#1275: it printed only its
        # startup banner — "> build · glm-…" — and produced nothing) holds no real
        # work even when its pid still passes the liveness gate. An opencode worker
        # runs as a ``node`` image, so AFTER it exits a recycled ``node`` pid that
        # lands in the sidecar's spawn window passes the weak create-time fallback
        # and would pin the lane FOREVER (#1398: ``docs`` stayed LANE_BUSY behind
        # dead 122-byte no-ops while real docs work could not dispatch). Drop such a
        # lane so a lane held ONLY by dead no-op workers reports FREE. This is safe:
        # a genuinely live worker streams kilobytes past the stub floor within
        # seconds so it never classifies as a banner no-op, and on a LIVE tick the
        # authoritative fenced git-ref lease (:func:`acquire_lane_lease`) still
        # serializes a just-started worker across hosts.
        if classify_no_commit_reason(log) == NO_COMMIT_BANNER_NOOP:
            continue
        lanes.add(lane)
    return lanes


def recently_attempted_issues(runs_dir: Path, *, cooldown_min: int,
                              now_ts: float | None = None) -> set[int]:
    """Issue numbers attempted within the last ``cooldown_min`` minutes — read from
    the mtime of their ``resolve-<N>-*.log`` or ``.witness`` sidecar. This is the
    anti-churn gate: a hard issue (e.g. a mislabeled epic) that a worker could not
    land must NOT be re-picked every tick — re-dispatching it re-storms a known
    drain while the rest of the lane's backlog goes untouched. A witnessed dead slot
    may have already lost its ``.log`` to sidecar cleanup, so the durable witness
    sidecar also cools the issue. After the cooldown it becomes eligible again (the
    repo may have changed, or a fresh worker may get further). 0 disables the gate."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    attempts = (*runs_dir.glob("resolve-*.log"),
                *runs_dir.glob(f"resolve-*{WITNESS_SIDECAR_SUFFIX}"))
    for attempt in attempts:
        m = _LOG_ISSUE_RE.search(attempt.name)
        if not m:
            continue
        try:
            if attempt.stat().st_mtime >= horizon:
                recent.add(int(m.group(1)))
        except OSError:
            continue
    return recent


def live_repair_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Contract-repair workers still ALIVE (``repair-<N>-<stamp>.pid``,
    identity-gated exactly like the resolve sidecars). The dispatcher admits at
    most ONE repair worker at a time -- it holds no lane lease (it edits GitHub
    issues, not repo files), so this scan is its serializer."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for pid_file in sorted(runs_dir.glob("repair-*.pid")):
        m = _REPAIR_LOG_RE.search(pid_file.name)
        if not m:
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
            continue
        try:
            pid: int | None = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        out.append({"pid": pid, "pid_file": pid_file.name, "issue": int(m.group(1))})
    return out


def recently_repaired_issues(runs_dir: Path, *, cooldown_min: int,
                             now_ts: float | None = None) -> set[int]:
    """Issues a contract-repair worker ATTEMPTED within the window -- the mtime of
    each ``repair-<N>-*.log`` plus every issue in its ``.issues`` batch sidecar.
    Anti-churn for un-repairables: an issue a worker could not honestly bring to
    contract must not be re-groomed every tick. A SUCCESSFUL repair needs no
    escape hatch here -- its edit bumps updatedAt, which re-admits the issue past
    the hold ledger, and a passing review never reaches the repair path again.
    0 disables the gate."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    for log in runs_dir.glob("repair-*.log"):
        m = _REPAIR_LOG_RE.search(log.name)
        if not m:
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        recent.add(int(m.group(1)))
        try:
            extra = log.with_suffix(REPAIR_ISSUES_SIDECAR_SUFFIX).read_text(
                encoding="utf-8")
            recent.update(int(x) for x in extra.split(",") if x.strip())
        except (OSError, ValueError):
            pass
    return recent


def terminate_issue_worker_tree(pid: int) -> dict[str, Any]:
    """Kill one detached issue worker and its descendants.

    The resolver's Windows path launches a hidden-console cmd/opencode tree, so
    `taskkill /T` is the honest process-tree reaper. On POSIX the worker is
    spawned with `start_new_session=True`, making its pid the process-group root.
    """
    if pid <= 0:
        return {"ok": False, "returncode": None, "error": "invalid pid"}
    try:
        if os.name == "nt":
            proc = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                  capture_output=True, text=True, timeout=30,
                                  creationflags=no_window_creationflags())
            return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                    "stdout": (proc.stdout or "").strip()[-500:],
                    "stderr": (proc.stderr or "").strip()[-500:]}
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return {"ok": True, "returncode": 0}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "error": str(exc)}


def reap_timed_out_workers(
    runs_dir: Path,
    *,
    timeout_s: int | None,
    live: bool,
    now_ts: float | None = None,
    probe: Any | None = None,
    killer: Any | None = None,
) -> dict[str, Any]:
    """Find resolver workers older than the wall-clock cap and optionally reap.

    This is deliberately sidecar-identity gated: a stale `.pid` file can only
    nominate a process if `dispatch_preflight.resolve_sidecar_pid_is_live`
    proves the current PID still belongs to that sidecar. Dry-run reports
    `would_reap`; the scheduled live tick kills the worker tree before preflight
    counts capacity.
    """
    if not timeout_s or timeout_s <= 0:
        return {"timeout_s": timeout_s, "live": live, "candidates": [],
                "reaped": [], "would_reap": [], "disabled": True}
    import time
    now = now_ts if now_ts is not None else time.time()
    kill = killer or terminate_issue_worker_tree
    candidates: list[dict[str, Any]] = []
    reaped: list[dict[str, Any]] = []
    would_reap: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return {"timeout_s": timeout_s, "live": live, "candidates": [],
                "reaped": [], "would_reap": []}
    for pid_file in sorted((*runs_dir.glob("resolve-*.pid"),
                            *runs_dir.glob("repair-*.pid"))):
        m = _ANY_WORKER_LOG_RE.search(pid_file.name)
        if not m:
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            st = pid_file.stat()
        except (OSError, ValueError):
            continue
        age_s = max(0.0, now - st.st_mtime)
        if age_s < timeout_s:
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(pid_file, probe=probe):
            continue
        rec: dict[str, Any] = {
            "issue": int(m.group(1)),
            "pid": pid,
            "pid_file": pid_file.name,
            "log": pid_file.with_suffix(".log").name,
            "age_s": round(age_s, 1),
        }
        candidates.append(dict(rec))
        if live:
            outcome = kill(pid)
            rec["kill"] = outcome
            reaped.append(rec)
        else:
            would_reap.append(rec)
    return {"timeout_s": timeout_s, "live": live, "candidates": candidates,
            "reaped": reaped, "would_reap": would_reap}


def prune_dead_sidecars(
    runs_dir: Path,
    *,
    live: bool,
    min_age_s: float = 60.0,
    now_ts: float | None = None,
    probe: Any | None = None,
) -> dict[str, Any]:
    """Sweep ``resolve-*.pid`` sidecars whose worker is no longer alive.

    A worker that exits NORMALLY leaves its ``.pid`` (and sibling ``.log`` /
    ``.backend`` / ``.wave`` / ``.account`` / ``.lease``) sidecars behind forever — the reaper
    only KILLS live runaways, it never deletes the corpses. Left alone they pile
    up (379 were found on one host) and become landmines for the create-time
    spawn-window liveness fallback: a recycled PID landing in a stale sidecar's
    window was miscounted as a live worker, pinning the dispatcher at cap against
    ghosts. Sweeping every tick keeps the worker count the preflight reads honest
    and the directory bounded.

    A sidecar is pruned only when its identity-gated liveness check says the
    process is gone, and only once it is at least ``min_age_s`` old (so a sidecar
    written microseconds ago, before its child has fully materialised, is never
    swept out from under a just-spawned worker). Dry-run reports ``would_prune``;
    ``live`` actually unlinks. The matching non-``.pid`` process metadata sidecars
    are removed with the ``.pid`` so a half-pruned run never confuses the wave
    auditor; resolve ``.log`` files and ``.witness`` records are retained as the
    durable transcript/cooldown evidence. Repair logs remain disposable and are
    swept with their repair-only sidecars.
    """
    import time
    now = now_ts if now_ts is not None else time.time()
    pruned: list[str] = []
    would_prune: list[str] = []
    if not runs_dir.is_dir():
        return {"live": live, "pruned": pruned, "would_prune": would_prune}
    resolve_sibling_suffixes = (
        ".backend", ".wave", ".account", LEASE_SIDECAR_SUFFIX,
        BASE_SHA_SIDECAR_SUFFIX,
    )
    repair_sibling_suffixes = (
        ".log", ".backend", ".account", LEASE_SIDECAR_SUFFIX,
        BASE_SHA_SIDECAR_SUFFIX, REPAIR_ISSUES_SIDECAR_SUFFIX,
    )
    for pid_file in sorted((*runs_dir.glob("resolve-*.pid"),
                            *runs_dir.glob("repair-*.pid"))):
        if not _ANY_WORKER_LOG_RE.search(pid_file.name):
            continue
        try:
            age_s = max(0.0, now - pid_file.stat().st_mtime)
        except OSError:
            continue
        if age_s < min_age_s:
            continue
        if dispatch_preflight.resolve_sidecar_pid_is_live(pid_file, probe=probe):
            continue
        if not live:
            would_prune.append(pid_file.name)
            continue
        stem = pid_file.with_suffix("")
        try:
            pid_file.unlink()
            pruned.append(pid_file.name)
        except OSError:
            continue
        sibling_suffixes = (
            repair_sibling_suffixes
            if pid_file.name.startswith("repair-")
            else resolve_sibling_suffixes
        )
        for suf in sibling_suffixes:
            sib = stem.with_suffix(suf)
            try:
                sib.unlink()
            except OSError:
                pass
    return {"live": live, "pruned": pruned, "would_prune": would_prune}


def pick_target_issue(numbers: list[int], skip: set[int]) -> int | None:
    """The first lane issue not in ``skip`` (live workers ∪ recently-attempted)."""
    for n in numbers:
        if n not in skip:
            return n
    return None


# Priority-label weights for the dispatchorder Candidate.priority field (internal/dispatchorder
# dispatchorder.go). These mirror tools/issue_triage.py's PRIORITY map and internal/dispatchtick's
# PriorityWeight* so the picker, the triage scorer, and the dispatch-order leaf never disagree
# about how heavy a priority/P* label is. Higher == do-first; the leaf sorts kept units
# priority-desc first, then by recency.
DISPATCH_PRIORITY_WEIGHT = {"priority/P0": 1000, "priority/P1": 400, "priority/P2": 150}


def candidate_priority(labels: Any) -> int:
    """The dispatchorder Candidate.priority integer an issue earns from its labels: the HEAVIEST
    priority/P* label it carries (P0 > P1 > P2), or 0 when it carries none. Accepts labels as
    ``gh``'s list of ``{"name": ...}`` dicts or plain strings; unknown entries are ignored.

    An unlabeled unit maps to 0 -- dispatchorder's documented "unknown/lowest" -- so an
    all-unlabeled candidate set carries priority 0 and orders by recency exactly as it did before
    the field existed (the additive-no-regression posture #3222 requires)."""
    best = 0
    for lab in labels or []:
        name = lab.get("name") if isinstance(lab, dict) else lab
        weight = DISPATCH_PRIORITY_WEIGHT.get(str(name or "").strip(), 0)
        if weight > best:
            best = weight
    return best


# A "depends-on:/blocked-by: #N" prerequisite marker in an issue body, followed by one or more
# "#N" references (comma/and/&-separated). The dispatchorder leaf (internal/dispatchorder) holds a
# keep-eligible unit while any prerequisite it names is still an OPEN candidate this tick.
_PREREQ_MARKER = re.compile(
    r"(?im)\b(?:depends[ -]?on|blocked[ -]?by)\b[:\s]*"
    r"((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)")
_ISSUE_REF = re.compile(r"#(\d+)")


def candidate_blocked_by(body: Any) -> list[str]:
    """The dispatchorder Candidate.blocked_by list an issue earns from its body: the issue numbers
    it declares as prerequisites via a "depends-on:/blocked-by: #N" marker (one or many, comma/
    and/&-separated), as string IDs in the leaf's ID space. Prose that merely contains the words
    ("it depends on the weather") never matches -- the marker must be immediately followed by a
    ``#N`` reference.

    An issue with no such marker maps to an empty list -- dispatchorder's "unique/never blocked"
    -- so a body-free or marker-free candidate set carries no prerequisite edges and dispatches
    exactly as it did before the field existed (the additive-no-regression posture #3224 requires).
    The leaf fails open on a prerequisite that is already closed, so a stale ``#N`` never wedges
    the dependent. Order-preserving and de-duplicated, so the result is deterministic per body."""
    text = body if isinstance(body, str) else ""
    out: list[str] = []
    seen: set[str] = set()
    for marker in _PREREQ_MARKER.finditer(text):
        for num in _ISSUE_REF.findall(marker.group(1)):
            if num not in seen:
                seen.add(num)
                out.append(num)
    return out


def contract_scan_stream(eligible_by_lane: list[Any] | None,
                         skip: set[int]) -> list[tuple[str, int]]:
    """The cross-lane candidate stream the bounded contract scan walks: round-robin
    ACROSS eligible lanes (rank order preserved: busiest first) and oldest-first
    WITHIN a lane, skipping live/cooled/held issues. This is the lane-level half of
    the head-of-line-blocking fix: with a lane-local scan, one fat lane whose head
    stratum is all pre-contract-era thin issues (docs: 232) starves every other
    lane's READY issues out of the scan window for ticks on end; interleaving gives
    each eligible lane's oldest candidate a look inside a single window. Pure:
    plain data in, [(lane, issue), ...] out."""
    queues: list[tuple[str, list[int]]] = []
    for entry in eligible_by_lane or []:
        try:
            lane, numbers = entry[0], entry[1]
        except (TypeError, IndexError, KeyError):
            continue
        nums = [n for n in (numbers or []) if n not in skip]
        if lane and nums:
            queues.append((str(lane), nums))
    stream: list[tuple[str, int]] = []
    depth = 0
    while any(depth < len(nums) for _, nums in queues):
        for lane, nums in queues:
            if depth < len(nums):
                stream.append((lane, nums[depth]))
        depth += 1
    return stream


def contract_held_issues(runs_dir: Path, *, ttl_h: int = DEFAULT_CONTRACT_HOLD_TTL_H,
                         now_ts: float | None = None,
                         updated_ts: dict[int, float] | None = None) -> set[int]:
    """Issue numbers whose contract review HELD within the last ``ttl_h`` hours --
    read from the ledger live ticks append via ``record_contract_holds``. This is
    what makes the bounded contract scan CUMULATIVE: with an oldest-first pick, a
    deep stratum of pre-contract-era issues at the head of a lane would otherwise be
    re-reviewed identically every tick and the scan window would never advance (the
    #1207 wedge: 8 reviews per tick, the same 8 thin heads, forever). Skipping
    recently-held issues moves the window ~scan-budget issues per tick, so a
    hundreds-deep thin backlog is swept in hours. After the TTL an issue is
    re-reviewed, so a contract backfill re-enters it with no manual reset.
    0 disables the gate.

    ``updated_ts`` ({issue: GitHub updatedAt epoch}, from ``open_issue_updated_map``)
    re-admits an issue whose body changed AFTER its latest held verdict without
    waiting out the TTL -- the turnaround that makes the contract-REPAIR pipeline
    work: a repair worker's edit (or a human backfill) bumps updatedAt, so the very
    next tick re-reviews the issue instead of skipping it for a day. Fail-open in
    the conservative direction: no updated timestamp for an issue -> it stays held."""
    return {int(r.get("issue") or 0) for r in contract_held_records(
        runs_dir, ttl_h=ttl_h, now_ts=now_ts, updated_ts=updated_ts)}


def _missing_fields_from_hold_reason(reason: str) -> list[str]:
    out: list[str] = []
    for token in reason.split(","):
        token = token.strip()
        if token.startswith("missing:"):
            field = token.split(":", 1)[1].strip()
            if field:
                out.append(field)
    return out


def contract_held_records(
    runs_dir: Path, *, ttl_h: int = DEFAULT_CONTRACT_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
    only: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Recent contract-held ledger rows, deduped to the latest verdict per issue.

    ``contract_held_issues`` uses the same reader for its skip set. The richer
    rows feed the repair-worker arm when the skip set itself emptied the picker:
    those issues are still work, just grooming work instead of resolution work.
    """
    if ttl_h <= 0:
        return []
    ledger = runs_dir / _CONTRACT_HOLD_LEDGER
    if not ledger.is_file():
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    held: dict[int, tuple[float, int, dict[str, Any]]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    only_set = {int(n) for n in only} if only is not None else None
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
            ts = float(row.get("ts") or 0)
            if ts >= horizon:
                n = int(row.get("issue"))
                if only_set is not None and n not in only_set:
                    continue
                if updated_ts and (updated_ts.get(n) or 0.0) > ts:
                    continue
                reason = str(row.get("reason") or "")
                missing = row.get("missing_fields")
                if not isinstance(missing, list):
                    missing = _missing_fields_from_hold_reason(reason)
                rec = {
                    "issue": n,
                    "number": n,
                    "score": int(row.get("score") or 0),
                    "reason": reason,
                    "title": str(row.get("title") or "")[:200],
                    "missing_fields": [str(m) for m in missing if m],
                }
                prev = held.get(n)
                if prev is None or ts > prev[0] or (ts == prev[0] and idx > prev[1]):
                    held[n] = (ts, idx, rec)
        except (TypeError, ValueError):
            continue
    return [rec for _ts, _idx, rec in sorted(held.values(),
                                            key=lambda item: (item[0], item[1]))]


def record_contract_holds(runs_dir: Path, rows: list[dict[str, Any]], *,
                          live: bool, now_ts: float | None = None) -> None:
    """Append this tick's contract-HELD issues to the skip ledger. Live ticks only:
    a dry-run must stay side-effect-free (the payload still reports the holds)."""
    if not live or not rows:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _CONTRACT_HOLD_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "utc": iso, "ts": now, "issue": int(r.get("issue") or 0),
                    "score": int(r.get("score") or 0),
                    "reason": str(r.get("reason") or "")[:200],
                    "title": str(r.get("title") or "")[:200],
                    "missing_fields": [str(m) for m in
                                       (r.get("missing_fields") or []) if m][:24],
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass  # fail-open: a missed skip re-reviews next tick, never blocks one


def record_contract_forced_bypass(runs_dir: Path, row: dict[str, Any], *,
                                  live: bool, now_ts: float | None = None) -> None:
    """Append one operator-forced contract-gate bypass to the audit ledger (#2637).

    Live ticks only: a dry-run stays side-effect-free (the payload still reports the
    would-be bypass). Each row carries the operator's structured ``--force-reason``
    alongside the gate's own hold reason, so the ledger answers *why* the readiness
    guard was overridden, not just that a spawn carried ``bypassed=true``."""
    if not live or not row:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _CONTRACT_FORCED_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "utc": iso, "ts": now,
                "issue": int(row.get("issue") or 0),
                "lane": str(row.get("lane") or "")[:80],
                "score": int(row.get("score") or 0),
                "reason": str(row.get("reason") or "")[:300],
                "gate_reason": str(row.get("gate_reason") or "")[:300],
                "title": str(row.get("title") or "")[:200],
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass  # fail-open: a missed audit row never blocks a live tick


def contract_forced_bypass_count(runs_dir: Path) -> int:
    """Total operator-forced contract-gate bypasses recorded so far (#2637) — the
    operator-visible tally of how often the readiness guard has been overridden.
    Fail-open: an absent or unreadable ledger reports 0."""
    ledger = runs_dir / _CONTRACT_FORCED_LEDGER
    if not ledger.is_file():
        return 0
    try:
        return sum(1 for line in
                   ledger.read_text(encoding="utf-8", errors="replace").splitlines()
                   if line.strip())
    except OSError:
        return 0


def multi_lane_held_records(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_MULTI_LANE_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
    only: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Recent MULTI_LANE_SCOPE rows, deduped to the latest verdict per issue."""
    if ttl_h <= 0:
        return []
    ledger = runs_dir / _MULTI_LANE_HOLD_LEDGER
    if not ledger.is_file():
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    held: dict[int, tuple[float, int, dict[str, Any]]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    only_set = {int(n) for n in only} if only is not None else None
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
            ts = float(row.get("ts") or 0)
            if ts < horizon:
                continue
            n = int(row.get("issue"))
            if only_set is not None and n not in only_set:
                continue
            if updated_ts and (updated_ts.get(n) or 0.0) > ts:
                continue
            uncovered_lanes = row.get("uncovered_lanes") or []
            rec = {
                "issue": n,
                "number": n,
                "lane": str(row.get("lane") or ""),
                "title": str(row.get("title") or "")[:200],
                "reason": str(row.get("reason") or "")[:240],
                "uncovered_lanes": [str(ln) for ln in uncovered_lanes if ln][:24],
            }
            prev = held.get(n)
            if prev is None or ts > prev[0] or (ts == prev[0] and idx > prev[1]):
                held[n] = (ts, idx, rec)
        except (TypeError, ValueError):
            continue
    return [rec for _ts, _idx, rec in sorted(held.values(),
                                            key=lambda item: (item[0], item[1]))]


def multi_lane_held_issues(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_MULTI_LANE_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
) -> set[int]:
    """Issue numbers held for operator split/reroute after MULTI_LANE_SCOPE."""
    return {int(r.get("issue") or 0) for r in multi_lane_held_records(
        runs_dir, ttl_h=ttl_h, now_ts=now_ts, updated_ts=updated_ts)}


def record_multi_lane_holds(runs_dir: Path, rows: list[dict[str, Any]], *,
                            live: bool, now_ts: float | None = None) -> None:
    """Append live MULTI_LANE_SCOPE holds so the next tick advances."""
    if not live or not rows:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _MULTI_LANE_HOLD_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "utc": iso,
                    "ts": now,
                    "issue": int(r.get("issue") or 0),
                    "lane": str(r.get("lane") or "")[:80],
                    "title": str(r.get("title") or "")[:200],
                    "reason": str(r.get("reason") or "")[:240],
                    "uncovered_lanes": [str(ln) for ln in
                                        (r.get("uncovered_lanes") or []) if ln][:24],
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass  # fail-open: a missed skip repeats one refusal, never blocks work


def collision_held_records(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_COLLISION_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
    only: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Recent DIRTY_PATH_COLLISION / SAME_ISSUE_WIP holds, deduped to the latest
    verdict per issue. Mirrors ``multi_lane_held_records`` over its own ledger so a
    still-colliding issue leaves the candidate stream instead of being re-refused
    every tick. A local-tree collision (``dirty_path`` / ``same_issue_wip``) clears
    only on a LOCAL commit/revert, so it is NOT re-admitted on a remote ``updatedAt``
    bump -- that void re-entered the same colliding head every tick while the local
    dirty condition was unchanged. Any OTHER (content-keyed) hold is still re-admitted
    early when the issue's gh ``updatedAt`` is newer than the hold, and the advance
    loop re-runs the live collision check on the re-admitted head. The 3h TTL bounds
    every hold regardless."""
    if ttl_h <= 0:
        return []
    ledger = runs_dir / _COLLISION_HOLD_LEDGER
    if not ledger.is_file():
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    held: dict[int, tuple[float, int, dict[str, Any]]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    only_set = {int(n) for n in only} if only is not None else None
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
            ts = float(row.get("ts") or 0)
            if ts < horizon:
                continue
            n = int(row.get("issue"))
            if only_set is not None and n not in only_set:
                continue
            kind = str(row.get("kind") or "")
            # A local-tree collision (dirty_path / same_issue_wip) clears on a LOCAL
            # commit or revert, not on a remote body edit. Re-admitting such a hold
            # just because gh ``updatedAt`` bumped voids the hold while the local
            # dirty condition is unchanged -- the head re-enters the pool and
            # re-collides every tick (the #3045-style retry loop). Gate the early
            # ``updated_ts`` re-admit to holds whose verdict a body edit can actually
            # change (contract / multi-lane-scope live on their own ledgers, but a
            # future collision kind keyed on issue CONTENT belongs here); the 3h TTL
            # still expires local-collision holds, and the advance loop re-runs the
            # live collision check on anything it does re-admit.
            _local_collision = kind in ("dirty_path", "same_issue_wip")
            if (updated_ts and not _local_collision
                    and (updated_ts.get(n) or 0.0) > ts):
                continue
            rec = {
                "issue": n,
                "number": n,
                "kind": kind,
                # Every row on THIS ledger is working-tree co-tenancy by
                # construction; default rows written before #4321 to that class so
                # a reader never has to special-case the backfill.
                "refusal_class": str(row.get("refusal_class")
                                     or REFUSAL_CLASS_WORKTREE_COTENANCY),
                "lane": str(row.get("lane") or ""),
                "title": str(row.get("title") or "")[:200],
                "reason": str(row.get("reason") or "")[:240],
                "dirty_paths": [str(p) for p in (row.get("dirty_paths") or []) if p][:24],
            }
            prev = held.get(n)
            if prev is None or ts > prev[0] or (ts == prev[0] and idx > prev[1]):
                held[n] = (ts, idx, rec)
        except (TypeError, ValueError):
            continue
    return [rec for _ts, _idx, rec in sorted(held.values(),
                                            key=lambda item: (item[0], item[1]))]


def collision_held_issues(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_COLLISION_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
) -> set[int]:
    """Issue numbers held after a dirty-path / same-issue-WIP collision this window."""
    return {int(r.get("issue") or 0) for r in collision_held_records(
        runs_dir, ttl_h=ttl_h, now_ts=now_ts, updated_ts=updated_ts)}


def record_collision_holds(runs_dir: Path, rows: list[dict[str, Any]], *,
                           live: bool, now_ts: float | None = None) -> None:
    """Append live dirty-path / same-issue-WIP collision holds so the next tick's
    scan starts past them. Fail-open (a missed skip repeats one refusal, never
    blocks work), and dry-run ticks (``live`` false) never touch the ledger."""
    if not live or not rows:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _COLLISION_HOLD_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            for r in rows:
                issue = int(r.get("issue") or 0)
                if issue <= 0:
                    continue
                f.write(json.dumps({
                    "utc": iso,
                    "ts": now,
                    "issue": issue,
                    "kind": str(r.get("kind") or "")[:40],
                    # #4321: the contention CLASS, so a ledger reader never folds a
                    # working-tree co-tenancy refusal in with a lane-lease one.
                    "refusal_class": REFUSAL_CLASS_WORKTREE_COTENANCY,
                    "lane": str(r.get("lane") or "")[:80],
                    "title": str(r.get("title") or "")[:200],
                    "reason": str(r.get("reason") or "")[:240],
                    "dirty_paths": [str(p) for p in
                                    (r.get("dirty_paths") or []) if p][:24],
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass  # fail-open: a missed skip repeats one refusal, never blocks work


def _parse_gh_timestamp(value: Any) -> float | None:
    """Epoch seconds from a gh ISO-8601 timestamp (``2026-07-02T18:33:12Z``);
    None on anything unparseable (fail-open, the caller keeps the issue held)."""
    s = str(value or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def open_issue_updated_map(root: Path, *, cap: int = 1000,
                           runner: Any = subprocess.run) -> dict[int, float]:
    """``{number: updatedAt-epoch}`` for open issues -- ONE bulk ``gh issue list``
    call per tick. Feeds ``contract_held_issues``'s re-admission check so a
    contract backfill (a repair worker's edit, or a human's) re-enters the pick
    on the NEXT tick instead of waiting out the hold TTL. Fail-open: any error
    returns {} (no re-admission that tick; the TTL stays the backstop)."""
    try:
        proc = runner(
            ["gh", "issue", "list", "--state", "open", "--limit", str(cap),
             "--json", "number,updatedAt"],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, creationflags=no_window_creationflags())
        rows = json.loads(proc.stdout or "[]") if proc.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    out: dict[int, float] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            n = int(row.get("number"))
        except (TypeError, ValueError):
            continue
        ts = _parse_gh_timestamp(row.get("updatedAt"))
        if ts is not None:
            out[n] = ts
    return out


def build_worker_command(backend: str, prompt: str, model: str | None,
                         effort: str | None = None, ultracode: bool = False) -> list[str]:
    """The argv for one detached issue-resolution worker, per backend. Both forms
    run the SAME issue-resolution prompt (with its ``#N``-in-subject rule), so the
    resulting commit is witnessable by the closure auditor regardless of backend.

    ``model``/``effort`` pin the claude worker's model and reasoning effort via the
    CLI flags (``--model``/``--effort``, both real in Claude Code >=2.1) — the only
    way to apply ``xhigh``, which has no ``settings.json`` equivalent.

    ``ultracode`` puts the claude worker in ultracode (xhigh reasoning PLUS dynamic
    multi-agent workflow orchestration) via ``--settings {"ultracode":true}``. It is
    mutually exclusive with ``effort`` on emit — ultracode already implies xhigh — so
    when set it wins and ``--effort`` is suppressed. This mirrors the Go
    ``dispatchtick.BuildWorkerCommand`` emit order exactly (model, then
    ultracode-xor-effort, then prompt) so the two launch surfaces stay parity."""
    if backend == "claude":
        cmd = ["claude", "-p", "--permission-mode", "bypassPermissions"]
        if model:
            cmd += ["--model", model]  # pin the exact model (reproducible/traced)
        if ultracode:
            # xhigh reasoning + multi-agent orchestration; session-only, no settings.json field
            cmd += ["--settings", tier_launch.ULTRACODE_SETTINGS_ARG]
        elif effort:
            cmd += ["--effort", effort]  # reasoning effort; no settings.json field
        # Detached Claude workers receive the prompt on stdin in spawn_issue_worker.
        # Keeping it out of argv avoids Windows CreateProcess's command-line limit.
        return cmd
    if backend == "opencode":
        # Keep the full dispatch prompt out of argv. The live spawn path writes it
        # beside the log and attaches it with ``--file``; this short notice keeps
        # process listings small while preserving the issue-worker liveness marker.
        # Detached issue workers should not inherit account/project MCP tools; the
        # built-in tool path still runs through fak guard, and --pure avoids a model
        # spending the first turn on plugin tools that the guard will refuse.
        cmd = ["opencode", "run", "--pure", "--print-logs",
               "--dangerously-skip-permissions"]
        if model:
            cmd += ["-m", model]  # pin the exact model so the run is reproducible/traced
        cmd.append(OPENCODE_PROMPT_NOTICE)
        return cmd
    if backend == "codex":
        # `codex exec` is the headless analogue of `claude -p` / `opencode run`;
        # --dangerously-bypass-approvals-and-sandbox is the full-access mode (we run
        # in the repo, already externally bounded by fak guard + the trunk guard).
        # --skip-git-repo-check keeps it from refusing in a worktree edge case.
        cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
               "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
        cmd.append(prompt)
        return cmd
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def _command_basename(path: str) -> str:
    """Return a command basename for POSIX or Windows-shaped paths."""
    return os.path.basename(str(path).replace("\\", "/")).lower()


def unwrap_opencode_npm_shim(exe: str) -> str:
    """Return the real opencode executable behind the Windows npm shim, if present."""
    if _command_basename(exe) not in {"opencode", "opencode.exe", "opencode.cmd",
                                      "opencode.bat", "opencode.ps1"}:
        return ""
    raw = os.fspath(exe)
    parent_idx = max(raw.rfind("/"), raw.rfind("\\"))
    if parent_idx < 0:
        return ""
    parent = raw[:parent_idx]
    if parent in ("", "."):
        return ""
    sep = raw[parent_idx]
    target = sep.join((parent, "node_modules", "opencode-ai", "bin", "opencode.exe"))
    try:
        return target if os.path.isfile(target) else ""
    except OSError:
        return ""


def resolve_worker_executable(backend: str, name: str) -> str:
    exe = shutil.which(name) or name
    if backend == "opencode" and os.name == "nt":
        target = unwrap_opencode_npm_shim(exe)
        if target:
            return target
    return exe


def attach_opencode_prompt_file(command: list[str], prompt_path: Path | str) -> list[str]:
    """Attach the staged dispatch prompt while keeping the final notice as text."""
    out = list(command)
    prompt = str(prompt_path)
    if not out or not prompt.strip():
        return out
    if len(out) == 1:
        return [*out, "--file", prompt]
    last = out[-1]
    return [*out[:-1], "--file", prompt, "--", last]


def resolve_opencode_command(command: list[str]) -> list[str]:
    """Resolve the opencode token even when it is nested behind ``fak guard --``."""
    out = list(command)
    for idx, token in enumerate(out):
        if _command_basename(token) in {"opencode", "opencode.exe", "opencode.cmd",
                                        "opencode.bat", "opencode.ps1"} \
                and idx + 1 < len(out) and out[idx + 1] == "run":
            out[idx] = resolve_worker_executable("opencode", token)
            break
    return out


def _opencode_config_home(account_dir: str, runs_dir: Path) -> str:
    """opencode loads its config from ``$XDG_CONFIG_HOME/opencode``. The switcher's
    account dir is either the canonical ``<config_home>/opencode`` (the default
    account) or a sibling ``<config_home>/opencode-<tag>``. For the canonical dir the
    parent IS the XDG home; for a sibling we mint a tiny pin dir whose ``opencode``
    entry is a junction to the account, so opencode loads exactly that account -- we
    never silently mis-attribute a run to the wrong account."""
    p = Path(account_dir)
    if p.name == "opencode":
        return str(p.parent)
    pin = runs_dir / ".opencode-pins" / p.name
    link = pin / "opencode"
    pin.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        try:
            os.symlink(account_dir, link, target_is_directory=True)
        except OSError:
            # Windows without symlink privilege: a directory junction needs none.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), account_dir],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           creationflags=no_window_creationflags())
    if not link.exists():
        raise RuntimeError(f"could not pin opencode account dir {account_dir!r}")
    return str(pin)


def _github_cli_config_dir(base: dict[str, str] | None = None) -> str | None:
    """Return the ambient gh config dir, if one exists.

    opencode account pinning uses XDG_CONFIG_HOME. GitHub CLI also consults that
    variable unless GH_CONFIG_DIR is explicit, so preserve the normal gh config
    path before the worker's model-account pin hides it.
    """
    e = base if base is not None else os.environ
    explicit = e.get("GH_CONFIG_DIR")
    if explicit:
        return explicit
    candidates: list[Path] = []
    appdata = e.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "GitHub CLI")
    home = e.get("HOME") or e.get("USERPROFILE")
    if home:
        candidates.append(Path(home) / ".config" / "gh")
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _opencode_temp_dir(runs_dir: Path, lane: str) -> Path:
    safe_lane = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(lane or "worker")).strip("-")
    if not safe_lane:
        safe_lane = "worker"
    return runs_dir / ".opencode-tmp" / safe_lane


def opencode_worker_env(account_dir: str | None, lane: str, workspace: Path,
                        runs_dir: Path) -> dict[str, str]:
    """Child env for an opencode/glm worker: the account pinned via XDG_CONFIG_HOME
    (NOT CLAUDE_CONFIG_DIR / oauth token, which are claude-only), plus the same
    self-describing dispatch vars the claude path stamps."""
    env = dispatch_worker.child_env(lane, "opencode", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["XDG_CONFIG_HOME"] = _opencode_config_home(account_dir, runs_dir)
    gh_config = _github_cli_config_dir()
    if gh_config:
        env.setdefault("GH_CONFIG_DIR", gh_config)
    scratch = _opencode_temp_dir(runs_dir, lane)
    scratch.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        env[key] = str(scratch)
    return env


def codex_worker_env(account_dir: str | None, lane: str, workspace: Path) -> dict[str, str]:
    """Child env for a codex (`codex exec`) worker. Codex authenticates from the
    ambient ``~/.codex`` (one ChatGPT login) rather than the multi-account switcher,
    so there is no per-account dir to pin — just clear the claude-only vars and stamp
    the self-describing dispatch vars. When a per-account CODEX_HOME dir IS supplied
    (future multi-account codex), honor it so the run is attributed to that account."""
    env = dispatch_worker.child_env(lane, "codex", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["CODEX_HOME"] = account_dir
    return env


# Windows process-creation flags for a detached worker.
#   DETACHED_PROCESS  — the child gets NO console at all.
#   CREATE_NO_WINDOW  — the child gets a console, but it is HIDDEN (no window).
# Every detached worker — batch shim OR native exe — gets CREATE_NO_WINDOW, never
# DETACHED_PROCESS, for TWO reasons:
#   1. A .cmd/.bat launcher (opencode.CMD, the npm shim) is run BY cmd.exe, which
#      REQUIRES a console: under DETACHED_PROCESS the batch wrapper dies at the
#      "Terminate batch job (Y/N)?" prompt producing ZERO output — every dispatched
#      glm/opencode worker was DOA (0-byte log) until it got a (hidden) console.
#   2. A native worker (claude.exe) given DETACHED_PROCESS has no console, so every
#      console tool it spawns — git, gh, fak, the shell — pops its OWN visible
#      window: the "random popup windows" seen during a dispatched run. CREATE_NO_WINDOW
#      gives the worker one HIDDEN console the whole tool subtree inherits, so it
#      still outlives this dispatcher but never flashes a window. (Same conclusion as
#      claude_agent_chat.detached_creationflags.)
_DETACHED_PROCESS = 0x00000008  # retained to document the rejected alternative
_CREATE_NO_WINDOW = 0x08000000


def win_creationflags(exe: str) -> int:
    """The Windows creationflags for spawning ``exe`` detached: ALWAYS a HIDDEN
    console (CREATE_NO_WINDOW). A .cmd/.bat shim needs a console to live; a native
    exe needs one so its console grandchildren don't each pop a visible window.
    ``exe`` is accepted for call-site stability but no longer branches the result."""
    del exe  # both batch shims and native exes take the hidden-console path
    return _CREATE_NO_WINDOW


def wave_membership_env(rank: int, wave_id: str, size: int,
                        shortfall: int) -> dict[str, str]:
    """The env vars that stamp a worker's place in its wave: which ``rank`` it is
    in ``[0, size)``, which ``wave_id`` it belongs to, the granted ``size`` of the
    group, and the recorded ``shortfall`` (lanes the wave fell short of the
    request). The deterministic counterpart of ``fleet_accounts.allocate_wave``'s
    per-lane membership stamp, carried into ``child_env``.

    NOT a collective: these vars LABEL an otherwise-independent detached worker —
    they grant no barrier/gather/all-to-all. A wave stays N lanes whose only shared
    fabric is git + the ``dos arbitrate`` lease."""
    return {
        "FLEET_WAVE_ID": str(wave_id),
        "FLEET_WAVE_RANK": str(int(rank)),
        "FLEET_WAVE_SIZE": str(int(size)),
        "FLEET_WAVE_SHORTFALL": str(int(shortfall)),
    }


def write_wave_sidecar(out_log: Path, rank: int, wave_id: str, size: int,
                       shortfall: int) -> dict[str, Any]:
    """Write the per-worker membership sidecar next to ``.pid``/``.backend`` so an
    auditor enumerates the whole wave from the filesystem — without trusting any
    worker's self-report. Mirrors the env stamp; deterministic at allocation."""
    rec = {"wave_id": str(wave_id), "rank": int(rank), "size": int(size),
           "shortfall": int(shortfall)}
    out_log.with_suffix(WAVE_SIDECAR_SUFFIX).write_text(
        json.dumps(rec, sort_keys=True), encoding="utf-8")
    return rec


def write_account_sidecar(out_log: Path, account: dict[str, Any] | None) -> dict[str, Any]:
    """Write the switcher account selected for this worker next to its log.

    A 0-byte worker log used to tell us only which issue died. Stamping the account
    at launch time makes later silent-death scans attributable without trusting the
    worker to start and self-report.
    """
    src = account or {}
    rec = {k: src.get(k) for k in ("tag", "tier", "model", "dir")
           if src.get(k) is not None}
    if rec:
        out_log.with_suffix(ACCOUNT_SIDECAR_SUFFIX).write_text(
            json.dumps(rec, sort_keys=True), encoding="utf-8")
    return rec


def write_lease_sidecar(out_log: Path, lease: dict[str, Any] | None) -> dict[str, Any]:
    """Write the fenced lease token selected for this worker.

    The release path must not infer holder/generation from the current dispatcher
    process, because the worker is audited by a later tick. Persist only the
    holder-checked release inputs; a fail-open/no-lease spawn writes nothing.
    """
    src = lease or {}
    if not src.get("acquired"):
        return {}
    rec = {k: src.get(k) for k in ("id", "holder", "generation", "session_id", "tree")
           if src.get(k) is not None}
    if rec.get("id") and rec.get("holder"):
        out_log.with_suffix(LEASE_SIDECAR_SUFFIX).write_text(
            json.dumps(rec, sort_keys=True), encoding="utf-8")
        return rec
    return {}


def read_lease_sidecar(out_log: Path) -> dict[str, Any] | None:
    try:
        rec = json.loads(out_log.with_suffix(LEASE_SIDECAR_SUFFIX).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or not rec.get("id") or not rec.get("holder"):
        return None
    return rec


EARLY_EXIT_TAIL_CHARS = 8192


def probe_spawned_worker(proc: subprocess.Popen, out_log: Path,
                         wait_s: float = 0.0) -> dict[str, Any]:
    """Briefly check whether a just-spawned worker died before it could log.

    A healthy issue worker can be silent for many seconds while Claude starts, so a
    live process with a 0-byte log is not a failure. The failure we can witness here
    is narrower: the process has already exited AND the log is still empty.
    """
    if wait_s <= 0:
        return {"checked": False}
    try:
        returncode = proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        return {"checked": True, "alive": True, "wait_s": wait_s}
    try:
        log_bytes = out_log.stat().st_size
    except OSError:
        log_bytes = 0
    rec: dict[str, Any] = {
        "checked": True,
        "alive": False,
        "wait_s": wait_s,
        "returncode": returncode,
        "log_bytes": log_bytes,
        "silent": log_bytes == 0,
    }
    if log_bytes:
        try:
            rec["tail"] = out_log.read_text(encoding="utf-8", errors="replace")[-EARLY_EXIT_TAIL_CHARS:]
        except OSError:
            pass
    return rec


def enumerate_wave(runs_dir: Path, wave_id: str) -> dict[str, Any]:
    """Enumerate a wave from its ``.wave`` sidecars on disk — the auditor's view of
    a typed group. Returns ``{wave_id, granted, size, shortfall, ranks, members}``
    where ``ranks`` is the sorted list of stamped ranks (a complete wave reads
    ``0..granted-1``) and ``shortfall`` is the recorded under-fill. Reads only the
    filesystem, never a worker's self-report."""
    runs_dir = Path(runs_dir)
    want = str(wave_id)
    members: list[dict[str, Any]] = []
    for side in runs_dir.glob(f"resolve-*{WAVE_SIDECAR_SUFFIX}"):
        try:
            rec = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(rec.get("wave_id")) != want:
            continue
        members.append(rec)
    members.sort(key=lambda r: int(r.get("rank", -1)))
    ranks = [int(r.get("rank", -1)) for r in members]
    size = int(members[0].get("size", 0)) if members else 0
    shortfall = int(members[0].get("shortfall", 0)) if members else 0
    return {"wave_id": want, "granted": len(members), "size": size,
            "shortfall": shortfall, "ranks": ranks, "members": members}


def spawn_issue_worker(command: list[str], env: dict[str, str], cwd: Path,
                       log_dir: Path, issue: int, lane: str,
                       backend: str,
                       membership: dict[str, Any] | None = None,
                       account: dict[str, Any] | None = None,
                       lease: dict[str, Any] | None = None,
                       base_sha: str | None = None,
                       spawn_probe_s: float = 0.0,
                       log_prefix: str = "resolve",
                       prompt_payload: str | None = None,
                       worktree_git: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Launch a detached worker (claude or opencode) on one issue; record pid.

    The log keeps the backend-neutral ``resolve-<N>-<stamp>.log`` name so the close
    arm, silent-worker scan, and in-flight de-dup all see it uniformly. A
    contract-repair spawn passes ``log_prefix="repair"`` so those resolve-only
    scans never see it, while the reap/prune sweeps and the preflight seat count
    (which cover both prefixes) still do.

    ``membership`` (``{rank, wave_id, size, shortfall}``, as ``allocate_wave`` stamps
    each lane) is OPTIONAL: when given, the worker's rank/wave identity is stamped
    into ``child_env`` (``FLEET_WAVE_*``) AND a ``.wave`` sidecar so the wave is a
    legible typed group enumerable from disk. It does NOT make the wave a collective
    — the worker stays a detached lane."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"{log_prefix}-{issue}-{stamp}.log"
    # #3181: opt-in per-worker worktree isolation behind FLEET_WORKER_WORKTREE, the same
    # gate + fail-open helper the Go spine (#3168) uses. When on, the worker edits in its
    # own detached worktree pinned at base_sha (the repo HEAD this dispatcher already
    # records) with GOCACHE/GOTMPDIR redirected inside it; the ``.worktree`` sidecar hands
    # the land+reap sweep the path + base SHA to land the diff under the lane lease this
    # dispatcher holds. Any worktree fault fails open to the shared-trunk ``cwd`` — flag
    # off is byte-identical to today.
    spawn_cwd, env, wt_info = worker_worktree.isolate_spawn(
        Path(cwd), lane, str(issue), cwd, env, base_sha=base_sha, git=worktree_git)
    prompt_file: Path | None = None
    prompt_stdin = None
    if backend == "opencode":
        if prompt_payload is None:
            raise ValueError("opencode worker spawn requires prompt_payload")
        prompt_file = out_log.with_suffix(OPENCODE_PROMPT_FILE_SUFFIX)
        prompt_file.write_text(prompt_payload, encoding="utf-8")
        command = resolve_opencode_command(
            attach_opencode_prompt_file(command, prompt_file))
    elif backend == "claude" and prompt_payload is not None:
        prompt_file = out_log.with_suffix(CLAUDE_PROMPT_FILE_SUFFIX)
        prompt_file.write_text(prompt_payload, encoding="utf-8")
        prompt_stdin = open(prompt_file, "r", encoding="utf-8")
    exe = resolve_worker_executable(backend, command[0])
    argv = [exe, *command[1:]]
    if membership is not None:
        env = {**env, **wave_membership_env(**membership)}
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = win_creationflags(exe)
    else:
        kwargs["start_new_session"] = True
    fh = open(out_log, "w", encoding="utf-8")
    try:
        # Flush a one-line spawn header BEFORE exec so a later 0-byte log means
        # "died before exec" (the OS never ran the child) — distinguishable from
        # "spawned, exec'd, then wrote nothing". The child inherits this fd and
        # appends after the header. Kept short (well under _STUB_LOG_MAX_BYTES) and
        # banner-free so it never trips the stub/cap-banner classifiers.
        fh.write("# fak-spawn %s issue=%s lane=%s backend=%s argv0=%s\n" % (
            stamp, issue, lane, backend, os.path.basename(exe)))
        fh.flush()
        proc = subprocess.Popen(argv, cwd=str(spawn_cwd), env=env,
                                stdin=prompt_stdin if prompt_stdin is not None else subprocess.DEVNULL,
                                stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    finally:
        if prompt_stdin is not None:
            prompt_stdin.close()
        fh.close()
    (out_log.with_suffix(".pid")).write_text(str(proc.pid), encoding="utf-8")
    # Per-worker backend sidecar: makes each run's backend traceable from disk,
    # independent of the (multi-task, last-writer) tick record.
    (out_log.with_suffix(".backend")).write_text(backend, encoding="utf-8")
    # Per-worker base SHA (#1324 proposal #2): record repo HEAD at launch so a later
    # tick re-audits the commit THIS worker lands (the newest commit in base..HEAD
    # citing its #issue), not whatever a sibling pushed to HEAD. Optional + best
    # effort: a None/empty base simply omits the sidecar and the witness falls back
    # to recent history.
    if base_sha:
        (out_log.with_suffix(BASE_SHA_SIDECAR_SUFFIX)).write_text(base_sha, encoding="utf-8")
    # #3181: record the isolated worktree in the SAME sidecar contract the shared
    # witness sweep already consumes for the Go spine — `fak dispatch witness --live`
    # (cmd/fak/dispatch_tick_witness.landAndReapWorkerWorktree) scans this same
    # `.dispatch-runs` dir, reads `<log>.worktree` as a PLAIN path, the pinned base from
    # `<log>.basesha`, and the scoped commit paths from `<log>.lease-tree.json`, then on
    # the worker's exit applies its diff-since-base onto the trunk as its own stamped
    # commit (under the lane lease this dispatcher holds) and reaps the worktree. The
    # base sidecar MUST carry the SHA the worktree was pinned to (not a bare HEAD): the
    # worker commits IN its detached worktree, so a HEAD-diff would be empty and its
    # whole change would evaporate. Best effort — a write failure never blocks the spawn.
    if wt_info.get("worktree"):
        try:
            out_log.with_suffix(".worktree").write_text(
                str(wt_info["worktree"]), encoding="utf-8")
            if wt_info.get("base_sha"):
                out_log.with_suffix(BASE_SHA_SIDECAR_SUFFIX).write_text(
                    str(wt_info["base_sha"]), encoding="utf-8")
            lease_tree = [t for t in ((lease or {}).get("tree") or [])
                          if isinstance(t, str)]
            if lease_tree:
                out_log.with_suffix(".lease-tree.json").write_text(
                    json.dumps(lease_tree), encoding="utf-8")
        except OSError:
            pass
    result: dict[str, Any] = {"pid": proc.pid, "log": str(out_log), "issue": issue,
                              "lane": lane, "backend": backend}
    if wt_info.get("worktree"):
        result["worktree"] = wt_info["worktree"]
    if prompt_file is not None:
        result["prompt_file"] = str(prompt_file)
    acct = write_account_sidecar(out_log, account)
    if acct:
        result["account"] = acct
    lease_rec = write_lease_sidecar(out_log, lease)
    if lease_rec:
        result["lease"] = lease_rec
    if membership is not None:
        result["membership"] = write_wave_sidecar(out_log, **membership)
    early = probe_spawned_worker(proc, out_log, spawn_probe_s)
    if early.get("checked"):
        result["early_exit"] = early
    return result


# --- Weekly-cap gate -------------------------------------------------------
# A worker spawned on a quota-exhausted (but still logged-in) account dies
# immediately with a ~65-byte banner, e.g.:
#   You've hit your weekly limit · resets Jun 25, 1pm (America/Los_Angeles)
#   You've hit your weekly limit · resets 4am (America/Los_Angeles)
# The spawn preflight cannot see this — it checks host/headroom/account-logged-in,
# NOT remaining quota — so absent this gate every tick re-spawns a doomed worker
# for the entire (multi-day) reset window: it ships nothing and floods the logs.
# These helpers read the banner back from recent worker logs and persist a hold so
# the dispatcher stops spawning until the quota resets, turning a silent spin into
# an honest WEEKLY_CAPPED hold. Everything here is FAIL-OPEN: any error resolves to
# "not capped", so the gate can only ever ADD a refusal, never wedge the loop.

_CAP_BANNER_RE = re.compile(
    r"hit your[\w\s]*limit|limit\s+exhausted|rate[_ -]limited|HTTP\s+429",
    re.IGNORECASE)
# The codex (OpenAI/ChatGPT) backend hits its own quota wall with a different banner
# than Claude's: "You've hit your usage limit. Visit https://chatgpt.com/codex/...
# purchase more credits or try again at Jul 1st, 2026 8:41 PM." It matches the phrase
# regex above ("hit your usage limit"), but its reset clause is "try again at <date>"
# (no "(America/Los_Angeles)" suffix), so the Claude reset parsers below miss it and
# the hold would fall back to the short cooldown — re-spawning doomed codex workers
# until the real (Jul-1) reset. _CODEX_RESET_RE recovers the codex reset moment so the
# hold runs to the real wall instead.
_CODEX_RESET_RE = re.compile(
    r"try again at\s+([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))",
    re.IGNORECASE)
_RESET_AT_RE = re.compile(
    r"(?:will\s+)?reset\s+at\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE)
# A SESSION limit is a short rolling window (resets in hours); a WEEKLY limit is a
# multi-day quota wall. They share the "hit your … limit" banner but must NOT share
# the hold: treating a transient session limit as a weekly hold — and projecting a
# bare time-of-day reset that already passed a full day forward — falsely walls an
# account that actually has room (the gem8 false-cap). Classify the banner so a
# session limit gets a short cooldown bounded to its real reset.
_CAP_WEEKLY_RE = re.compile(r"weekly[\w/\s]*limit", re.IGNORECASE)
_CAP_SESSION_RE = re.compile(r"session\s+limit", re.IGNORECASE)
_CAP_RESET_RE = re.compile(r"resets\s+(.+?)\s*\(America/Los_Angeles\)", re.IGNORECASE)
_CAP_RESET_FALLBACK_RE = re.compile(r"resets\s+([^\r\n]+)", re.IGNORECASE)
# The guard/gateway names a cap wall as a RELATIVE Go-duration wait, not an absolute
# "resets <when>" clause — e.g.
#   fak-turn trace=guard FAILED reason=rate_limited wire=anthropic_messages \
#     kind=weekly_limit announced_wait=1h7m0s
# (internal/gateway/debug_stats.go emits `announced_wait=<dur>`; #2610). Without this
# the parser discarded the window and the hold fell back to a blind 60-min cooldown —
# so a weekly-capped seat whose real reset is hours out was re-offered after an hour,
# re-spawning straight into the same cap. Capture the duration so the hold runs to the
# ANNOUNCED window and the status surface can name the retry window. The `[=:≈~]` class
# also accepts the prose "≈" form a human transcript / issue body may carry.
_ANNOUNCED_WAIT_RE = re.compile(
    r"announced_wait\s*[=:≈~]?\s*(\d+h(?:\d+m)?(?:\d+s)?|\d+m(?:\d+s)?|\d+s)",
    re.IGNORECASE)
_ANNOUNCED_WAIT_DUR_RE = re.compile(
    r"^\s*(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?\s*$", re.IGNORECASE)
# A session-limit hold never exceeds this even if its reset clause parses to a far
# future moment (a stale, already-passed bare time-of-day must not become a ~24h
# wall). Weekly limits keep the full parsed reset.
_SESSION_HOLD_MAX_MIN = 90
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    start=1)}
# America/Los_Angeles is PDT (UTC-7) over the summer reset windows this gate sees;
# a PT wall-clock time + this offset == UTC. (A ±1h DST error only nudges the hold
# edge and self-corrects on the next probe — fine for a throttle.)
_PT_TO_UTC = dt.timedelta(hours=7)


def _backend_of_log(log: Path) -> str:
    """The backend that produced a worker log, from its ``.backend`` sidecar; if
    the sidecar is missing (#1452: 128/468 legacy + opencode logs never got one),
    fall back to the ``backend=<x>`` field of the log's ``# fak-spawn`` header
    line so an opencode/codex worker is NOT silently misattributed to ``claude``
    — that misattribution blinds the opencode/codex cap+health scan to its own
    quota-walled logs (it filters by backend) while polluting claude's. Only a log
    with neither a sidecar nor a parseable header defaults to ``claude``."""
    try:
        sidecar = log.with_suffix(".backend").read_text(encoding="utf-8").strip()
        if sidecar:
            return sidecar
    except OSError:
        pass
    try:
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        return "claude"
    m = re.search(r"\bbackend=([A-Za-z0-9_]+)", header)
    return m.group(1) if m else "claude"


def _account_tag_of_log(log: Path) -> str | None:
    """Return the switcher account tag stamped next to ``log``, if any.

    Account sidecars are written before the child can produce output. A quota
    banner without this sidecar is not positive evidence for whichever account
    the next tick happens to route to, so account-scoped cap checks ignore it.
    """
    try:
        rec = json.loads(log.with_suffix(ACCOUNT_SIDECAR_SUFFIX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(rec, dict):
        tag = rec.get("tag") or rec.get("account")
        return str(tag) if tag else None
    if isinstance(rec, str) and rec:
        return rec
    return None


# How many trailing bytes of a worker log to inspect for the quota banner. A walled
# worker emits the banner as the LAST thing it writes, so the tail is where it lives;
# bounding the read keeps a multi-MB productive log cheap to scan and stops its early
# agent prose from false-matching the phrase.
_CAP_TAIL_BYTES = 4096


def _log_tail_text(log: Path, *, nbytes: int = _CAP_TAIL_BYTES) -> str:
    """The last ``nbytes`` of ``log`` decoded as utf-8 (errors replaced), or "" on
    any read error. Used to inspect a log's ending for the quota banner without
    reading the whole (possibly large) file."""
    try:
        size = log.stat().st_size
        with log.open("rb") as fh:
            if size > nbytes:
                fh.seek(-nbytes, os.SEEK_END)
            raw = fh.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _log_is_cap_banner(log: Path) -> bool:
    """True when ``log``'s tail is the quota-limit banner (Claude or codex). Lets the
    backend-health classifier treat a credit-walled worker as a stub regardless of
    byte size — a codex wall log clears the size floor on its startup banner alone."""
    return _cap_hit_from_text(_log_tail_text(log)) is not None


# A PERMANENT auth gap is a different death than a transient credit wall. A backend with
# no API key / no resolved subscription login (e.g. codex: "provider env $OPENAI_API_KEY
# is empty and no ChatGPT subscription auth.json was resolved. Run `codex login`…") dies
# IDENTICALLY on every spawn and cannot recover without an operator. _BACKEND_REPROBE_MIN
# assumes the wall lifts on its own (codex resets nightly), so applied to an auth gap the
# re-probe burns one worker SEAT every 30 min forever — hundreds of doomed spawns +
# BACKEND_UNHEALTHY refusals per day, and the hold's "detecting recovery" framing is a lie
# (a missing key never self-heals). Detect the auth-gap signature so the health gate makes
# the hold STICKY (an unauthenticated backend we simply stopped spawning leaves an empty
# window, which must NOT read as recovery) and backs the re-probe WAY off. FAIL-CLOSED to
# "not an auth gap" on any doubt — a false positive would wedge a recoverable backend — so
# match only unambiguous, operator-only phrases the guard/provider preflight emits.
_AUTH_GAP_RE = re.compile(
    r"API_KEY is empty"
    r"|no ChatGPT subscription auth\.json"
    r"|no auth\.json was resolved"
    r"|Run `(?:codex|claude) login`",
    re.IGNORECASE)


def _log_is_auth_gap(log: Path) -> bool:
    """True when a stub log's tail is the PERMANENT auth-gap banner (missing API key / no
    resolved login), as opposed to a transient credit wall (``_log_is_cap_banner``). Such
    a death cannot self-heal without an operator, so the health gate holds it stickily and
    stops re-probing it at the credit-wall cadence."""
    return bool(_AUTH_GAP_RE.search(_log_tail_text(log)))


def _cap_hit_from_text(text: str, *, evidence_log: str = "") -> dict[str, Any] | None:
    """Parse a quota/rate-limit hit from already-captured worker text."""
    if not _CAP_BANNER_RE.search(text or ""):
        return None
    m = (_CODEX_RESET_RE.search(text) or _RESET_AT_RE.search(text) or _CAP_RESET_RE.search(text)
         or _ANNOUNCED_WAIT_RE.search(text) or _CAP_RESET_FALLBACK_RE.search(text))
    reset_text = m.group(1).strip() if m else ""
    if _CAP_WEEKLY_RE.search(text):
        kind = "weekly"
    elif _CAP_SESSION_RE.search(text):
        kind = "session"
    else:
        kind = "session"
    return {"reset_text": reset_text, "evidence_log": evidence_log, "kind": kind}


def _write_cap_hold(runs_dir: Path, *, product: str, account_tag: str | None,
                    hit: dict[str, Any], now_ts: float, fallback_min: int,
                    source: str) -> dict[str, Any]:
    now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)  # naive UTC
    kind = hit.get("kind") or "session"
    until = (_parse_relative_wait(str(hit.get("reset_text") or ""), now_utc)
             or _parse_reset_to_utc(str(hit.get("reset_text") or ""), now_utc)
             or now_utc + dt.timedelta(minutes=fallback_min))
    if kind == "session":
        session_cap = now_utc + dt.timedelta(minutes=_SESSION_HOLD_MAX_MIN)
        until = min(until, session_cap)
    state = {"product": product, "account": account_tag, "kind": kind,
             "reset_text": hit.get("reset_text") or "",
             "evidence_log": hit.get("evidence_log") or "",
             "detected": now_utc.isoformat() + "Z", "until": until.isoformat() + "Z"}
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        for state_path in _cap_state_paths(runs_dir, product=product,
                                           account_tag=account_tag):
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass
    return {"capped": True, "until": state["until"], "kind": kind,
            "reset_text": state["reset_text"], "source": source,
            "evidence_log": state["evidence_log"]}


def _safe_cap_account_component(account_tag: str | None) -> str | None:
    if not account_tag:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(account_tag).strip())
    return safe or None


def _cap_state_paths(runs_dir: Path, *, product: str,
                     account_tag: str | None) -> list[Path]:
    paths: list[Path] = []
    safe_account = _safe_cap_account_component(account_tag)
    if safe_account:
        paths.append(runs_dir / f"account-cap-{product}-{safe_account}.json")
    # Keep writing the legacy per-product file so older launchers still see the
    # most recent cap. Readers prefer the account-specific file above.
    paths.append(runs_dir / f"account-cap-{product}.json")
    return paths


def _scan_recent_cap_banner(runs_dir: Path, *, product: str, lookback_min: int,
                            now_ts: float, account_tag: str | None = None) -> dict[str, Any] | None:
    """The most-recent worker log (this backend, mtime within ``lookback_min``)
    whose TAIL is the quota-limit banner → ``{reset_text, evidence_log}``, else None.
    Scanning only the tail (not the whole file, and not size-gated) catches a codex
    wall log — which clears any byte floor on its ~700-byte startup banner before the
    error — while keeping a real multi-MB worker log cheap to inspect; the specific
    phrase still keeps a prose log that merely mentions "limit" mid-run from
    false-matching, because the banner only appears as the worker's final output."""
    if not runs_dir.is_dir():
        return None
    horizon = now_ts - lookback_min * 60
    best: tuple[float, str, str, str] | None = None
    for log in runs_dir.glob("resolve-*.log"):
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon:
            continue
        if _backend_of_log(log) != product:
            continue
        if account_tag and _account_tag_of_log(log) != account_tag:
            continue
        hit = _cap_hit_from_text(_log_tail_text(log), evidence_log=log.name)
        if not hit:
            continue
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, str(hit["reset_text"]), str(hit["evidence_log"]), str(hit["kind"]))
    if best is None:
        return None
    return {"reset_text": best[1], "evidence_log": best[2], "kind": best[3]}


def _persisted_hold_matches_account(runs_dir: Path, state: dict[str, Any],
                                    account_tag: str | None) -> bool:
    """True when a persisted cap hold is still attributable to ``account_tag``.

    Old state written before account-scoped scanning can name the next selected
    account even though the evidence log was generic. If the state points at an
    evidence log, require that log's account sidecar to still match before
    honoring the hold.
    """
    if not account_tag:
        return True
    evidence = str(state.get("evidence_log") or "")
    if not evidence:
        return True
    return _account_tag_of_log(runs_dir / evidence) == account_tag


def _parse_reset_to_utc(reset_text: str, now_utc: dt.datetime) -> dt.datetime | None:
    """Best-effort parse of a banner reset clause ('Jun 25, 1pm', '4am',
    '11:30pm') as America/Los_Angeles wall-clock → a naive-UTC datetime, clamped to
    (now, now+8d]. None when no time-of-day is present (caller falls back to a short
    cooldown)."""
    if not reset_text:
        return None
    t = reset_text.lower()
    iso = _iso_to_utc(reset_text.strip())
    if iso:
        if iso <= now_utc:
            return None
        return min(iso, now_utc + dt.timedelta(days=8))
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
    if not tm:
        return None
    hour = int(tm.group(1)) % 12 + (12 if tm.group(3) == "pm" else 0)
    minute = int(tm.group(2) or 0)
    mo = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})", t)
    if mo:
        month, day = _MONTHS[mo.group(1)], int(mo.group(2))
        try:
            cand = dt.datetime(now_utc.year, month, day, hour, minute) + _PT_TO_UTC
        except ValueError:
            return None
        if cand < now_utc - dt.timedelta(days=1):
            try:
                cand = dt.datetime(now_utc.year + 1, month, day, hour, minute) + _PT_TO_UTC
            except ValueError:
                return None
    else:
        now_pt = now_utc - _PT_TO_UTC
        cand_pt = now_pt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if cand_pt <= now_pt:
            cand_pt += dt.timedelta(days=1)
        cand = cand_pt + _PT_TO_UTC
    if cand <= now_utc:
        return None
    return min(cand, now_utc + dt.timedelta(days=8))


def _parse_relative_wait(reset_text: str, now_utc: dt.datetime) -> dt.datetime | None:
    """A RELATIVE cap wait ('1h7m0s', '90m', '6h50m0s') -> now+dur as a naive-UTC
    reset instant, clamped to (now, now+8d]. None when ``reset_text`` is not a bare
    Go-duration -- an absolute 'resets <when>' clause then falls through to
    :func:`_parse_reset_to_utc`. This is how a guard/gateway cap crash names its wall
    (``announced_wait=<dur>``, #2610) vs Claude's absolute 'resets <when>' banner, so
    a weekly cap cools to its ANNOUNCED window instead of a blind fallback."""
    m = _ANNOUNCED_WAIT_DUR_RE.match(reset_text or "")
    if not m or not any(m.groups()):
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    if hours == 0 and minutes == 0 and seconds == 0:
        return None
    cand = now_utc + dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return min(cand, now_utc + dt.timedelta(days=8))


def _iso_to_utc(s: str) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def check_weekly_cap(runs_dir: Path, *, product: str, account_tag: str | None,
                     now_ts: float | None = None, lookback_min: int = 45,
                     fallback_min: int = 60) -> dict[str, Any]:
    """Is ``account_tag`` quota-capped right now? Detects the limit banner from
    recent worker logs and persists a hold in ``account-cap-<product>.json`` so it
    survives the ticks AFTER spawns stop (no fresh banner is produced once we stop
    launching doomed workers). Returns ``{"capped": bool, ...}``. FAIL-OPEN: any
    exception → ``{"capped": False}``."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)  # naive UTC
        hit = _scan_recent_cap_banner(runs_dir, product=product,
                                      lookback_min=lookback_min, now_ts=now_ts,
                                      account_tag=account_tag)
        if hit:
            return _write_cap_hold(runs_dir, product=product, account_tag=account_tag,
                                   hit=hit, now_ts=now_ts, fallback_min=fallback_min,
                                   source="banner")
        # No fresh banner: honor a persisted, unexpired hold for THIS account.
        for state_path in _cap_state_paths(runs_dir, product=product,
                                           account_tag=account_tag):
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            until = _iso_to_utc(state.get("until") or "")
            if state.get("account") == account_tag and until and now_utc < until:
                if not _persisted_hold_matches_account(runs_dir, state, account_tag):
                    try:
                        state_path.unlink()
                    except OSError:
                        pass
                    continue
                return {"capped": True, "until": state.get("until"),
                        "reset_text": state.get("reset_text", ""), "source": "state"}
            if until and now_utc >= until:
                try:
                    state_path.unlink()
                except OSError:
                    pass
        return {"capped": False}
    except Exception:
        return {"capped": False}


# --- backend-health reallocation gate -----------------------------------------
# The weekly-cap gate above catches ONE dead-backend shape: the explicit "hit your
# limit" banner. But a backend can spin just as silently WITHOUT that banner — a
# codex spawn that dies credit-walled (0-byte log) or a glm/opencode worker that
# prints only its startup banner ("> build · glm-5.2", ~32 bytes) and exits without
# a turn. Those pass the 5s early-exit probe (alive then) and slip the cap regex, so
# the dispatcher keeps feeding the backend budget while it ships nothing — and the
# lane it owned (e.g. docs, the biggest backlog) is abandoned rather than handed to
# a backend that IS producing turns.
#
# check_backend_health() is the sibling gate, shaped exactly like check_weekly_cap:
# a fast on-disk signal (a MAJORITY of stub logs for this backend) declares it DEAD,
# persisted in backend-health-<product>.json with the same write/honor/expire and
# FAIL-OPEN discipline (any error -> healthy, so the gate can only ever ADD a hold).
# The dead backend self-suppresses (BACKEND_UNHEALTHY) but lets ONE re-probe worker
# through per interval so it can prove recovery; the healthy backend reads the
# sidecar and claims the freed lane + budget (see evaluate()).

# A worker log at/under this size produced no real turn: a 0-byte credit-wall death
# or a banner-only (~32-byte) exit. A real turn streams kilobytes. Matches the size
# floor the weekly-cap banner scanner already trusts (_scan_recent_cap_banner: 512).
_STUB_LOG_MAX_BYTES = 512
# The sample floor for the majority-stub verdict: fewer than N classified logs in the
# window is not enough evidence to call a backend dead, so a single blip (a real worker
# can rarely exit early for benign reasons) never trips the gate. N is small so the gate
# reacts within a few ticks, not hours.
_BACKEND_DEAD_STREAK = 3
# How long a DEAD hold suppresses spawns before letting one re-probe worker through.
# A credit wall lifts on its own (codex resets nightly); the re-probe is how the
# backend earns its budget back without an operator.
_BACKEND_REPROBE_MIN = 30
# A backend held dead by a PERMANENT auth gap (no API key / no login) must not re-probe on
# the credit-wall cadence: every re-probe is a doomed spawn that burns a seat and dies on
# the same missing key. Back it off to a slow heartbeat — enough that the backend still
# auto-recovers within a few hours of an operator fixing auth, without the 30-min churn.
# (See _AUTH_GAP_RE / _log_is_auth_gap.)
_BACKEND_AUTHGAP_REPROBE_MIN = 360


def _classify_backend_logs(runs_dir: Path, *, product: str, lookback_min: int,
                           now_ts: float, alive: set[int] | None,
                           probe: Any | None) -> list[dict[str, Any]]:
    """The recent resolve logs for ONE backend, newest first, each tagged
    ``{"stamp", "productive", "size"}``. Productive = a log over the real-turn floor;
    a stub (<= floor) only counts as evidence of death when its pid is provably dead
    (a still-running claude worker writes nothing until its final message, so a live
    0-byte log is NOT a stub — reuses the silent_workers dead-pid guard)."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    horizon = now_ts - lookback_min * 60
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon:
            continue
        if _backend_of_log(log) != product:
            continue
        # Size alone is not productivity. A codex credit-wall log carries a ~700-byte
        # startup banner (workdir/model/session id/hook lines) BEFORE the quota error,
        # so it clears the byte floor while landing zero real turns. A log whose body
        # is the quota banner is never a real turn regardless of size -> read the tail
        # and treat a _CAP_BANNER_RE hit as a stub, so a credit-walled codex worker
        # joins the dead-streak instead of falsely breaking it.
        productive = st.st_size > _STUB_LOG_MAX_BYTES and not _log_is_cap_banner(log)
        if not productive:
            # Only a DEAD pid over a stub log is evidence of a doomed spawn; a live
            # one is still running (claude streams nothing until the end).
            pid_file = log.with_suffix(".pid")
            if pid_file.exists() and dispatch_preflight.resolve_sidecar_pid_is_live(
                    pid_file, alive=alive, probe=probe):
                continue  # still running -> not (yet) evidence either way
        out.append({"stamp": st.st_mtime, "productive": productive, "size": st.st_size,
                    "log": log.name})
    out.sort(key=lambda r: r["stamp"], reverse=True)
    return out


def check_backend_health(runs_dir: Path, *, product: str, lane: str | None = None,
                         now_ts: float | None = None, lookback_min: int = 90,
                         alive: set[int] | None = None,
                         probe: Any | None = None) -> dict[str, Any]:
    """Is ``product`` producing real turns right now, or spinning dead? Reads the
    backend's recent worker logs: over the lookback window, MORE stubs (banner-only/
    0-byte over a dead pid) than productive logs -> DEAD; a window it is not mostly
    stubbing -> HEALTHY (and a productive log there clears a stale hold). The
    majority-stub rule is the same ``stub > productive`` rollup the status card shows
    (dispatch_status.backend_stub_rates), so the card and this spawn gate agree (#3247);
    a sample floor of ``_BACKEND_DEAD_STREAK`` logs keeps a thin window from tripping it.
    Persists a hold in ``backend-health-<product>.json`` so it survives the ticks after
    spawns stop, and gates ONE re-probe spawn per ``_BACKEND_REPROBE_MIN`` so a dead
    backend can earn its budget back. Returns ``{state, ...}``. FAIL-OPEN: any error ->
    healthy."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)
        state_path = runs_dir / f"backend-health-{product}.json"
        if alive is None and probe is None:
            try:
                import psutil  # type: ignore
                alive = {p.pid for p in psutil.process_iter()}
            except ImportError:
                alive = None  # cannot prove a pid dead -> classify no stubs (no false death)
        logs = _classify_backend_logs(runs_dir, product=product, lookback_min=lookback_min,
                                      now_ts=now_ts, alive=alive, probe=probe)
        productive = sum(1 for r in logs if r["productive"])
        stub = len(logs) - productive
        # Read any persisted hold up FRONT: a sticky auth-gap hold must survive its doomed
        # stubs aging out of the window (below), so the prior state is needed before the
        # majority-stub decision, not only after it.
        prior: dict[str, Any] = {}
        if state_path.exists():
            try:
                prior = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
        prior_auth_gap = prior.get("state") == "dead" and bool(prior.get("auth_gap"))
        # Is THIS window's death a permanent auth gap? Only when the stub logs are
        # DOMINATED by the auth-gap banner (fail-closed: a lone match never flips it).
        stub_logs = [r["log"] for r in logs if not r["productive"]]
        authgap_stub = sum(1 for name in stub_logs if _log_is_auth_gap(runs_dir / name))
        authgap_now = authgap_stub * 2 > len(stub_logs) and authgap_stub > 0
        # DEAD = the backend stubs MOST of its recent spawns (#3247). This is the same
        # `stub > productive` rollup the status card computes (dispatch_status.
        # backend_stub_rates), so the card and this spawn gate can never disagree: the
        # card used to read "majority-stub [codex=4/4 stub]" while the gate read healthy
        # off a single productive log, and the codex cron kept feeding seats to a
        # backend that returned nothing. The sample floor keeps a thin window (one or
        # two logs) from tripping the gate on noise -- reuses the streak constant, so
        # a 3-stub/0-productive window is still exactly as dead as it was before.
        majority_stub = len(logs) >= _BACKEND_DEAD_STREAK and stub > productive
        # A sticky auth-gap hold keeps the backend dead even once its doomed stubs age out
        # of the lookback: we STOPPED spawning it, so the empty window is not recovery
        # evidence — only a productive log (an operator fixed auth and something ran) may
        # clear it. A transient credit wall is NOT sticky (prior_auth_gap is False for it),
        # so it still ages out to healthy exactly as before and #3247's auto-recovery holds.
        sticky_auth_gap = prior_auth_gap and productive == 0
        if not (majority_stub or sticky_auth_gap):
            # A productive log in a window the backend is NOT mostly stubbing is the
            # witnessed restore -- clear any stale hold so spawns resume with no
            # operator. Stub logs age out of the lookback on their own, so a recovered
            # backend crosses back over the majority line within one window.
            if productive and state_path.exists():
                try:
                    state_path.unlink()
                except OSError:
                    pass
            return {"state": "healthy",
                    "source": "productive" if productive else "no-streak"}
        # DEAD. auth_gap LATCHES once seen and is only cleared by a productive log (the
        # `not (majority_stub or sticky_auth_gap)` branch above), so a backend that dies on
        # a missing key stays held even after its stubs scroll out of the window.
        auth_gap = authgap_now or prior_auth_gap
        reprobe_min = _BACKEND_AUTHGAP_REPROBE_MIN if auth_gap else _BACKEND_REPROBE_MIN
        # Honor an existing hold (keep its since/lane); otherwise open one.
        since = prior.get("since") if prior.get("state") == "dead" else None
        since = since or (now_utc.isoformat() + "Z")
        # The lane this backend would otherwise own — recorded so the healthy tick can
        # claim it. Keep a prior non-empty lane if this tick didn't resolve one.
        abandoned = lane or prior.get("abandoned_lane") or ""
        last_probe = _iso_to_utc(prior.get("last_reprobe") or "") if prior else None
        due = (last_probe is None
               or now_utc - last_probe >= dt.timedelta(minutes=reprobe_min))
        # The stub logs are the evidence (newest first) — the spawns this backend burned.
        # On a sticky hold whose stubs already aged out, keep the prior evidence so the
        # surface still names the cause.
        evidence = (stub_logs[:_BACKEND_DEAD_STREAK]
                    or list(prior.get("evidence_logs") or []))
        stub_rate = round(stub / len(logs), 3) if logs else (prior.get("stub_rate") or 1.0)
        state = {
            "product": product, "state": "dead", "since": since,
            "abandoned_lane": abandoned,
            "auth_gap": auth_gap,
            "evidence_logs": evidence,
            "stub": stub, "productive": productive,
            "stub_rate": stub_rate,
            "detected": now_utc.isoformat() + "Z",
            "last_reprobe": (now_utc.isoformat() + "Z") if due else prior.get("last_reprobe"),
            "reprobe_min": reprobe_min,
        }
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError:
            pass
        # reprobe_due: caller (the DEAD backend's own tick) may let ONE worker through
        # to test recovery. We stamped last_reprobe above so the next tick holds again.
        return {"state": "dead", "since": since, "abandoned_lane": abandoned,
                "reprobe_due": bool(due), "evidence_logs": evidence,
                "stub": stub, "productive": productive, "auth_gap": auth_gap,
                "reprobe_min": reprobe_min, "stub_rate": stub_rate,
                "source": "auth-gap" if auth_gap else "majority-stub"}
    except Exception:
        return {"state": "healthy", "source": "error"}


def read_dead_backends(runs_dir: Path, *, exclude: str | None = None,
                       now_ts: float | None = None) -> list[dict[str, Any]]:
    """Sibling backends currently held DEAD, read from backend-health-*.json — the
    healthy tick's view of what budget/lane is free to claim. ``exclude`` drops the
    caller's own product. Read-only / best-effort (a corrupt sidecar is skipped)."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for path in runs_dir.glob("backend-health-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("state") != "dead":
            continue
        if exclude and st.get("product") == exclude:
            continue
        out.append(st)
    return out


def loop_id_for_payload(payload: dict[str, Any]) -> str:
    backend = str(payload.get("backend") or "claude").strip() or "claude"
    return f"{LOOP_ID_PREFIX}/{backend}"


def default_loop_ledger(root: Path) -> Path:
    configured = os.environ.get("FAK_LOOP_LEDGER")
    if configured:
        return Path(configured)
    return root / ".fak" / "loops.jsonl"


def fak_loop_cmd(root: Path) -> list[str]:
    configured = os.environ.get("FAK_BIN")
    if configured:
        return split_command_env(configured)
    return ["go", "run", "./cmd/fak"]


def split_command_env(value: str) -> list[str]:
    """Split a command-valued env var without eating Windows path separators."""
    return shlex.split(value, posix=(os.name != "nt"))


# --- Fenced lane-lease gate (residual of #1310) ------------------------------
# The same-host log-scan gate (live_resolution_lanes) serializes two ticks ON ONE
# HOST run sequentially, but it is blind across machines and has a TOCTOU window
# (the lane= header is written near the END of a tick; two evaluate() processes
# racing both read "lane free"). These helpers close both gaps by holding the
# fenced refs/fak/locks/resolve-<lane> lease via `fak leaseref acquire` — an
# atomic update-ref compare-and-swap that rides ordinary git fetch/push, so a peer
# on another clone SEES the lease after a fetch. The log scan stays the fast path;
# this is the authoritative pre-spawn admission. EVERYTHING here is FAIL-OPEN: any
# error (no fak binary, a broken store, an unparseable verdict) resolves to "not
# held / acquired" so a lease-layer fault can only ever DROP the extra protection,
# never wedge the loop — the same discipline as check_weekly_cap.

# A structured fence refusal (LEASE_HELD / LEASE_CONTENDED / STALE_LEASE /
# NO_LEASE) exits 3 from `fak leaseref` (cmd/fak/leaseref.go: leaserefRefused); 0
# is acquired, 1 a git/store failure, 2 a usage error. We fence-refuse (fail
# CLOSED) only on 3, and fail OPEN on everything else.
LEASE_REFUSED_RC = 3


def _fak_bin(root: Path) -> list[str] | None:
    """The on-disk `fak` argv for a lease subprocess, or None when only the
    `go run ./cmd/fak` fallback is available. We DELIBERATELY refuse the go-run
    fallback for the lease path: building cmd/fak on a peer-dirty shared trunk
    routinely fails (the cmd-lane-undispatchable law), and a lease gate that
    shells a 30s flaky `go run` every tick would be worse than no gate. With no
    FAK_BIN the gate fails open (the same-host log scan still applies)."""
    configured = os.environ.get("FAK_BIN")
    if configured:
        return split_command_env(configured)
    exe = shutil.which("fak")
    return [exe] if exe else None


def lease_holder() -> str:
    """A holder identity stable across this session's separate dispatcher ticks.
    Mirrors dos_fleet_lease.default_owner: the harness sets CLAUDE_CODE_SESSION_ID
    once per session; fall back to host:pid so two unrelated sessions never share a
    holder (a shared holder would let one RENEW the other's lease)."""
    sid = os.environ.get("FAK_LEASE_OWNER") or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        return sid.strip()
    host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "host"
    return f"{host}:{os.getpid()}"


def lease_owner_override() -> str | None:
    owner = (os.environ.get("FAK_LEASE_OWNER") or "").strip()
    return owner or None


_LEASE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,199}$")
_LEASE_SESSION_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def lease_session_id() -> str | None:
    """Best-effort session binding for `fak leaseref liveness`.

    The lease gate must never fail only because a harness exported a malformed
    session id, so invalid values are skipped instead of passed to the CLI's
    stricter ref-segment validator.
    """
    sid = (os.environ.get("FAK_LEASE_SESSION_ID")
           or os.environ.get("CLAUDE_CODE_SESSION_ID") or "").strip()
    if (not sid or not _LEASE_SESSION_ID_RE.match(sid)
            or sid.startswith("session-")):
        return None
    return sid


def _safe_lease_session_segment(raw: str) -> str | None:
    segment = _LEASE_SESSION_SAFE_RE.sub("-", raw.strip()).strip("-.")
    if not segment:
        return None
    if segment.startswith("session-"):
        segment = "dispatch-" + segment
    segment = segment[:200].rstrip("-.")
    if not segment or not _LEASE_SESSION_ID_RE.match(segment):
        return None
    return segment


def synthetic_lease_session_id() -> str:
    host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "host"
    return (_safe_lease_session_segment(f"dispatch-{host}-{os.getpid()}")
            or f"dispatch-{os.getpid()}")


def lease_id_for_lane(lane: str) -> str:
    return f"{LEASE_ID_PREFIX}{lane}"


def _run_lease(root: Path, args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    """Run one `fak leaseref` subcommand, returning {rc, stdout, verdict}. Never
    raises: an exec failure is reported as rc 127 so the caller fails open."""
    bin_argv = _fak_bin(root)
    if not bin_argv:
        return {"rc": 127, "stdout": "", "verdict": None, "skipped": "no fak binary"}
    cmd = [*bin_argv, "leaseref", *args]
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": 127, "stdout": "", "verdict": None, "error": str(exc)}
    out = (proc.stdout or "").strip()
    doc: Any = None
    try:
        doc = json.loads(out) if out else None
    except ValueError:
        doc = None
    return {"rc": proc.returncode, "stdout": out, "verdict": doc}


def publish_lease_session(root: Path, session_id: str | None, ttl_s: int,
                          *, runner: Any | None = None) -> dict[str, Any]:
    """Best-effort heartbeat for a synthetic lease session.

    Publishing is advisory read-side evidence for liveness. A missing/old `fak`
    binary must not block dispatch, so failures are carried as fail-open evidence
    and the caller still attempts the actual lease acquire.
    """
    if not session_id:
        return {"published": False, "skipped": "missing session id"}
    run = runner or _run_lease
    res = run(root, ["session-publish", "--session", session_id, "--ttl",
                     str(int(ttl_s)), "--state", "RUNNING"])
    if res.get("rc") == 0:
        return {"published": True, "session_id": session_id}
    return {"published": False, "session_id": session_id, "fail_open": True,
            "rc": res.get("rc"), "detail": res.get("error") or res.get("skipped")}


def acquire_lane_lease(root: Path, lane: str, *, tree: list[str], ttl_s: int,
                       holder: str | None = None,
                       runner: Any | None = None) -> dict[str, Any]:
    """ATOMICALLY take the fenced refs/fak/locks/resolve-<lane> lease before a
    spawn. Returns {"acquired": bool, "refused": bool, "id", "holder", ...}.

    - acquired=True  -> exit 0: we hold the lease; carry the id+holder/generation
                        so the worker sidecar can be released after a later
                        dead-pid witness. TTL + reap remain the crash backstop.
    - refused=True   -> exit 3: a LIVE peer (this host OR another clone after a
                        fetch) holds the lane; the caller refuses LANE_LEASE_HELD.
    - acquired=False, refused=False -> FAIL OPEN: no fak binary / a git-store
                        failure / an unparseable verdict; the caller proceeds on
                        the same-host log scan alone, exactly as before the lease.
    """
    run = runner or _run_lease
    session_id = lease_session_id()
    requested_holder = holder or lease_owner_override()
    session_publish: dict[str, Any] | None = None
    if not session_id and not requested_holder:
        session_id = synthetic_lease_session_id()
        session_publish = publish_lease_session(root, session_id, ttl_s,
                                                runner=run)
    lease_id = lease_id_for_lane(lane)
    args = ["acquire", "--id", lease_id, "--ttl", str(int(ttl_s))]
    if requested_holder:
        args += ["--holder", requested_holder]
    elif not session_id:
        requested_holder = lease_holder()
        args += ["--holder", requested_holder]
    if session_id:
        args += ["--session", session_id]
    for t in tree:
        args += ["--tree", t]
    res = run(root, args)
    rc = res.get("rc")
    verdict = res.get("verdict") or {}
    if rc == 0:
        rec = verdict.get("record") if isinstance(verdict, dict) else None
        rec_holder = (rec or {}).get("holder") if isinstance(rec, dict) else None
        actual_holder = rec_holder or requested_holder
        out = {"acquired": True, "refused": False, "id": lease_id, "holder": actual_holder,
               "generation": (rec or {}).get("generation") if isinstance(rec, dict) else None,
               "session_id": session_id,
               "tree": tree}
        if session_publish is not None:
            out["session_publish"] = session_publish
        return out
    if rc == LEASE_REFUSED_RC:
        v = verdict.get("verdict") if isinstance(verdict, dict) else None
        out = {"acquired": False, "refused": True, "id": lease_id, "holder": requested_holder,
               "session_id": session_id,
               "reason": (v or {}).get("reason") if isinstance(v, dict) else None,
               "fence_verdict": v, "tree": tree}
        if session_publish is not None:
            out["session_publish"] = session_publish
        return out
    # rc in {1,2,127} or unparseable -> fail open (no protection added, loop never wedges).
    out = {"acquired": False, "refused": False, "id": lease_id, "holder": requested_holder,
           "session_id": session_id,
           "fail_open": True, "rc": rc, "detail": res.get("error") or res.get("skipped"),
           "tree": tree}
    if session_publish is not None:
        out["session_publish"] = session_publish
    return out


def release_lane_lease(root: Path, lease: dict[str, Any] | None,
                       *, runner: Any | None = None) -> dict[str, Any]:
    """Release a previously-acquired lane lease with holder/generation fencing.

    This never uses ``--force``. A release failure is non-fatal: the original lease
    TTL plus ``reap_expired_leases`` remains the crash backstop, so this helper can
    only improve concurrency by freeing a finished worker's lane earlier.
    """
    src = lease or {}
    lease_id = str(src.get("id") or "").strip()
    holder = str(src.get("holder") or "").strip()
    if not lease_id or not holder:
        return {"released": False, "skipped": "missing lease id/holder"}
    args = ["release", "--id", lease_id, "--holder", holder]
    generation = src.get("generation")
    if generation is not None:
        args += ["--generation", str(generation)]
    run = runner or _run_lease
    res = run(root, args)
    rc = res.get("rc")
    verdict = res.get("verdict") or {}
    if rc == 0:
        return {"released": True, "refused": False, "id": lease_id,
                "holder": holder, "generation": generation}
    if rc == LEASE_REFUSED_RC:
        v = verdict.get("verdict") if isinstance(verdict, dict) else None
        return {"released": False, "refused": True, "id": lease_id,
                "holder": holder, "generation": generation,
                "reason": (v or {}).get("reason") if isinstance(v, dict) else None,
                "fence_verdict": v}
    return {"released": False, "refused": False, "id": lease_id,
            "holder": holder, "generation": generation, "fail_open": True,
            "rc": rc, "detail": res.get("error") or res.get("skipped")}


def reap_expired_leases(root: Path, *, runner: Any | None = None) -> dict[str, Any]:
    """Delete expired (reapable) refs/fak/locks/* leases — the crashed-holder
    backstop. A worker (or a same-tick spawn that died at exec) that never gets to
    drop its lane lease has it swept here once the TTL lapses, so a crash can't
    wedge a lane forever. Fail-open: a reap failure is reported, never raised.

    The normal path now releases with the holder/generation sidecar once
    :func:`witness_exited_workers` proves the worker pid is dead. This reap remains
    the crash/failure backstop: if the process dies before sidecar write, the release
    command is unavailable, or the ref store refuses, the expired lease is still
    bounded by TTL."""
    run = runner or _run_lease
    res = run(root, ["reap"])
    return {"ok": res.get("rc") == 0, "rc": res.get("rc"), "stdout": res.get("stdout")}


def live_lane_lease_lanes(root: Path, *, runner: Any | None = None) -> dict[str, Any]:
    """Read live refs/fak/locks lane leases for planner-side collision avoidance.

    This is advisory visibility for dry-run/ranking, not the spawn fence itself:
    live spawning still calls acquire_lane_lease() and must win the atomic lease.
    A read failure fails open so a broken local fak binary does not wedge the loop.
    """
    run = runner or _run_lease
    res = run(root, ["live"])
    if res.get("rc") != 0:
        return {"lanes": [], "fail_open": True, "rc": res.get("rc"),
                "detail": res.get("error") or res.get("skipped")}
    doc = res.get("verdict")
    if not isinstance(doc, list):
        return {"lanes": [], "fail_open": True, "rc": res.get("rc"),
                "detail": "unparseable leaseref live output"}
    lanes: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in doc:
        if not isinstance(row, dict):
            continue
        records.append(row)
        lane = str(row.get("lane") or row.get("id") or "").strip()
        if lane.startswith(LEASE_ID_PREFIX):
            lane = lane[len(LEASE_ID_PREFIX):]
        if lane:
            lanes.add(lane)
    return {"lanes": sorted(lanes), "records": records}


# A per-process cache of the dos.toml lane->trees map so a tick that probes several
# lanes shells `dos doctor` at most once. Keyed by workspace string.
_LANE_TREE_CACHE: dict[str, dict[str, list[str]]] = {}


def lane_tree(root: Path, lane: str) -> list[str]:
    """The repo-relative file-tree globs the fenced lease should cover for `lane`,
    from dos.toml's [lanes.trees] (reused via issue_lane_router.lane_taxonomy), with
    an `internal/<lane>/**` fallback. The lease tree only matters for the arbiter's
    disjointness reasoning; the lease id (resolve-<lane>) is what actually
    serializes. Fail-open: a `dos doctor` failure falls back to the convention so
    the gate still acquires SOMETHING rather than refusing to protect."""
    key = str(root)
    trees = _LANE_TREE_CACHE.get(key)
    if trees is None:
        trees = {}
        try:
            import issue_lane_router  # noqa: PLC0415  (lazy: keep the import optional)
            _concurrent, trees, _exclusive = issue_lane_router.lane_taxonomy(root)
        except Exception:  # noqa: BLE001  (fail-open: any failure -> convention fallback)
            trees = {}
        _LANE_TREE_CACHE[key] = trees
    globs = trees.get(lane)
    if globs:
        return list(globs)
    return [f"internal/{lane}/**"]


# --- Multi-lane scope guard (#2615) ----------------------------------------
# A contract score of 100 proves an issue is well-DESCRIBED; it says nothing about
# whether the ONE lane the router collapsed it onto can protect every file family
# the body names. The router already sees the collision — it computes `path_lanes`
# across all named paths and flags `path-ambiguous` when they span >1 lane — but it
# silently picks a single lane, and the fenced lease then covers only that lane's
# tree. So a broad mechanical issue (#2233: callers across `.claude/**`, `.github/**`,
# `docs/**`, tools/…) admitted under a narrow lane (`ci` = `.github/**`) reads as
# WOULD_SPAWN while the worker is scoped to edit far outside the leased tree — the
# exact "safe-looking but unsafe in the shared tree" admission this guard refuses.
#
# The rule is the issue's own assumption: the LEASE TREE, not the label or lane
# string, is the collision authority. A named file family the chosen lane's lease
# does NOT cover but that belongs to ANOTHER concurrent lane is an unprotected
# mutation region -> refuse (or require an operator split / --force). Deliberately
# conservative in the refuse direction: an unprotected broad rewrite in the shared
# trunk is worse than a false refusal an operator can --force or split. Only rooted,
# recognized families count (issue_lane_router.named_repo_paths), so a bare prose
# path mention or a `tools/*.py` glob never trips it.
def multi_lane_scope(text: str, chosen_lane: str, chosen_tree: list[str],
                     trees: dict[str, list[str]], lane_set: set[str]) -> dict[str, Any]:
    """PURE: file families ``text`` names that ``chosen_lane``'s lease can't cover.

    ``chosen_tree`` is the lease's glob set (from :func:`lane_tree`); ``trees`` is the
    full ``dos.toml`` lane->globs map; ``lane_set`` the routable (concurrent) lanes.
    A named path is UNCOVERED when the lease tree does not match it AND it belongs to
    at least one OTHER routable lane. Returns ``{multi_lane, uncovered:[{path,lanes}],
    uncovered_lanes, covered_paths, chosen_lane, chosen_tree}``."""
    import issue_lane_router as ilr  # noqa: PLC0415  (lazy, same as lane_tree)
    lease_view = {"__lease__": list(chosen_tree or [])}
    covered: list[str] = []
    uncovered: list[dict[str, Any]] = []
    for p in ilr.named_repo_paths(text):
        if ilr.path_matches_lane(p, lease_view):
            covered.append(p)
            continue
        other = sorted(ln for ln in ilr.path_matches_lane(p, trees)
                       if ln in lane_set and ln != chosen_lane)
        if other:
            uncovered.append({"path": p, "lanes": other})
    uncovered_lanes = sorted({ln for u in uncovered for ln in u["lanes"]})
    return {"multi_lane": bool(uncovered), "chosen_lane": chosen_lane,
            "chosen_tree": list(chosen_tree or []), "covered_paths": covered,
            "uncovered": uncovered, "uncovered_lanes": uncovered_lanes}


def scan_multi_lane_scope(root: Path, text: str, chosen_lane: str) -> dict[str, Any]:
    """Resolve the taxonomy and run :func:`multi_lane_scope` for a live dispatch.

    Fail-open: a taxonomy read failure (no ``dos`` binary / broken store) returns
    ``{multi_lane: False, unavailable: True}`` so the guard can only ADD a refusal,
    never wedge the loop — the same discipline as the contract and lease gates."""
    try:
        import issue_lane_router as ilr  # noqa: PLC0415
        concurrent, trees, _exclusive = ilr.lane_taxonomy(root)
    except Exception as exc:  # noqa: BLE001  (fail-open: no taxonomy -> no guard)
        return {"multi_lane": False, "unavailable": True, "reason": str(exc)}
    if not trees:
        return {"multi_lane": False, "unavailable": True,
                "reason": "dos doctor returned no lane trees"}
    scan = multi_lane_scope(text, chosen_lane, lane_tree(root, chosen_lane),
                            trees, set(concurrent) or set(trees))
    scan["unavailable"] = False
    return scan


def multi_lane_scope_reason(issue: int, scan: dict[str, Any]) -> str:
    """One actionable line naming the uncovered families + the split/override path."""
    fam = ", ".join(f"{u['path']} ({'/'.join(u['lanes'])})"
                    for u in scan.get("uncovered") or [])
    return (f"issue #{issue} names file families outside lane "
            f"'{scan.get('chosen_lane')}' lease tree {scan.get('chosen_tree')}: {fam} "
            f"— the lease is the collision authority, and a narrow "
            f"'{scan.get('chosen_lane')}' lease would NOT protect "
            f"{scan.get('uncovered_lanes')} while the worker edits them. Split into "
            f"per-lane child issues, or re-dispatch under a lane whose lease covers "
            f"every named family (or --force to accept the under-scoped lease)")


# --- Dirty-path collision guard (#2977) -------------------------------------
# The lease checks protect live workers from each other, but a shared checkout may
# already contain uncommitted peer WIP with no live worker owning it. If the next
# issue explicitly names one of those dirty files, launching a new resolver would
# send it straight into a local pathspec collision. This is a collision guard, not
# a readiness gate: --force does not bypass it.
def parse_git_status_paths(status: str) -> list[str]:
    """Repo-relative paths from ``git status --porcelain=v1`` output."""
    out: list[str] = []
    for raw in (status or "").splitlines():
        if len(raw) < 4:
            continue
        spec = raw[3:].strip()
        parts = spec.split(" -> ", 1) if " -> " in spec else [spec]
        for p in parts:
            p = normalize_repo_path(p)
            if p and p not in out:
                out.append(p)
    return out


def normalize_repo_path(path: str) -> str:
    p = str(path or "").strip().strip('"').replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def dirty_repo_paths(root: Path) -> dict[str, Any]:
    rc, out = _git_capture(root, [
        "-c", "core.quotepath=false",
        "status", "--porcelain=v1", "--untracked-files=all",
    ], timeout=15)
    if rc != 0:
        return {"paths": [], "unavailable": True, "rc": rc}
    return {"paths": parse_git_status_paths(out), "unavailable": False}


def _path_text_variants(path: str) -> list[str]:
    p = normalize_repo_path(path)
    variants = [p] if p else []
    if p.startswith("fak/"):
        variants.append(p[len("fak/"):])
    elif p.startswith(("internal/", "cmd/", "experiments/")):
        variants.append("fak/" + p)
    return list(dict.fromkeys(variants))


def text_mentions_repo_path(text: str, path: str) -> bool:
    hay = (text or "").replace("\\", "/")
    for variant in _path_text_variants(path):
        if re.search(r"(?<![\w./-])" + re.escape(variant) + r"(?![\w./-])", hay):
            return True
    return False


def dirty_path_collision(text: str, dirty_paths: list[str]) -> dict[str, Any]:
    matches: list[str] = []
    for p in dirty_paths or []:
        norm = normalize_repo_path(p)
        if norm and text_mentions_repo_path(text, norm) and norm not in matches:
            matches.append(norm)
    return {"collides": bool(matches), "dirty_paths": matches}


def dirty_path_collision_reason(issue: int, scan: dict[str, Any]) -> str:
    paths = ", ".join(str(p) for p in (scan.get("dirty_paths") or [])[:8])
    more = ""
    if len(scan.get("dirty_paths") or []) > 8:
        more = f" (+{len(scan.get('dirty_paths') or []) - 8} more)"
    return (f"issue #{issue} names dirty local path(s) already modified in this "
            f"checkout: {paths}{more} — refusing DIRTY_PATH_COLLISION so a new "
            f"worker cannot overwrite peer WIP; wait for those paths to commit/"
            f"clear, or dispatch a disjoint issue")


_SAME_ISSUE_WIP_RE = re.compile(
    r"\b(uncommitted|working[- ]tree|working tree changes|left .* working[- ]tree|"
    r"worktree changes|left .* worktree|not committed|without a commit|"
    r"no commit|commit blocked)\b",
    re.IGNORECASE,
)
_SAME_ISSUE_WIP_CLAIMS = frozenset({"CLAIM_NO_COMMIT", "CLAIM_UNWITNESSED"})


def _read_log_tail(path: Path, *, max_bytes: int = 128 * 1024) -> str:
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes), os.SEEK_SET)
            except OSError:
                pass
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def same_issue_wip_collision(
    runs_dir: Path,
    issue: int | None,
    dirty_paths: list[str],
    *,
    lookback_min: int = SAME_ISSUE_WIP_LOOKBACK_MIN,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Detect same-issue uncommitted WIP left by a previous finished resolver.

    This is intentionally narrower than the generic dirty-path guard: it needs a
    recent ``resolve-<issue>-*.log`` (or no-commit witness) for the SAME issue and
    a still-dirty repo path named in that artifact. Unrelated dirty paths and old
    logs fail open so a hot shared tree can still dispatch disjoint work.

    Orientation docs (#4321) are excluded from the named-path intersection: the
    worker prompt names them by rote in every log, so their presence proves only
    that the worker was told to read them. See SAME_ISSUE_WIP_ORIENTATION_DOCS.
    Excluded names are still reported under ``orientation_paths`` so the
    suppression is auditable rather than silent.
    """
    if issue is None or lookback_min <= 0 or not runs_dir.is_dir() or not dirty_paths:
        return {"collides": False, "dirty_paths": [], "orientation_paths": []}
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - lookback_min * 60
    matches: list[dict[str, Any]] = []
    orientation: list[str] = []
    for log in sorted(runs_dir.glob(f"resolve-{int(issue)}-*.log"),
                      key=lambda p: p.stat().st_mtime if p.exists() else 0,
                      reverse=True):
        try:
            mtime = log.stat().st_mtime
        except OSError:
            continue
        if mtime < horizon:
            continue
        text = _read_log_tail(log)
        witness_claim = ""
        witness_path = log.with_suffix(WITNESS_SIDECAR_SUFFIX)
        if witness_path.exists():
            try:
                doc = json.loads(witness_path.read_text(encoding="utf-8"))
                witness_claim = str(doc.get("claim") or "")
            except (OSError, ValueError, AttributeError):
                witness_claim = ""
        if not (_SAME_ISSUE_WIP_RE.search(text)
                or witness_claim in _SAME_ISSUE_WIP_CLAIMS):
            continue
        dirty = []
        for p in dirty_paths:
            norm = normalize_repo_path(p)
            if not norm or not text_mentions_repo_path(text, norm):
                continue
            if is_orientation_doc(norm):
                if norm not in orientation:
                    orientation.append(norm)
                continue
            if norm not in dirty:
                dirty.append(norm)
        if dirty:
            matches.append({
                "log": log.name,
                "path": str(log),
                "mtime": mtime,
                "witness_claim": witness_claim or None,
                "dirty_paths": dirty,
            })
    if not matches:
        return {"collides": False, "dirty_paths": [],
                "orientation_paths": orientation}
    paths: list[str] = []
    for match in matches:
        for p in match.get("dirty_paths") or []:
            if p not in paths:
                paths.append(p)
    return {
        "collides": True,
        "issue": int(issue),
        "dirty_paths": paths,
        "orientation_paths": orientation,
        "evidence": matches[:3],
    }


def same_issue_wip_reason(issue: int, scan: dict[str, Any]) -> str:
    paths = ", ".join(str(p) for p in (scan.get("dirty_paths") or [])[:8])
    more = ""
    if len(scan.get("dirty_paths") or []) > 8:
        more = f" (+{len(scan.get('dirty_paths') or []) - 8} more)"
    evidence = scan.get("evidence") or []
    first = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
    log = first.get("log") if first else None
    claim = first.get("witness_claim") if first else None
    source = f" in {log}" if log else ""
    if claim:
        source += f" ({claim})"
    return (f"issue #{issue} has recent same-issue uncommitted WIP{source} "
            f"naming dirty local path(s): {paths}{more} — refusing "
            f"SAME_ISSUE_WIP so a second resolver cannot stack onto unfinished "
            f"work; continue/commit those paths first, or dispatch a disjoint issue")


# --- Commit-time diff-witness binding (residual of #1310/#1324, proposal #2) ---
# The fenced lane-lease above is the UPSTREAM half of the verified loop: it denies a
# colliding spawn before it starts. This is the DOWNSTREAM half: a worker's slot only
# counts as PRODUCTIVE once the commit it actually landed passes `dos commit-audit`
# (diff-witnessed). Before this, the slot counted as productive on a bare worker
# `exit 0` / a non-empty log, so an unwitnessed — or wrong-issue — commit silently
# claimed the slot. Here each finished worker's commit is re-audited through the same
# non-forgeable witness rung the close arm already uses (issue_resolve_witnessed
# .reverify), pulled forward to the dispatch slot, and the verdict is RECORDED:
# CLAIM_WITNESSED / CLAIM_UNWITNESSED / CLAIM_NO_COMMIT.
#
# Per-worker commit-sha tracking: the .basesha sidecar stamped at launch scopes the
# re-audit to the commit THIS worker landed (base..HEAD citing its #issue), not
# whatever a sibling pushed to HEAD. EVERYTHING here is FAIL-OPEN and DEAD-PID gated
# (a live worker may not have committed yet — never mis-blame it), the same discipline
# as prune_dead_sidecars and the backend-health classifier, so it can only ever ADD a
# recorded verdict, never wedge the loop. When a lease sidecar exists, the same sweep
# also releases it with holder/generation fencing; TTL+reap remains the fallback.
CLAIM_WITNESSED = "CLAIM_WITNESSED"
CLAIM_UNWITNESSED = "CLAIM_UNWITNESSED"
CLAIM_NO_COMMIT = "CLAIM_NO_COMMIT"
# The DOS witness rung a "truly resolved" commit must clear — the same non-forgeable
# keep-bit issue_closure_audit / issue_resolve_witnessed grade against.
_WITNESS_OK = "diff-witnessed"

# Why a FINISHED worker landed no resolving commit. A CLAIM_NO_COMMIT is not one
# thing: a SELF_MODIFY / POLICY_BLOCK guard refusal is a worker that TRIED and was
# STRUCTURALLY blocked (re-dispatching it re-blocks identically — an anti-churn
# cooldown should hold it, not re-storm it); an AUTH_WALL is a transient credit cap
# (re-probe once the window resets); a BANNER_NOOP is a DOA backend (the
# check_backend_health DEAD streak's per-spawn evidence). UNKNOWN preserves the prior
# opaque behavior. The reason is recorded in the .witness sidecar so a downstream
# picker can route by it instead of re-grepping raw worker logs.
NO_COMMIT_SELF_MODIFY = "self_modify"
NO_COMMIT_POLICY_BLOCK = "policy_block"
NO_COMMIT_AUTH_WALL = "auth_wall"
NO_COMMIT_OFF_TRUNK = "off_trunk"
NO_COMMIT_BANNER_NOOP = "banner_noop"
NO_COMMIT_PREVIEW_CONFIRM = "preview_confirm_feedback"
NO_COMMIT_MISSING_LOG = "missing_log_artifact"
NO_COMMIT_RESTART_EXHAUSTED = "restart_exhausted"
NO_COMMIT_UNKNOWN = "unknown"

# An opencode/glm worker that prints only its startup banner ("> build · glm-…") and
# exits — the documented banner-only no-op (#1275).
_NOOP_BANNER_RE = re.compile(r">\s*build\s*[·:]", re.IGNORECASE)
# The opencode/GLM (zai-coding-plan) quota wording, distinct from the Claude/codex
# "hit your … limit" that _CAP_BANNER_RE already matches.
_GLM_WALL_RE = re.compile(r"Limit Exhausted|limit will reset at|usage limit reached",
                          re.IGNORECASE)
_MISSING_API_KEY_RE = re.compile(
    r"Missing environment variable:\s*`?[A-Z0-9_]*API_KEY`?",
    re.IGNORECASE)
_PREVIEW_CONFIRM_FEEDBACK_RE = re.compile(
    r"(REQUIRE_WITNESS\s*/\s*ESCALATE|preview-confirm|_fak_confirm)",
    re.IGNORECASE)
# Guard writes resource/audit summaries after its managed-context terminal. The live
# epilogue can exceed the generic 4 KiB quota-banner tail, so give this precise typed
# signature a larger still-bounded window without broadening every classifier.
_RESTART_EXHAUSTED_TAIL_BYTES = 16 * 1024
_RESTART_EXHAUSTED_RE = re.compile(
    r"managed-context status (?P<status>restart_exhausted|reset_limit) "
    r"(?:limit=(?P<limit>\d+) )?(?:count=(?P<count>\d+) )?"
    r"(?:reason|dominant_cause)=(?P<cause>[^\s]+)", re.IGNORECASE)


def classify_restart_exhaustion(log: Path) -> dict | None:
    """Extract the typed guard terminal from the resolver process log."""
    tail = _log_tail_text(log, nbytes=_RESTART_EXHAUSTED_TAIL_BYTES)
    matches = list(_RESTART_EXHAUSTED_RE.finditer(tail))
    if not matches:
        return None
    match = matches[-1]
    count = match.group("count") or match.group("limit")
    return {"reason": NO_COMMIT_RESTART_EXHAUSTED,
            "restart_count": int(count) if count else None,
            "dominant_cause": match.group("cause")}


def classify_no_commit_reason(log: Path) -> str:
    """Why did a finished worker land no resolving commit? Classify from the log TAIL
    (the guard summary + final turn live at the end) so the witness records a
    STRUCTURED reason rather than an opaque CLAIM_NO_COMMIT. Lets an anti-churn picker
    tell a re-blockable guard refusal (self_modify / policy_block) from a transient
    wall (auth_wall) or a DOA backend (banner_noop). Pure + FAIL-OPEN: a missing log
    artifact gets a typed missing_log_artifact reason; no recognized signature stays
    UNKNOWN (never a false positive)."""
    try:
        if not log.exists():
            return NO_COMMIT_MISSING_LOG
    except OSError:
        return NO_COMMIT_MISSING_LOG
    restart = classify_restart_exhaustion(log)
    if restart is not None:
        return NO_COMMIT_RESTART_EXHAUSTED
    tail = _log_tail_text(log)
    if "SELF_MODIFY" in tail:
        return NO_COMMIT_SELF_MODIFY
    if "POLICY_BLOCK" in tail:
        return NO_COMMIT_POLICY_BLOCK
    if (_log_is_cap_banner(log) or _GLM_WALL_RE.search(tail)
            or _MISSING_API_KEY_RE.search(tail)):
        return NO_COMMIT_AUTH_WALL
    if "OFF_TRUNK" in tail:
        return NO_COMMIT_OFF_TRUNK
    if _PREVIEW_CONFIRM_FEEDBACK_RE.search(tail) and "exiting loop" in tail.lower():
        return NO_COMMIT_PREVIEW_CONFIRM
    try:
        small = log.stat().st_size <= _STUB_LOG_MAX_BYTES
    except OSError:
        small = False
    if small and _NOOP_BANNER_RE.search(tail):
        return NO_COMMIT_BANNER_NOOP
    return NO_COMMIT_UNKNOWN


# --- SPAWN_FAILED cause attribution (#2635) --------------------------------
# The ~1-in-25 baseline early-exit — a spawned worker that dies inside the <5s
# probe window (probe_spawned_worker) with an empty/stub log — is self-healing
# failover noise, but it is only ever LABELLED verdict=SPAWN_FAILED, never
# ATTRIBUTED. Lumping the causes means "it's just baseline noise" is an
# assumption, not a measured fact: a regression in ONE sub-cause could hide
# inside the ~4% aggregate. classify_spawn_failed_cause stamps each early-exit
# witness with ONE cause bucket from the log tail + exit code ALREADY captured,
# so the read-only fold (dispatch_status.spawn_failed_cause_breakdown) can report
# the trailing rate PER CAUSE instead of one opaque constant.
SPAWN_CAUSE_WEEKLY_LIMIT = "weekly_limit"
SPAWN_CAUSE_STALE_CRED = "stale_cred"
SPAWN_CAUSE_CHILD_CRASH = "child_crash"
SPAWN_CAUSE_EXEC_RACE = "exec_race"
SPAWN_CAUSE_UNKNOWN = "unknown"
SPAWN_FAILED_CAUSES = (
    SPAWN_CAUSE_WEEKLY_LIMIT, SPAWN_CAUSE_STALE_CRED, SPAWN_CAUSE_CHILD_CRASH,
    SPAWN_CAUSE_EXEC_RACE, SPAWN_CAUSE_UNKNOWN,
)

# A child_crash tail names a runtime fault the worker hit AFTER it exec'd — a real
# crash (interpreter traceback / Go panic / OOM / segfault / exec failure; cf.
# #2170), as opposed to a clean lease/exec race that never wrote past the pre-exec
# spawn header. Kept to unambiguous crash signatures so a benign line never
# false-blames a healthy exit.
_SPAWN_CRASH_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|panic:|goroutine \d+ \[running\]"
    r"|Segmentation fault|SIGSEGV|signal:\s*(?:killed|segmentation|abort)"
    r"|core dumped|std::bad_alloc|std::terminate"
    r"|out of memory|OOMKilled|Cannot allocate memory"
    r"|fatal error:"
    r"|exec format error|executable file not found|fork/exec",
    re.IGNORECASE)


def _spawn_header_only(tail: str) -> bool:
    """True when ``tail`` holds nothing past the pre-exec ``# fak-spawn`` header
    line(s) :func:`spawn_issue_worker` flushes BEFORE exec — i.e. the child never
    wrote a byte of its own. That is the on-disk signature of a same-tick
    lease/exec race (the OS ran nothing, or the child died at exec). An empty tail
    counts as header-only."""
    for line in tail.splitlines():
        if line.strip() and not line.startswith("# fak-spawn"):
            return False
    return True


def classify_spawn_failed_cause(early: dict[str, Any]) -> str:
    """Attribute ONE early-exit SPAWN_FAILED event to a cause bucket (#2635).

    ``early`` is the witness :func:`probe_spawned_worker` records for a worker that
    exited inside the probe window: it carries the worker-log ``tail``, the process
    ``returncode``, ``silent`` (0-byte log), and ``log_bytes``. Classify from those
    ALREADY-captured signals — no new probe, no new instrumentation:

    - ``weekly_limit``: the provider quota / 429 banner (``_cap_hit_from_text`` /
      ``_GLM_WALL_RE``) — the case #2610 cools the seat down for,
    - ``stale_cred``: a permanent auth gap — missing key / no resolved login
      (``_AUTH_GAP_RE`` / ``_MISSING_API_KEY_RE``): a seat that cannot authenticate,
    - ``child_crash``: the child exec'd then died on a runtime fault (traceback /
      panic / OOM / segfault / exec failure — ``_SPAWN_CRASH_RE``; cf. #2170),
    - ``exec_race``: an empty / header-only log — the child never wrote past the
      pre-exec spawn header: a same-tick lease/exec race,
    - ``unknown``: a non-empty tail with no recognized signature — kept HONEST so a
      genuinely new cause surfaces as a rising ``unknown`` share, not a silent
      misfile into one of the known buckets.

    Pure + FAIL-OPEN: precedence runs most-specific first (quota, then cred, then
    crash) and any doubt falls through to ``exec_race``/``unknown`` — never a false
    ``weekly_limit``/``stale_cred`` attribution."""
    tail = str(early.get("tail") or "")
    if _cap_hit_from_text(tail) is not None or _GLM_WALL_RE.search(tail):
        return SPAWN_CAUSE_WEEKLY_LIMIT
    if _AUTH_GAP_RE.search(tail) or _MISSING_API_KEY_RE.search(tail):
        return SPAWN_CAUSE_STALE_CRED
    if _SPAWN_CRASH_RE.search(tail):
        return SPAWN_CAUSE_CHILD_CRASH
    if early.get("silent") or _spawn_header_only(tail):
        return SPAWN_CAUSE_EXEC_RACE
    return SPAWN_CAUSE_UNKNOWN


# The RE-BLOCKABLE terminal guard refusals: a worker that TRIED and was STRUCTURALLY
# blocked by a guard (SELF_MODIFY / POLICY_BLOCK). Re-dispatching the SAME issue
# re-hits the SAME guard and burns the budget again for no commit (#1396: two ticks
# re-picked #1338 -> SELF_MODIFY then POLICY_BLOCK, ~4.75M token-equiv, 0 commit). An
# AUTH_WALL is deliberately NOT in this set -- it is a transient credit cap that the
# picker SHOULD re-probe after its cooldown window; a BANNER_NOOP is owned by the
# backend-health self-suppress gate; OFF_TRUNK / UNKNOWN fall through to the plain
# time cooldown. Only re-blockable guard feedback is held by structure.
_HOLD_NO_COMMIT_REASONS = frozenset({
    NO_COMMIT_SELF_MODIFY,
    NO_COMMIT_POLICY_BLOCK,
    NO_COMMIT_PREVIEW_CONFIRM,
})


def held_no_commit_issues(witnessed: dict[str, Any] | None) -> set[int]:
    """Issue numbers to HOLD this tick because their last FINISHED worker landed no
    resolving commit for a RE-BLOCKABLE structural reason (self_modify / policy_block).
    Read from the in-memory witness result :func:`witness_exited_workers` just recorded
    -- the same ``reason`` it stamped into each ``.witness`` sidecar, so the picker
    routes by the recorded reason instead of re-grepping raw worker logs. This is the
    pick-held-invariant rung for the bare verb (#1396): a SELF_MODIFY / POLICY_BLOCK
    exit on issue N is a guard refusal a re-dispatch would hit identically, so the
    picker must HOLD N this tick rather than re-storm it. An AUTH_WALL is NOT held here
    (it re-probes after the time cooldown window); a BANNER_NOOP is owned by the
    backend-health gate. Pure + FAIL-OPEN: a missing/odd record is skipped, never
    raised, so it can only ever ADD a hold, never wedge the picker."""
    held: set[int] = set()
    for rec in ((witnessed or {}).get("no_commit") or []):
        if isinstance(rec, dict) and rec.get("reason") in _HOLD_NO_COMMIT_REASONS:
            try:
                held.add(int(rec["issue"]))
            except (KeyError, TypeError, ValueError):
                continue
    return held


def _subject_cites_issue(subject: str, issue: int) -> bool:
    """True when a commit ``subject`` names ``#<issue>`` at a word boundary — the
    same binding key issue_closure_audit uses. A subject that does not name the
    worker's issue is not this worker's resolving commit (the "wrong-issue commit"
    the slot must not claim). The leading boundary keeps ``#1324`` from matching a
    glued ``...#1324`` token while still matching a normal ``(#1324)``."""
    return re.search(rf"(?<![\w-])#{int(issue)}\b", subject or "") is not None


def worker_resolving_sha(root: Path, issue: int, *, base_sha: str | None = None,
                         git: Any | None = None, scan_limit: int = 300) -> str | None:
    """The newest commit whose SUBJECT cites ``#<issue>`` — the commit THIS worker
    landed for its assigned issue. Scoped to ``base_sha..HEAD`` (the per-worker
    window recorded at spawn) when the base is known, else the most recent
    ``scan_limit`` commits. Returns None when no such commit exists — the worker
    landed nothing for its issue, or committed a wrong-issue subject — so the slot
    claims nothing. Fail-open: any git error yields None."""
    git = git or _git_capture
    rev_args = ["log", "--no-color", "--pretty=format:%H\x1f%s"]
    if base_sha:
        rev_args.append(f"{base_sha}..HEAD")
    else:
        rev_args += ["-n", str(int(scan_limit))]
    rc, out = git(root, rev_args)
    if rc != 0 or not out:
        return None
    for line in out.splitlines():
        sha, _sep, subject = line.partition("\x1f")
        if sha and _subject_cites_issue(subject, issue):
            return sha.strip()
    return None


def _run_commit_audit(root: Path, sha: str, *, timeout: int = 60) -> dict[str, Any]:
    """Default ``dos commit-audit <sha> --json`` runner -> the row for ``sha`` (the
    command emits a JSON array, one row per audited sha). Never raises; an exec/parse
    failure yields ``{}`` so the caller fails open to "not witnessed"."""
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(["dos", "commit-audit", sha, "--workspace", str(root),
                               "--json"], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = (proc.stdout or "").strip()
    try:
        parsed = json.loads(out) if out else []
    except ValueError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for row in parsed:
            if isinstance(row, dict) and row.get("sha") and str(sha).startswith(str(row["sha"])):
                return row
        if parsed and isinstance(parsed[0], dict):
            return parsed[0]
    return {}


def audit_commit_witness(root: Path, sha: str, *, runner: Any | None = None) -> dict[str, Any]:
    """Grade ``sha`` through the DOS witness rung: ``{witnessed, verdict, witness}``.
    ``witnessed`` is True only on verdict OK AND a diff-witness — the same keep-bit
    issue_resolve_witnessed.reverify uses at close time, applied here at the dispatch
    slot. Fail-open: a missing/empty audit -> witnessed False (the conservative slot
    verdict, never a silent productive claim)."""
    run = runner or _run_commit_audit
    doc = run(root, sha) or {}
    verdict = str(doc.get("verdict") or "")
    witness = str(doc.get("witness") or "")
    witnessed = verdict.upper() == "OK" and witness == _WITNESS_OK
    return {"witnessed": witnessed, "verdict": verdict or None, "witness": witness or None}


_ISSUE_TOKEN_RE = re.compile(r"(?<![\w-])#(\d+)\b")


def _run_commit_audit_rows(root: Path, ref: str, *, timeout: int = 60) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(["dos", "commit-audit", ref, "--workspace", str(root),
                               "--json"], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        parsed = json.loads((proc.stdout or "").strip() or "[]")
    except ValueError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    return []


def locally_witnessed_issues(root: Path, *, base_ref: str = "origin/main",
                             git: Any | None = None,
                             audit_rows: Any | None = None) -> set[int]:
    """Open-loop duplicate guard for a hot shared tree.

    A resolving commit may be present locally but not yet pushed/closed because a
    pre-push guard or a peer commit blocks the range. Those issues are still open
    on GitHub, so the normal picker would respawn them. Skip only issues cited by
    commits in ``base_ref..HEAD`` that the DOS witness already grades OK.
    """
    git = git or _git_capture
    rc, out = git(root, ["log", "--no-color",
                         "--pretty=format:%H%x1f%B%x1e",
                         f"{base_ref}..HEAD"])
    if rc != 0 or not out:
        return set()
    cited: dict[str, set[int]] = {}
    for entry in out.split("\x1e"):
        entry = entry.strip()
        if not entry:
            continue
        sha, sep, text = entry.partition("\x1f")
        if not sep or not sha:
            continue
        nums = {int(m.group(1)) for m in _ISSUE_TOKEN_RE.finditer(text or "")}
        if nums:
            cited[sha.strip()] = nums
    if not cited:
        return set()
    rows = audit_rows(root, f"{base_ref}..HEAD") if audit_rows else _run_commit_audit_rows(
        root, f"{base_ref}..HEAD")
    witnessed_shas = {
        str(r.get("sha") or "") for r in rows
        if str(r.get("verdict") or "").upper() == "OK"
        and str(r.get("witness") or "") == _WITNESS_OK
    }
    out_issues: set[int] = set()
    for sha, nums in cited.items():
        if any(sha.startswith(w) or w.startswith(sha) for w in witnessed_shas if w):
            out_issues.update(nums)
    return out_issues


def _candidate_issue_numbers(eligible_lanes: list[Any] | None,
                             fallback: list[int] | None = None) -> set[int]:
    out: set[int] = set()
    for entry in eligible_lanes or []:
        try:
            nums = entry[1] or []
        except (TypeError, IndexError, KeyError):
            continue
        for n in nums:
            try:
                out.add(int(n))
            except (TypeError, ValueError):
                continue
    if not out:
        for n in fallback or []:
            try:
                out.add(int(n))
            except (TypeError, ValueError):
                continue
    return out


def commit_audit_abstain_holds(root: Path, candidates: set[int], *,
                               git: Any | None = None,
                               audit_runner: Any | None = None,
                               scan_limit: int = 300) -> list[dict[str, Any]]:
    """Recent candidate-bound test commits whose witness ABSTAINs become holds.

    A still-open issue with a matching test-only commit should not be normal
    resolver fuel again: either the closure witness needs to learn the shape, or a
    human needs to handle the mismatch. Fail-open on git/audit errors.
    """
    wanted = {int(n) for n in candidates if n is not None}
    if not wanted:
        return []
    git = git or _git_capture
    audit = audit_runner or _run_commit_audit
    rc, out = git(root, ["log", "--no-color", "-n", str(int(scan_limit)),
                         "--pretty=format:%H%x1f%B%x1e"])
    if rc != 0 or not out:
        return []
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in out.split("\x1e"):
        entry = entry.strip()
        if not entry:
            continue
        sha, sep, text = entry.partition("\x1f")
        if not sep or not sha:
            continue
        nums = {int(m.group(1)) for m in _ISSUE_TOKEN_RE.finditer(text or "")} & wanted
        nums -= seen
        if not nums:
            continue
        first_line = next((line.strip() for line in (text or "").splitlines()
                           if line.strip()), "")
        row = audit(root, sha.strip()) or {}
        verdict = str(row.get("verdict") or "").upper()
        if verdict != "ABSTAIN":
            continue
        test_files = row.get("test_files") or []
        test_subject = first_line.startswith("test(") or first_line.startswith("test:")
        if not test_files and not test_subject:
            continue
        for issue in sorted(nums):
            records.append({
                "issue": issue,
                "sha": sha.strip(),
                "code": "COMMIT_AUDIT_ABSTAIN",
                "verdict": verdict,
                "witness": row.get("witness"),
                "claim_kind": row.get("claim_kind"),
                "reason": row.get("reason") or "commit-audit abstained",
                "test_files": test_files,
            })
            seen.add(issue)
    return records


# --- Pre-dispatch OPEN_WITNESSED closure guard (#5071) ---
# On 2026-07-16 the live resolver dispatched #2850 even though two resolving
# commits (f8aff29dfd, 8d0d6f620f) were already on main AND origin/main: the
# locally_witnessed_issues guard above only covers the UNPUSHED origin/main..HEAD
# window, so an issue that is open on GitHub while its resolving commit sits in
# full trunk ancestry (the closure auditor's OPEN_WITNESSED bucket) was admitted
# as engineering fuel, burned a ~18-minute seat, and wrote a false CLAIM_NO_COMMIT.
# A witness-gated loop must CONSUME already-shipped open issues by closing them,
# never redispatch them. This guard joins issue selection to the same
# resolving-commit witness issue_closure_audit grades against, and the close is
# routed through the EXISTING trusted close arm (issue_resolve_witnessed.py) so
# every close still re-verifies per-sha and keeps that arm's pushed / reopen /
# coverage / readback gates. Everything here is FAIL-OPEN: a git or audit error
# witnesses nothing (the tick dispatches exactly as before).
OPEN_WITNESSED = "OPEN_WITNESSED"


def open_witnessed_dispositions(root: Path, candidates: set[int], *,
                                git: Any | None = None,
                                audit_runner: Any | None = None,
                                scan_limit: int = 600) -> list[dict[str, Any]]:
    """Typed OPEN_WITNESSED dispositions for candidates already resolved in trunk.

    A candidate is disposed OPEN_WITNESSED when a commit in HEAD ancestry (the
    trunk the dispatcher stands on — a commit outside that ancestry is never
    consulted) classifies RESOLVING for it under the closure auditor's shared
    grammar (issue_closure_audit.classify_refs — a mere body mention is
    insufficient) AND the DOS witness grades that commit OK / diff-witnessed
    with a resolution-binding claim kind. Each disposition carries the commit
    witness (sha, subject, verdict, witness, claim_kind) so the tick can report
    the witnessed SHA and route the close. An unwitnessed resolving cite keeps
    scanning older commits; an issue with no witnessed resolving commit is NOT
    disposed and stays eligible. Fail-open on any git/audit error.
    """
    wanted = {int(n) for n in candidates if n is not None}
    if not wanted:
        return []
    git = git or _git_capture
    audit = audit_runner or _run_commit_audit
    rc, out = git(root, ["log", "--no-color", "-n", str(int(scan_limit)),
                         "--pretty=format:%H%x1f%s%x1f%b%x1e"])
    if rc != 0 or not out:
        return []
    rows: list[dict[str, Any]] = []
    decided: set[int] = set()
    audited: dict[str, dict[str, Any]] = {}
    for entry in out.split("\x1e"):
        entry = entry.strip("\n")
        if not entry.strip():
            continue
        parts = entry.split("\x1f")
        if len(parts) < 2:
            continue
        sha = parts[0].strip()
        subject = parts[1].strip()
        body = parts[2] if len(parts) > 2 else ""
        if not sha:
            continue
        refs = issue_closure_audit.classify_refs(subject, body)
        nums = sorted(n for n, kind in refs.items()
                      if kind == issue_closure_audit.RESOLVING
                      and n in wanted and n not in decided)
        if not nums:
            continue
        row = audited.get(sha)
        if row is None:
            row = audit(root, sha) or {}
            audited[sha] = row
        verdict = str(row.get("verdict") or "").upper()
        witness = str(row.get("witness") or "")
        if verdict != "OK" or witness != _WITNESS_OK:
            # A resolving cite without the diff-witness keep-bit is not shipped
            # evidence (preserve the existing witness rules); keep scanning.
            continue
        if not issue_closure_audit.commit_binds_resolution(row, ""):
            # A doc/triage claim never resolves a (presumed non-docs) issue; the
            # empty title is the conservative direction — the issue stays
            # eligible rather than being closed on a doc claim.
            continue
        for issue in nums:
            rows.append({
                "issue": issue, "sha": sha, "subject": subject,
                "code": OPEN_WITNESSED,
                "verdict": row.get("verdict"), "witness": row.get("witness"),
                "claim_kind": row.get("claim_kind"),
                "close_via": "tools/issue_resolve_witnessed.py",
            })
            decided.add(issue)
        if decided >= wanted:
            break
    return rows


def _run_close_arm(root: Path, cmd: list[str], *, timeout: int = 300) -> tuple[int, str]:
    """Default close-arm runner -> (returncode, stdout). Never raises."""
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or "").strip()


def close_open_witnessed(root: Path, runs_dir: Path, rows: list[dict[str, Any]], *,
                         live: bool, runner: Any | None = None) -> dict[str, Any]:
    """Route OPEN_WITNESSED dispositions through the EXISTING trusted close path.

    The dispatcher never closes an issue itself: it hands the witnessed rows to
    issue_resolve_witnessed.py (the close-resolved arm) as a synthetic
    OPEN_WITNESSED audit via --audit-json, so every close still re-verifies its
    sha through `dos commit-audit` at close time and keeps the arm's pushed /
    reopen / coverage / state-readback gates. A non-live invocation plans the
    closes dry-run (the arm's default); a failed or unparseable invocation
    fails open — the skip-set already prevented the redispatch this tick, so
    closure merely lags to a later tick or a standalone close-arm run.
    """
    if not rows:
        return {}
    run = runner or _run_close_arm
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    audit_path = runs_dir / f"open-witnessed-{stamp}.audit.json"
    synthetic = {
        "schema": "issue-closure-audit/synthetic/open-witnessed-dispatch/1",
        "issues": [{"number": int(r["issue"]), "bucket": OPEN_WITNESSED,
                    "title": str(r.get("title") or ""),
                    "witnessed_commits": [{"sha": str(r.get("sha") or ""),
                                           "subject": str(r.get("subject") or "")}]}
                   for r in rows if r.get("issue") is not None],
    }
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    except OSError as exc:
        return {"invoked": False, "error": f"could not write synthetic audit: {exc}"}
    cmd = [_py(), str(root / "tools" / "issue_resolve_witnessed.py"),
           "--workspace", str(root), "--audit-json", str(audit_path),
           "--limit", str(len(rows)), "--json"]
    if live:
        cmd.append("--live")
    rc, out = run(root, cmd)
    try:
        doc = json.loads(out or "{}")
    except ValueError:
        doc = {}
    if not isinstance(doc, dict) or not doc:
        return {"invoked": True, "ok": False, "returncode": rc, "live": live,
                "error": (out or "close arm produced no JSON")[-300:]}
    return {"invoked": True, "ok": bool(doc.get("ok")), "returncode": rc,
            "live": live, "verdict": doc.get("verdict"),
            "counts": doc.get("counts"),
            "closed_numbers": doc.get("closed_numbers"),
            "audit_json": str(audit_path)}


def witness_exited_workers(runs_dir: Path, root: Path, *, live: bool,
                           alive: set[int] | None = None, probe: Any | None = None,
                           git: Any | None = None,
                           audit_runner: Any | None = None,
                           lease_runner: Any | None = None) -> dict[str, Any]:
    """Bind each FINISHED worker's slot to a `dos commit-audit` witness (#1324
    proposal #2). For every ``resolve-<N>-<stamp>.log`` whose pid is provably DEAD and
    not yet witnessed (no ``.witness`` sidecar), find the commit it landed for its
    issue (:func:`worker_resolving_sha`) and grade it: a diff-witnessed commit ->
    CLAIM_WITNESSED (the slot was productive); an unwitnessed or wrong-issue commit ->
    CLAIM_UNWITNESSED; no resolving commit at all -> CLAIM_NO_COMMIT. The verdict is
    recorded in a ``.witness`` sidecar on live ticks so a bare ``exit 0`` / non-empty
    log never SILENTLY counts as productive.

    Dead-pid gated (a still-running worker may not have committed yet — never
    mis-blame it) and FAIL-OPEN throughout, exactly like :func:`prune_dead_sidecars`."""
    out: dict[str, Any] = {"live": live, "audited": [], "witnessed": [],
                           "unwitnessed": [], "no_commit": [],
                           "lease_released": [], "lease_release_failed": [],
                           "lease_release_retried": []}
    if not runs_dir.is_dir():
        return out
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore
            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None
    bucket_of = {CLAIM_WITNESSED: "witnessed", CLAIM_UNWITNESSED: "unwitnessed",
                 CLAIM_NO_COMMIT: "no_commit"}
    for log in sorted(runs_dir.glob("resolve-*.log")):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        witness_path = log.with_suffix(WITNESS_SIDECAR_SUFFIX)
        already_witnessed = witness_path.exists()
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue  # no pid -> cannot prove the worker finished -> not yet auditable
        if dispatch_preflight.resolve_sidecar_pid_is_live(pid_file, alive=alive, probe=probe):
            continue  # still running -> it may not have committed yet
        if already_witnessed:
            # Older ticks may have stamped the immutable commit witness before the
            # fenced lane-lease release path existed or after a transient release
            # failure. Retry the release for dead workers, but never re-audit the
            # commit verdict.
            if live:
                try:
                    witness_doc = json.loads(witness_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    witness_doc = {}
                prior_release = witness_doc.get("lease_release") if isinstance(witness_doc, dict) else {}
                if not (isinstance(prior_release, dict) and prior_release.get("released")):
                    lease = read_lease_sidecar(log)
                    if lease:
                        rel = release_lane_lease(root, lease, runner=lease_runner)
                        out["lease_release_retried"].append({
                            "log": log.name,
                            "id": lease.get("id"),
                            "reason": "already_witnessed_dead_worker",
                            "released": bool(rel.get("released")),
                        })
                        if rel.get("released"):
                            out["lease_released"].append(rel)
                        else:
                            out["lease_release_failed"].append(rel)
                        if isinstance(witness_doc, dict):
                            witness_doc["lease_release"] = rel
                            try:
                                witness_path.write_text(
                                    json.dumps(witness_doc, sort_keys=True),
                                    encoding="utf-8")
                            except OSError:
                                pass
            continue  # audited once; a commit's diff (so its verdict) is immutable
        issue = int(m.group(1))
        try:
            base = log.with_suffix(BASE_SHA_SIDECAR_SUFFIX).read_text(
                encoding="utf-8").strip() or None
        except OSError:
            base = None
        sha = worker_resolving_sha(root, issue, base_sha=base, git=git)
        if not sha:
            rec = {"issue": issue, "log": log.name, "sha": None,
                   "claim": CLAIM_NO_COMMIT, "verdict": None, "witness": None,
                   "reason": classify_no_commit_reason(log)}
            exhaustion = classify_restart_exhaustion(log)
            if exhaustion:
                rec.update(exhaustion)
        else:
            w = audit_commit_witness(root, sha, runner=audit_runner)
            rec = {"issue": issue, "log": log.name, "sha": sha,
                   "claim": CLAIM_WITNESSED if w["witnessed"] else CLAIM_UNWITNESSED,
                   "verdict": w["verdict"], "witness": w["witness"]}
        if live:
            lease = read_lease_sidecar(log)
            if lease:
                rel = release_lane_lease(root, lease, runner=lease_runner)
                rec["lease_release"] = rel
                if rel.get("released"):
                    out["lease_released"].append(rel)
                else:
                    out["lease_release_failed"].append(rel)
        out["audited"].append(rec)
        out[bucket_of[rec["claim"]]].append(rec)
        if live:
            try:
                log.with_suffix(WITNESS_SIDECAR_SUFFIX).write_text(
                    json.dumps(rec, sort_keys=True), encoding="utf-8")
            except OSError:
                pass
    return out


def is_dos_run_id(run_id: object) -> bool:
    return bool(_RID_RE.fullmatch(str(run_id or "")))


def mint_dos_run_id(root: Path, process: str, parent: str | None = None) -> str | None:
    """Mint the DOS RID that `dos status` accepts, fail-open to caller fallback."""
    cmd = ["dos", "run-id", "mint", process, "--root", str(root)]
    if parent:
        cmd += ["--parent", parent]
    kwargs: dict[str, Any] = {
        "cwd": root,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 30,
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        doc = json.loads(proc.stdout)
    except ValueError:
        return None
    rid = str(doc.get("run_id") or "")
    return rid if proc.returncode == 0 and is_dos_run_id(rid) else None


def _loop_metric_args(metrics: dict[str, int]) -> list[str]:
    out: list[str] = []
    for key in sorted(metrics):
        out += ["--metric", f"{key}={int(metrics[key])}"]
    return out


def _loop_evidence_args(evidence: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for kind, ref in evidence:
        if kind and ref:
            out += ["--evidence", f"{kind}={ref}"]
    return out


def _dispatch_collision_evidence(root: Path, payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Structured lane/lane_kind/mode/tree evidence for a refused dispatch tick
    (#4322): the canonical lane (not scraped from summary prose), its kind
    ("cluster" -- every lane this dispatcher fences is a tree-scoped
    refs/fak/locks lease, the same constant leaseref.ArbiterLaneKind names on
    the Go `fak dispatch tick` producer so a reader sees one grammar across
    both), its serialization mode (exclusive/shared, from the same
    EXCLUSIVE_LANES taxonomy dispatch_tick.go's tax.Exclusive check mirrors),
    and the requested tree/paths that collided. So per-lane collision rate and
    the WIP-vs-lease split are computable straight from loops.jsonl instead of
    regex-scraped out of the summary prose. Additive: an unrecognized evidence
    kind is simply unread by existing consumers.

    The WIP-vs-lease split above is what the ``refusal_class`` pair makes actually
    computable (#4321). Lane/tree/paths alone do not carry it: a lane-lease refusal
    and a working-tree co-tenancy refusal emit the SAME lane and tree, so a reader
    had to infer the split from verdict-name prose -- and both mechanisms spell
    themselves "collision" here (the lane-lease refusal's own reason text says
    "refusing COLLISION_RISK"), which is exactly how ~369 working-tree refusals were
    folded in with lease contention. Emitting the class as its own field makes the
    two separable by data. Emitted only for a verdict refusal_class() classifies, so
    a non-contention tick is unchanged.

    Deliberately silent on whether a LANE_LEASE_HELD blocker was live or
    stranded/dead -- that instrumentation is owned by #4324 and is not
    fabricated here.
    """
    out: list[tuple[str, str]] = []
    contention = refusal_class(str(payload.get("verdict") or ""))
    if contention:
        out.append(("refusal_class", contention))
    lane = str(payload.get("lane") or "").strip()
    if not lane:
        return out
    out.append(("lane", lane))
    out.append(("lane_kind", "cluster"))
    out.append(("mode", "exclusive" if lane in issue_lane_router.EXCLUSIVE_LANES else "shared"))
    lease = payload.get("lease") if isinstance(payload.get("lease"), dict) else {}
    tree = lease.get("tree") if isinstance(lease.get("tree"), list) else None
    if not tree:
        try:
            tree = lane_tree(root, lane)
        except Exception:  # noqa: BLE001  (fail-open: evidence stays lane-only)
            tree = None
    if tree:
        out.append(("tree", ",".join(str(t) for t in tree)))
    dirty = (payload.get("dirty_path_collision")
             if isinstance(payload.get("dirty_path_collision"), dict) else {})
    paths = dirty.get("dirty_paths") if isinstance(dirty.get("dirty_paths"), list) else None
    if paths:
        out.append(("paths", ",".join(str(p) for p in paths)))
    return out


def append_loop_event(root: Path, ledger: Path, event: dict[str, Any],
                      *, source: str = "issue_resolve_dispatch") -> dict[str, Any]:
    """Append one canonical loop-ledger row through `fak loop append`.

    Fail-open by design: a dispatcher tick must not fail just because the optional
    observability append could not find a binary. The error is returned into the
    tick payload for audit, while the dispatch verdict stays grounded in preflight.
    """
    cmd = fak_loop_cmd(root) + [
        "loop", "append",
        "--ledger", str(ledger),
        "--loop", str(event["loop_id"]),
        "--kind", str(event["kind"]),
        "--source", source,
        "--principal", str(event.get("principal") or event.get("backend") or "dispatcher"),
    ]
    for flag, key in (("--run", "run_id"), ("--status", "status"),
                      ("--reason", "reason"), ("--summary", "summary")):
        if event.get(key):
            cmd += [flag, str(event[key])]
    cmd += _loop_metric_args(event.get("metrics") or {})
    # `verified_state` rides the EVIDENCE channel, never a flag of its own. loopmgr.Event
    # has no VerifiedState field and `fak loop append` defines no --verified-state, so
    # passing one made the flag parser exit 2 ("flag provided but not defined") BEFORE
    # anything was appended. Only the witness event carries verified_state, so
    # fire/admit/end landed and every witness row was dropped — and because this wrapper
    # is deliberately fail-open the usage error went into the tick payload, where no
    # reader looks. `fak loop health` consequently read issue-resolve-progress at
    # 0-of-546 witnessed with witness_collapse=true for the whole life of the ledger.
    # The Go twin (cmd/fak/dispatch_progress.go) already encodes it the canonical way,
    # as an EvidenceRef{Kind: "verified_state"}; match it.
    evidence = list(event.get("evidence") or [])
    if event.get("verified_state"):
        evidence.append(("verified_state", str(event["verified_state"])))
    cmd += _loop_evidence_args(evidence)
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60,
                              creationflags=no_window_creationflags())
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "kind": event.get("kind"), "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "kind": event.get("kind"),
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip()[-500:],
        "stdout": (proc.stdout or "").strip()[-500:],
    }


def loop_run_id(payload: dict[str, Any], root: Path | None = None,
                mint: Any | None = None) -> str:
    existing = str(payload.get("run_id") or "")
    if is_dos_run_id(existing):
        return existing
    backend = str(payload.get("backend") or "claude").strip() or "claude"
    if root is not None:
        minted = (mint or mint_dos_run_id)(root, f"issue-resolve-dispatch-{backend}")
        if is_dos_run_id(minted):
            return str(minted)
    if existing:
        return existing
    spawned = payload.get("spawned") or {}
    if spawned.get("pid"):
        return f"resolve-{payload.get('target_issue') or 'none'}-{spawned.get('pid')}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"resolve-tick-{backend}-{stamp}"


def record_loop_tick(root: Path, payload: dict[str, Any],
                     *,
                     ledger: Path | None = None,
                     append: Any | None = None,
                     mint: Any | None = None) -> dict[str, Any]:
    """Lower one issue-resolve dispatcher tick into loop ledger events.

    The rows describe the DISPATCHER tick, not the spawned worker's eventual issue
    resolution. A successful spawn is therefore `claimed_done` for the tick only;
    the independent commit/issue witness remains the close/audit arm.
    """
    ledger = ledger or default_loop_ledger(root)
    append = append or append_loop_event
    run_id = str(loop_run_id(payload, root=root, mint=mint))
    payload["run_id"] = run_id
    loop_id = loop_id_for_payload(payload)
    pre = payload.get("preflight") or {}
    spawned = payload.get("spawned") or {}
    metrics: dict[str, int] = {
        "live": 1 if payload.get("live") else 0,
        "lane_issue_count": int(payload.get("lane_issue_count") or 0),
        "max_workers": int(payload.get("max_workers") or 0),
        "preflight_live": int(pre.get("live") or 0),
        "preflight_cap": int(pre.get("cap") or 0),
    }
    if payload.get("target_issue") is not None:
        metrics["target_issue"] = int(payload["target_issue"])
    if payload.get("prompt_chars") is not None:
        metrics["prompt_chars"] = int(payload["prompt_chars"])
    if spawned.get("pid"):
        metrics["pid"] = int(spawned["pid"])
    evidence: list[tuple[str, str]] = []
    if payload.get("target_issue") is not None:
        evidence.append(("issue", str(payload["target_issue"])))
    if spawned.get("log"):
        evidence.append(("log", str(spawned["log"])))
    if (payload.get("account") or {}).get("tag"):
        evidence.append(("account", str((payload.get("account") or {}).get("tag"))))
    admitted = bool(payload.get("ok")) and payload.get("action") in {"would_spawn", "spawned"}
    if not admitted:
        # #4322: a refused tick gets the structured lane/lane_kind/mode/tree
        # fields a reader needs for exact per-lane collision accounting,
        # instead of regex-scraping the lane out of the summary prose.
        evidence.extend(_dispatch_collision_evidence(root, payload))

    events: list[dict[str, Any]] = [{
        "loop_id": loop_id, "run_id": run_id, "kind": "fire",
        "backend": payload.get("backend"), "summary": f"issue dispatch tick lane={payload.get('lane') or '-'}",
        "metrics": metrics, "evidence": evidence,
    }]
    events.append({
        "loop_id": loop_id, "run_id": run_id, "kind": "admit",
        "backend": payload.get("backend"),
        "status": "admitted" if admitted else "refused",
        "reason": str(payload.get("verdict") or payload.get("action") or ""),
        "summary": str(payload.get("reason") or "")[:200],
        "metrics": metrics, "evidence": evidence,
    })
    if payload.get("action") == "spawned":
        events.append({
            "loop_id": loop_id, "run_id": run_id, "kind": "start",
            "backend": payload.get("backend"),
            "status": "running",
            "reason": "SPAWNED",
            "summary": str(payload.get("reason") or "")[:200],
            "metrics": metrics, "evidence": evidence,
        })
    if payload.get("ok"):
        events.append({
            "loop_id": loop_id, "run_id": run_id, "kind": "end",
            "backend": payload.get("backend"),
            "status": "claimed_done",
            "reason": str(payload.get("verdict") or payload.get("action") or ""),
            "summary": str(payload.get("reason") or "")[:200],
            "metrics": metrics, "evidence": evidence,
        })
    rows = [append(root, ledger, ev) for ev in events]
    return {
        "ledger": str(ledger),
        "loop_id": loop_id,
        "run_id": run_id,
        "events": rows,
        "ok": all(r.get("ok") for r in rows) if rows else True,
    }


def append_fleet_trend_row(root: Path, payload: dict[str, Any], *,
                           append: Any | None = None,
                           now: str | None = None) -> dict[str, Any]:
    """#4594: feed the fleet-status trend ledger one partial row per live tick.

    A single tick natively knows only its preflight live-worker count; the
    tolerant fleet_trend reader accepts a partial ``{ts, live}`` row, which is
    exactly the gauge the net-worker-decline alarm (#4591) consumes. The
    ``usable``/``sessions``/``escalate`` aggregates come from a `fleet_top`
    snapshot, not a tick, so they stay honestly absent rather than a fabricated
    zero. Best-effort: a trend append must never fail the tick."""
    do_append = append or fleet_trend.append
    pre = payload.get("preflight") or {}
    ledger = str(root / fleet_trend.DEFAULT_LEDGER)
    ts = now or dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    try:
        row = do_append(ledger, {"live": float(pre.get("live") or 0)}, ts)
        return {"ok": True, "ledger": ledger, "row": row}
    except Exception as exc:
        return {"ok": False, "ledger": ledger, "error": str(exc)}


def _contract_missing_fields(contract: dict[str, Any]) -> list[str]:
    review = contract.get("review") if isinstance(contract.get("review"), dict) else {}
    return [str(m) for m in (review.get("missing_fields") or []) if m]


def _maybe_dispatch_contract_repair(
    root: Path, runs_dir: Path, payload: dict[str, Any], *,
    rows: list[dict[str, Any]], live: bool, backend: str,
    acct: dict[str, Any], repair_batch: int, repair_cooldown_min: int,
    spawn_probe_s: float, product: str,
) -> dict[str, Any] | None:
    """The self-serve arm of the readiness gate: when a WHOLE scan window fails
    the issue-contract floor, dispatch ONE worker to bring the held issues up to
    contract themselves (``gh issue edit``, verified against the same gate)
    instead of idling until a human grooms the backlog -- with a ~700-issue
    pre-schema backlog, a scan-only tick otherwise converges to a permanent hold.

    Returns the completed tick payload (repair spawned / planned / in flight), or
    None when repair cannot run (disabled, or every held candidate is inside the
    repair cooldown) -- the caller falls through to the plain ISSUE_CONTRACT_HOLD
    with ``payload["contract_repair"]`` saying why. Admission: at most ONE repair
    worker at a time (serialized by the live repair-sidecar scan; it takes no
    lane lease because it never touches repo files), and the spawned worker
    occupies a preflight seat like any resolution worker."""
    if repair_batch <= 0:
        payload["contract_repair"] = {"skipped": "disabled (--repair-batch 0)"}
        return None
    in_flight = live_repair_workers(runs_dir)
    if in_flight:
        payload["contract_repair"] = {"in_flight": in_flight}
        payload.update({
            "ok": True, "action": "repair_in_flight", "verdict": "REPAIR_IN_FLIGHT",
            "reason": (f"every scanned candidate fails the contract gate and a "
                       f"contract-repair worker (pid {in_flight[0].get('pid')}) is "
                       f"already grooming the backlog — waiting for it, not "
                       f"stacking a second groomer onto the same issues")})
        return payload
    cooled = recently_repaired_issues(runs_dir, cooldown_min=repair_cooldown_min)
    batch = [r for r in rows if int(r.get("issue") or 0)
             and int(r.get("issue") or 0) not in cooled][:repair_batch]
    if not batch:
        payload["contract_repair"] = {
            "skipped": (f"all {len(rows)} held candidate(s) inside the "
                        f"{repair_cooldown_min}-min repair cooldown")}
        return None
    prompt_rows = [{
        "number": r.get("issue"),
        "title": r.get("title"),
        "missing_fields": r.get("missing_fields") or [],
        "reasons": [t for t in str(r.get("reason") or "").split(", ")
                    if t.startswith("ISSUE_")][:4],
    } for r in batch]
    rec = issue_worker_prompt.build_repair(
        prompt_rows, workspace=root, min_score=DEFAULT_ISSUE_CONTRACT_MIN_SCORE)
    payload["contract_repair"] = {"batch": rec["issues"],
                                  "prompt_chars": rec["prompt_chars"]}
    nums = ",".join(str(n) for n in rec["issues"])
    if not live:
        model, effort = worker_model_effort(backend, acct)
        preview_prompt = f"<contract-repair prompt, {rec.get('prompt_chars')} chars>"
        preview_env = dispatch_worker.child_env(REPAIR_LANE, backend, root)
        preview_command, preview_guarded = dispatch_worker.guarded_launch_command(
            build_worker_command(backend, preview_prompt, model, effort),
            REPAIR_LANE, backend, root, preview_env)
        payload["command"] = preview_command
        payload["guarded"] = preview_guarded
        payload["launch_gate"] = launch_gate_for_guard(preview_guarded, backend)
        payload.update({
            "ok": True, "action": "would_repair", "verdict": "WOULD_REPAIR",
            "reason": (f"every scanned candidate fails the contract gate; would "
                       f"spawn 1 {backend} contract-repair worker on issue(s) "
                       f"{nums} to bring them up to contract")})
        return payload
    model, effort = worker_model_effort(backend, acct)
    if backend == "claude":
        env = issue_dispatch.worker_env(acct.get("dir"), REPAIR_LANE, root)
    elif backend == "codex":
        env = codex_worker_env(acct.get("dir"), REPAIR_LANE, root)
    else:
        env = opencode_worker_env(acct.get("dir"), REPAIR_LANE, root, runs_dir)
    env["FLEET_REPAIR_ISSUES"] = nums
    command, guarded = dispatch_worker.guarded_launch_command(
        build_worker_command(backend, rec["prompt"], model, effort),
        REPAIR_LANE, backend, root, env)
    if guarded:
        dispatch_worker.guard_env_augment(env)
    payload["command"] = command
    payload["guarded"] = guarded
    spawned = spawn_issue_worker(command, env, root, runs_dir, rec["issues"][0],
                                 REPAIR_LANE, backend, account=acct,
                                 spawn_probe_s=spawn_probe_s, log_prefix="repair",
                                 prompt_payload=rec["prompt"] if backend in ("claude", "opencode") else None)
    try:
        Path(str(spawned.get("log") or "")).with_suffix(
            REPAIR_ISSUES_SIDECAR_SUFFIX).write_text(nums, encoding="utf-8")
    except OSError:
        pass  # cooldown degrades to first-issue-only; never blocks the spawn
    early = spawned.get("early_exit") or {}
    if early.get("checked") and not early.get("alive"):
        cap_hit = _cap_hit_from_text(
            str(early.get("tail") or ""),
            evidence_log=Path(str(spawned.get("log") or "")).name)
        if cap_hit:
            import time
            payload["quota_cap"] = _write_cap_hold(
                runs_dir, product=product, account_tag=acct.get("tag"),
                hit=cap_hit, now_ts=time.time(), fallback_min=60,
                source="early_exit")
        payload.update({
            "ok": False, "action": "spawn_failed", "verdict": "SPAWN_FAILED",
            "spawned": spawned,
            "cause": classify_spawn_failed_cause(early),
            # #4591: a repair spawn burns the same seat — count it against the
            # seat streak too so a dead seat can't hide behind repair rotations.
            "seat_streak": bump_spawn_failure_streak_seat(
                runs_dir, acct.get("tag"), backend),
            "reason": (f"{backend} contract-repair worker pid {spawned['pid']} "
                       f"for issue(s) {nums} exited within {early.get('wait_s')}s "
                       f"with code {early.get('returncode')}"
                       + (" and produced an empty log" if early.get("silent") else ""))})
        _record(runs_dir, payload)
        return payload
    # A clean repair launch proves the seat is alive: break its streak (#4591).
    clear_spawn_failure_streak_seat(runs_dir, acct.get("tag"), backend)
    payload.update({
        "ok": True, "action": "repair_spawned", "verdict": "REPAIR_SPAWNED",
        "spawned": spawned,
        "reason": (f"spawned {backend} contract-repair worker pid {spawned['pid']} "
                   f"on issue(s) {nums} — it edits the ISSUES up to contract "
                   f"(no lane lease; the repo tree is read-only for it)")})
    _record(runs_dir, payload)
    return payload


def seat_adaptive_target(pre: dict[str, Any], *, fallback: int, ceiling: int,
                         ramp_delta: int) -> tuple[int, dict[str, Any]]:
    """Seat-adaptive effective worker cap for one tick (#3246) — a pure fold.

    Reads the seat/host signal the preflight probe already returned and sizes the
    tick's effective cap as ``min(live + seat_free, host_cap, ceiling,
    live + ramp_delta)`` so a newly added account (more free seats) converts to
    live workers on the next tick with no cron edit. Every term is a TOTAL live
    worker population, not a delta: free seats admit ``seat_free`` workers ABOVE
    the current live count, ``host_cap`` and the hard ceiling are already totals,
    and the ramp admits at most ``ramp_delta`` NEW workers per tick (0 disables).
    The hard ceiling never shrinks an explicit operator cap: a configured
    ``fallback`` (--max-workers, plus any realloc bonus) above it raises it.

    FAIL-OPEN: when the doc carries no usable seat signal (no ``seat.free`` /
    ``live``, e.g. a preflight predating the seat pool or a hermetic stub), the
    configured ``fallback`` is returned unchanged so the tick sizes exactly as
    before. No I/O — the second preflight run at the resized cap (the caller's
    job) keeps the DoS gate authoritative. Returns ``(target, info)`` where
    ``info`` is the payload's ``seat_adaptive`` audit block.
    """
    def _num(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    limiter = (pre.get("capacity_limiter")
               if isinstance(pre.get("capacity_limiter"), dict) else {}) or {}
    raw = limiter.get("raw") if isinstance(limiter.get("raw"), dict) else {}
    seat = pre.get("seat") if isinstance(pre.get("seat"), dict) else {}
    seat_free = _num(seat.get("free"))
    if seat_free is None:
        seat_free = _num(raw.get("seat_free"))
    live = _num(pre.get("live"))
    if live is None:
        live = _num(raw.get("live"))
    host_cap = _num(pre.get("host_cap"))
    if host_cap is None:
        host_cap = _num(raw.get("host_cap"))
    hard_ceiling = max(int(ceiling), int(fallback))
    info: dict[str, Any] = {
        "enabled": True, "fallback_max_workers": int(fallback),
        "hard_ceiling": hard_ceiling, "ramp_delta": int(ramp_delta),
    }
    if seat_free is None or live is None:
        info.update({"signal_available": False, "effective_target": int(fallback),
                     "binding": "fallback_max_workers"})
        return int(fallback), info
    terms: list[tuple[str, int]] = [("seat_free", live + seat_free),
                                    ("hard_ceiling", hard_ceiling)]
    if host_cap is not None:
        terms.append(("host_cap", host_cap))
    if ramp_delta > 0:
        terms.append(("ramp_delta", live + ramp_delta))
    binding, target = min(terms, key=lambda t: t[1])
    target = max(target, 0)
    info.update({"signal_available": True, "seat_free": seat_free, "live": live,
                 **({"host_cap": host_cap} if host_cap is not None else {}),
                 "terms": dict(terms), "effective_target": target,
                 "binding": binding})
    return target, info


# Backend-health SPAWN gate (#3247). The standing codex cron kept dispatching into a
# backend that was returning banner-only/0-byte stub logs — every seat burned on a
# no-op that produced zero real output, and seat-adaptive sizing (above) only made it
# worse by handing the dead backend MORE seats. The gate below is the stateless,
# per-tick complement to check_backend_health's persistent self-suppress: it reads the
# SAME per-backend stub_rate the status card computes and, when the backend is majority-
# stub, sizes the tick to 0 workers BEFORE the seat math runs so no amount of free seats
# can reopen a dead backend. The threshold matches the `stub > productive` majority the
# status card and check_backend_health already use to call a backend dead.
_HEALTH_SKIP_STUB_RATE = 0.5


def recent_backend_stub_rate(runs_dir: Path, *, product: str,
                             lookback_min: int = 90, now_ts: float | None = None,
                             alive: set[int] | None = None,
                             probe: Any | None = None) -> float | None:
    """The recent stub_rate for one backend ``product`` — the fraction of its recent
    worker logs that were banner-only/0-byte no-ops — read from the status card's own
    ``dispatch_status.backend_stub_rates`` rollup (#3247) so the spawn gate and the
    health card can never disagree. Returns ``None`` when the signal is unavailable (no
    recent logs for the backend, the status module is missing, or any read error): a
    ``None`` tells :func:`gate_spawn_on_health` to FAIL-OPEN, so unknown health never
    wedges dispatch. The import is lazy to avoid an import cycle with dispatch_status
    (which imports this module for its own fold)."""
    try:
        import dispatch_status  # noqa: E402  (lazy: dispatch_status imports us back)
    except ImportError:
        return None
    try:
        rows = dispatch_status.backend_stub_rates(
            runs_dir, lookback_min=lookback_min, now_ts=now_ts,
            alive=alive, probe=probe)
    except Exception:
        return None
    for row in rows or []:
        if row.get("product") == product and row.get("stub_rate") is not None:
            try:
                return float(row.get("stub_rate"))
            except (TypeError, ValueError):
                return None
    return None


def gate_spawn_on_health(planned: int, stub_rate: float | None,
                         threshold: float = _HEALTH_SKIP_STUB_RATE
                         ) -> tuple[int, str | None]:
    """Gate a tick's planned worker count on the backend's recent health (#3247) — a
    PURE fold that composes with :func:`seat_adaptive_target`. When the backend is
    majority-stub (``stub_rate >= threshold``) it returns ``(0, reason)`` so the tick
    plans zero spawns REGARDLESS of what the seat math would size, carrying a legible
    health-skip ``reason`` for the payload; otherwise it returns ``(planned, None)``
    unchanged. FAIL-OPEN + auto-restoring: a ``None`` (or non-numeric) stub_rate is
    never gated, and because the live rate is re-read every tick, a backend whose
    stub_rate recovers below ``threshold`` resumes spawning on its own — no persisted
    hold, no operator, no reprobe timer."""
    if stub_rate is None:
        return planned, None
    try:
        rate = float(stub_rate)
    except (TypeError, ValueError):
        return planned, None
    if rate < threshold:
        return planned, None
    reason = (
        f"backend is majority-stub (recent stub_rate={round(rate, 3)} >= {threshold}); "
        f"planning 0 spawns instead of {planned} — every seat would burn on a "
        f"banner-only/0-byte no-op producing zero real output. Spawns auto-restore the "
        f"first tick its stub_rate recovers below {threshold} (the live signal is "
        f"re-read each tick; no persistent hold)")
    return 0, reason


def gate_opencode_gateway(planned: int, account: dict, *, probe=None) -> tuple[int, str | None]:
    """Suppress opencode only for a conclusive existing-probe GATEWAY_DOWN verdict."""
    if planned <= 0 or not account:
        return planned, None
    probe = probe or account_probe.probe_opencode_account
    try:
        verdict = probe(account)
    except Exception:
        return planned, None
    if verdict.get("status") != "GATEWAY_DOWN":
        return planned, None
    detail = verdict.get("block_reason") or "guard gateway unreachable"
    return 0, f"gateway_down: backend_unhealthy: {detail}"


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
              live: bool, refresh: bool = True, cooldown_min: int = 120,
              backend: str = "claude",
              exclude_lanes: set[str] | None = None,
              worker_timeout_s: int | None = DEFAULT_WORKER_TIMEOUT_S,
              spawn_probe_s: float = DEFAULT_SPAWN_PROBE_S,
              realloc_ceiling: int = DEFAULT_REALLOC_CEILING,
              seat_adaptive: bool = DEFAULT_SEAT_ADAPTIVE,
              seat_ceiling: int = DEFAULT_SEAT_CEILING,
              seat_ramp_delta: int = DEFAULT_SEAT_RAMP_DELTA,
              record_loop: bool = False,
              loop_ledger: Path | None = None,
              issue_override: int | None = None,
              force: bool = False,
              force_reason: str = "",
              contract_scan: int = DEFAULT_CONTRACT_SCAN,
              contract_hold_ttl_h: int = DEFAULT_CONTRACT_HOLD_TTL_H,
              repair_batch: int = DEFAULT_REPAIR_BATCH,
              repair_cooldown_min: int = DEFAULT_REPAIR_COOLDOWN_MIN,
              lease_runner: Any | None = None) -> dict[str, Any]:
    # lease_runner is the injectable `fak leaseref` seam (default: a real
    # subprocess). Tests pass a canned runner returning {rc, verdict} so the whole
    # acquire/refuse/release path runs with no real git and no real fak binary.
    def finish(payload: dict[str, Any]) -> dict[str, Any]:
        if record_loop:
            payload["loop_ledger"] = record_loop_tick(root, payload, ledger=loop_ledger)
        # #4594: every LIVE tick feeds the fleet-status trend ledger so
        # climbing-vs-draining renders unattended. Dry runs never mutate
        # runtime state (same posture as the lease/witness sweeps above).
        if live:
            payload["fleet_trend"] = append_fleet_trend_row(root, payload)
        _record(root / RUNS_DIRNAME, payload)
        return payload

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    product = _BACKEND_PRODUCT[backend]
    runs_dir = root / RUNS_DIRNAME
    reg = issue_dispatch.refresh_registry(root) if refresh else {"skipped": True}
    reaped = reap_timed_out_workers(runs_dir, timeout_s=worker_timeout_s, live=live)
    # Commit-time diff-witness binding (#1324 proposal #2) — BEFORE the corpse sweep
    # below, which would delete a finished worker's log/sidecars before it could be
    # audited. Each dead-pid worker's slot is graded through `dos commit-audit`: an
    # unwitnessed / wrong-issue commit is recorded CLAIM_UNWITNESSED instead of
    # silently counting as productive. Live ticks only (the audit + the .witness
    # sidecar write are the side effects); fail-open.
    witnessed = (witness_exited_workers(runs_dir, root, live=live,
                                        lease_runner=lease_runner) if live
                 else {"skipped": True})
    # Sweep the dead sidecars the reaper leaves behind, BEFORE preflight counts
    # capacity — otherwise stale `.pid` files accumulate and a recycled PID landing
    # in one's spawn window is miscounted as a live worker, pinning the cap.
    pruned = prune_dead_sidecars(runs_dir, live=live)
    # Sweep EXPIRED fenced lane leases (residual of #1310) — the crashed-holder
    # backstop. A worker that died without releasing its refs/fak/locks/resolve-<lane>
    # lease has it deleted here once its TTL lapses, so a crash can't wedge a lane
    # forever. Live ticks only (a dry-run never mutates lease state); fail-open.
    leases_reaped = reap_expired_leases(root, runner=lease_runner) if live else {"skipped": True}

    # Backend-health reallocation (the cross-read half). A HEALTHY backend claims the
    # budget + lane a DEAD sibling abandoned: it raises its effective cap by one slot
    # per dead sibling (bounded by realloc_ceiling) and prefers the freed lane in the
    # busiest-pick. Fail-open: no sibling held dead -> realloc is a no-op and behavior
    # is exactly as before. The DEAD backend's own self-suppress gate is below, after
    # preflight/weekly-cap, so a dead backend never also tries to claim from itself.
    own_health = check_backend_health(runs_dir, product=product, lane=lane)
    realloc: dict[str, Any] = {"claimed_lanes": [], "bonus": 0, "from": []}
    eff_max_workers = max_workers
    if own_health.get("state") != "dead":
        dead_siblings = read_dead_backends(runs_dir, exclude=product)
        if dead_siblings:
            bonus = min(len(dead_siblings), max(0, realloc_ceiling))
            eff_max_workers = max_workers + bonus
            realloc = {
                "claimed_lanes": [d.get("abandoned_lane") for d in dead_siblings
                                  if d.get("abandoned_lane")],
                "bonus": bonus,
                "from": [d.get("product") for d in dead_siblings],
            }

    pre = issue_dispatch.preflight(root, max_workers=eff_max_workers, work_kind=work_kind,
                                   product=product)
    # Backend-health spawn gate (#3247) — decided BEFORE the seat sizing so a majority-
    # stub backend plans 0 REGARDLESS of what the seat math would size. Read the live
    # per-backend stub_rate the status card already computes (recent_backend_stub_rate ->
    # dispatch_status.backend_stub_rates); when the backend is mostly returning stub
    # output (>= _HEALTH_SKIP_STUB_RATE), gate the effective cap to 0 and SKIP the seat
    # re-sizing below so free seats can never reopen a dead backend. The legible refusal
    # is emitted alongside the sibling health/cap gates further down (once the preflight/
    # account render helpers are in scope). STATELESS + auto-restoring: the rate is
    # re-read every tick, so recovery lifts the gate with no operator and no persisted
    # hold. FAIL-OPEN: no recent logs / None rate -> not gated, sized exactly as before.
    health_stub_rate = recent_backend_stub_rate(runs_dir, product=product)
    eff_max_workers, health_skip_reason = gate_spawn_on_health(
        eff_max_workers, health_stub_rate)
    # Seat-adaptive tick sizing (#3246): the configured --max-workers is a fail-safe,
    # not the throttle. When the probe above carries a seat signal, re-size the
    # effective cap to min(live + seat_free, host_cap, ceiling, live + ramp) and
    # re-run the preflight AT that cap — the preflight stays the authoritative DoS
    # floor (a REFUSE_* at the resized cap still stops growth); only the redundant
    # configured_max term moves. No signal -> no re-run, exactly the old sizing. Skipped
    # entirely when the health gate above fired, so seat math can never raise a majority-
    # stub backend back above the 0 it was pinned to.
    seat_sizing: dict[str, Any] | None = None
    if seat_adaptive and health_skip_reason is None:
        seat_target, seat_sizing = seat_adaptive_target(
            pre, fallback=eff_max_workers, ceiling=seat_ceiling,
            ramp_delta=seat_ramp_delta)
        if seat_sizing.get("signal_available") and seat_target != eff_max_workers:
            eff_max_workers = seat_target
            seat_sizing["reprobed"] = True
            pre = issue_dispatch.preflight(root, max_workers=eff_max_workers,
                                           work_kind=work_kind, product=product)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}
    gateway_skip_reason = None
    if backend == "opencode" and pre_ok:
        eff_max_workers, gateway_skip_reason = gate_opencode_gateway(
            eff_max_workers, acct)
    account_cap_reroute: dict[str, Any] | None = None

    def _preflight_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _preflight_public(doc: dict[str, Any]) -> dict[str, Any]:
        out = {"verdict": doc.get("verdict"), "reason": doc.get("reason"),
               "cap": doc.get("cap"), "live": doc.get("live")}
        for key in ("headroom", "max_workers", "host_cap"):
            if doc.get(key) is not None:
                out[key] = doc.get(key)
        limiter = doc.get("capacity_limiter")
        if isinstance(limiter, dict):
            out["capacity_limiter"] = limiter
        seat = doc.get("seat")
        if isinstance(seat, dict):
            out["seat"] = {k: seat.get(k) for k in (
                "total", "free", "leased", "depleted", "unattributed_live")
                if seat.get(k) is not None}
        return out

    def _preflight_refusal_hint(doc: dict[str, Any]) -> dict[str, Any] | None:
        if doc.get("verdict") != "REFUSE_AT_CAP":
            return None
        limiter = doc.get("capacity_limiter") if isinstance(doc.get("capacity_limiter"), dict) else {}
        raw = limiter.get("raw") if isinstance(limiter.get("raw"), dict) else {}
        term = str(limiter.get("term") or "")
        maxw = _preflight_int(raw.get("max_workers"))
        if maxw is None:
            maxw = _preflight_int(doc.get("max_workers"))
        host_cap = _preflight_int(raw.get("host_cap"))
        if host_cap is None:
            host_cap = _preflight_int(doc.get("host_cap"))
        live_count = _preflight_int(doc.get("live"))
        cap_count = _preflight_int(doc.get("cap"))
        needed = live_count + 1 if live_count is not None else None
        configured_cap = term == "max_workers" or (
            maxw is not None and cap_count == maxw
            and host_cap is not None and host_cap > maxw)
        host_headroom_available = (
            host_cap is None or live_count is None or live_count < host_cap)
        needed_within_host = needed is None or host_cap is None or needed <= host_cap
        if configured_cap and host_headroom_available and needed_within_host:
            return {
                "kind": "configured_max_workers",
                "message": (
                    f"configured --max-workers={maxw} is the binding cap; "
                    + (f"rerun with --max-workers >= {needed} " if needed else "raise --max-workers ")
                    + (f"(still bounded by host_cap={host_cap}) "
                       if host_cap is not None else "")
                    + "only if the operator intends to use available host headroom"),
                "max_workers": maxw,
                "required_min": needed,
                "host_cap": host_cap,
            }
        if limiter:
            return {
                "kind": term or str(limiter.get("primary") or "capacity"),
                "message": (f"capacity limiter {limiter.get('primary') or '?'}"
                            f"/{term or '?'} is binding; do not route around "
                            "the preflight refusal"),
            }
        return None

    def _account_public(account: dict[str, Any]) -> dict[str, Any]:
        return {k: account.get(k) for k in ("tag", "tier", "model", "dir")}

    def _same_account(left: dict[str, Any], right: dict[str, Any]) -> bool:
        for key in ("tag", "dir"):
            lv = str(left.get(key) or "")
            rv = str(right.get(key) or "")
            if lv and rv and lv == rv:
                return True
        return False

    def _weekly_capped_payload(blocked_cap: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": _preflight_public(pre),
            "account": _account_public(acct),
            "weekly_cap": blocked_cap, "ok": False, "action": "weekly_capped",
            "verdict": "WEEKLY_CAPPED",
            "reason": (f"account '{acct.get('tag')}' is weekly-capped (resets "
                       f"{blocked_cap.get('reset_text') or '?'}); holding spawn until "
                       f"{blocked_cap.get('until')} — re-arm is automatic at the reset"),
        }
        if account_cap_reroute:
            payload["account_cap_reroute"] = account_cap_reroute
        return finish(payload)

    def _seat_cooled_payload(state: dict[str, Any],
                             reroute: dict[str, Any] | None) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": _preflight_public(pre),
            "account": _account_public(acct),
            "seat_streak": state, "ok": False, "action": "seat_cooled",
            "verdict": "SEAT_COOLED",
            "reason": (f"seat '{state.get('seat')}' has {state.get('streak')} "
                       f"spawn-fails in a row (across targets) — cooling the seat "
                       f"instead of feeding it another issue; run `fak accounts "
                       f"status` and re-login or replace the seat; one re-probe "
                       f"spawn is admitted every {SEAT_STREAK_REPROBE_MIN} min "
                       f"(#4591)"),
        }
        if reroute:
            payload["seat_cool_reroute"] = reroute
        return finish(payload)

    # Active opencode gateway gate (#3866): unlike the passive stub-rate signal, this
    # catches a worker that hangs before producing a terminal log. It is deliberately
    # stateless and fail-open except for the existing prober's typed GATEWAY_DOWN result.
    if gateway_skip_reason is not None:
        return finish({
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": _preflight_public(pre), "account": _account_public(acct),
            "gateway_health_gate": {"status": "backend_unhealthy",
                                    "reason": "gateway_down", "planned": 0},
            "ok": False, "action": "backend_health_skip", "verdict": "GATEWAY_DOWN",
            "reason": gateway_skip_reason,
        })

    # Backend-health spawn gate (#3247) — the majority-stub decision taken before the
    # seat sizing (which already pinned eff_max_workers to 0) is EMITTED here, alongside
    # the sibling capacity/health gates, now that the preflight/account render helpers
    # are in scope. Plan 0 spawns this tick with a legible health-skip reason instead of
    # feeding --max-workers to doomed workers. `backend_health_skip` is a BENIGN action
    # (a correctly-declined tick — see BENIGN_ACTIONS), never a dispatcher malfunction.
    if health_skip_reason is not None:
        return finish({
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": _preflight_public(pre),
            "account": _account_public(acct),
            "backend_health_skip": {
                "backend": backend, "stub_rate": health_stub_rate,
                "threshold": _HEALTH_SKIP_STUB_RATE, "planned": 0},
            "ok": False, "action": "backend_health_skip",
            "verdict": "BACKEND_HEALTH_SKIP", "reason": health_skip_reason,
        })

    # Weekly-cap gate — BEFORE the lane router's gh work. A logged-in account can
    # still be quota-exhausted; the preflight returns SPAWN_OK regardless. Without
    # this, every tick spawns a worker that instantly dies on the limit banner,
    # ships nothing, and floods the logs for the whole reset window. Read that
    # banner back from recent worker logs and HOLD until the stated reset instead.
    # Fail-open (check_weekly_cap can only return capped on positive evidence).
    cap = (check_weekly_cap(runs_dir, product=product, account_tag=acct.get("tag"))
           if pre_ok else {"capped": False})
    if pre_ok and cap.get("capped"):
        reroute_pre = issue_dispatch.preflight(
            root, max_workers=eff_max_workers, work_kind=work_kind, product=product)
        reroute_ok = reroute_pre.get("verdict") == "SPAWN_OK"
        reroute_acct = reroute_pre.get("account") or {}
        account_cap_reroute = {
            "attempted": True,
            "from": _account_public(acct),
            "preflight": _preflight_public(reroute_pre),
            "account": _account_public(reroute_acct),
        }
        if reroute_ok and reroute_acct and not _same_account(acct, reroute_acct):
            reroute_cap = check_weekly_cap(
                runs_dir, product=product, account_tag=reroute_acct.get("tag"))
            account_cap_reroute["weekly_cap"] = reroute_cap
            if not reroute_cap.get("capped"):
                pre, pre_ok, acct, cap = reroute_pre, True, reroute_acct, reroute_cap
            else:
                account_cap_reroute["reason"] = "rerouted account is also capped"
                return _weekly_capped_payload(cap)
        else:
            account_cap_reroute["reason"] = (
                "reroute did not find a different account"
                if reroute_ok else "reroute preflight refused")
            return _weekly_capped_payload(cap)

    # Seat-keyed spawn-failure cool gate (#4591) — the keying fix. The
    # target-keyed streak (tick_exit_code / #2636) cannot see a dead needs_login
    # seat cycling across DIFFERENT issues: every new target restarts that
    # counter at 1, so the selector re-hands the same dead seat each tick and
    # the fleet drains with no cooldown and no red card. Here the SAME
    # run-length is read SEAT-keyed: a routed seat whose streak has reached
    # SPAWN_FAILED_RED_STREAK is COOLED — try one reroute to a different seat
    # (mirroring the weekly-cap reroute above), else decline the tick with a
    # named verdict. A cooled seat earns one re-probe spawn every
    # SEAT_STREAK_REPROBE_MIN minutes (reprobe_due) so recovery after an
    # operator re-login is detected; a clean launch clears the streak.
    # Fail-open: an unreadable ledger reads streak 0 and gates nothing.
    seat_streak_state = (seat_spawn_failure_state(runs_dir, acct.get("tag"), backend)
                         if pre_ok else {"cooled": False})
    seat_cool_reroute: dict[str, Any] | None = None
    if (pre_ok and seat_streak_state.get("cooled")
            and not seat_streak_state.get("reprobe_due")):
        reroute_pre = issue_dispatch.preflight(
            root, max_workers=eff_max_workers, work_kind=work_kind, product=product)
        reroute_ok = reroute_pre.get("verdict") == "SPAWN_OK"
        reroute_acct = reroute_pre.get("account") or {}
        seat_cool_reroute = {
            "attempted": True,
            "from": _account_public(acct),
            "seat_streak": seat_streak_state,
            "preflight": _preflight_public(reroute_pre),
            "account": _account_public(reroute_acct),
        }
        if reroute_ok and reroute_acct and not _same_account(acct, reroute_acct):
            reroute_state = seat_spawn_failure_state(
                runs_dir, reroute_acct.get("tag"), backend)
            seat_cool_reroute["reroute_seat_streak"] = reroute_state
            if reroute_state.get("cooled") and not reroute_state.get("reprobe_due"):
                seat_cool_reroute["reason"] = "rerouted seat is also cooled"
                return _seat_cooled_payload(seat_streak_state, seat_cool_reroute)
            pre, pre_ok, acct = reroute_pre, True, reroute_acct
            seat_streak_state = reroute_state
        else:
            seat_cool_reroute["reason"] = (
                "reroute did not find a different seat"
                if reroute_ok else "reroute preflight refused")
            return _seat_cooled_payload(seat_streak_state, seat_cool_reroute)

    # Backend-health gate (the self-suppress half) — AFTER weekly-cap, same shape. If
    # THIS backend is spinning dead (a streak of banner-only/0-byte deaths the cap
    # regex misses), hold the spawn so we stop feeding budget to a doomed worker. But
    # let ONE re-probe worker through per interval (reprobe_due) so the backend can
    # earn its budget back the moment it recovers — a fully-held backend can never
    # come back. Fail-open: check_backend_health returns dead only on positive
    # streak evidence. (own_health was read above for the realloc no-op decision.)
    if pre_ok and own_health.get("state") == "dead" and not own_health.get("reprobe_due"):
        _lane = own_health.get("abandoned_lane") or "?"
        if own_health.get("auth_gap"):
            # A permanent auth gap, not a recoverable credit wall: naming the operator
            # fix (and the backed-off re-probe) stops the "detecting recovery" framing from
            # implying this self-heals — it does not until credentials are set.
            _reason = (f"backend '{backend}' is UNAUTHENTICATED — every spawn dies instantly "
                       f"on a missing API key / no resolved login (see its evidence logs) "
                       f"since {own_health.get('since')}; planning 0 spawns. Set the backend's "
                       f"credentials (e.g. export OPENAI_API_KEY / `codex login`) or drop it "
                       f"from the backend rotation; its lane '{_lane}' is reallocated to a "
                       f"healthy backend, and a re-probe is admitted only every "
                       f"{own_health.get('reprobe_min')} min (not {_BACKEND_REPROBE_MIN}) "
                       f"because a missing key cannot self-heal")
        else:
            _reason = (f"backend '{backend}' is majority-stub "
                       f"({own_health.get('stub')}/"
                       f"{(own_health.get('stub') or 0) + (own_health.get('productive') or 0)}"
                       f" recent spawns banner-only/0-byte, stub_rate="
                       f"{own_health.get('stub_rate')}) since {own_health.get('since')}; "
                       f"planning 0 spawns — its lane "
                       f"'{_lane}' is reallocated "
                       f"to a healthy backend, and one re-probe worker is admitted every "
                       f"{_BACKEND_REPROBE_MIN} min to detect recovery")
        return finish({
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "timed_out_workers": reaped, "pruned_sidecars": pruned,
            "preflight": _preflight_public(pre),
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
            "backend_health": own_health, "ok": False, "action": "backend_unhealthy",
            "verdict": "BACKEND_UNHEALTHY", "reason": _reason,
        })

    # Healthy backend: claim any lane a dead sibling abandoned. Dropping the freed
    # lane from the dead sibling's exclude set lets this backend's busiest-pick own it
    # (it was kept off this backend by --exclude-lane lane partitioning). The freed
    # budget was already folded into eff_max_workers above.
    effective_exclude = set(exclude_lanes or set())
    if realloc["claimed_lanes"]:
        effective_exclude -= {lc for lc in realloc["claimed_lanes"] if lc}
    # Pre-spawn lane-lease gate (#1310): the dispatcher HOLDS the target lane
    # before launching instead of trusting the worker to self-arbitrate. fak's
    # taxonomy is one worker per leaf tree, so a second worker on a lane that
    # already has a live one co-edits the same files. Fold the held lanes into the
    # busiest-pick exclude so the auto-pick reroutes to a FREE lane (queued
    # elsewhere); an explicitly-named lane that is held is refused below
    # (COLLISION_RISK) rather than raced.
    local_held_lanes = live_resolution_lanes(runs_dir)
    live_leases = live_lane_lease_lanes(root)
    lease_held_lanes = set(live_leases.get("lanes") or [])
    held_lanes = local_held_lanes | lease_held_lanes
    # Part C (#2062): SOFT-exclude any lane whose recent finished sessions burned
    # turns yet closed nothing (a poison-pill sink like a GPU-less host re-grabbing a
    # P1 GPU epic). The lane→tree map comes from the cheap dos.toml taxonomy (dos
    # doctor, no gh); the git ancestry-close join inside the fold runs only for
    # candidate lanes. An explicitly-pinned --lane deliberately BYPASSES the soft
    # exclude (operator override), matching the operator-exclude semantics. Fully
    # fail-open: any error yields no soft exclude. lane_issue_numbers applies the
    # starvation floor (re-seats the last eligible lane under low_yield_relief).
    try:
        _concurrent, low_yield_trees, _exclusive = issue_lane_router.lane_taxonomy(root)
    except Exception:
        low_yield_trees = {}
    low_yield = low_yield_soft_excludes(root, runs_dir, lane_trees=low_yield_trees)
    soft_exclude = set(low_yield.get("exclude") or set())
    if lane:
        soft_exclude.discard(lane)  # explicit pin bypasses the soft demote
    # Pass soft_exclude only when the loop actually flagged a lane: an empty set is
    # identical to the default, so the common (no-low-yield) case keeps the picker
    # call byte-identical rather than threading a no-op kwarg every tick.
    _soft = {"soft_exclude": soft_exclude} if soft_exclude else {}
    pick = lane_issue_numbers(root, lane, exclude=effective_exclude | held_lanes,
                              **_soft)
    low_yield_excluded_lanes = pick.get("low_yield_excluded") or []
    low_yield_relief = pick.get("low_yield_relief") or []
    # Evidence rows only for the lanes actually demoted this tick (never-silent-drop).
    _excluded_set = set(low_yield_excluded_lanes)
    low_yield_evidence = [r for r in (low_yield.get("lanes") or [])
                          if r.get("lane") in _excluded_set]
    chosen_lane = pick.get("lane")
    live_issues = live_resolution_issues(runs_dir)
    cooled = recently_attempted_issues(runs_dir, cooldown_min=cooldown_min)
    # Skip a live worker's issue, a recently-attempted one (the time cooldown), AND an
    # issue whose last finished worker was STRUCTURALLY guard-blocked this tick
    # (self_modify / policy_block) -- re-dispatching it re-blocks identically, so HOLD
    # it (the pick-held-invariant rung, #1396) instead of re-storming the same un-
    # landable drain. The held set is read from the recorded no-commit reason the
    # witness sweep above just bound, so an auth_wall still re-probes after its window.
    held_no_commit = held_no_commit_issues(witnessed)
    local_witnessed = locally_witnessed_issues(root)
    # ...AND an issue whose contract review HELD within the TTL -- re-reviewing it
    # would hold identically (the body has not grown a contract in the meantime),
    # so skipping it lets the bounded scan advance to unreviewed issues instead of
    # pinning the scan window to the same thin heads every tick.
    # A prior hold is re-admitted early when the issue changed AFTER its held
    # verdict: a GitHub updatedAt bump (a human backfill; one bulk gh call,
    # fail-open) OR a fresh local contract overlay (a repair worker's landed
    # backfill — the write path a worker can actually complete on this host).
    if (contract_hold_ttl_h > 0 or DEFAULT_MULTI_LANE_HOLD_TTL_H > 0
            or DEFAULT_COLLISION_HOLD_TTL_H > 0):
        refreshed_ts = open_issue_updated_map(root)
        for n, ts in contract_overlay_times(runs_dir).items():
            refreshed_ts[n] = max(refreshed_ts.get(n) or 0.0, ts)
    else:
        refreshed_ts = None
    contract_held_prior = contract_held_issues(
        runs_dir, ttl_h=contract_hold_ttl_h, updated_ts=refreshed_ts)
    multi_lane_held_prior = multi_lane_held_issues(
        runs_dir, ttl_h=DEFAULT_MULTI_LANE_HOLD_TTL_H, updated_ts=refreshed_ts)
    # ...AND an issue whose last tick collided with dirty local WIP (dirty-path or
    # same-issue-WIP). The dirty tree does not shrink between 5-minute ticks, so a
    # colliding head would re-collide identically; holding it lets the bounded scan
    # advance to disjoint work instead of re-refusing the same 2-3 heads forever
    # (the durability the raw #2977/#2975 gates lacked). The short TTL re-checks the
    # live tree periodically, so a committed/reverted path re-admits on its own.
    collision_held_prior = collision_held_issues(
        runs_dir, ttl_h=DEFAULT_COLLISION_HOLD_TTL_H, updated_ts=refreshed_ts)
    # The cross-lane candidate stream the bounded contract scan walks (busiest
    # lane's oldest candidate first, then each other eligible lane's, round-robin).
    # Falls back to the single chosen lane when the router fold predates the
    # eligible_by_lane key (hermetic test stubs).
    eligible_lanes = pick.get("eligible_by_lane")
    if not eligible_lanes and chosen_lane:
        eligible_lanes = [[chosen_lane, pick.get("numbers") or []]]
    candidates = _candidate_issue_numbers(eligible_lanes, pick.get("numbers") or [])
    audit_abstain_holds = commit_audit_abstain_holds(root, candidates)
    audit_abstain_held = {int(r["issue"]) for r in audit_abstain_holds
                          if r.get("issue") is not None}
    # #5071: a candidate whose resolving commit is ALREADY witnessed in trunk
    # ancestry (the closure auditor's OPEN_WITNESSED bucket) is bookkeeping lag,
    # not engineering fuel — exclude it from selection BEFORE lease/spawn and
    # route it to the trusted close arm instead of burning a seat on a redo
    # (the witnessed #2850 redispatch).
    open_witnessed_rows = open_witnessed_dispositions(root, candidates)
    open_witnessed_held = {int(r["issue"]) for r in open_witnessed_rows
                           if r.get("issue") is not None}
    # Part B (per-node hardware-capability gate): drop candidate issues whose
    # declared required_caps this node does NOT advertise (FLEET_NODE_CAPS, default
    # empty => GPU-less). A capability-skipped issue is NOT cooled/leased/labeled —
    # it stays OPEN and visible so a node that opts in (e.g. FLEET_NODE_CAPS=gpu)
    # picks it. This is the honest "leave it for the compute node" behavior: a
    # GPU-less host stops re-grabbing an 8xH100 serving epic it structurally cannot
    # run (the #2062 tools-lane burn), instead of falsely stopping it.
    node_caps_set = dispatch_worker.node_caps()
    caps_by_issue = {int(k): v for k, v in (pick.get("caps_by_issue") or {}).items()}
    capability_skipped = {n for n in candidates
                          if not (set(caps_by_issue.get(n, [])) <= node_caps_set)}
    capability_skipped_issues = [
        {"issue": n, "required_caps": sorted(caps_by_issue.get(n, []))}
        for n in sorted(capability_skipped)]
    skip = (live_issues | cooled | held_no_commit | contract_held_prior
            | multi_lane_held_prior | collision_held_prior
            | local_witnessed | audit_abstain_held | open_witnessed_held
            | capability_skipped)
    scan_stream = contract_scan_stream(eligible_lanes, skip)
    if scan_stream:
        chosen_lane, target = scan_stream[0]
    else:
        target = pick_target_issue(pick.get("numbers") or [], skip)
    # Operator override: pin an explicit, already-vetted target issue instead of
    # the lane's freshest-first auto-pick. The full safety chain STILL runs on the
    # pinned issue -- preflight cap, the issue-contract gate, the lane lease, and
    # the detached spawn -- so an override can only NARROW what spawns, never
    # bypass a guard. Used to dispatch a specific contract-passing top issue when
    # the busiest lane's freshest issue is a thin one the gate would (rightly) HOLD.
    if issue_override is not None:
        target = int(issue_override)

    payload: dict[str, Any] = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "work_kind": work_kind, "force": bool(force),
        "contract_scan": int(contract_scan), "repair_batch": int(repair_batch),
        "registry_refresh": reg,
        "timed_out_workers": reaped,
        # Only surface the realloc block when it actually changed something, so the
        # common (no dead sibling) path's payload is byte-identical to before.
        **({"reallocation": {"effective_max_workers": eff_max_workers, **realloc}}
           if realloc["bonus"] or realloc["claimed_lanes"] else {}),
        # Surface the lease reap only on a live tick where it ran, so a dry-run
        # payload stays byte-identical to before the lease gate.
        **({"leases_reaped": leases_reaped} if live else {}),
        # Surface the commit-time witness verdicts (#1324 proposal #2) only on a live
        # tick where the sweep ran, so a dry-run payload stays byte-identical.
        **({"witnessed_slots": witnessed} if live else {}),
        **({"account_cap_reroute": account_cap_reroute} if account_cap_reroute else {}),
        # Seat-cool reroute audit (#4591): present only when the cooled-seat gate
        # rerouted this tick to a different seat, so the common path stays
        # byte-identical to before.
        **({"seat_cool_reroute": seat_cool_reroute} if seat_cool_reroute else {}),
        "preflight": _preflight_public(pre),
        # Seat-adaptive sizing audit (#3246): which term bound the effective cap this
        # tick (seat_free / host_cap / hard_ceiling / ramp_delta), or that the tick
        # fell back to the configured --max-workers for want of a seat signal. Only
        # present when the feature is enabled, so --no-seat-adaptive payloads stay
        # byte-identical to before.
        **({"seat_adaptive": seat_sizing} if seat_sizing else {}),
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen_lane, "lane_issue_count": len(pick.get("numbers") or []),
        "cooled_recently": sorted(cooled), "target_issue": target,
        # Surface the structurally-held issues only when something is actually held, so
        # the common (nothing held) payload stays byte-identical to before (#1396).
        **({"held_no_commit": sorted(held_no_commit)} if held_no_commit else {}),
        **({"multi_lane_held_prior": len(multi_lane_held_prior)}
           if multi_lane_held_prior else {}),
        **({"locally_witnessed": sorted(local_witnessed)} if local_witnessed else {}),
        **({"commit_audit_abstain_held": audit_abstain_holds}
           if audit_abstain_holds else {}),
        # Never a silent drop (#5071): name every candidate disposed
        # OPEN_WITNESSED with its witnessing SHA, so the tick REPORTS the
        # evidence it excluded on. Empty => key omitted (byte-identical).
        **({"open_witnessed": open_witnessed_rows} if open_witnessed_rows else {}),
        # Surface proactive self-source-tree holds (lane_issue_numbers) only when
        # something is actually held, for the same byte-identical-common-case reason.
        **({"self_modify_held": pick.get("self_modify_held")} if pick.get("self_modify_held") else {}),
        **({"safe_lanes_busy": pick.get("safe_lanes_busy")} if pick.get("safe_lanes_busy") else {}),
        # Never a silent drop: name every issue this node structurally skipped for
        # want of a hardware capability, so the visible-skip is auditable (mirrors the
        # router's skipped_human_blocked contract). Empty => key omitted (byte-identical).
        **({"capability_skipped_issues": capability_skipped_issues}
           if capability_skipped_issues else {}),
        **({"node_caps": sorted(node_caps_set)} if node_caps_set else {}),
        # Never a silent drop: name every lane the low-yield loop demoted this tick
        # (#2062), with its turn/close evidence, and surface low_yield_relief when the
        # starvation floor re-seated a flagged lane because it was the last eligible
        # one. Empty => keys omitted (byte-identical no-low-yield common case).
        **({"low_yield_excluded_lanes": sorted(low_yield_excluded_lanes)}
           if low_yield_excluded_lanes else {}),
        **({"low_yield_lanes": low_yield_evidence} if low_yield_evidence else {}),
        **({"low_yield_relief": sorted(low_yield_relief)} if low_yield_relief else {}),
        **({"lease_held_lanes": sorted(lease_held_lanes)} if lease_held_lanes else {}),
        **({"lane_leases": live_leases} if live_leases.get("fail_open") else {}),
        "already_live": sorted(live_issues), "held_lanes": sorted(held_lanes),
    }

    # #5071: transition OPEN_WITNESSED candidates through the defined closure
    # path (the trusted close arm) on a LIVE tick — a close consumes no worker
    # seat, so it runs regardless of the spawn outcome below. Dry-run ticks
    # report the dispositions + explicit close action only (byte-identical
    # common case, and never a gh side effect from a plan).
    if live and open_witnessed_rows:
        payload["open_witnessed_close"] = close_open_witnessed(
            root, runs_dir, open_witnessed_rows, live=True)

    if not pre_ok:
        hint = _preflight_refusal_hint(pre)
        if hint:
            payload["preflight_hint"] = hint
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return finish(payload)
    guard_livelock = active_guard_livelock_hold(root, runs_dir)
    if guard_livelock.get("active"):
        reason = str(guard_livelock.get("reason") or "active guard livelock")
        payload["active_guard_livelock"] = guard_livelock
        payload["launch_gate"] = launch_gate_blocked(
            "ACTIVE_GUARD_LIVELOCK",
            reason,
            "wait-for-loop-worker-to-exit-or-fix-guard-result-livelock")
        payload.update({
            "ok": False,
            "action": "active_guard_livelock",
            "verdict": "ACTIVE_GUARD_LIVELOCK",
            "reason": reason,
        })
        return finish(payload)
    compact_runaway = active_compact_runaway_hold(root, runs_dir)
    if compact_runaway.get("active"):
        # Lane-scope the hold (#5858). A compact runaway is ONE worker's local
        # context condition; it justifies not piling a SECOND worker onto the
        # same lane, but it is no reason to idle every disjoint lane. Held
        # fleet-wide it converted one worker's overshoot into a 20-free-slot
        # stall -- the same unit-fault-becomes-fleet-block shape as a lane lease
        # held by a long-dead worker. Fail CLOSED when any runaway's lane is
        # unknown (unreadable spawn header): we cannot prove disjointness then.
        runaway_lanes = set(compact_runaway.get("lanes") or ())
        lane_unknown = bool(compact_runaway.get("lane_unknown", True))
        collides = (
            lane_unknown
            or not chosen_lane
            or str(chosen_lane) in runaway_lanes
        )
        compact_runaway = {
            **compact_runaway,
            "scoped_lane": chosen_lane,
            "collides": collides,
        }
        payload["active_compact_runaway"] = compact_runaway
        if collides:
            reason = str(compact_runaway.get("reason") or "active compact runaway")
            payload["launch_gate"] = launch_gate_blocked(
                "ACTIVE_COMPACT_RUNAWAY",
                reason,
                "wait-for-loop-worker-to-exit-or-fix-compact-control")
            payload.update({
                "ok": False,
                "action": "active_compact_runaway",
                "verdict": "ACTIVE_COMPACT_RUNAWAY",
                "reason": reason,
            })
            return finish(payload)
    if not chosen_lane:
        if pick.get("self_modify_held"):
            held = sorted(pick.get("self_modify_held"))
            safe_busy = sorted(pick.get("safe_lanes_busy") or [])
            blockers = []
            if safe_busy:
                blockers.append({
                    "code": "SAFE_LANES_BUSY",
                    "reason": "safe non-self-source lanes already have live workers: "
                    + ", ".join(safe_busy),
                    "next_action": "wait-for-safe-lane-lease",
                })
            # The concrete, RUNNABLE worktree-isolated escape the next_action names
            # (#1334). Point at it explicitly so the hold is actionable — prepare an
            # isolated detached worktree at trunk HEAD, edit the self-source files in
            # it, land the diff back onto the trunk as one stamped signed-off commit,
            # then reap. `land`/`reap` args elided to <...> for the operator to fill.
            safe_lane = held[0] if held else "cmd"
            worktree_command = (
                f"python tools/worker_worktree.py --root . prepare "
                f"--lane {safe_lane} --key <issue>  # edit in .path, then: "
                f"python tools/worker_worktree.py --root . land --worktree <path> "
                f"--msg-file <msg> --path <files>; ... reap --worktree <path>")
            blockers.append({
                "code": "SELF_MODIFY_HOLD",
                "reason": "every remaining open issue lane maps to the shared source tree"
                if safe_busy else "every open issue lane maps to the shared source tree",
                "next_action": "route-non-self-source-lane-or-enable-worktree-isolated-resolver",
                "command": worktree_command,
            })
            if safe_busy:
                reason = (f"safe non-self-source lanes are busy ({safe_busy}) and "
                          f"every remaining open issue lane is self-source ({held}); "
                          f"waiting rather than risking build-poisoning the shared trunk "
                          f"(or land via the worktree-isolated path: {worktree_command})")
            else:
                reason = (f"every lane with open issues is self-source ({held}); "
                          f"refusing to risk build-poisoning the shared trunk — land it "
                          f"via the worktree-isolated path (#1334): {worktree_command}")
            payload.update({"ok": False, "action": "no_lane", "verdict": "SELF_MODIFY_HOLD",
                            "launch_gate": {"ready": False, "blockers": blockers},
                            "reason": reason})
        else:
            payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                            "reason": "no lane has open issues (router empty/error)"})
        return finish(payload)
    if chosen_lane in held_lanes:
        # The lane lease is held by a live worker (only reachable via an explicit
        # --lane, since the auto-pick excluded held lanes above). Refuse instead of
        # racing a second worker onto the same leaf tree — the upstream half of the
        # verified loop (#1310): deny the collision by structure, before the spawn.
        payload.update({"ok": False, "action": "lane_busy", "verdict": "LANE_BUSY",
                        "reason": (f"lane '{chosen_lane}' already holds a live "
                                   f"resolution worker (held lanes "
                                   f"{sorted(held_lanes)}); refusing COLLISION_RISK "
                                   f"— the dispatcher holds the lane lease before "
                                   f"launching, it does not race a second worker "
                                   f"onto the same leaf tree")})
        return finish(payload)
    if target is not None:
        missing_caps = sorted(set(caps_by_issue.get(int(target), [])) - node_caps_set)
        if missing_caps:
            # Only reachable when an operator PINNED an incapable issue/lane
            # (--issue/--lane) past the capability skip. Structural capability is not
            # an operator preference, so we refuse with a reason rather than silently
            # route around it — the honest counterpart to the visible auto-pick skip.
            payload.update({"ok": False, "action": "node_incapable",
                            "verdict": "NODE_INCAPABLE", "target_issue": int(target),
                            "required_caps": sorted(caps_by_issue.get(int(target), [])),
                            "node_caps": sorted(node_caps_set),
                            "reason": (f"issue #{int(target)} requires hardware "
                                       f"capabilities {missing_caps} this node does not "
                                       f"advertise (FLEET_NODE_CAPS={sorted(node_caps_set)}); "
                                       f"leaving it OPEN for a capable node rather than "
                                       f"running it here — set FLEET_NODE_CAPS to opt in")})
            return finish(payload)
    if target is None:
        held_candidate_numbers: set[int] = set()
        for entry in eligible_lanes or []:
            try:
                for n in entry[1] or []:
                    if int(n) in contract_held_prior:
                        held_candidate_numbers.add(int(n))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        held_rows = contract_held_records(
            runs_dir, ttl_h=contract_hold_ttl_h, updated_ts=refreshed_ts,
            only=held_candidate_numbers) if held_candidate_numbers else []
        if held_rows:
            payload["contract_held_prior"] = len(contract_held_prior)
            repaired = _maybe_dispatch_contract_repair(
                root, runs_dir, payload, rows=held_rows, live=live,
                backend=backend, acct=acct, repair_batch=repair_batch,
                repair_cooldown_min=repair_cooldown_min,
                spawn_probe_s=spawn_probe_s, product=product)
            if repaired is not None:
                return finish(repaired)
        lanes_considered = [e[0] for e in (eligible_lanes or []) if e and e[1]]
        where = (f"lane '{chosen_lane}'" if len(lanes_considered) <= 1
                 else f"all {len(lanes_considered)} eligible lanes")
        repair_note = (payload.get("contract_repair") or {}).get("skipped")
        repair_suffix = f" (contract-repair: {repair_note})" if repair_note else ""
        payload.update({"ok": False, "action": "no_issue", "verdict": "NO_ISSUE",
                        "reason": (f"every open issue on {where} is "
                                   f"either live ({sorted(live_issues)}), locally "
                                   f"witnessed ({sorted(local_witnessed)}), in "
                                   f"cooldown ({sorted(cooled)}), commit-audit-abstain "
                                   f"held ({sorted(audit_abstain_held)}), "
                                   f"multi-lane-held ({sorted(multi_lane_held_prior)}), "
                                   f"or contract-held "
                                   f"within the last {contract_hold_ttl_h}h "
                                   f"({len(contract_held_prior)} issue(s) awaiting "
                                   f"a contract backfill) — nothing fresh to "
                                   f"dispatch this tick{repair_suffix}")})
        return finish(payload)

    rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
    contract = issue_contract_review(root, rec.get("issue_record"), target)
    dirty_paths_doc = dirty_repo_paths(root)

    def _contract_holds(c: dict[str, Any]) -> bool:
        return bool(c.get("unavailable") or not c.get("ok") or
                    int(c.get("score") or 0) < DEFAULT_ISSUE_CONTRACT_MIN_SCORE)

    # Contract-ready pick: the readiness gate holds THIN issues, not the whole tick.
    # With the oldest-first pick, one pre-contract issue at the head of a lane wedged
    # the loop permanently -- every tick re-picked the same held head and launched
    # nothing while the lane had contract-ready issues deeper in (#1207 held the
    # 232-issue docs lane this way). Scan forward -- bounded at `contract_scan`
    # reviews, oldest-first order preserved -- to the oldest issue that PASSES the
    # floor. A pinned --issue keeps its exact old semantics (the operator vetted that
    # one issue; never silently substitute a different one), --force still accepts
    # the thin head unscanned, and an UNAVAILABLE review (the contract tool itself
    # failed) stops the scan -- every further review would fail identically.
    contract_skipped: list[dict[str, Any]] = []
    scan_pos = 0  # position of `target` in the cross-lane stream
    while (_contract_holds(contract) and not force and issue_override is None
           and not contract.get("unavailable")
           and len(contract_skipped) + 1 < max(1, contract_scan)):
        if scan_pos + 1 >= len(scan_stream):
            break
        contract_skipped.append({
            "issue": target,
            "lane": chosen_lane,
            "title": rec.get("title"),
            "score": int(contract.get("score") or 0),
            "reason": issue_contract_hold_reason(contract)[:200],
            "missing_fields": _contract_missing_fields(contract),
        })
        scan_pos += 1
        chosen_lane, target = scan_stream[scan_pos]
        rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
        contract = issue_contract_review(root, rec.get("issue_record"), target)

    def _candidate_text(r: dict[str, Any]) -> str:
        return f"{r.get('title') or ''}\n{(r.get('issue_record') or {}).get('body') or ''}"

    def _advance_candidate() -> None:
        nonlocal chosen_lane, target, rec, contract, scan_pos
        scan_pos += 1
        chosen_lane, target = scan_stream[scan_pos]
        rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
        contract = issue_contract_review(root, rec.get("issue_record"), target)

    # Collision/scope guards are also per-candidate holds for auto-picks. If the
    # oldest contract-ready issue would collide with dirty local WIP, or names
    # files outside its lane lease, scan forward within the same bounded window.
    # Pinned issues still hold exactly. Dirty-path collisions are never force-
    # bypassed; multi-lane scope keeps its existing --force escape hatch.
    dirty_path_skipped: list[dict[str, Any]] = []
    same_issue_wip_skipped: list[dict[str, Any]] = []
    multi_lane_skipped: list[dict[str, Any]] = []
    scope_scan: dict[str, Any] | None = None
    dirty_scan: dict[str, Any] | None = None
    wip_scan: dict[str, Any] | None = None
    scan_limit = max(1, contract_scan)
    while issue_override is None and not _contract_holds(contract):
        wip_scan = same_issue_wip_collision(
            runs_dir, target, list(dirty_paths_doc.get("paths") or []))
        dirty_scan = dirty_path_collision(
            _candidate_text(rec), list(dirty_paths_doc.get("paths") or []))
        if wip_scan.get("collides"):
            if scan_pos + 1 >= len(scan_stream):
                break
            if (len(contract_skipped) + len(same_issue_wip_skipped)
                    + len(dirty_path_skipped) + len(multi_lane_skipped) + 1 >= scan_limit):
                break
            same_issue_wip_skipped.append({
                "issue": target,
                "lane": chosen_lane,
                "title": rec.get("title"),
                "reason": same_issue_wip_reason(target, wip_scan)[:240],
                "dirty_paths": wip_scan.get("dirty_paths"),
                "evidence": wip_scan.get("evidence"),
            })
            _advance_candidate()
            scope_scan = None
            dirty_scan = None
        elif dirty_scan.get("collides"):
            if scan_pos + 1 >= len(scan_stream):
                break
            if (len(contract_skipped) + len(same_issue_wip_skipped)
                    + len(dirty_path_skipped) + len(multi_lane_skipped) + 1 >= scan_limit):
                break
            dirty_path_skipped.append({
                "issue": target,
                "lane": chosen_lane,
                "title": rec.get("title"),
                "reason": dirty_path_collision_reason(target, dirty_scan)[:240],
                "dirty_paths": dirty_scan.get("dirty_paths"),
            })
            _advance_candidate()
            scope_scan = None
        elif not force:
            scope_scan = scan_multi_lane_scope(root, _candidate_text(rec), chosen_lane)
            if not scope_scan.get("multi_lane"):
                break
            if scan_pos + 1 >= len(scan_stream):
                break
            if (len(contract_skipped) + len(same_issue_wip_skipped)
                    + len(dirty_path_skipped) + len(multi_lane_skipped) + 1 >= scan_limit):
                break
            multi_lane_skipped.append({
                "issue": target,
                "lane": chosen_lane,
                "title": rec.get("title"),
                "reason": multi_lane_scope_reason(target, scope_scan)[:240],
                "uncovered_lanes": scope_scan.get("uncovered_lanes"),
            })
            _advance_candidate()
            dirty_scan = None
        else:
            break
        while (_contract_holds(contract) and not contract.get("unavailable")
               and issue_override is None
               and scan_pos + 1 < len(scan_stream)
               and (len(contract_skipped) + len(same_issue_wip_skipped)
                    + len(dirty_path_skipped) + len(multi_lane_skipped) + 1 < scan_limit)):
            contract_skipped.append({
                "issue": target,
                "lane": chosen_lane,
                "title": rec.get("title"),
                "score": int(contract.get("score") or 0),
                "reason": issue_contract_hold_reason(contract)[:200],
                "missing_fields": _contract_missing_fields(contract),
            })
            _advance_candidate()

    payload["target_issue"] = target
    payload["lane"] = chosen_lane
    payload["lane_issue_count"] = (pick.get("by_lane_count") or {}).get(
        chosen_lane, payload.get("lane_issue_count"))
    payload["prompt_chars"] = rec.get("prompt_chars")
    payload["issue_title"] = rec.get("title")
    # The picked unit's dispatchorder priority, mapped from its priority/P* label (P0>P1>P2, 0
    # when unlabeled). This is the point candidates are built, so the do-first weight the
    # dispatch-order leaf ranks on is derived here rather than thrown away (#3222).
    payload["target_priority"] = candidate_priority(
        ((rec.get("issue_record") or {}).get("labels")) or [])
    # The picked unit's dispatchorder prerequisite edges, parsed from the issue body's
    # "depends-on:/blocked-by: #N" markers. Derived here, at the point candidates are built, so the
    # dispatch-order leaf can hold a dependent until its prerequisite closes (#3224). Empty for an
    # issue that names no prerequisite -- the additive no-regression case.
    target_blocked_by = candidate_blocked_by(
        (rec.get("issue_record") or {}).get("body"))
    if target_blocked_by:
        payload["target_blocked_by"] = target_blocked_by
    if contract_skipped:
        payload["contract_skipped"] = contract_skipped
    if same_issue_wip_skipped:
        payload["same_issue_wip_skipped"] = same_issue_wip_skipped
    if dirty_path_skipped:
        payload["dirty_path_skipped"] = dirty_path_skipped
    if multi_lane_skipped:
        payload["multi_lane_skipped"] = multi_lane_skipped
    if contract_held_prior:
        payload["contract_held_prior"] = len(contract_held_prior)
    if collision_held_prior:
        payload["collision_held_prior"] = len(collision_held_prior)
    payload["issue_contract_gate"] = {
        "ok": bool(contract.get("ok")),
        "unavailable": bool(contract.get("unavailable")),
        "score": int(contract.get("score") or 0),
        "spine_priority": int(contract.get("spine_priority") or 0),
        "min_score": DEFAULT_ISSUE_CONTRACT_MIN_SCORE,
        "reason": issue_contract_hold_reason(contract),
    }
    contract_hold = _contract_holds(contract)
    # Persist every held verdict (skipped heads + a held final target) so the next
    # tick's scan starts past them -- the cumulative half of the contract-ready pick.
    # An UNAVAILABLE verdict is a tool failure, not an issue verdict: never ledger it.
    ledger_rows = list(contract_skipped)
    if (contract_hold and not force and not contract.get("unavailable")
            and issue_override is None):
        ledger_rows.append({"issue": target,
                            "lane": chosen_lane,
                            "title": rec.get("title"),
                            "score": int(contract.get("score") or 0),
                            "reason": issue_contract_hold_reason(contract)[:200],
                            "missing_fields": _contract_missing_fields(contract)})
    record_contract_holds(runs_dir, ledger_rows, live=live)
    record_multi_lane_holds(runs_dir, multi_lane_skipped, live=live)
    # Give every collision the scan advanced PAST the same durable hold the terminal
    # collision below gets, so next tick's skip set starts beyond them.
    record_collision_holds(
        runs_dir,
        [{**r, "kind": "same_issue_wip"} for r in same_issue_wip_skipped]
        + [{**r, "kind": "dirty_path"} for r in dirty_path_skipped],
        live=live)
    if contract_hold and not force:
        # Self-serve before idling: a genuine all-scanned-fail tick dispatches a
        # contract-REPAIR worker on the held candidates instead of just holding.
        # Never on a pinned --issue (operator vetted THAT issue, not grooming) or
        # an UNAVAILABLE review (tool failure, not an issue verdict).
        if issue_override is None and not contract.get("unavailable"):
            repaired = _maybe_dispatch_contract_repair(
                root, runs_dir, payload, rows=ledger_rows, live=live,
                backend=backend, acct=acct, repair_batch=repair_batch,
                repair_cooldown_min=repair_cooldown_min,
                spawn_probe_s=spawn_probe_s, product=product)
            if repaired is not None:
                return finish(repaired)
        reason = issue_contract_hold_reason(contract)
        if contract_skipped:
            lanes_scanned = {r.get("lane") for r in contract_skipped} | {chosen_lane}
            span = (f"issue(s) across {len(lanes_scanned)} lanes"
                    if len(lanes_scanned) > 1 else "lane issue(s)")
            reason = (f"scanned {len(contract_skipped) + 1} {span}, none "
                      f"contract-ready; last (#{target}): {reason}")
        repair_note = (payload.get("contract_repair") or {}).get("skipped")
        if repair_note:
            reason = f"{reason} (contract-repair: {repair_note})"
        payload.update({
            "ok": False,
            "action": "issue_contract_hold",
            "verdict": "ISSUE_CONTRACT_HOLD",
            "reason": reason,
        })
        return finish(payload)
    if contract_hold and force:
        # Operator force (--force): the readiness gate WOULD hold (typically the
        # issue lacks the full agent-context contract the always-on loop demands),
        # but the operator has explicitly accepted a best-effort spawn. Every
        # downstream SAFETY guard still applies -- the preflight cap, the lane lease
        # (no two workers on one leaf tree), the spawn-probe liveness check, the
        # worker-timeout reaper -- and the worker prompt is honest-block-first, so a
        # force can only relax READINESS, never a safety invariant.
        #
        # A bare --force is NOT enough (#2637): silently downgrading a failed gate to
        # `bypassed=true` telemetry is exactly the leak this issue closes. The
        # override must carry a structured --force-reason, which is recorded in the
        # run artifact AND appended to the durable forced-bypass audit ledger so an
        # operator can see WHEN and HOW OFTEN the guard is being bypassed. Without a
        # reason the tick REFUSES to spawn rather than quietly overriding the gate.
        gate_reason = issue_contract_hold_reason(contract)
        reason_text = str(force_reason or "").strip()
        if not reason_text:
            hold_reason = (
                f"--force on a failed issue-contract gate (#{target}: {gate_reason}) "
                f"requires a structured --force-reason so the override is recorded in "
                f"the run ledger; refusing to spawn")
            payload["launch_gate"] = launch_gate_blocked(
                "FORCE_REASON_REQUIRED", hold_reason,
                "re-run with --force --force-reason "
                "\"<why this spawn is worth bypassing the contract gate>\"")
            payload.update({
                "ok": False,
                "action": "force_reason_required",
                "verdict": "FORCE_REASON_REQUIRED",
                "reason": hold_reason,
            })
            _record(runs_dir, payload)
            return finish(payload)
        record_contract_forced_bypass(
            runs_dir,
            {"issue": target, "lane": chosen_lane,
             "score": int(contract.get("score") or 0),
             "reason": reason_text, "gate_reason": gate_reason,
             "title": rec.get("title")},
            live=live)
        payload["issue_contract_forced"] = {
            "bypassed": True,
            "gate_reason": gate_reason,
            "reason": reason_text,
            "bypass_count": contract_forced_bypass_count(runs_dir),
        }

    # Same-issue WIP guard (#2975): hold before command pricing if a previous
    # finished resolver for THIS issue left named local working-tree files dirty.
    if wip_scan is None:
        wip_scan = same_issue_wip_collision(
            runs_dir, target, list(dirty_paths_doc.get("paths") or []))
    if wip_scan.get("collides"):
        payload["same_issue_wip"] = wip_scan
        reason = same_issue_wip_reason(target, wip_scan)
        payload["launch_gate"] = launch_gate_blocked(
            "SAME_ISSUE_WIP", reason,
            "continue-or-commit-same-issue-wip-or-pick-disjoint-issue")
        payload.update({
            "ok": False,
            "action": "same_issue_wip",
            "verdict": "SAME_ISSUE_WIP",
            "refusal_class": REFUSAL_CLASS_WORKTREE_COTENANCY,
            "reason": reason,
        })
        if issue_override is None:
            record_collision_holds(runs_dir, [{
                "kind": "same_issue_wip", "issue": target, "lane": chosen_lane,
                "title": rec.get("title"), "reason": reason[:240],
                "dirty_paths": wip_scan.get("dirty_paths"),
            }], live=live)
        _record(runs_dir, payload)
        return finish(payload)

    # Dirty-path collision guard (#2977): hold before command pricing if the
    # chosen issue explicitly names a path already dirty in this checkout. This
    # protects uncommitted peer WIP that has no live lease left to protect it.
    if dirty_scan is None:
        dirty_scan = dirty_path_collision(
            _candidate_text(rec), list(dirty_paths_doc.get("paths") or []))
    if dirty_scan.get("collides"):
        payload["dirty_path_collision"] = dirty_scan
        reason = dirty_path_collision_reason(target, dirty_scan)
        payload["launch_gate"] = launch_gate_blocked(
            "DIRTY_PATH_COLLISION", reason,
            "wait-for-or-commit-dirty-paths-or-pick-disjoint-issue")
        payload.update({
            "ok": False,
            "action": "dirty_path_collision",
            "verdict": "DIRTY_PATH_COLLISION",
            "refusal_class": REFUSAL_CLASS_WORKTREE_COTENANCY,
            "reason": reason,
        })
        if issue_override is None:
            record_collision_holds(runs_dir, [{
                "kind": "dirty_path", "issue": target, "lane": chosen_lane,
                "title": rec.get("title"), "reason": reason[:240],
                "dirty_paths": dirty_scan.get("dirty_paths"),
            }], live=live)
        _record(runs_dir, payload)
        return finish(payload)

    # Multi-lane scope guard (#2615): the contract gate and the lane-disjointness
    # gate are SEPARATE — a score of 100 never proves the chosen lane's lease covers
    # the issue's mutation region. Refuse an issue whose body names a file family the
    # chosen lane's lease tree does not cover (and that belongs to another lane),
    # rather than admitting a broad rewrite under a narrow lease. Runs on a pinned
    # --issue too (a collision guard, not a readiness one); --force accepts the
    # under-scoped lease transparently. Fail-open on an unavailable taxonomy.
    if scope_scan is None:
        scope_scan = scan_multi_lane_scope(root, _candidate_text(rec), chosen_lane)
    if scope_scan.get("multi_lane") and not force:
        hold_row = {
            "issue": target,
            "lane": chosen_lane,
            "title": rec.get("title"),
            "reason": multi_lane_scope_reason(target, scope_scan),
            "uncovered_lanes": scope_scan.get("uncovered_lanes"),
        }
        record_multi_lane_holds(runs_dir, [hold_row], live=live)
        payload["multi_lane_scope"] = scope_scan
        payload["multi_lane_hold"] = hold_row
        payload.update({
            "ok": False,
            "action": "multi_lane_scope",
            "verdict": "MULTI_LANE_SCOPE",
            "reason": multi_lane_scope_reason(target, scope_scan),
        })
        _record(runs_dir, payload)
        return finish(payload)
    if scope_scan.get("multi_lane") and force:
        payload["multi_lane_scope_forced"] = {
            "bypassed": True,
            "uncovered_lanes": scope_scan.get("uncovered_lanes"),
            "uncovered": scope_scan.get("uncovered"),
        }

    model, effort = worker_model_effort(backend, acct)
    preview_prompt = f"<resolve #{target} prompt, {rec.get('prompt_chars')} chars>"
    preview_env = dispatch_worker.child_env(chosen_lane, backend, root)
    preview_command, preview_guarded = dispatch_worker.guarded_launch_command(
        build_worker_command(backend, preview_prompt, model, effort),
        chosen_lane, backend, root, preview_env)
    payload["command"] = preview_command
    payload["guarded"] = preview_guarded
    payload["launch_gate"] = launch_gate_for_guard(preview_guarded, backend)

    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 {backend} issue-resolution worker on "
                                   f"#{target} (lane '{chosen_lane}') under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')}, "
                                   f"{acct.get('model') or backend})")})
        return finish(payload)

    # Fenced lane-lease ACQUIRE (residual of #1310) — the authoritative pre-spawn
    # admission, layered atop the same-host log scan above. ATOMICALLY take
    # refs/fak/locks/resolve-<lane> before launching; a structured fence refusal
    # (a live peer holds the lane, on THIS host or another clone after a fetch)
    # returns LANE_LEASE_HELD instead of racing a second worker onto the leaf tree.
    # Fail-open: a missing fak binary / broken store proceeds on the log scan
    # alone (lease.acquired False, lease.refused False) so the loop never wedges.
    ttl_s = int((worker_timeout_s or DEFAULT_WORKER_TIMEOUT_S) + LEASE_TTL_MARGIN_S)
    lease = acquire_lane_lease(root, chosen_lane, tree=lane_tree(root, chosen_lane),
                               ttl_s=ttl_s, runner=lease_runner)
    payload["lease"] = lease
    if lease.get("refused"):
        payload.update({"ok": False, "action": "lane_leased", "verdict": "LANE_LEASE_HELD",
                        "refusal_class": REFUSAL_CLASS_LANE_LEASE,
                        "reason": (f"lane '{chosen_lane}' lease is held by a live peer "
                                   f"(fence {lease.get('reason') or lease.get('fence_verdict') or '?'}); "
                                   f"refusing COLLISION_RISK — the fenced "
                                   f"refs/fak/locks/{lease.get('id')} lease serializes "
                                   f"this lane across machines, not just this host")})
        _record(runs_dir, payload)
        return finish(payload)

    if backend == "claude":
        env = issue_dispatch.worker_env(acct.get("dir"), chosen_lane, root)
    elif backend == "codex":
        env = codex_worker_env(acct.get("dir"), chosen_lane, root)
    else:
        env = opencode_worker_env(acct.get("dir"), chosen_lane, root, runs_dir)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    command, guarded = dispatch_worker.guarded_launch_command(
        build_worker_command(backend, rec["prompt"], model, effort),
        chosen_lane, backend, root, env)
    if guarded:
        dispatch_worker.guard_env_augment(env)
    payload["command"] = command
    payload["guarded"] = guarded
    payload["launch_gate"] = launch_gate_for_guard(guarded, backend)
    # Per-worker commit-sha tracking (#1324 proposal #2): stamp repo HEAD now, before
    # the worker can commit, so a later tick re-audits the commit THIS worker lands
    # (base..HEAD citing #target). Fail-open: a git error -> no base, and the witness
    # falls back to recent history.
    rc_head, head_out = _git_capture(root, ["rev-parse", "HEAD"])
    base_sha = head_out.strip() if rc_head == 0 and head_out.strip() else None
    spawned = spawn_issue_worker(command, env, root, runs_dir, target, chosen_lane, backend,
                                 account=acct, lease=lease, base_sha=base_sha,
                                 spawn_probe_s=spawn_probe_s,
                                 prompt_payload=rec["prompt"] if backend in ("claude", "opencode") else None)
    early = spawned.get("early_exit") or {}
    if early.get("checked") and not early.get("alive"):
        if lease.get("acquired"):
            release = release_lane_lease(root, lease, runner=lease_runner)
            payload["lease_release"] = release
            if not release.get("released"):
                payload["lease_held_until_ttl"] = {"id": lease.get("id"),
                                                   "holder": lease.get("holder"),
                                                   "release": release}
        cap_hit = _cap_hit_from_text(
            str(early.get("tail") or ""),
            evidence_log=Path(str(spawned.get("log") or "")).name)
        if cap_hit:
            import time
            payload["quota_cap"] = _write_cap_hold(
                runs_dir, product=product, account_tag=acct.get("tag"),
                hit=cap_hit, now_ts=time.time(), fallback_min=60,
                source="early_exit")
        # A transient <5s child crash is self-healing noise, not a tick malfunction:
        # count it against THIS target and only redden the health bit once the same
        # target keeps failing (failover not healing it) — see tick_exit_code (#2636).
        # The SEAT streak is bumped alongside (#4591): the same crash also counts
        # against the routed seat, so a dead seat cycling across DIFFERENT targets
        # still accrues one run-length and the cool gate above stops re-handing it.
        payload.update({"ok": False, "action": "spawn_failed",
                        "verdict": "SPAWN_FAILED",
                        "spawned": spawned,
                        "cause": classify_spawn_failed_cause(early),
                        "spawn_failed_streak": bump_spawn_failure_streak(
                            runs_dir, target, backend),
                        "seat_streak": bump_spawn_failure_streak_seat(
                            runs_dir, acct.get("tag"), backend),
                        "reason": (f"{backend} worker pid {spawned['pid']} for "
                                   f"#{target} exited within {early.get('wait_s')}s "
                                   f"with code {early.get('returncode')}"
                                   + (" and produced an empty log" if early.get("silent") else ""))})
        _record(runs_dir, payload)
        return finish(payload)
    # A worker that launched cleanly breaks any prior same-target SPAWN_FAILED streak
    # and the routed SEAT's cross-target streak (#4591).
    clear_spawn_failure_streak(runs_dir, target, backend)
    clear_spawn_failure_streak_seat(runs_dir, acct.get("tag"), backend)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned {backend} issue-resolution worker pid "
                               f"{spawned['pid']} on #{target} (lane '{chosen_lane}') "
                               f"under '{acct.get('tag')}' ({acct.get('model') or backend})")})
    _record(runs_dir, payload)
    return finish(payload)


def resolve_spawn_burst(flag: int | None = None,
                        env: Any | None = None) -> dict[str, Any]:
    """Resolve one tick's spawn-burst limit — ``flag > env > default``, clamped.

    Precedence: an explicit ``--spawn-burst`` wins; otherwise ``FAK_DISPATCH_SPAWN_BURST``
    (the zero-touch way to arm the ALREADY-REGISTERED scheduled task without re-writing
    its action); otherwise ``DEFAULT_SPAWN_BURST`` (1 = exactly today's behavior).

    Every path is clamped to ``[1, SPAWN_BURST_HARD_CEILING]`` and NOTHING here raises:
    a garbage env value falls back to the default rather than failing the tick, and no
    value can exceed the ceiling. That is the whole point — the burst bound is the
    fail-safe against the #3153 spawn-churn lockup, so it must not be defeatable by
    configuration. Returns the audit block the tick payload carries under ``burst``."""
    env = os.environ if env is None else env
    source = "default"
    requested: int = DEFAULT_SPAWN_BURST
    env_invalid: str | None = None
    raw_env = env.get(SPAWN_BURST_ENV)
    if raw_env not in (None, ""):
        try:
            requested = int(str(raw_env).strip())
            source = f"env:{SPAWN_BURST_ENV}"
        except (TypeError, ValueError):
            env_invalid = str(raw_env)
    if flag is not None:
        try:
            requested = int(flag)
            source = "flag:--spawn-burst"
        except (TypeError, ValueError):
            pass
    limit = max(1, min(int(requested), SPAWN_BURST_HARD_CEILING))
    out: dict[str, Any] = {
        "limit": limit, "requested": int(requested), "source": source,
        "hard_ceiling": SPAWN_BURST_HARD_CEILING,
        "clamped": limit != int(requested),
    }
    if env_invalid is not None:
        out["env_invalid"] = env_invalid
    return out


def resolve_spawn_burst_stagger(flag: float | None = None,
                                env: Any | None = None) -> float:
    """Seconds to wait BETWEEN spawns inside one burst — ``flag > env > default``.

    Spreading N launches over N*stagger seconds is what turns a fan-out from a spawn
    BURST (the #3153 lockup shape: many process trees created inside one scheduler
    window) into a paced arrival. It also gives each sub-tick's own preflight a fresh
    read of host load before the next admission. Clamped to
    ``[0, SPAWN_BURST_STAGGER_MAX_S]`` so the pacing can never outlive the 5-minute
    tick window; never raises."""
    env = os.environ if env is None else env
    value = DEFAULT_SPAWN_BURST_STAGGER_S
    raw_env = env.get(SPAWN_BURST_STAGGER_ENV)
    if raw_env not in (None, ""):
        try:
            value = float(str(raw_env).strip())
        except (TypeError, ValueError):
            value = DEFAULT_SPAWN_BURST_STAGGER_S
    if flag is not None:
        try:
            value = float(flag)
        except (TypeError, ValueError):
            pass
    if value != value or value < 0:  # NaN or negative
        value = 0.0
    return float(min(value, SPAWN_BURST_STAGGER_MAX_S))


def evaluate_burst(root: Path, *, spawn_burst: int = DEFAULT_SPAWN_BURST,
                   burst_stagger_s: float = DEFAULT_SPAWN_BURST_STAGGER_S,
                   sleeper: Any | None = None,
                   **kw: Any) -> dict[str, Any]:
    """One TICK that may spawn up to ``spawn_burst`` workers, filling live headroom.

    This is the arrival-rate fix. :func:`evaluate` spawns at most ONE worker per call,
    so a PT5M tick against a ~5-min worker lifetime pins steady-state concurrency near
    1 regardless of ``--max-workers``. This driver re-enters ``evaluate`` while the
    previous sub-tick actually SPAWNED, up to the burst limit.

    Independent admission, by construction: each iteration is a FULL ``evaluate`` call,
    so every gate re-runs per spawn — preflight (host cap / seat pool / live<cap),
    weekly-cap, backend health, seat-cool, live-issue and live-lane de-dup, the
    contract / dirty-path / same-issue-WIP / multi-lane scans, and the fenced
    ``refs/fak/locks/resolve-<lane>`` lease. No admission decision is hoisted over N
    spawns. The de-dup is self-consistent because it reads the FILESYSTEM, not memory:
    sub-tick N's worker leaves a ``.pid`` sidecar (so it counts as live and consumes
    headroom), a spawn-header log (so its lane reads HELD), and a fresh log mtime (so
    its issue is in the attempt cooldown) — all of which sub-tick N+1 re-reads. The
    collision holds sub-tick N ledgered are likewise re-read, so the fan-out advances to
    DISJOINT candidates instead of re-refusing the same blocked head.

    Stops at the FIRST non-``spawned`` verdict (nothing to spawn, at cap, lane leased,
    collision, contract hold, spawn failure). A refusal means the fleet is at a real
    boundary; re-attempting it inside the same tick would only re-refuse.

    DRY RUNS DO NOT FAN OUT: a ``--live=False`` sub-tick spawns nothing, so no headroom
    is consumed and every sub-tick would re-pick the SAME issue and report it N times.
    A dry run therefore evaluates exactly once and says so.

    ``sleeper`` is the injectable stagger seam (tests pass a recorder; production uses
    ``time.sleep``). Returns the FIRST sub-tick's payload — the canonical verdict for
    this tick, and the one whose ``action`` :func:`tick_exit_code` grades — augmented
    with a ``burst`` audit block. With the default limit of 1 the payload is
    byte-identical to a plain ``evaluate``."""
    limit = max(1, min(int(spawn_burst), SPAWN_BURST_HARD_CEILING))
    live = bool(kw.get("live"))
    stagger = max(0.0, float(burst_stagger_s))
    if limit <= 1:
        return evaluate(root, **kw)
    if not live:
        payload = evaluate(root, **kw)
        payload["burst"] = {
            "limit": limit, "attempts": 1, "spawned": 0, "stagger_s": stagger,
            "skipped": ("dry_run — a dry-run sub-tick spawns nothing, so headroom "
                        "never moves and every further sub-tick would re-pick the "
                        "same issue; re-run with --live to fan out"),
        }
        return payload

    payloads: list[dict[str, Any]] = []
    spawned = 0
    stopped_on = "burst_limit"
    sub_kw = dict(kw)
    for n in range(limit):
        if n:
            # The registry refresh is a per-TICK concern (route off current account
            # evidence), not a per-spawn one: sub-tick 1 already re-derived it seconds
            # ago. Skipping it on the follow-ons drops one subprocess per extra spawn —
            # which is itself churn reduction on the axis that freezes this host.
            sub_kw["refresh"] = False
            _burst_stagger_sleep(stagger, sleeper)
        payload = evaluate(root, **sub_kw)
        payloads.append(payload)
        if str(payload.get("action") or "") != "spawned":
            stopped_on = str(payload.get("verdict") or payload.get("action") or "unknown")
            break
        spawned += 1

    out = dict(payloads[0])
    out["burst"] = {
        "limit": limit,
        "attempts": len(payloads),
        "spawned": spawned,
        "stagger_s": stagger,
        "stopped_on": stopped_on,
        "sub_ticks": [{
            "n": i + 1,
            "action": p.get("action"),
            "verdict": p.get("verdict"),
            "issue": p.get("target_issue"),
            "lane": p.get("lane"),
            "pid": (p.get("spawned") or {}).get("pid"),
        } for i, p in enumerate(payloads)],
    }
    if spawned > 1:
        extra = [f"#{r['issue']}" for r in out["burst"]["sub_ticks"][1:]
                 if r.get("action") == "spawned" and r.get("issue")]
        out["reason"] = (f"{out.get('reason')} (+{spawned - 1} more this tick: "
                         f"{' '.join(extra)}; burst {spawned}/{limit})")
    if len(payloads) > 1:
        # Each sub-tick already wrote its own last-resolve-tick artifact via finish();
        # re-record the AGGREGATE last so the on-disk tick record is the whole tick
        # (a SPAWNED verdict plus its burst block), not the trailing refusal.
        _record(root / RUNS_DIRNAME, out)
    return out


def _burst_stagger_sleep(seconds: float, sleeper: Any | None = None) -> float:
    """Pace one inter-spawn gap. Injectable so tests never really sleep."""
    if seconds <= 0:
        return 0.0
    if sleeper is None:
        import time
        time.sleep(seconds)
    else:
        sleeper(seconds)
    return seconds


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    # Scope the tick record by backend so concurrent tasks (e.g. the opus
    # FleetIssueDispatch and the glm FleetIssueDispatchGlm) don't clobber each
    # other's trace; also keep the legacy unscoped file (last-writer) for any
    # manual reader that still expects it.
    backend = payload.get("backend") or "claude"
    blob = json.dumps(payload, indent=2)
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"last-resolve-tick-{backend}.json").write_text(blob, encoding="utf-8")
        (runs_dir / "last-resolve-tick.json").write_text(blob, encoding="utf-8")
    except OSError:
        pass


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    lines = [
        f"issue-resolve-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  "
        f"backend={p.get('backend')}  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live)",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)",
        f"  target    : #{p.get('target_issue')}  {(p.get('issue_title') or '')[:54]}",
        f"  -> {p.get('reason')}",
    ]
    sa = p.get("seat_adaptive") or {}
    if sa.get("signal_available"):
        lines.insert(2, f"  seats     : adaptive cap {sa.get('effective_target')} "
                        f"(binding {sa.get('binding')}; live {sa.get('live')} + "
                        f"free {sa.get('seat_free')}, ceiling {sa.get('hard_ceiling')}, "
                        f"ramp +{sa.get('ramp_delta')}/tick)")
    hint = p.get("preflight_hint") or {}
    if hint.get("message"):
        lines.append(f"  hint      : {hint.get('message')}")
    if p.get("spawned"):
        s = p["spawned"]
        lines.append(f"  spawned pid={s.get('pid')} issue=#{s.get('issue')} log={s.get('log')}")
    burst = p.get("burst") or {}
    if burst.get("limit"):
        if burst.get("skipped"):
            lines.append(f"  burst     : limit {burst.get('limit')} — {burst['skipped']}")
        else:
            rows = " ".join(
                f"#{r.get('issue')}:{r.get('verdict')}"
                for r in (burst.get("sub_ticks") or []))
            lines.append(f"  burst     : {burst.get('spawned')}/{burst.get('limit')} "
                         f"spawned in {burst.get('attempts')} sub-tick(s), stopped on "
                         f"{burst.get('stopped_on')} "
                         f"(stagger {burst.get('stagger_s')}s) — {rows}")
    cs = p.get("contract_skipped") or []
    if cs:
        lanes = sorted({str(r.get("lane")) for r in cs if r.get("lane")})
        lines.append(f"  scan      : {len(cs)} contract-held head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    ws = p.get("same_issue_wip_skipped") or []
    if ws:
        lanes = sorted({str(r.get("lane")) for r in ws if r.get("lane")})
        lines.append(f"  scan      : {len(ws)} same-issue WIP head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    ds = p.get("dirty_path_skipped") or []
    if ds:
        lanes = sorted({str(r.get("lane")) for r in ds if r.get("lane")})
        lines.append(f"  scan      : {len(ds)} dirty-path head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    ms = p.get("multi_lane_skipped") or []
    if ms:
        lanes = sorted({str(r.get("lane")) for r in ms if r.get("lane")})
        lines.append(f"  scan      : {len(ms)} multi-lane head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    icf = p.get("issue_contract_forced") or {}
    if icf.get("bypassed"):
        lines.append(f"  forced    : contract gate bypassed — {icf.get('reason')} "
                     f"(gate: {icf.get('gate_reason')}; total bypasses "
                     f"{icf.get('bypass_count')})")
    cr = p.get("contract_repair") or {}
    if cr.get("batch"):
        lines.append(f"  repair    : issue(s) {cr['batch']} -> {p.get('verdict')}")
    elif cr.get("skipped"):
        lines.append(f"  repair    : skipped — {cr['skipped']}")
    elif cr.get("in_flight"):
        lines.append(f"  repair    : in flight (pid {cr['in_flight'][0].get('pid')})")
    rl = p.get("reallocation") or {}
    if rl.get("bonus") or rl.get("claimed_lanes"):
        frm = ",".join(x for x in (rl.get("from") or []) if x)
        claimed = ",".join(x for x in (rl.get("claimed_lanes") or []) if x)
        lines.append(f"  realloc   : +{rl.get('bonus')} workers, lane(s) [{claimed}] "
                     f"claimed from dead [{frm}] -> eff cap {rl.get('effective_max_workers')}")
    timed_out = p.get("timed_out_workers") or {}
    reaped = timed_out.get("reaped") or []
    would_reap = timed_out.get("would_reap") or []
    if reaped:
        lines.append(f"  reaped timed-out workers: {len(reaped)}")
    elif would_reap:
        lines.append(f"  would reap timed-out workers: {len(would_reap)}")
    ws = p.get("witnessed_slots") or {}
    if ws.get("audited"):
        lines.append(f"  witnessed slots: {len(ws.get('witnessed') or [])} witnessed, "
                     f"{len(ws.get('unwitnessed') or [])} CLAIM_UNWITNESSED, "
                     f"{len(ws.get('no_commit') or [])} no-commit")
    def _gate_suffix(gate: dict[str, Any]) -> str:
        blockers = gate.get("blockers") or []
        bits = []
        for b in blockers[:3]:
            bits.append(f"{b.get('code')}: {b.get('next_action')}")
        return "; ".join(bits) if bits else "launch gate is not ready"

    if not p.get("live") and p.get("action") == "would_spawn":
        gate = p.get("launch_gate") or {}
        if gate.get("ready"):
            lines.append("  DRY-RUN — re-run with --live to spawn the issue worker")
        else:
            lines.append(f"  DRY-RUN — NOT launch-ready ({_gate_suffix(gate)})")
    elif not p.get("live"):
        gate = p.get("launch_gate") or {}
        if gate.get("ready") is False:
            lines.append(f"  launch gate: BLOCKED ({_gate_suffix(gate)})")
    return "\n".join(lines)


# main()'s return value is a scheduled-task / loop LastResult. It must flag only
# a genuine MALFUNCTION, never the fleet correctly declining to spawn. These are
# evaluate()'s "ran correctly" actions: the tick either dispatched work (spawned
# / would_spawn) or BENIGNLY declined because there was nothing to spawn or the
# box is at capacity — no eligible issue, no free lane, lane busy/leased, host
# refused at cap, account weekly-capped, backend unhealthy. All exit 0 so Task
# Scheduler — and fleet_control_pane.scheduler_result_needs_action, which treats
# any non-zero LastResult as needs-action — don't read a healthy idle/capped
# fleet as a crash. Only SPAWN_FAILED (a worker that died on launch), an unknown
# action, or an unhandled exception (which exits 1 on its own) is a real failure.
# Before this, a routine NO_ISSUE/WEEKLY_CAPPED tick exited 1, so the always-on
# FleetIssueDispatch/Codex tasks were chronically 0x1 and indistinguishable from
# a real crash.
BENIGN_ACTIONS = frozenset({
    "spawned", "would_spawn",                            # work dispatched
    "repair_spawned", "would_repair",                    # contract grooming dispatched
    "repair_in_flight",                                  # a groomer is already on it
    "no_issue", "no_lane", "lane_busy", "lane_leased",   # nothing to spawn
    "same_issue_wip",                                    # same issue has local WIP
    "dirty_path_collision",                              # local WIP collision guard
    "active_guard_livelock",                             # live guard loop fuse
    "active_compact_runaway",                            # live compact-control loop fuse
    "multi_lane_scope",                                  # issue spans lanes the lease can't cover (#2615)
    "issue_contract_hold",                               # issue needs scope before spawn
    "force_reason_required",                             # --force lacked a structured --force-reason (#2637)
    "refused",                                           # preflight backpressure (host at cap / no account)
    "weekly_capped", "backend_unhealthy",                # pool unavailable; declined correctly
    "backend_health_skip",                               # backend majority-stub; planned 0 spawns (#3247)
    "seat_cooled",                                       # routed seat's spawn-fail streak cooled it (#4591)
})


# A SPAWN_FAILED is NOT a dispatcher malfunction: at the ~1-in-25 baseline the tick
# did its job (picked a target, passed the contract gate, took the lane lease, launched)
# and the CHILD died in <5s — which the dispatcher's own backend/account failover
# retries on a later tick. A SINGLE such crash must not redden the scheduled-task health
# bit (LastTaskResult), which the control-pane watchdog reads as green/red. Only a spawn
# that keeps failing on the SAME target this many times IN A ROW — failover not healing
# it — is a real, non-self-healing fault the watchdog should see red (#2636).
SPAWN_FAILED_RED_STREAK = 3


def _spawn_failure_streak_path(runs_dir: Path, backend: str) -> Path:
    # Per-backend so the opus FleetIssueDispatch and the glm FleetIssueDispatchGlm keep
    # independent same-target run-lengths and never clobber each other's counter.
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(backend or "claude")) or "claude"
    return runs_dir / f"spawn-failure-streak-{safe}.json"


def _read_spawn_failure_streaks(runs_dir: Path, backend: str) -> dict[str, int]:
    try:
        doc = json.loads(
            _spawn_failure_streak_path(runs_dir, backend).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in doc.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _write_spawn_failure_streaks(runs_dir: Path, backend: str,
                                 streaks: dict[str, int]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        _spawn_failure_streak_path(runs_dir, backend).write_text(
            json.dumps(streaks, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # fail-open: a streak we can't persist just resets to single-transient


def bump_spawn_failure_streak(runs_dir: Path, target: int | str, backend: str) -> int:
    """Record one more consecutive SPAWN_FAILED for ``target`` on ``backend`` and return
    the new run-length. Reset by :func:`clear_spawn_failure_streak` on the next
    successful spawn of the same target, so the returned count is the number of spawn
    failures IN A ROW — the 'failover not healing it' signal :func:`tick_exit_code`
    reddens on (#2636). Fail-open: an unreadable ledger just starts the count at 1."""
    streaks = _read_spawn_failure_streaks(runs_dir, backend)
    count = streaks.get(str(target), 0) + 1
    streaks[str(target)] = count
    _write_spawn_failure_streaks(runs_dir, backend, streaks)
    return count


def clear_spawn_failure_streak(runs_dir: Path, target: int | str, backend: str) -> None:
    """A successful spawn of ``target`` broke the streak — failover healed it, so the
    same-target SPAWN_FAILED run-length resets to zero (#2636)."""
    streaks = _read_spawn_failure_streaks(runs_dir, backend)
    if str(target) in streaks:
        del streaks[str(target)]
        _write_spawn_failure_streaks(runs_dir, backend, streaks)


# --- Seat-keyed spawn-failure streak (#4591) --------------------------------
# The pair above is TARGET-keyed: it reddens when failover keeps failing on the
# SAME issue. It is blind to the inverse drain: a dead needs_login SEAT that the
# selector keeps re-handing across DIFFERENT issues restarts the target counter
# at 1 every tick, so it never builds a streak, never cools, and bleeds the
# fleet one issue at a time with no alert (the observed drain that recurred
# after the #2059/#2075 stale-cred detectors shipped). This pair keys the SAME
# run-length by account/seat tag instead, persisted per backend in
# runs/spawn-failure-streak-seat-<backend>.json as
# {seat_tag: {"count": N, "last_ts": epoch}}. A seat at/over
# SPAWN_FAILED_RED_STREAK is COOLED: the evaluate() gate reroutes to a
# different seat or declines the tick instead of feeding the dead seat another
# issue, and one re-probe spawn is admitted every SEAT_STREAK_REPROBE_MIN
# minutes so a re-logged-in seat can clear itself (a permanently-skipped seat
# could never break its own streak).
SEAT_STREAK_REPROBE_MIN = 30


def _spawn_failure_streak_seat_path(runs_dir: Path, backend: str) -> Path:
    # Per-backend for the same reason as the target-keyed ledger: independent
    # counters that never clobber each other.
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(backend or "claude")) or "claude"
    return runs_dir / f"spawn-failure-streak-seat-{safe}.json"


def _read_seat_failure_streaks(runs_dir: Path, backend: str) -> dict[str, dict[str, float]]:
    try:
        doc = json.loads(
            _spawn_failure_streak_seat_path(runs_dir, backend).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for k, v in doc.items():
        if isinstance(v, dict):
            try:
                out[str(k)] = {"count": int(v.get("count") or 0),
                               "last_ts": float(v.get("last_ts") or 0.0)}
            except (TypeError, ValueError):
                continue
        else:
            try:  # tolerate a bare-int row (the target-ledger shape)
                out[str(k)] = {"count": int(v), "last_ts": 0.0}
            except (TypeError, ValueError):
                continue
    return out


def _write_seat_failure_streaks(runs_dir: Path, backend: str,
                                streaks: dict[str, dict[str, float]]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        _spawn_failure_streak_seat_path(runs_dir, backend).write_text(
            json.dumps(streaks, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # fail-open, mirroring the target-keyed ledger


def bump_spawn_failure_streak_seat(runs_dir: Path, seat_tag: str | None,
                                   backend: str,
                                   now_ts: float | None = None) -> int:
    """Record one more consecutive SPAWN_FAILED for ``seat_tag`` on ``backend`` —
    REGARDLESS of which issue the seat was launched at — and return the new
    run-length. The #4591 keying fix: a dead seat cycling across N distinct
    targets accrues ONE seat streak of N instead of N fresh target streaks of 1.
    A blank tag records nothing (returns 0); fail-open like the target ledger."""
    import time
    tag = str(seat_tag or "").strip()
    if not tag:
        return 0
    streaks = _read_seat_failure_streaks(runs_dir, backend)
    prior = streaks.get(tag) or {}
    row = {"count": int(prior.get("count") or 0) + 1,
           "last_ts": float(time.time() if now_ts is None else now_ts)}
    streaks[tag] = row
    _write_seat_failure_streaks(runs_dir, backend, streaks)
    return int(row["count"])


def clear_spawn_failure_streak_seat(runs_dir: Path, seat_tag: str | None,
                                    backend: str) -> None:
    """A clean launch ON THIS SEAT — any target — broke the seat's streak. This
    is deliberately seat-scoped, not same-target-scoped: the target-keyed clear
    above was what let a dead seat's evidence evaporate on every issue rotation."""
    tag = str(seat_tag or "").strip()
    if not tag:
        return
    streaks = _read_seat_failure_streaks(runs_dir, backend)
    if tag in streaks:
        del streaks[tag]
        _write_seat_failure_streaks(runs_dir, backend, streaks)


def seat_spawn_failure_state(runs_dir: Path, seat_tag: str | None, backend: str,
                             now_ts: float | None = None) -> dict[str, Any]:
    """The routed seat's current seat-keyed streak, folded for the evaluate()
    cool gate (#4591): ``cooled`` once the run-length reaches
    SPAWN_FAILED_RED_STREAK, ``reprobe_due`` once SEAT_STREAK_REPROBE_MIN
    minutes have passed since the last recorded failure (admit ONE probe spawn
    so recovery is detectable). Fail-open: a blank tag or unreadable ledger
    reads streak 0, not cooled."""
    import time
    tag = str(seat_tag or "").strip()
    row = _read_seat_failure_streaks(runs_dir, backend).get(tag) if tag else None
    count = int((row or {}).get("count") or 0)
    last_ts = float((row or {}).get("last_ts") or 0.0)
    now = float(time.time() if now_ts is None else now_ts)
    cooled = count >= SPAWN_FAILED_RED_STREAK
    return {"seat": tag or None, "backend": backend, "streak": count,
            "last_ts": last_ts or None, "cooled": cooled,
            "reprobe_min": SEAT_STREAK_REPROBE_MIN,
            "reprobe_due": bool(cooled and last_ts
                                and now - last_ts >= SEAT_STREAK_REPROBE_MIN * 60)}


def tick_exit_code(payload: dict[str, Any]) -> int:
    """Process exit code for one dispatch tick: 0 when the tick ran correctly
    (dispatched work, or benignly declined per BENIGN_ACTIONS), non-zero only on a
    genuine malfunction such as an unrecognised action or a non-dict payload. A
    SPAWN_FAILED is the ~1-in-25 self-healing baseline (correct tick, child died on
    launch, failover retries it): a SINGLE same-target crash stays 0 and only a run of
    ``SPAWN_FAILED_RED_STREAK`` in a row on the same target (``spawn_failed_streak``,
    stamped by :func:`bump_spawn_failure_streak`) goes 1. Keeps a healthy tick — idle,
    capped, OR self-healing a transient child crash — from reporting failure to Task
    Scheduler / the control-pane watchdog (#2636)."""
    if not isinstance(payload, dict):
        return 1
    action = str(payload.get("action") or "")
    if action in BENIGN_ACTIONS:
        return 0
    if action == "spawn_failed":
        try:
            streak = int(payload.get("spawn_failed_streak") or 1)
        except (TypeError, ValueError):
            streak = 1
        return 1 if streak >= SPAWN_FAILED_RED_STREAK else 0
    return 1


def _emit_low_yield_excludes(root: Path) -> int:
    """Emit the #2062 low-yield soft-exclude lane set as JSON (read-only) and exit.

    The live Go dispatch launcher (``fak dispatch wave`` / ``fak dispatch tick``)
    shells out for this so the SAME trust-gated fold that soft-demotes a poison-pill
    lane inside this Python picker (see ``low_yield_soft_excludes`` and its call at the
    ``evaluate`` busiest-pick) also steers the Go fleet launcher -- instead of the two
    paths disagreeing, with the Go picker happily re-storming a lane the fold already
    flagged as >=40 turns / 0 closes. Reuses the exact lane->tree taxonomy resolution
    the picker uses. Fail-open throughout: any error yields an empty exclude set and a
    zero exit, so a caller that shells out for it is never starved by this probe.
    """
    runs_dir = root / RUNS_DIRNAME
    try:
        _concurrent, low_yield_trees, _exclusive = issue_lane_router.lane_taxonomy(root)
    except Exception:
        low_yield_trees = {}
    low_yield = low_yield_soft_excludes(root, runs_dir, lane_trees=low_yield_trees)
    payload = {
        "exclude": sorted(str(x) for x in (low_yield.get("exclude") or set())),
        "lanes": low_yield.get("lanes") or [],
        "flagged": low_yield.get("flagged") or [],
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One guarded tick that spawns an issue-RESOLUTION worker (dry-run by default).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=20,
                    help="fail-safe cap on live workers, enforced by the preflight "
                         "(default: 20). With seat-adaptive sizing (the default, "
                         "#3246) this binds only when the preflight returns no seat "
                         "signal; with --no-seat-adaptive it is the hard cap.")
    ap.add_argument("--no-seat-adaptive", action="store_true",
                    help="disable seat-adaptive tick sizing (#3246): keep the "
                         "configured --max-workers as the binding cap instead of "
                         "growing to min(live+seat_free, host_cap, --seat-ceiling, "
                         "live+--seat-ramp-delta) when the preflight carries a "
                         "seat signal")
    ap.add_argument("--seat-ceiling", type=int, default=DEFAULT_SEAT_CEILING,
                    help=f"hard fail-safe ceiling on the seat-adaptive cap (default "
                         f"{DEFAULT_SEAT_CEILING}). An explicit --max-workers above "
                         f"it raises it — seat sizing never shrinks an operator's "
                         f"configured cap.")
    ap.add_argument("--seat-ramp-delta", type=int, default=DEFAULT_SEAT_RAMP_DELTA,
                    help=f"max NEW workers one tick may plan above the live count "
                         f"(canary-safe ramp, #3246; default "
                         f"{DEFAULT_SEAT_RAMP_DELTA}; 0 disables the ramp bound so "
                         f"the cap jumps straight to the seat/host minimum)")
    ap.add_argument("--spawn-burst", type=int, default=None,
                    help=f"max workers ONE tick may spawn, filling live headroom "
                         f"(default {DEFAULT_SPAWN_BURST} = one spawn per tick, exactly "
                         f"the pre-fan-out behavior; hard-clamped to "
                         f"{SPAWN_BURST_HARD_CEILING}). This is SEPARATE from "
                         f"--max-workers: --max-workers bounds the total LIVE "
                         f"population, this bounds NEW processes per tick — the churn "
                         f"axis behind the host-lockup class (#3153). Every spawn in "
                         f"the burst re-runs the full admission chain (preflight, seat, "
                         f"collision scans, lane lease) independently. Also settable "
                         f"fleet-wide via {SPAWN_BURST_ENV} (flag wins). Dry runs never "
                         f"fan out.")
    ap.add_argument("--spawn-burst-stagger-s", type=float, default=None,
                    help=f"seconds between spawns inside one burst (default "
                         f"{DEFAULT_SPAWN_BURST_STAGGER_S}; env "
                         f"{SPAWN_BURST_STAGGER_ENV}; clamped to "
                         f"{SPAWN_BURST_STAGGER_MAX_S}). Paces process creation so a "
                         f"fan-out arrives as a ramp, not a burst, and each sub-tick's "
                         f"preflight re-reads host load before the next launch.")
    ap.add_argument("--work-kind", default=None,
                    help="switcher work kind (engineering->t1, gardening->t2). Default "
                         "follows --backend: engineering for claude (t1 opus pool), "
                         "gardening for opencode/glm (t2 pool). The opencode pool has NO "
                         "t1 account, so an engineering route there REFUSE_NO_ACCOUNTs — "
                         "which silently stalled the docs lane until this default landed.")
    ap.add_argument("--lane", default=None,
                    help="explicit lane (default: the lane with the most open issues)")
    ap.add_argument("--issue", type=int, default=None,
                    help="pin an explicit target issue #N (operator override of the "
                         "freshest-first auto-pick). The full safety chain still runs "
                         "on it (preflight cap, contract gate, lane lease, spawn). "
                         "Requires --lane so the lane lease/tree matches the issue.")
    ap.add_argument("--backend", choices=BACKENDS, default="claude",
                    help="worker backend: claude (opus, t1) or opencode (glm-5.2, t2, "
                         "a separate quota pool). Default claude.")
    ap.add_argument("--exclude-lane", default="",
                    help="comma-separated lanes to drop from the busiest-pick (e.g. an "
                         "opus task excludes 'docs' so a glm task owns it). Ignored when "
                         "--lane is set.")
    ap.add_argument("--live", action="store_true",
                    help="actually spawn the worker (default: dry-run / plan only)")
    ap.add_argument("--force", action="store_true",
                    help="operator best-effort: downgrade the issue-contract readiness "
                         "HOLD to advisory and spawn anyway. REQUIRES --force-reason so "
                         "the override is recorded in the run ledger (#2637); a bare "
                         "--force on a failed gate is refused FORCE_REASON_REQUIRED. "
                         "Every SAFETY guard still applies (preflight cap, lane lease, "
                         "spawn probe, worker-timeout). Use to dispatch the top real "
                         "issues when the always-on gate demands a fuller agent-context "
                         "contract than the backlog currently carries.")
    ap.add_argument("--force-reason", default="",
                    help="structured operator reason recorded when --force bypasses a "
                         "failed issue-contract gate (#2637). Written to the run "
                         f"artifact (issue_contract_forced.reason) and appended to the "
                         f"{RUNS_DIRNAME}/{_CONTRACT_FORCED_LEDGER} audit ledger whose "
                         "count exposes how often the guard has been bypassed.")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh")
    ap.add_argument("--cooldown-min", type=int, default=120,
                    help="skip an issue attempted within this many minutes (anti-churn; "
                         "0 disables). Default 120 — stops re-storming one un-landable issue.")
    ap.add_argument("--worker-timeout-s", type=int, default=DEFAULT_WORKER_TIMEOUT_S,
                    help=f"wall-clock cap for live resolver workers before this tick "
                         f"reaps their process tree (default: {DEFAULT_WORKER_TIMEOUT_S}; "
                         "0 disables)")
    ap.add_argument("--spawn-probe-s", type=float, default=DEFAULT_SPAWN_PROBE_S,
                    help=f"seconds to wait after a live spawn to catch immediate "
                         f"empty-log exits (default: {DEFAULT_SPAWN_PROBE_S}; 0 disables)")
    ap.add_argument("--max-realloc-workers", type=int, default=DEFAULT_REALLOC_CEILING,
                    help=f"ceiling on extra worker slots a HEALTHY backend may claim "
                         f"from dead siblings in one tick (backend-health reallocation; "
                         f"default {DEFAULT_REALLOC_CEILING}, 0 disables the budget bump)")
    ap.add_argument("--contract-scan", type=int, default=DEFAULT_CONTRACT_SCAN,
                    help=f"max lane issues (oldest first) one tick may contract-review "
                         f"while scanning past held THIN heads for a spawnable target "
                         f"(default {DEFAULT_CONTRACT_SCAN}; 1 restores the old "
                         f"head-only behavior). The floor itself is never relaxed.")
    ap.add_argument("--contract-hold-ttl-h", type=int,
                    default=DEFAULT_CONTRACT_HOLD_TTL_H,
                    help=f"skip an issue whose contract review HELD within this many "
                         f"hours (ledger: {RUNS_DIRNAME}/{_CONTRACT_HOLD_LEDGER}), so "
                         f"the bounded scan advances across ticks instead of "
                         f"re-reviewing the same thin heads (default "
                         f"{DEFAULT_CONTRACT_HOLD_TTL_H}; 0 disables). An issue whose "
                         f"GitHub updatedAt is newer than its held verdict re-enters "
                         f"early — a contract backfill needs no TTL wait.")
    ap.add_argument("--repair-batch", type=int, default=DEFAULT_REPAIR_BATCH,
                    help=f"when EVERY scanned candidate fails the contract gate, "
                         f"spawn one contract-repair worker on up to this many held "
                         f"issues to bring them up to contract themselves via gh "
                         f"issue edit (default {DEFAULT_REPAIR_BATCH}; 0 disables "
                         f"repair dispatch — the tick then just HOLDs)")
    ap.add_argument("--repair-cooldown-min", type=int,
                    default=DEFAULT_REPAIR_COOLDOWN_MIN,
                    help=f"skip an issue a repair worker already attempted within "
                         f"this many minutes (anti-churn for un-repairables; default "
                         f"{DEFAULT_REPAIR_COOLDOWN_MIN}; 0 disables)")
    ap.add_argument("--loop-ledger", default="",
                    help="append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
    ap.add_argument("--no-loop-ledger", action="store_true",
                    help="disable loop-ledger append for this tick")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--low-yield-excludes", action="store_true",
                    help="emit the #2062 low-yield soft-exclude lane set as JSON and "
                         "exit (read-only; consumed by the Go dispatch wave/tick picker)")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    if args.low_yield_excludes:
        return _emit_low_yield_excludes(root)
    if args.issue is not None and not args.lane:
        ap.error("--issue requires --lane so the lane lease/tree matches the pinned issue")
    exclude_lanes = {s.strip() for s in args.exclude_lane.split(",") if s.strip()}
    # The opencode/glm pool is tier 2 only; engineering routes to t1 and finds
    # nothing there. Derive the work-kind from the backend unless set explicitly.
    work_kind = args.work_kind or ("gardening" if args.backend == "opencode" else "engineering")
    burst = resolve_spawn_burst(args.spawn_burst)
    payload = evaluate_burst(root,
                       spawn_burst=burst["limit"],
                       burst_stagger_s=resolve_spawn_burst_stagger(
                           args.spawn_burst_stagger_s),
                       max_workers=args.max_workers, work_kind=work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh,
                       cooldown_min=args.cooldown_min, backend=args.backend,
                       exclude_lanes=exclude_lanes,
                       worker_timeout_s=dispatch_worker.normalize_timeout(args.worker_timeout_s),
                       spawn_probe_s=max(0.0, args.spawn_probe_s),
                       realloc_ceiling=max(0, args.max_realloc_workers),
                       seat_adaptive=not args.no_seat_adaptive,
                       seat_ceiling=max(1, args.seat_ceiling),
                       seat_ramp_delta=max(0, args.seat_ramp_delta),
                       record_loop=not args.no_loop_ledger,
                       loop_ledger=(Path(args.loop_ledger).resolve()
                                    if args.loop_ledger else None),
                       issue_override=args.issue,
                       force=args.force,
                       force_reason=args.force_reason,
                       contract_scan=max(1, args.contract_scan),
                       contract_hold_ttl_h=max(0, args.contract_hold_ttl_h),
                       repair_batch=max(0, args.repair_batch),
                       repair_cooldown_min=max(0, args.repair_cooldown_min))
    # Surface HOW the burst limit was resolved (flag / env / default, and whether the
    # hard ceiling clamped it) alongside what it did, so an operator reading one tick
    # artifact can tell an armed fan-out from a defaulted one.
    if isinstance(payload.get("burst"), dict):
        payload["burst"].update({k: v for k, v in burst.items() if k != "limit"})
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return tick_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
