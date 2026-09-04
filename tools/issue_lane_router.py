#!/usr/bin/env python3
"""Route OPEN GitHub issues to the dos.toml lane whose file-tree they touch.

The DOS supervisor dispatches by LANE (from dos.toml), and a lane-worker picks
work from the plan portfolio — issues are invisible to it. So a live supervisor
run only resolves tickets that happen to ride along on plan-lane work; it cannot
TARGET the backlog. The closure auditor (`tools/issue_closure_audit.py`) proved
the cost: closure_rate sits near zero because nothing aims the fleet at tickets.

This tool is the missing aim. For each open issue it picks the lane whose `trees`
globs the issue most likely touches, with a confidence ladder
(path-confirmed > exact-scope > alias > label > keyword > none) so the supervisor can
prefer high-confidence routes and the worker can fold its lane's issues into the
dispositions sidecar it already builds. UNROUTED is a first-class, surfaced
output — an issue with no defensible lane is never force-fit (and an exclusive
lane is never auto-handed to a heuristic-driven worker).

Read-only BY DEFAULT: routing never edits, labels, or closes an issue; the dispatch
worker (and the operator) consume the map. The one exception is the operator-gated
`--apply-labels` backfill, which reconciles each issue's `class:*` work-class label
(frontdoor / infra / dev — the axis that separates the public release path and fleet
infra from product dev leaves). It DRY-RUNS by default and writes to GitHub only under
the explicit `--apply-labels-write` flag. Closing tickets / writing sidecars stays gated.

Run from the repo ROOT (dos resolves its lane taxonomy from the nearest dos.toml;
from fak/ it scaffolds a throwaway wrong-lane config).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from dispatch_worker import no_window_creationflags

SCHEMA = "fleet-issue-lane-router/1"

# Mirrored from issue_triage.py (small; triage is not an importable package).
_SCOPE_RE = re.compile(r"\b(\w+)\(([^)]+)\)")  # feat(scope), fix(scope), ...
# Bare `prefix:` start form (e.g. `abi: ...`, `RSI: ...`) — the loose convention
# used for scopeless types. Only the leading token before the first colon.
_BARE_PREFIX_RE = re.compile(r"^([A-Za-z][\w-]*):\s")

# Concrete repo paths an issue body/title may name (the strongest routing signal).
# The root alternation splits on its leading char: word-rooted roots (tools, docs,
# fak/...) keep a `\b` so a partial prefix like `mytools/` never matches; DOT-rooted
# roots (.github, .claude) cannot use `\b` — a `\b` needs a word char on one side, but
# `.github` is non-word-then-word preceded by a backtick/space (both non-word), so the
# boundary is absent and `\b\.github` silently never matched a `.github/...` path in its
# natural `\`.github/...\`` position. A `(?<![\w.])` lookbehind anchors the dotted root
# instead: it matches when preceded by a non-word/non-dot char (start, space, backtick)
# and still rejects an embedded `x.github`. Without this a workflow-only finding (e.g. a
# scheduled `.github/workflows/security-audit.yml` gate) path-confirmed NO lane.
_PATH_RE = re.compile(
    r"((?:\b(?:fak/(?:internal|cmd|experiments)|tools|scripts|docs|visuals)"
    r"|(?<![\w.])\.(?:github|claude))"
    r"/[\w./-]+)"
)

# Concrete BARE-rooted repo paths (`internal/...`, `cmd/...`, `experiments/...`) —
# the families _PATH_RE only extracts behind the `fak/` doc-link prefix. Deliberately
# NOT part of the main path rung (prose names these constantly; giving them wholesale
# routing power would re-route the backlog). They are consulted ONLY as the
# stronger-binding probe when a `.github/**` path hit looks witness-only (#2609), so
# e.g. #2464's `Paths: internal/modver/` binding outranks the workflow key space the
# body merely models. The lookbehind rejects `fak/internal/...` (already extracted
# above) and any `x/internal/...` subpath or `myinternal/` partial token.
_BINDING_PATH_RE = re.compile(
    r"((?<![\w./-])(?:internal|cmd|experiments)/[\w./-]+)"
)

# An issue body's own explicit lane declaration — the strongest witness-vs-binding
# cue a body carries. Two forms in the wild: the contract-overlay `## Lane` section
# (lane name on the next non-empty line) and the inline `Lane: `x`` field row
# (#2464's `Lane: `modver` · Paths: …`). Anchored at line start so prose like
# "route it to the right lane: whichever" never matches.
_BODY_LANE_SECTION_RE = re.compile(
    r"^#{2,}[ \t]*lane[ \t]*\r?\n\s*`?([A-Za-z][\w-]*)`?[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE)
_BODY_LANE_FIELD_RE = re.compile(
    r"^[ \t]*lane[ \t]*:[ \t]*`?([A-Za-z][\w-]*)`?",
    re.IGNORECASE | re.MULTILINE)


def body_declared_lane(body: str | None) -> str | None:
    """The lane the issue body itself DECLARES (`## Lane` section or a `Lane: x`
    field line), lowercased, or None. This is the issue author's routing key;
    route_issue consults it as a stronger-binding signal when a `.github/**`
    path hit is witness-only (#2609). The caller validates it against the live
    lane set — an unknown or exclusive declaration is ignored, never force-fit."""
    for rx in (_BODY_LANE_SECTION_RE, _BODY_LANE_FIELD_RE):
        m = rx.search(body or "")
        if m:
            return m.group(1).lower()
    return None

# Lanes that are exclusive in dos.toml — NEVER auto-route a worker onto these from
# a heuristic; that is exactly the collision the arbiter exists to prevent.
#
# This literal is only a FALLBACK. The EFFECTIVE exclusive set is derived from
# `dos doctor --json` `lanes.exclusive` at route time (`lane_taxonomy`, threaded
# into `route_issue(..., exclusive=...)`), so a lane newly marked exclusive or
# renamed in dos.toml is refused by DERIVATION, not by a hand-edited literal that
# has already drifted once (#4027: `dos` was exclusive in dos.toml but missing
# here, masked only incidentally by the `global` `**/*` tree). It is consulted
# only when `dos doctor` yields no exclusive list (e.g. run outside a workspace).
EXCLUSIVE_LANES = {"abi", "release", "global"}
EXCLUSIVE_UNBLOCK_ACTION = (
    "operator: handle this issue on the human-owned lane or split out a non-exclusive "
    "scope; do not spawn an issue worker for the blocked lane"
)

# ---------------------------------------------------------------------------
# Trust-critical pre-route (#3122)
# ---------------------------------------------------------------------------
#
# Mirrored from `internal/dispatchtick/selfmodify.go` (TrustCriticalTreePrefixes /
# TrustCriticalFileGlobs / trustCriticalTextRE). These are the witness-machinery
# trees a GUARDED worker must never SHIP an edit to — letting an autonomous loop
# rewrite its own referee is the RSI hazard #1397 protects against.
#
# The Go side already holds this at PICK time (`SelfModifyHoldForPick`), which stays
# in place as defense-in-depth. What was missing is the ROUTING-time arm: a
# trust-critical issue whose scope/label/keyword aliases to a SAFE lane (a
# `fix(dispatch):` title lands on `tools`) sits at that shippable lane's front
# forever, because its lane tree never reveals the hazard — so the picker re-fetches
# and re-refuses it every single tick, invisibly to `dispatch route --json`.
#
# NOTE the deliberate NARROWNESS, and do not widen it to the broad "self-source"
# (`cmd/**`, `internal/**`) set the #3122 body's wording suggests: the live guard
# floor permits shipping internal/gateway, internal/agent, cmd/fak, ... , so holding
# the whole Go module here would hold issues the picker dispatches happily and starve
# the dispatch surface. The router must hold exactly what the picker holds, only
# earlier. `selfmodify.go`'s `IsSelfSourceTree` is the OTHER, broader predicate (build
# isolation) — it is not this one.
TRUST_CRITICAL_TREE_PREFIXES = (
    "internal/abi/",
    "internal/kernel/",
    "internal/adjudicator/",
    "internal/policy/",
    "internal/registrations/",
    "internal/architest/",
    "internal/shipgate/",
)
TRUST_CRITICAL_FILE_GLOBS = ("dos.toml", ".dos/", "policy.json", "VERSION")

# The text arm of the same predicate: a trust-critical path/glob named anywhere in an
# issue's title or body, with an optional `./` or `fak/` doc-link prefix. The leading
# boundary (start of text, or a non-path char — a newline qualifies) keeps it from
# matching inside a longer word. `re.ASCII` pins `\w` to Go's `[0-9A-Za-z_]` so the
# Python port and `trustCriticalTextRE` agree character-for-character.
_TRUST_CRITICAL_TEXT_RE = re.compile(
    r"(?:^|[^\w./-])((?:\./|fak/)?internal/"
    r"(?:abi|kernel|adjudicator|policy|registrations|architest|shipgate)[\w*./-]*)",
    re.ASCII)

TRUST_CRITICAL_UNBLOCK_ACTION = (
    "operator: re-route to the lane that owns the named trust-critical tree, or run it "
    "UNGUARDED (the worktree-isolated escape); a guarded worker on a shippable lane can "
    "investigate this but can never ship it"
)


def _normalize_tree(glob: str) -> str:
    """Canonicalize one lane-tree glob for prefix matching — the port of
    selfmodify.go's normalizeTree. A leading `./` or `fak/` module prefix is
    stripped and backslashes are normalized, so a Windows-authored or doc-link
    glob matches the same as a POSIX repo-relative one."""
    g = glob.strip().replace("\\", "/")
    for prefix in ("./", "fak/"):
        if g.startswith(prefix):
            g = g[len(prefix):]
    return g


def is_trust_critical_tree(glob: str) -> bool:
    """Whether one lane-tree glob is rooted in the trust-critical witness machinery
    (the port of selfmodify.go's IsTrustCriticalTree). This is the ship-HOLD
    predicate, a strict subset of the broad self-source set."""
    g = _normalize_tree(glob)
    if not g:
        return False
    if any(g.startswith(p) for p in TRUST_CRITICAL_TREE_PREFIXES):
        return True
    return any(g == f or g.startswith(f) for f in TRUST_CRITICAL_FILE_GLOBS)


def lane_is_trust_critical(lane: str | None, trees: dict[str, list[str]]) -> bool:
    """Whether a LANE's own declared tree is trust-critical — i.e. the issue is
    routed CORRECTLY and the pick-time lane-tree arm (`SelfModifyHold`) already sees
    the hazard. Such a lane is deliberately left alone by the routing-time rung: the
    mis-route arm exists only for the case the lane tree HIDES."""
    if lane is None:
        return False
    return any(is_trust_critical_tree(g) for g in trees.get(lane, []))


def issue_text_targets_trust_critical(text: str) -> str | None:
    """The first trust-critical tree an issue's text (title + body) names, or None —
    the port of selfmodify.go's IssueTextTargetsTrustCritical. A reference to a
    merely-self-source tree (`internal/gateway`, `cmd/fak`) is deliberately NOT
    matched: the guard permits shipping those, so holding them would starve the
    dispatch surface."""
    m = _TRUST_CRITICAL_TEXT_RE.search(text or "")
    return m.group(1) if m else None

# Default cap for the `gh issue list` fetch. MUST stay above the real open-issue
# count: `gh issue list` returns NEWEST-first, so a fetch that hits the cap silently
# drops the OLDEST open issues, which then never reach a lane and so are never
# dispatch candidates (an unrouted issue is skipped, but an unFETCHED one is not even
# counted). Measured 2026-08-06 at 1383 open: the historical 1000 default truncated,
# hiding 383 open issues — 203 of them cleanly ROUTABLE — from every dispatch picker,
# while `counts.open` reported 897 as if that were the whole backlog. The cost of a
# larger cap is zero until the backlog reaches it (gh returns only what exists; the
# full 1383-issue fetch measured 34s against the callers' 130s timeout), so this
# matches the value `.github/workflows/issue-lane-router.yml` already pins for the
# label backfill. `compute_coverage` still flags truncation if the backlog outgrows it.
DEFAULT_ISSUE_LIMIT = 3000

# Scope token -> lane, when the scope is not itself a lane name. Conservative and
# derived from real issue scopes vs the real lane roster. Override via --config.
SCOPE_ALIAS: dict[str, str] = {
    "cuda": "compute", "gpu": "compute", "vulkan": "compute", "metal": "compute",
    "multi-gpu": "compute", "moe": "compute",
    "serve": "gateway", "anthropic": "gateway",
    "inkernel": "engine",
    "qwen35": "model", "qwen36": "model",
    "loader": "ggufload",
    "swebench": "experiments", "demo": "experiments", "simpledemo": "experiments",
    "fanbench": "bench",
    "terminal-bench": "bench",
    "testing": "ci",
    "simd": "model",
    "rehydrate": "sessionimage",
    "devex": "devindex",
    "readme": "docs", "getting-started": "docs", "fak": "docs",
    "adopt": "docs", "licensing": "docs",
    "dashboard": "metrics", "observability": "metrics",
    "dos": "tools", "control-pane": "tools", "rsi": "tools",
    "dispatch": "tools", "scrub": "tools", "ops": "tools",
    "grafana": "tools", "support-maturity": "tools", "cachevalue": "tools",
    "tooling": "tools",
    "mobile": "examples", "edge": "examples",
    "install": "cmd",
    "adjudication": "adjudicator",
}

# Label name -> lane (rung 4, weakest). Labels are coarse, so confidence is low.
LABEL_ALIAS: dict[str, str] = {
    "gpu": "compute", "compute": "compute", "performance": "compute",
    "cuda": "compute", "multi-gpu": "compute", "moe": "compute",
    "documentation": "docs",
    "model": "model", "model-arch": "model", "model-support": "model",
    "loader": "ggufload",
    "security": "policy", "trust-floor": "policy",
    "build": "ci",
    "rsi": "tools", "dispatch": "tools",
    "agentic-serving": "gateway",
}

# Last-rung lexical fallback for unscoped issue titles/bodies. This is deliberately
# small and lane-named: it catches obvious routing words that were otherwise rotting
# with no conventional scope, without dumping ambiguous work into a catch-all lane.
KEYWORD_ALIAS: dict[str, str] = {
    "promptmmu": "promptmmu",
    "cuda": "compute",
    "a100": "compute",
    "gpu": "compute",
    "multi-gpu": "compute",
    "benchmark": "bench",
    "dashboard": "metrics",
    "observability": "metrics",
    "telemetry": "metrics",
    "tooling": "tools",
    "backlog": "tools",
}

# Confidence ordering (higher wins; used for sort + override decisions).
CONFIDENCE_RANK = {
    "path-confirmed": 5,
    "exact-scope": 4,
    "alias": 3,
    "label": 2,
    "keyword": 1,
    "none": 0,
}

# Hardware-capability signals. An issue carrying one of these needs a host that
# declares the capability (FLEET_NODE_CAPS) to run; a GPU-less worker skips it and
# leaves it OPEN + visible for a GPU node (see issue_required_caps + the dispatcher's
# capability gate). Deliberately keyed on UNAMBIGUOUS accelerator signals only: a bare
# `moe`/`agentic-serving` label ROUTES to compute/gateway (it's real lane work) but is
# NOT itself hardware-gated — that code is often unit-testable on a GPU-less host, so
# gating it would falsely strand legitimate local work. Keyword literals are lowercase
# so they never trip the uppercase-bounded hardware-name scrubber (scrub_hardware_names).
GPU_CAP_LABELS = {"cuda", "gpu", "multi-gpu"}
GPU_CAP_KEYWORDS = ("h100", "a100", "dgx", "nvidia")

# The maintainer-applied "this needs sanctioned physical hardware" label. Unlike the
# accelerator signals above — which INFER the requirement from prose — this one is a
# human judgement, so it is precise by construction and needs no regex. It carries its
# own `hardware` capability rather than `gpu`, because the work it gates is not always
# accelerator work: #4750 projects desired state into systemd and #4754 wants a real
# crash/reboot/partition host, neither of which a lone GPU satisfies. A node that can
# serve it declares FLEET_NODE_CAPS=hardware; a datacenter GPU box declares
# `gpu,hardware`. The two caps are ANDed by the dispatcher's subset test, so a
# single-GPU dev node no longer claims multi-node lab campaigns it cannot finish.
#
# The measured gap this closes (#4835): the label alone was invisible here, so a P0
# needing 8-GPU hardware plus a live lab witness returned NO caps and was dispatched 13
# times on this GPU-less host AFTER the Part-B gate landed. Its own parent #4784 IS
# gated — only because its body happens to spell the accelerator bare, while #4835
# writes the digit-suffixed node label, which _has_keyword's whole-token boundary
# rejects. That near-miss is deliberately NOT cured by loosening the matcher to accept a
# numeric/plural suffix: doing so newly gates #5595 and #5594, which merely CITE the lab
# box while being ordinary locally-runnable work (#5595 names it only to describe
# #4835's parked-ness). A false skip silently starves the backlog with no operator
# signal, so the precise human label wins over the broader regex.
HARDWARE_CAP_LABEL = "gated/hardware"
HARDWARE_CAP = "hardware"

# Shift-left resource requirement labels and signals (#10965).
# Recognises static hardware/resource capability requirement axis labels and explicit
# body execution target declarations (Execution boundary / Execution Target / Requires).
REQUIRES_GPU_LABELS = {"requires:gpu", "requires:gpu:single", "requires:gpu:multi", "requires:cuda"}
REQUIRES_HARDWARE_LABELS = {"requires:hardware", "requires:lab-hw", "requires:lab"}
REQUIRES_METAL_LABELS = {"requires:metal"}
REQUIRES_QUOTA_LABELS = {"requires:quota"}
REQUIRES_NONE_LABEL = "requires:none"

_EXECUTION_BOUNDARY_RE = re.compile(
    r"^[#*\-\s]*(?:\*\*)?(?:execution\s+boundary(?:\s*/\s*resource\s+requirement)?|execution\s+target|requires):?(?:\*\*)?:?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_EXECUTION_BOUNDARY_BLOCK_RE = re.compile(
    r"^[#*\-\s]*(?:\*\*)?(?:execution\s+boundary(?:\s*/\s*resource\s+requirement)?|execution\s+target|requires):?(?:\*\*)?:?\s*\n+([^\n#]+)",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Work-CLASS axis (infra / frontdoor / dev) — orthogonal to the lane an issue
# routes to. The lane says WHAT FILE-TREE the work touches; the class says WHAT
# KIND of work it is, so an operator can select "product leaves only" (hide the
# CI/dispatch/observability plumbing) or "the public release path" (the fenced
# front-door bucket) from the same issue-views the lane router already feeds.
#
# The class is DERIVED, not hand-labeled: it falls out of the lane an issue
# already routes to (LANE_CLASS below), with a cross-cutting front-door override
# on top (FRONT_DOOR_* signals) because the release path spans docs/cmd/ci and is
# not captured by any single leaf lane. Vocabulary matches the branch-regime ADR
# (docs/branch-regime.md #1694): front-door / development / release roles.
#
#  - frontdoor: the public release path — install/README/getting-started front
#    door, release promotion + version-everything, the branch-regime cutover.
#  - infra: fleet machinery — CI/CD, dispatch/supervisor loops, observability,
#    slack cadence, build, testing infra, host maintenance.
#  - dev: product/kernel leaves — the residual default (engine, model, gateway,
#    compute, recall, …), the clean day-to-day dispatch surface.
CLASS_INFRA = "infra"
CLASS_FRONTDOOR = "frontdoor"
CLASS_DEV = "dev"
WORK_CLASSES = (CLASS_FRONTDOOR, CLASS_INFRA, CLASS_DEV)

# The GitHub label carrying each class (the `--apply-labels` backfill target). One
# label per class so the axis is selectable in GitHub's own UI/saved-views and in
# the `.github/issue-views.json` class views (front-door / dev-leaves / infra).
CLASS_LABEL = {
    CLASS_FRONTDOOR: "class:frontdoor",
    CLASS_INFRA: "class:infra",
    CLASS_DEV: "class:dev",
}
# color + one-line description for `gh label create` (branch-regime vocabulary).
CLASS_LABEL_SPEC = {
    # NB: GitHub caps label descriptions at 100 chars — keep this string <=100 or
    # `gh label create` 422s and the class:frontdoor label is silently never made.
    CLASS_FRONTDOOR: ("B60205", "Work class: public release / front-door path "
                                "(install, README, release promotion)"),
    CLASS_INFRA: ("5319E7", "Work class: fleet infrastructure "
                            "(CI/CD, dispatch loops, observability, slack, build, testing)"),
    CLASS_DEV: ("0E8A16", "Work class: product/kernel dev leaf (the default day-to-day work)"),
}
ALL_CLASS_LABELS = set(CLASS_LABEL.values())

# Lane -> class seed. Any lane NOT listed defaults to `dev` (the product residual).
# Conservative and commented: only lanes that are unambiguously fleet-plumbing are
# `infra`; only lanes that are unambiguously release-path are `frontdoor`. The
# mixed lanes (`tools`, `docs`, `cmd`) are NOT hard-classed here — they resolve via
# the issue's own scope/label/path signals in derive_class (a `tools` dispatch
# issue is infra; a `tools` kernel-helper issue is dev; a `docs` README issue is
# frontdoor; a `docs` design-note issue is dev).
LANE_CLASS: dict[str, str] = {
    # infra — fleet machinery.
    "ci": CLASS_INFRA,               # .github/** CI/CD pipelines
    "metrics": CLASS_INFRA,          # observability / telemetry surface
    "slackwire": CLASS_INFRA,        # slack control-surface transport
    "slackoutbox": CLASS_INFRA,      # slack outbound cadence
    "dispatchauto": CLASS_INFRA,     # dispatch/supervisor automation
    "loopdrive": CLASS_INFRA,        # loop-driver plumbing
    "rsiloop": CLASS_INFRA,          # RSI loop engine (self-improvement machinery)
    "tracesink": CLASS_INFRA,        # trace/telemetry sink
    "operatorbrief": CLASS_INFRA,    # operator-facing status plumbing
    # frontdoor — public release path (lane-level; cmd/docs stay signal-gated).
    "appversion": CLASS_FRONTDOOR,   # per-module derived versions (version-everything)
    "shipgate": CLASS_FRONTDOOR,     # ship/promotion gate
    "release": CLASS_FRONTDOOR,      # release lane (exclusive; operator-gated)
    # everything else -> dev via lane_class()'s default.
}

# Lanes whose class genuinely depends on the issue, not the lane alone. For these
# the front-door / infra signal sets decide; absent a signal they fall to `dev`.
MIXED_LANES = {"tools", "docs", "cmd"}

# Front-door SIGNAL set (fires independent of lane — the release path is
# cross-cutting). Any hit classes the issue `frontdoor`, the fenced bucket.
# Widest-signal on purpose: a false-positive INTO the fenced bucket (an operator
# reviews it) is safer than a release-path issue leaking into the default dev
# stream. Labels, scopes, and path/keyword surfaces are all checked.
FRONT_DOOR_LABELS = {
    "version-everything", "adoption", "popularization", "brand",
}
FRONT_DOOR_SCOPES = {
    "release", "install", "readme", "getting-started", "start-here",
    "promote", "promotion", "version", "branch-regime", "branchregime",
    "front-door", "frontdoor", "appversion",
}
# Public front-door surfaces named in an issue title/body (regex, case-insensitive).
# Anchored to real release-path files so ordinary prose ("the main agent") never
# trips it — mirrors the narrow intent of docs/branch-regime-public-front-door-audit.md.
FRONT_DOOR_PATH_RE = re.compile(
    r"(?<![\w./-])("
    r"README\.md|INSTALL\.md|GETTING-STARTED\.md|START-HERE\.md"
    r"|install\.sh|docs/branch-regime[\w./-]*"
    r"|\.github/workflows/release[\w.-]*\.yml"
    r")",
    re.IGNORECASE,
)

# Infra SIGNAL set for the MIXED lanes (mainly `tools`): a dispatch/observability/
# CI/slack cue on a mixed-lane issue makes it infra rather than dev. Keyed on the
# same scope/label vocabulary the lane router already understands.
INFRA_SCOPES = {
    "dispatch", "observability", "ops", "grafana", "ci", "ci-cd", "cicd",
    "build", "slack", "watchdog", "metrics", "telemetry", "supervisor",
    "scoreboard", "nightrun",
}
INFRA_LABELS = {
    "ci-cd", "build", "dispatch", "observability", "slack-cadence",
    "slack-watchdog", "area:slack", "testing", "coverage", "deployment",
    "track/G-foundation", "velocity", "score-signal",
}

# Issues a required HUMAN/EXTERNAL action genuinely blocks (e.g. a legal trademark
# filing) — nothing an agent can land. Carrying the `blocked-by-human` GitHub label,
# they are dropped from the DISPATCH candidate set in collect() — the one chokepoint
# BOTH dispatch lanes read (issue_dispatch.pick_lane and issue_resolve_dispatch via
# this router) — so a worker never re-spins on an issue no agent can close. The label
# keeps them VISIBLE to a human, the skipped count is surfaced (never a silent drop),
# and the closure auditor is untouched (the issue still closes when the human acts).
# Apply the label RARELY — only when truly human-blocked, never as a difficulty dodge.
BLOCKED_BY_HUMAN_LABEL = "blocked-by-human"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def run_text(cmd: list[str], cwd: Path, *, timeout: int = 60) -> dict[str, Any]:
    """Run a command, return {stdout, stderr, returncode}; UTF-8/replace decode.

    git/dos/gh emit non-ASCII prose that the Windows default cp1252 codec cannot
    decode, which otherwise crashes the subprocess reader thread.
    """
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=no_window_creationflags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 1, "_error": str(exc)}
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def read_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return obj
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# ---------------------------------------------------------------------------
# Lane taxonomy (from dos doctor) + glob matching
# ---------------------------------------------------------------------------

def lane_taxonomy(
    workspace: Path,
) -> tuple[list[str], dict[str, list[str]], set[str]]:
    """Return (concurrent_lanes, {lane: [tree globs]}, exclusive_lanes) from
    `dos doctor --json`.

    The exclusive set is the single source of truth for the exclusive-lane
    refusal (#4027): it is read from `lanes.exclusive` here rather than a
    hand-maintained literal, so a lane newly marked exclusive in dos.toml is
    refused by derivation. Falls back to :data:`EXCLUSIVE_LANES` only when the
    payload carries no exclusive list (e.g. an older `dos` or a bad workspace)."""
    payload = read_json_from_text(
        run_text(["dos", "doctor", "--workspace", str(workspace), "--json"], workspace)["stdout"]
    )
    lanes = payload.get("lanes") or {}
    concurrent = [str(x) for x in (lanes.get("concurrent") or [])]
    trees = {str(k): [str(g) for g in v] for k, v in (lanes.get("trees") or {}).items()}
    exclusive = {str(x) for x in (lanes.get("exclusive") or [])} or set(EXCLUSIVE_LANES)
    return concurrent, trees, exclusive


def _glob_to_re(glob: str) -> re.Pattern[str]:
    """Translate a dos tree glob to a regex with proper segment boundaries.

    `**` matches any depth (including zero segments); `*` matches within one
    path segment only — so `fak/internal/gateway/**` matches
    `fak/internal/gateway/x.go` and `fak/internal/gateway/sub/x.go` but NOT
    `fak/internal/gatewayx/...` (no partial-segment match).
    """
    g = glob.replace("\\", "/")
    out = []
    i = 0
    while i < len(g):
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def named_repo_paths(text: str) -> list[str]:
    """Concrete repo paths an issue title/body NAMES, deduped in first-seen order.

    The canonical extraction the path-grep routing rung uses (:data:`_PATH_RE`),
    exposed so a caller (the dispatcher's multi-lane scope guard, #2615) reads the
    issue's file families through the SAME lens the router routes by, rather than
    re-deriving a second, drifting path regex. Only rooted paths under a recognized
    family (`internal/**`, `tools/`, `scripts/`, `docs/`, `.github/`, `.claude/`,
    `cmd/`, `visuals/`) match; a bare `Makefile` / `dos.toml` / glob like `tools/*.py` does
    not, so the set is deliberately the confidently-rooted families, not prose."""
    seen: list[str] = []
    for p in _PATH_RE.findall(text or ""):
        if p not in seen:
            seen.append(p)
    return seen


def path_matches_lane(path: str, trees: dict[str, list[str]]) -> list[str]:
    """All lanes whose tree globs match `path` (normalized, repo-relative)."""
    # Strip ONLY a leading `./` prefix — NOT `str.lstrip("./")`, which is a
    # character-set strip that also eats the leading `.` of a dotted root like
    # `.github/...`, leaving `github/...` that the `.github/**` glob then never
    # matches (so a workflow-only #978 finding path-confirmed NO lane). Same trap
    # gate_signal.py's `where`-path handling already fixed.
    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    # Issue text names files in the `fak/internal/...` doc-link convention (AGENTS.md
    # writes the repo as `fak/`), but the Go module is the repository ROOT and the
    # dos.toml trees are repo-relative (`internal/...`, `cmd/...`). Strip a leading
    # `fak/` so a doc-link path matches the real-layout tree. Without this, the
    # 2026-06-22 dos.toml prefix correction (fak/internal/** -> internal/**) would
    # have made path-confirmed routing silently go dark.
    if p.startswith("fak/"):
        p = p[len("fak/"):]
    hits = []
    for lane, globs in trees.items():
        for glob in globs:
            if _glob_to_re(glob).match(p):
                hits.append(lane)
                break
    return hits


# ---------------------------------------------------------------------------
# The router (pure)
# ---------------------------------------------------------------------------

def _scope_token(title: str) -> str | None:
    """The scope from `type(scope):` if present, else a bare `prefix:` start."""
    title = title or ""
    m = _SCOPE_RE.search(title)
    if m:
        return m.group(2).strip().lower()
    bare = _BARE_PREFIX_RE.match(title)
    return bare.group(1).strip().lower() if bare else None


def _norm_scope(token: str | None) -> str:
    """A scope/lane token with word separators folded out, for punctuation-insensitive
    matching (`cache-value` -> `cachevalue`).

    Lane names in `dos.toml` are unpunctuated (`cachevalue`, `sessionjournal`), but
    issue authors write the SAME leaf hyphenated in a Conventional-Commits scope
    (`feat(cache-value): ...`). The exact-scope rung is a literal `==`, so those
    titles matched nothing and the issues went UNROUTED — invisible to the lane
    picker forever, because an unrouted issue never enters `lanes[...]` and so is
    never a dispatch candidate. Folding the separator is a pure widening of the
    exact-scope rung: no two lanes share a normalized form (asserted in the tests),
    so a hit is unambiguous."""
    return re.sub(r"[-_ ]", "", (token or "").lower())


def _norm_lane_index(lane_set: set[str]) -> dict[str, str]:
    """Normalized-lane-name -> lane. Lanes whose normalized form is shared are
    DROPPED rather than guessed, so this index only ever resolves unambiguously."""
    seen: dict[str, list[str]] = {}
    for lane in lane_set:
        seen.setdefault(_norm_scope(lane), []).append(lane)
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


def _type_token(title: str) -> str | None:
    """The leading type from `type(scope):` for nonstandard issue families."""
    m = _SCOPE_RE.search(title or "")
    return m.group(1).strip().lower() if m else None


def _label_names(issue: dict[str, Any]) -> set[str]:
    return {str(lab.get("name", "")) for lab in issue.get("labels", [])}


def _has_keyword(text: str, keyword: str) -> bool:
    """Whole-token keyword match, treating '-' and '_' as part of a token."""
    return bool(re.search(r"(?<![\w-])" + re.escape(keyword.lower()) + r"(?![\w-])",
                          text.lower()))


def issue_required_caps(issue: dict[str, Any]) -> list[str]:
    """The hardware capabilities a host must declare (FLEET_NODE_CAPS) to run this
    issue, sorted and deduplicated.

    Contributes "gpu" when the issue carries an unambiguous accelerator signal — a
    cuda/gpu/multi-gpu label or scope, a named-accelerator keyword (h100/a100/dgx/
    nvidia) in the title/body, a requires:gpu/* label, or an explicit execution
    boundary declaring single/multi GPU / CUDA — and "hardware" when it carries
    HARDWARE_CAP_LABEL or requires:hardware/* or lab hardware / DGX, "metal" for
    requires:metal, and "quota" for requires:quota.

    Explicit `requires:none` or a "Standard runner" execution target declaration
    designates unconstrained CPU execution and suppresses inferred accelerator
    keywords. Pure + deterministic."""
    labels = {ln.lower() for ln in _label_names(issue)}
    caps: set[str] = set()
    suppress_inference = False

    if REQUIRES_NONE_LABEL in labels:
        suppress_inference = True

    for lab in labels:
        if lab in REQUIRES_GPU_LABELS:
            caps.add("gpu")
        elif lab in REQUIRES_HARDWARE_LABELS:
            caps.add(HARDWARE_CAP)
        elif lab in REQUIRES_METAL_LABELS:
            caps.add("metal")
        elif lab in REQUIRES_QUOTA_LABELS:
            caps.add("quota")

    if HARDWARE_CAP_LABEL in labels:
        caps.add(HARDWARE_CAP)
    if labels & GPU_CAP_LABELS:
        caps.add("gpu")

    body = str(issue.get("body") or "")
    targets: list[str] = []
    for m in _EXECUTION_BOUNDARY_RE.finditer(body):
        t = m.group(1).strip()
        if t:
            targets.append(t)
    if not targets:
        for m in _EXECUTION_BOUNDARY_BLOCK_RE.finditer(body):
            t = m.group(1).strip()
            if t:
                targets.append(t)

    for target in targets:
        if (re.search(r"\bstandard\s+runner\b", target, re.IGNORECASE) or
                re.search(r"\brequires:none\b", target, re.IGNORECASE)):
            suppress_inference = True
            continue

        # Extract explicit requires:* tags in target
        for tr in re.findall(r"requires:[\w:-]+", target, re.IGNORECASE):
            tr_lower = tr.lower()
            if tr_lower in REQUIRES_GPU_LABELS:
                caps.add("gpu")
            elif tr_lower in REQUIRES_HARDWARE_LABELS:
                caps.add(HARDWARE_CAP)
            elif tr_lower in REQUIRES_METAL_LABELS:
                caps.add("metal")
            elif tr_lower in REQUIRES_QUOTA_LABELS:
                caps.add("quota")
            elif tr_lower == REQUIRES_NONE_LABEL:
                suppress_inference = True

        # Check prose mentions
        if re.search(r"\bsingle[- ]gpu\b", target, re.IGNORECASE):
            caps.add("gpu")
        if re.search(r"\bcuda\b", target, re.IGNORECASE):
            caps.add("gpu")
        if re.search(r"\bmulti[- ]gpu\b", target, re.IGNORECASE):
            caps.add("gpu")
        if re.search(r"\bdgx\b", target, re.IGNORECASE):
            caps.add("gpu")
            caps.add(HARDWARE_CAP)
        if re.search(r"\blab\s+hardware\b", target, re.IGNORECASE):
            caps.add(HARDWARE_CAP)

        # Check for Metal, excluding generic template options like "(CUDA / Metal)" or "bare metal"
        clean_target = re.sub(r"\(\s*cuda\s*/\s*metal\s*\)", "", target, flags=re.IGNORECASE)
        clean_target = re.sub(r"\bbare\s+metal\b", "", clean_target, flags=re.IGNORECASE)
        clean_target = re.sub(r"requires:[\w:-]+", "", clean_target, flags=re.IGNORECASE)
        if re.search(r"\bmetal\b", clean_target, re.IGNORECASE):
            caps.add("gpu")
            caps.add("metal")

        if re.search(r"\bquota\b", target, re.IGNORECASE):
            caps.add("quota")

    if not suppress_inference:
        scope = _scope_token(str(issue.get("title") or ""))
        if scope in GPU_CAP_LABELS:
            caps.add("gpu")
        text = str(issue.get("title") or "") + "\n" + str(issue.get("body") or "")
        if any(_has_keyword(text, kw) for kw in GPU_CAP_KEYWORDS):
            caps.add("gpu")

    return sorted(caps)


def is_blocked_by_human(issue: dict[str, Any], *, label: str = BLOCKED_BY_HUMAN_LABEL) -> bool:
    """True when the issue carries the human/external-blocked label — see
    BLOCKED_BY_HUMAN_LABEL. Such issues are kept out of the dispatch candidate set."""
    return label in _label_names(issue)


# An epic is a PARENT/tracking issue (an umbrella over child issues), not a single
# closeable change. A worker told to close it "with the smallest correct change"
# burns a slot for minutes and ships a partial commit at best while the epic stays
# open — measured waste (~7% of open issues, ~1 in 6 spawns landed on one). Skipping
# the parent never strands a lane: collect() just routes the next closeable issue,
# and the epic still closes normally when its children do. Detected by the `epic`
# label OR the repo's `epic(scope):` / `epic:` commit-style title convention.
_EPIC_TITLE_RE = re.compile(r"^\s*epic\b\s*[\(:]", re.IGNORECASE)


def is_epic(issue: dict[str, Any]) -> bool:
    """True when the issue is an epic parent (label `epic` or an `epic(...)`/`epic:`
    title). Kept out of the dispatch candidate set like a human-blocked issue."""
    if "epic" in _label_names(issue):
        return True
    return bool(_EPIC_TITLE_RE.match(str(issue.get("title") or "")))


def is_dispatchable(issue: dict[str, Any]) -> bool:
    """A worker can be pointed at this issue: not human-blocked, not an epic parent."""
    return not is_blocked_by_human(issue) and not is_epic(issue)


def lane_class(lane: str | None) -> str:
    """The seed class for a lane from LANE_CLASS, defaulting to `dev` (the product
    residual). A MIXED lane (`tools`/`docs`/`cmd`) also returns `dev` here — its
    real class is decided by the issue's own signals in derive_class."""
    if not lane:
        return CLASS_DEV
    return LANE_CLASS.get(lane, CLASS_DEV)


def is_front_door(issue: dict[str, Any], scope: str | None, typ: str | None) -> bool:
    """True when a cross-cutting front-door signal fires: a front-door label, a
    release-path scope/type, or a public-front-door surface named in title/body.
    Independent of the routed lane — the release path spans docs/cmd/ci."""
    if _label_names(issue) & FRONT_DOOR_LABELS:
        return True
    if scope in FRONT_DOOR_SCOPES or typ in FRONT_DOOR_SCOPES:
        return True
    text = str(issue.get("title") or "") + "\n" + str(issue.get("body") or "")
    return bool(FRONT_DOOR_PATH_RE.search(text))


def is_infra_signal(issue: dict[str, Any], scope: str | None, typ: str | None) -> bool:
    """True when a mixed-lane issue carries a fleet-plumbing cue (dispatch,
    observability, CI, slack, build). Used to class a `tools`/`docs`/`cmd` issue
    `infra` rather than letting it fall to the `dev` residual."""
    if _label_names(issue) & INFRA_LABELS:
        return True
    return scope in INFRA_SCOPES or typ in INFRA_SCOPES


def derive_class(issue: dict[str, Any], lane: str | None) -> str:
    """The work-CLASS for a routed issue: frontdoor > infra > dev.

    Front-door wins outright (the fenced release-path bucket) whenever its
    cross-cutting signal fires, regardless of lane. Otherwise the lane's seed
    class (LANE_CLASS) decides, except a MIXED lane, where a fleet-plumbing cue
    promotes the issue to `infra`; absent any signal a mixed lane falls to `dev`.
    Pure + deterministic — same issue, same class."""
    scope = _scope_token(str(issue.get("title") or ""))
    typ = _type_token(str(issue.get("title") or ""))
    if is_front_door(issue, scope, typ):
        return CLASS_FRONTDOOR
    seed = lane_class(lane)
    if seed != CLASS_DEV:
        return seed  # a hard-classed infra/frontdoor lane
    # dev seed (product leaf OR a mixed lane): a fleet-plumbing cue makes it infra.
    if is_infra_signal(issue, scope, typ):
        return CLASS_INFRA
    return CLASS_DEV


def route_issue(
    issue: dict[str, Any],
    concurrent: list[str],
    trees: dict[str, list[str]],
    *,
    scope_alias: dict[str, str] | None = None,
    label_alias: dict[str, str] | None = None,
    keyword_alias: dict[str, str] | None = None,
    exclusive: set[str] | None = None,
) -> dict[str, Any]:
    """Route one issue to a lane via the confidence ladder, then apply the
    trust-critical mis-route hold. Pure + deterministic.

    The ladder itself is :func:`_route_by_ladder`; this wrapper adds the
    routing-time pre-route rung (#3122). Callers keep the same signature and the
    same payload shape — a held issue uses the existing :func:`_blocked_route`
    fields, so no consumer sees a new required key."""
    routed = _route_by_ladder(
        issue, concurrent, trees, scope_alias=scope_alias, label_alias=label_alias,
        keyword_alias=keyword_alias, exclusive=exclusive)
    return _hold_trust_critical_misroute(routed, issue, trees)


def _hold_trust_critical_misroute(
    routed: dict[str, Any], issue: dict[str, Any], trees: dict[str, list[str]],
) -> dict[str, Any]:
    """The routing-time arm of the self-modify hold (#3122): an issue whose TEXT
    targets fak's trust-critical witness machinery, but which the ladder aimed at a
    lane that is NOT that machinery, is held here instead of being handed to a
    guarded worker that can never ship it.

    This is the ROOT-CAUSE twin of the pick-time `dispatchtick.SelfModifyHoldForPick`
    (which stays in place as defense-in-depth, per #3122's acceptance). Without it the
    mis-routed issue sits at a shippable lane's FRONT: the picker re-fetches its text
    and re-refuses it every tick, forever, while `dispatch route --json` reports it as
    routed-and-ready. Holding at routing time makes the picker a cheap fast-path and
    makes the mis-route visible to operators.

    Two deliberate non-holds keep the rung from over-firing:

    * A lane whose OWN tree is trust-critical (a correctly-routed `adjudicator` /
      `policy` / ... issue) is left routed. The lane tree already reveals the hazard,
      so the pick-time lane-tree arm holds it with the better witness, and the lane
      attribution operators rely on survives.
    * An already-held row (lane is None — exclusive-lane hold or plain unrouted) is
      returned untouched, so this rung never overwrites a stronger verdict.
    """
    lane = routed.get("lane")
    if lane is None or lane_is_trust_critical(str(lane), trees):
        return routed
    text = str(issue.get("title") or "") + "\n" + str(issue.get("body") or "")
    tree = issue_text_targets_trust_critical(text)
    if tree is None:
        return routed
    return _blocked_route(
        issue, str(lane), f"trust-critical-text:{tree} (held from {lane})",
        "trust-critical",
        reason=(
            f"lane-policy:trust-critical issue text targets '{tree}' (fak's own witness "
            f"machinery, which a guarded worker may never ship) but the routing ladder "
            f"chose the shippable lane '{lane}'; held before spawn"),
        unblock_action=TRUST_CRITICAL_UNBLOCK_ACTION)


def _route_by_ladder(
    issue: dict[str, Any],
    concurrent: list[str],
    trees: dict[str, list[str]],
    *,
    scope_alias: dict[str, str] | None = None,
    label_alias: dict[str, str] | None = None,
    keyword_alias: dict[str, str] | None = None,
    exclusive: set[str] | None = None,
) -> dict[str, Any]:
    """The confidence ladder proper (path-grep -> exact-scope -> alias -> label ->
    keyword). Pure + deterministic.

    ``exclusive`` is the exclusive-lane set to refuse auto-routing onto; callers
    pass the value derived from `dos doctor` (`lane_taxonomy`'s third element).
    When omitted it falls back to the module :data:`EXCLUSIVE_LANES` literal so
    existing fixture-only callers keep their prior behavior (#4027)."""
    scope_alias = scope_alias or SCOPE_ALIAS
    label_alias = label_alias or LABEL_ALIAS
    keyword_alias = keyword_alias or KEYWORD_ALIAS
    exclusive = EXCLUSIVE_LANES if exclusive is None else exclusive
    title = str(issue.get("title") or "")
    body = str(issue.get("body") or "")
    lane_set = set(concurrent)

    # Rung 1: path-grep probe (strongest). Run first so it can override a wrong scope.
    # Track WHICH named paths supported each lane — the witness-vs-binding demotion
    # below needs to know when a lane's only evidence is `.github/**` prose.
    path_lanes: list[str] = []
    lane_paths: dict[str, list[str]] = {}
    for p in _PATH_RE.findall(title + "\n" + body):
        for lane in path_matches_lane(p, trees):
            if lane not in lane_set:
                continue
            if lane not in path_lanes:
                path_lanes.append(lane)
            lane_paths.setdefault(lane, []).append(p)

    scope = _scope_token(title)
    typ = _type_token(title)

    # Rung 2: exact scope == lane.
    scope_lane = None
    scope_conf = None
    # Set only by rung 3b, so the emitted signal names the issue's OWN scope token
    # (`scope:cache-value->cachevalue`). Without it the token falls back to the
    # Conventional-Commits TYPE and the audit trail reads `scope:feat->cachevalue`,
    # which tells an operator nothing about why the lane was chosen.
    folded_scope = False
    if scope and scope in lane_set and scope not in exclusive:
        scope_lane, scope_conf = scope, "exact-scope"
    # Rung 3: alias scope -> lane.
    elif scope and scope in scope_alias and scope_alias[scope] in lane_set:
        scope_lane, scope_conf = scope_alias[scope], "alias"
    elif typ and typ in scope_alias and scope_alias[typ] in lane_set:
        scope_lane, scope_conf = scope_alias[typ], "alias"
    # Rung 3b: punctuation-insensitive scope == lane (`cache-value` -> `cachevalue`).
    # Deliberately placed BELOW the explicit alias rungs, never above: two shipped
    # aliases (`terminal-bench` -> bench, `support-maturity` -> tools) ALSO match a
    # same-named lane once the hyphen is folded, and the hand-written alias is the
    # operator's stated intent. Running this last means the new rung can only ever
    # rescue a scope that matched NOTHING, so it cannot re-route any issue that
    # routes today. Graded `alias` (not `exact-scope`) for the same reason — a
    # separator-folded hit is weaker evidence than a literal lane name.
    elif scope and (_folded := _norm_lane_index(lane_set).get(_norm_scope(scope))):
        scope_lane, scope_conf, folded_scope = _folded, "alias", True

    # Rung 4: label -> lane.
    label_lane = None
    for lab in sorted(_label_names(issue)):
        if lab in label_alias and label_alias[lab] in lane_set:
            label_lane = label_alias[lab]
            break

    # Rung 5: explicit keyword -> lane. Weakest automatic signal, but better than
    # leaving obviously-lane-named issues permanently unrouted.
    keyword_lane = None
    keyword = None
    searchable = title + "\n" + body
    for key in sorted(keyword_alias):
        lane = keyword_alias[key]
        if lane in lane_set and _has_keyword(searchable, key):
            keyword, keyword_lane = key, lane
            break

    blocked_path_lanes = sorted(ln for ln in path_lanes if ln in exclusive)
    if blocked_path_lanes:
        lane = blocked_path_lanes[0]
        return _blocked_route(issue, lane, f"path:{lane}", "exclusive")

    # Exclusive-lane scope is operator-gated, never auto-routed.
    if scope in exclusive:
        return _blocked_route(issue, scope, f"exclusive-scope:{scope}", "exclusive")
    if scope_lane in exclusive:
        token = scope if (scope in scope_alias or folded_scope) else typ
        return _blocked_route(issue, scope_lane, f"scope:{token}->{scope_lane}", "exclusive")
    if label_lane in exclusive:
        return _blocked_route(issue, label_lane, f"label->{label_lane}", "exclusive")
    if keyword_lane in exclusive:
        return _blocked_route(issue, keyword_lane, f"keyword:{keyword}->{keyword_lane}", "exclusive")

    # Witness-vs-binding demotion (#2609): `.github/**` is the fleet's most-cited
    # WITNESS surface — bodies name workflow files as the key space being modeled,
    # the gate that fired, the CI log that proved a bug — without the dispatchable
    # work living there. A lane whose ONLY path evidence is `.github/**` prose is
    # demoted to a mention whenever the body carries a STRONGER binding elsewhere:
    # an explicit `## Lane`/`Lane:` body declaration, an exact-scope lane token, or
    # a concrete non-.github path (including the bare-rooted `internal/...` /
    # `cmd/...` forms the main path rung deliberately skips). With no stronger
    # binding the `.github/**` path stays authoritative — a workflow-only gate
    # issue (#978) still routes ci. Runs AFTER the exclusive holds above, which
    # decide on the full original path set and never weaken.
    body_lane = body_declared_lane(body)
    if body_lane is not None and (body_lane not in lane_set or body_lane in exclusive):
        body_lane = None
    witness_demoted: list[str] = []
    demote_note = ""
    witness_only = [ln for ln in path_lanes
                    if all(p.startswith(".github/") for p in lane_paths.get(ln, []))]
    if witness_only:
        others = [ln for ln in path_lanes if ln not in witness_only]
        binding_lanes: list[str] = []
        for p in _BINDING_PATH_RE.findall(title + "\n" + body):
            for lane in path_matches_lane(p, trees):
                if (lane in lane_set and lane not in exclusive
                        and lane not in witness_only and lane not in others
                        and lane not in binding_lanes):
                    binding_lanes.append(lane)
        strong_scope = scope_lane if scope_conf == "exact-scope" else None
        stronger = bool(others or binding_lanes
                        or (body_lane and body_lane not in witness_only)
                        or (strong_scope and strong_scope not in witness_only))
        if stronger:
            witness_demoted = witness_only
            path_lanes = others + binding_lanes
            demote_note = (" (witness-only .github demoted: "
                           + "|".join(sorted(witness_only)) + ")")

    path_lane: str | None = None
    path_ambiguous = False
    if len(path_lanes) == 1:
        path_lane = path_lanes[0]
    elif len(path_lanes) > 1:
        path_ambiguous = True

    # Resolve with override: path_lane wins outright (it's the non-forgeable signal).
    if path_lane is not None:
        weaker = scope_lane or label_lane
        conflict = weaker is not None and weaker != path_lane
        signal = f"path:{path_lane}" + (f" (overrode {weaker})" if conflict else "") + demote_note
        return _route(issue, path_lane, "path-confirmed", signal, conflict)

    if path_ambiguous:
        # Tie-break: prefer the body-declared lane (only armed by a witness
        # demotion), then the lane also matching scope/label; else lexicographic.
        routable_path_lanes = [ln for ln in path_lanes
                               if ln not in exclusive]
        prefer = (body_lane if (witness_demoted and body_lane in routable_path_lanes)
                  else (scope_lane if scope_lane in routable_path_lanes
                        else (label_lane if label_lane in routable_path_lanes else None)))
        pick = prefer or sorted(routable_path_lanes)[0]
        return _route(issue, pick, "path-confirmed",
                      f"path-ambiguous:{'|'.join(sorted(path_lanes))}" + demote_note,
                      True, unrouted_reason=None)

    # A witness demotion armed by the body's own lane declaration and nothing
    # else: route to the declared lane — an explicit lane-key binding, graded at
    # exact-scope confidence (it IS an exact lane name, declared by the issue).
    if witness_demoted and body_lane is not None:
        return _route(issue, body_lane, "exact-scope",
                      f"body-lane:{body_lane}" + demote_note, False)

    if scope_lane is not None:
        token = scope if (scope in lane_set or scope in scope_alias
                          or folded_scope) else typ
        return _route(issue, scope_lane, scope_conf,
                      f"scope:{token}->{scope_lane}" + demote_note, False)
    if label_lane is not None:
        return _route(issue, label_lane, "label", f"label->{label_lane}", False)
    if keyword_lane is not None:
        return _route(issue, keyword_lane, "keyword", f"keyword:{keyword}->{keyword_lane}", False)

    reason = "no scope, no repo-path, no aliasable label" if scope else "no scope/path/label signal"
    return _route(issue, None, "none", "unrouted", False, unrouted_reason=reason)


def _route(issue, lane, confidence, signal, conflict, *, unrouted_reason=None,
           class_lane: str | None = None) -> dict[str, Any]:
    # The work-class derives from the routed lane (or, for a blocked/unrouted issue,
    # the class_lane the caller passes so the class survives even without a dispatch
    # lane). Front-door still overrides via derive_class regardless of lane.
    return {
        "number": issue.get("number"),
        "title": str(issue.get("title") or "")[:80],
        "lane": lane,
        "confidence": confidence,
        "signal": signal,
        "signal_conflict": bool(conflict),
        "unrouted_reason": unrouted_reason,
        "class": derive_class(issue, lane if lane is not None else class_lane),
        "required_caps": issue_required_caps(issue),
    }


def _blocked_route(issue, lane: str, signal: str, policy: str, *,
                   reason: str | None = None,
                   unblock_action: str | None = None) -> dict[str, Any]:
    # `reason`/`unblock_action` are optional overrides for a non-exclusive hold
    # policy (the trust-critical pre-route, #3122). Omitted, they keep the original
    # exclusive-lane wording verbatim, so the field SHAPE is unchanged for every
    # existing consumer.
    lane = str(lane)
    out = _route(
        issue, None, "none", signal, False,
        unrouted_reason=(
            reason if reason is not None else
            f"lane-policy:{policy} lane '{lane}' is human-owned/operator-gated; "
            f"held before spawn"),
        class_lane=lane)
    out["blocked_lane"] = lane
    out["blocked_policy"] = policy
    out["unblock_action"] = (
        unblock_action if unblock_action is not None else EXCLUSIVE_UNBLOCK_ACTION)
    return out


# ---------------------------------------------------------------------------
# Fetch + payload
# ---------------------------------------------------------------------------

IssueFetcher = Callable[[Path], list[dict[str, Any]]]


def fetch_issues(workspace: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    # body IS required — the path-grep rung reads it. (closure-audit omits body;
    # do not blind-copy that fetcher.)
    res = run_text(
        ["gh", "issue", "list", "--state", "open", "--limit", str(limit),
         "--json", "number,title,labels,body"],
        workspace,
    )
    # A FAILED fetch (gh rate-limit, network, auth) exits non-zero; an empty repo
    # exits 0 with a "[]" body. These must NOT collapse to the same silent []: a
    # failed fetch read as "0 open issues" makes compute_coverage() report
    # complete-and-empty, so the router (and the by-default backfill) no-ops while
    # LOOKING successful. Fail loud instead — a red CI job beats a silent miss.
    if res.get("returncode"):
        raise RuntimeError(
            "gh issue list failed (rc="
            f"{res.get('returncode')}): {(res.get('stderr') or '').strip()[:300]}"
        )
    text = (res.get("stdout") or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def load_injected_issues(source: str) -> list[dict[str, Any]]:
    """Read the open-issue set from a ``gh issue list --json`` array instead of the
    built-in gh fetch — ``source`` is a file path, or ``-`` for stdin.

    This is what lets a NAMED VIEW drive routing: pipe
    ``issue_views.py show --view <slug> --json --fields number,title,labels,body``
    into ``--issues -`` and the view IS the backlog the router folds, rather than
    every tool re-fetching the whole open set (issue-views is the default
    selection surface; this makes it the literal one for the router too).

    Field-tolerant: the router only reads number/title/labels/body, and any field a
    view omits degrades gracefully — an absent ``body`` merely weakens the
    path-grep rung (scope/label/keyword routing still fire), so pass
    ``--fields …,body`` upstream when full path-confirmation fidelity matters.
    Raises ValueError on non-array / invalid JSON.
    """
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--issues input is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("--issues input must be a JSON array of gh issue objects")
    return data


def fetch_view_issues(workspace: Path, slug: str, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Fetch the open-issue backlog for a NAMED issue-view (``.github/issue-views.json``)
    by resolving it through the sibling ``issue_views`` tool, so e.g. ``--view current``
    routes exactly the slice the issue-views selection surface defines.

    Raises on a missing/invalid config or a gh failure — the caller fail-softs to the
    full open backlog so an unattended dispatch tick never starves on a bad view.
    """
    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import issue_views as iv  # sibling tool; resolves a view slug -> gh search argv

    cfg_root = iv.repo_root(workspace)
    cfg = iv.load_config(iv.default_config_path(cfg_root))
    view = iv.resolve_view(cfg, slug)
    args = iv.build_gh_args(cfg, view, limit=limit, json_fields="number,title,labels,body")
    res = run_text(args, workspace, timeout=90)
    if res.get("returncode"):
        raise RuntimeError(res.get("stderr") or f"gh exited {res.get('returncode')}")
    text = (res.get("stdout") or "").strip()
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else []


def build_payload(
    *,
    workspace: str,
    routes: list[dict[str, Any]],
    trees: dict[str, list[str]],
    max_unrouted_frac: float = 0.25,
    fetch_error: str | None = None,
    coverage: dict[str, Any] | None = None,
    skipped_blocked: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    skipped_blocked = skipped_blocked or []
    skipped = [
        {"number": i.get("number"), "title": str(i.get("title") or "")[:80]}
        for i in skipped_blocked
    ]
    by_conf: dict[str, int] = {k: 0 for k in CONFIDENCE_RANK}
    by_class: dict[str, int] = {k: 0 for k in WORK_CLASSES}
    class_issues: dict[str, list[int]] = {k: [] for k in WORK_CLASSES}
    lane_routes: dict[str, list[dict[str, Any]]] = {}
    for r in routes:
        by_conf[r["confidence"]] = by_conf.get(r["confidence"], 0) + 1
        cls = r.get("class") or CLASS_DEV
        by_class[cls] = by_class.get(cls, 0) + 1
        class_issues.setdefault(cls, []).append(int(r.get("number") or 0))
        lane = r["lane"]
        if lane:
            lane_routes.setdefault(lane, []).append(r)

    lanes: dict[str, dict[str, Any]] = {}
    for lane, lane_rs in lane_routes.items():
        # Order a lane's issues by routing confidence (strongest first), then issue
        # number desc — the SAME routed-ordering the flat `issues` list uses via
        # _route_sort_key. A dos-dispatch worker folds lanes[<its lane>].issues into
        # its dispositions; ordering them best-routed-first steers it at the ticket it
        # can most confidently target (a path-confirmed hit) instead of whatever gh
        # returned newest. Also carry the lane's own by_confidence so an operator (and
        # a future confidence-weighted lane picker) can see how well-aimed a lane is.
        lane_rs = sorted(lane_rs, key=_lane_issue_sort_key)
        lane_by_conf: dict[str, int] = {}
        lane_by_class: dict[str, int] = {}
        for r in lane_rs:
            lane_by_conf[r["confidence"]] = lane_by_conf.get(r["confidence"], 0) + 1
            rcls = r.get("class") or CLASS_DEV
            lane_by_class[rcls] = lane_by_class.get(rcls, 0) + 1
        lanes[lane] = {
            "tree": trees.get(lane, []),
            "count": len(lane_rs),
            "issues": [r["number"] for r in lane_rs],
            "by_confidence": lane_by_conf,
            "by_class": lane_by_class,
        }

    total = len(routes)
    unrouted = by_conf["none"]
    routed = total - unrouted
    frac = round(unrouted / total, 4) if total else 0.0

    coverage = coverage or {"complete": True, "notes": []}
    incomplete = not coverage.get("complete", True)
    coverage_note = "; ".join(coverage.get("notes") or [])

    if fetch_error:
        ok, verdict, finding = False, "FETCH_ERROR", "fetch_error"
        reason = fetch_error
        next_action = "fix the gh/dos read-back error, then re-run the lane router"
    elif incomplete:
        # The gh fetch hit its cap, so some open issues were never routed. Routing
        # only a slice of the backlog is itself an ACTION — surface it loudly.
        ok, verdict, finding = False, "ACTION", "incomplete_coverage"
        reason = (
            f"routed {routed}/{total} fetched, but the open-issue fetch was truncated, "
            f"so some open issues were never routed — {coverage_note}"
        )
        next_action = (
            "re-run with a higher --issue-limit (the wired loop now defaults above the "
            "current open-issue count) so every open issue is routed"
        )
    elif total and frac > max_unrouted_frac:
        ok, verdict, finding = False, "ACTION", "high_unrouted"
        reason = f"{unrouted}/{total} open issues UNROUTED (frac={frac} > {max_unrouted_frac})"
        next_action = "operator: add scopes/labels or extend SCOPE_ALIAS so workers can target these"
    else:
        ok, verdict, finding = True, "OK", "routed"
        blocked_note = f"; {len(skipped)} human-blocked skipped" if skipped else ""
        reason = (f"{routed}/{total} open issues routed to {len(lanes)} lane(s); "
                  f"{unrouted} UNROUTED{blocked_note}")
        next_action = "dos-dispatch workers fold lanes[<their lane>].issues into the dispositions sidecar"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "coverage": coverage,
        "counts": {
            "open": total, "routed": routed, "unrouted": unrouted,
            "unrouted_frac": frac, "by_confidence": by_conf,
            "by_class": by_class,
            "skipped_human_blocked": len(skipped),
        },
        "classes": {
            cls: {
                "count": by_class.get(cls, 0),
                "issues": sorted(class_issues.get(cls, []), reverse=True),
            }
            for cls in WORK_CLASSES
        },
        "lanes": dict(sorted(lanes.items(), key=lambda kv: (-kv[1]["count"], kv[0]))),
        "unrouted_scopes": unrouted_scope_clusters(routes),
        "issues": sorted(routes, key=_route_sort_key),
        "skipped_human_blocked": sorted(skipped, key=lambda s: -int(s.get("number") or 0)),
    }


def unrouted_scope_clusters(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bucket the truly-UNROUTED issues by their scope token, most-common first.

    A flat "269 UNROUTED" is not actionable; the OPERATOR next-action ("add
    scopes/labels or extend SCOPE_ALIAS") needs to know WHICH scope families are
    rotting and how big each is. This clusters the unrouted rows (lane is None AND
    not an exclusive/human-blocked hold — those carry a `blocked_lane` and are a
    different triage) by the same `type(scope):` / bare-`prefix:` key the router
    routes by, so a recurring family (e.g. a real `internal/<scope>/` leaf that has
    no declared dos.toml lane yet, or a genuinely-ambiguous scope) surfaces as one
    named bucket instead of scattered singletons. Scopeless rows fold into a single
    `(no-scope)` bucket. Pure + deterministic: sorted by count desc, then scope asc.

    Each bucket is {scope, count, issues:[numbers desc]}. Held (blocked_lane) rows
    are excluded — they are surfaced by the existing exclusive-lane render path."""
    clusters: dict[str, list[int]] = {}
    for r in routes:
        if r.get("lane") is not None or r.get("blocked_lane"):
            continue
        scope = _scope_token(str(r.get("title") or "")) or "(no-scope)"
        clusters.setdefault(scope, []).append(int(r.get("number") or 0))
    out = [
        {"scope": scope, "count": len(nums), "issues": sorted(nums, reverse=True)}
        for scope, nums in clusters.items()
    ]
    out.sort(key=lambda c: (-c["count"], c["scope"]))
    return out


def _route_sort_key(r: dict[str, Any]) -> tuple[int, int, int]:
    # UNROUTED first (operator triage), then by confidence desc, then issue number desc.
    unrouted_first = 0 if r["lane"] is None else 1
    return (unrouted_first, -CONFIDENCE_RANK.get(r["confidence"], 0), -int(r.get("number") or 0))


def _lane_issue_sort_key(r: dict[str, Any]) -> tuple[int, int]:
    # Within a lane every issue is routed, so order by confidence desc then number
    # desc — the routed tail of _route_sort_key, applied per lane.
    return (-CONFIDENCE_RANK.get(r["confidence"], 0), -int(r.get("number") or 0))


def compute_coverage(*, issues_fetched: int, issue_limit: int) -> dict[str, Any]:
    """Flag when the open-issue fetch hit its cap (so some issues went unrouted).

    `gh issue list` returns newest-first; a fetch that returned exactly the limit
    almost certainly dropped older open issues, which then never reach a lane.
    Routing a *slice* of the backlog while reporting `routed/open` as if it were
    the whole open set is the silent-truncation failure this surfaces.
    """
    truncated = issues_fetched >= issue_limit
    notes: list[str] = []
    if truncated:
        notes.append(
            f"gh fetch returned {issues_fetched} open issue(s) = the --issue-limit cap; "
            f"older open issues may be unrouted — raise --issue-limit"
        )
    return {
        "complete": not truncated,
        "truncated": truncated,
        "issues_fetched": issues_fetched,
        "issue_limit": issue_limit,
        "notes": notes,
    }


def collect(
    workspace: Path,
    *,
    issue_limit: int = DEFAULT_ISSUE_LIMIT,
    max_unrouted_frac: float = 0.25,
    fetcher: IssueFetcher | None = None,
    scope_alias: dict[str, str] | None = None,
    label_alias: dict[str, str] | None = None,
    keyword_alias: dict[str, str] | None = None,
    injected: bool = False,
) -> dict[str, Any]:
    root = workspace.resolve()
    concurrent, trees, exclusive = lane_taxonomy(root)
    fetch = fetcher or (lambda ws: fetch_issues(ws, limit=issue_limit))
    issues = fetch(root)
    if injected:
        # The issue set was supplied explicitly (a named-view slice piped in via
        # --issues), not fetched from gh — so the silent-truncation guard does not
        # apply: the slice IS the intended backlog, complete by construction. (A
        # view deliberately narrows the open set; flagging it "truncated" and
        # advising "raise --issue-limit" would misfire.)
        coverage = {
            "complete": True, "truncated": False, "injected": True,
            "issues_fetched": len(issues), "issue_limit": issue_limit,
            "notes": ["issues injected via --issues (a named-view slice); coverage "
                      "reflects the provided slice, not a full gh fetch"],
        }
    else:
        coverage = compute_coverage(issues_fetched=len(issues), issue_limit=issue_limit)
    # Drop non-dispatchable issues (human/external-blocked AND epic parents) from the
    # candidate set — the one chokepoint both lanes read — but surface them so the skip
    # is never silent. Epics stay visible to humans and still close when their children do.
    blocked = [i for i in issues if not is_dispatchable(i)]
    routable = [i for i in issues if is_dispatchable(i)]
    fetch_error = None
    if not concurrent:
        fetch_error = "dos doctor returned no lanes — run from the repo root (not fak/)"
    elif not issues and not injected:
        fetch_error = "gh returned no open issues (auth/network?)"
    routes = [
        route_issue(i, concurrent, trees, scope_alias=scope_alias, label_alias=label_alias,
                    keyword_alias=keyword_alias, exclusive=exclusive)
        for i in routable
    ]
    return build_payload(
        workspace=str(root), routes=routes, trees=trees,
        max_unrouted_frac=max_unrouted_frac, fetch_error=fetch_error, coverage=coverage,
        skipped_blocked=blocked,
    )


# Short labels for the per-lane confidence tag in the human render.
_CONF_ABBR = {"path-confirmed": "path", "exact-scope": "scope", "alias": "alias",
              "label": "label", "keyword": "kw"}


def _lane_conf_tag(by_conf: dict[str, int] | None) -> str:
    """Compact 'how well-aimed is this lane' tag, strongest rung first, zeros omitted
    (e.g. 'path 3·scope 4·kw 2'). '' when no per-lane confidence is present — so a
    low-confidence backlog (all keyword routes) is visible to an operator at a glance
    rather than hidden behind a bare issue count."""
    if not by_conf:
        return ""
    return "·".join(
        f"{_CONF_ABBR.get(k, k)} {by_conf[k]}"
        for k in sorted(by_conf, key=lambda x: -CONFIDENCE_RANK.get(x, 0))
        if by_conf.get(k))


# Short labels for the per-lane class mix in the human render.
_CLASS_ABBR = {CLASS_FRONTDOOR: "front", CLASS_INFRA: "infra", CLASS_DEV: "dev"}


def _lane_class_tag(by_class: dict[str, int] | None) -> str:
    """Compact class-mix tag for a lane, front-door first (e.g. 'front 2·dev 5').
    '' when a lane is single-class (its class is already obvious from LANE_CLASS);
    only surfaced when a lane genuinely SPANS classes — the mixed lanes (tools/
    docs/cmd) where the split is the whole point."""
    present = {k: v for k, v in (by_class or {}).items() if v}
    if len(present) < 2:
        return ""
    return "·".join(
        f"{_CLASS_ABBR.get(k, k)} {present[k]}"
        for k in WORK_CLASSES if present.get(k))


def render(payload: dict[str, Any]) -> str:
    c = payload.get("counts") or {}
    bc = c.get("by_class") or {}
    class_line = "  ·  ".join(f"{cls}={bc.get(cls, 0)}" for cls in WORK_CLASSES)
    lines = [
        f"issue-lane router: {payload.get('verdict')} ({payload.get('finding')})",
        f"routed={c.get('routed')}/{c.get('open')} unrouted={c.get('unrouted')} "
        f"(frac={c.get('unrouted_frac')})  by_conf={c.get('by_confidence')}",
        f"by class:  {class_line}",
        f"next: {payload.get('next_action')}",
    ]
    skipped = payload.get("skipped_human_blocked") or []
    if skipped:
        nums = ", ".join(f"#{s['number']}" for s in skipped[:10])
        lines.append(f"  human-blocked (skipped, not dispatched): {len(skipped)}  {nums}")
    coverage = payload.get("coverage") or {}
    if not coverage.get("complete", True):
        for note in coverage.get("notes") or []:
            lines.append(f"  ! partial coverage: {note}")
    lines.append("  lanes with ticket backlog:")
    for lane, grp in list((payload.get("lanes") or {}).items())[:20]:
        nums = ",".join(f"#{n}" for n in grp["issues"][:10])
        tag = _lane_conf_tag(grp.get("by_confidence"))
        ctag = _lane_class_tag(grp.get("by_class"))
        suffix = "".join([f"  [{tag}]" if tag else "", f"  {{{ctag}}}" if ctag else ""])
        lines.append(f"    {lane:<14} {grp['count']:>2}  {nums}" + suffix)
    unrouted = [r for r in payload.get("issues", []) if r["lane"] is None]
    if unrouted:
        lines.append(f"  UNROUTED ({len(unrouted)}) — operator triage:")
        for r in unrouted[:10]:
            action = f" -> {r['unblock_action']}" if r.get("unblock_action") else ""
            policy = f" [{r.get('blocked_policy')}:{r.get('blocked_lane')}]" if r.get("blocked_lane") else ""
            lines.append(f"    #{r['number']:<5} {r['unrouted_reason']}{policy}: {r['title']}{action}")
    clusters = payload.get("unrouted_scopes") or []
    if clusters:
        lines.append("  UNROUTED by scope (add a lane/alias to drain a whole family):")
        for cl in clusters[:12]:
            nums = ",".join(f"#{n}" for n in cl["issues"][:6])
            lines.append(f"    {cl['scope']:<20} {cl['count']:>3}  {nums}")
    return "\n".join(lines)


def render_md(payload: dict[str, Any], *, date: str) -> str:
    c = payload.get("counts") or {}
    out = [
        f"# Issue → lane routing — {date}",
        "",
        f"- routed **{c.get('routed')}/{c.get('open')}**, UNROUTED {c.get('unrouted')} "
        f"(frac {c.get('unrouted_frac')})",
        f"- by confidence: `{c.get('by_confidence')}`",
        f"- by class: `{c.get('by_class')}`",
        f"- verdict: `{payload.get('verdict')}` — {payload.get('reason')}",
    ]
    coverage = payload.get("coverage") or {}
    if not coverage.get("complete", True):
        out.append(f"- **coverage**: ⚠ partial — {'; '.join(coverage.get('notes') or [])}")
    elif coverage:
        out.append(f"- **coverage**: complete (issues_fetched={coverage.get('issues_fetched')})")
    out += [
        "",
        "## Per-lane backlog",
        "",
        "| lane | count | issues |",
        "|---|---|---|",
    ]
    for lane, grp in (payload.get("lanes") or {}).items():
        out.append(f"| {lane} | {grp['count']} | {', '.join('#'+str(n) for n in grp['issues'])} |")
    out += ["", "## UNROUTED (operator triage)", "",
            "| # | reason | blocked lane | unblock action | title |",
            "|---|---|---|---|---|"]
    for r in payload.get("issues", []):
        if r["lane"] is None:
            blocked = ""
            if r.get("blocked_lane"):
                blocked = f"{r.get('blocked_policy')}:{r.get('blocked_lane')}"
            out.append(f"| #{r['number']} | {r['unrouted_reason']} | "
                       f"{blocked or '—'} | {r.get('unblock_action') or '—'} | "
                       f"{r['title']} |")
    clusters = payload.get("unrouted_scopes") or []
    if clusters:
        out += ["", "## UNROUTED by scope (drain a whole family with one lane/alias)", "",
                "| scope | count | issues |", "|---|---|---|"]
        for cl in clusters:
            nums = ", ".join("#" + str(n) for n in cl["issues"])
            out.append(f"| `{cl['scope']}` | {cl['count']} | {nums} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Class-label backfill (the ONE write path — gated behind --apply-labels)
# ---------------------------------------------------------------------------

def plan_class_label_changes(
    routes: list[dict[str, Any]],
    current_labels: dict[int, set[str]],
) -> list[dict[str, Any]]:
    """Pure: compute the class-label add/remove diff per routed issue.

    `routes` are the router's per-issue result rows (each with `number` + `class`);
    `current_labels` maps issue number -> its current label-name set. For each issue
    whose class label is missing or whose sibling class labels are stale, emit
    {number, add: [...], remove: [...]}. Issues already correct are omitted (so the
    backfill is idempotent). Deterministic — sorted by issue number."""
    changes: list[dict[str, Any]] = []
    for r in routes:
        num = int(r.get("number") or 0)
        if not num:
            continue
        want = CLASS_LABEL.get(r.get("class") or CLASS_DEV)
        have = current_labels.get(num, set())
        have_class = have & ALL_CLASS_LABELS
        add = [] if (want in have_class) else [want]
        remove = sorted(have_class - {want})
        if add or remove:
            changes.append({"number": num, "add": add, "remove": remove})
    return sorted(changes, key=lambda change: change["number"])


def bound_class_label_changes(
    changes: list[dict[str, Any]], limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select a deterministic bounded prefix and report the unapplied remainder."""
    selected = changes[:limit] if limit else changes
    return selected, len(changes) - len(selected)


def ensure_class_labels(workspace: Path, *, apply: bool) -> list[str]:
    """Create the three class:* labels if absent (idempotent `gh label create
    --force`). Returns the labels it (re)created. No-op preview when apply=False."""
    created: list[str] = []
    for cls in WORK_CLASSES:
        name = CLASS_LABEL[cls]
        color, desc = CLASS_LABEL_SPEC[cls]
        created.append(name)
        if apply:
            # --force upserts: create-or-update color/description, never errors on
            # an existing label. Matches the repo's ad-hoc `gh label create` pattern.
            res = run_text(["gh", "label", "create", name, "--color", color,
                            "--description", desc, "--force"], workspace)
            # A failed create (e.g. a >100-char description 422) must NOT be silent:
            # the per-issue `--add-label` writes would then fail en masse with a
            # confusing "'<label>' not found". Surface it so the run is diagnosable.
            if res.get("returncode"):
                print(f"WARNING: could not ensure label {name!r}: "
                      f"{(res.get('stderr') or '').strip()}", file=sys.stderr)
    return created


def apply_class_label_changes(
    workspace: Path, changes: list[dict[str, Any]], *, apply: bool,
) -> dict[str, Any]:
    """Apply (or preview) the class-label diff via `gh issue edit`. When apply is
    False this only reports what WOULD change — the tool's read-only-by-default
    contract. Returns a summary {applied, changed, errors}."""
    errors: list[str] = []
    if apply:
        for ch in changes:
            args = ["gh", "issue", "edit", str(ch["number"])]
            for lab in ch["add"]:
                args += ["--add-label", lab]
            for lab in ch["remove"]:
                args += ["--remove-label", lab]
            res = run_text(args, workspace, timeout=60)
            if res.get("returncode"):
                errors.append(f"#{ch['number']}: {res.get('stderr') or 'gh edit failed'}")
    return {"applied": apply, "changed": len(changes), "errors": errors}


def render_label_plan(changes: list[dict[str, Any]], *, applied: bool,
                      errors: list[str] | None = None) -> str:
    """Human render of the class-label backfill diff (dry-run or applied)."""
    head = "APPLIED" if applied else "DRY-RUN (no writes — pass --apply-labels-write to commit)"
    lines = [f"class-label backfill: {head} — {len(changes)} issue(s) need a change"]
    for ch in changes[:40]:
        parts = []
        if ch["add"]:
            parts.append("+" + ",".join(ch["add"]))
        if ch["remove"]:
            parts.append("-" + ",".join(ch["remove"]))
        lines.append(f"    #{ch['number']:<6} {' '.join(parts)}")
    if len(changes) > 40:
        lines.append(f"    … and {len(changes) - 40} more")
    for e in errors or []:
        lines.append(f"  ! {e}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Route open GitHub issues to dos.toml lanes (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--issue-limit", type=int, default=DEFAULT_ISSUE_LIMIT,
                    help="max open issues to fetch")
    ap.add_argument("--max-unrouted-frac", type=float, default=0.25,
                    help="ACTION verdict when UNROUTED fraction exceeds this")
    ap.add_argument("--config", default="", help="JSON file overriding scope_alias / label_alias")
    ap.add_argument("--md", default="", help="write a dated markdown report to this path")
    ap.add_argument("--issues", default="", metavar="PATH|-",
                    help="route a gh-issue-list JSON array read from a file or '-' "
                         "(stdin) instead of fetching via gh — lets a named view drive "
                         "routing, e.g. `python tools/issue_views.py show --view "
                         "ready-leaves --json --fields number,title,labels,body | "
                         "python tools/issue_lane_router.py --issues - --json`")
    ap.add_argument("--view", default="", metavar="SLUG",
                    help="scope candidates to a named issue-view from "
                         ".github/issue-views.json (e.g. 'current', 'm2-kv-cache') "
                         "instead of the full open backlog. FAIL-SOFT: an empty or "
                         "unresolved view falls through to the full backlog so an "
                         "unattended tick never starves. Ignored when --issues is set.")
    ap.add_argument("--apply-labels", action="store_true",
                    help="reconcile each issue's class:* GitHub label with its derived "
                         "work-class (frontdoor/infra/dev). DRY-RUN by default — prints "
                         "the label add/remove diff WITHOUT writing. Add "
                         "--apply-labels-write to actually commit the writes.")
    ap.add_argument("--apply-labels-write", action="store_true",
                    help="with --apply-labels, actually WRITE the class:* labels to "
                         "GitHub (create the labels + `gh issue edit`). The only "
                         "outward-facing action this tool takes; operator-gated.")
    ap.add_argument("--label-change-limit", type=int, default=0,
                    help="apply at most N deterministic class-label changes (0 = unlimited)")
    args = ap.parse_args(argv)
    if args.apply_labels_write and not args.apply_labels:
        ap.error("--apply-labels-write requires --apply-labels")
    if args.label_change_limit < 0:
        ap.error("--label-change-limit must be non-negative")

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    scope_alias = label_alias = None
    keyword_alias = None
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        scope_alias = cfg.get("scope_alias")
        label_alias = cfg.get("label_alias")
        keyword_alias = cfg.get("keyword_alias")

    fetcher: IssueFetcher | None = None
    injected = False
    if args.issues:
        try:
            _injected_rows = load_injected_issues(args.issues)
        except (ValueError, OSError) as exc:
            print(f"ERROR: --issues: {exc}", file=sys.stderr)
            return 2
        fetcher = lambda ws: _injected_rows  # noqa: E731
        injected = True
    elif args.view:
        try:
            _view_rows = fetch_view_issues(workspace, args.view, limit=args.issue_limit)
        except Exception as exc:  # noqa: BLE001 — fail-soft: the tick must never starve
            print(f"WARN: --view {args.view!r}: {exc}; using full open backlog",
                  file=sys.stderr)
            _view_rows = None
        if _view_rows and any(is_dispatchable(r) for r in _view_rows):
            fetcher = lambda ws: _view_rows  # noqa: E731
            injected = True
        elif _view_rows is not None:
            print(f"WARN: --view {args.view!r}: no dispatchable issues; "
                  "using full open backlog", file=sys.stderr)

    payload = collect(
        workspace, issue_limit=args.issue_limit, max_unrouted_frac=args.max_unrouted_frac,
        fetcher=fetcher, injected=injected,
        scope_alias=scope_alias, label_alias=label_alias,
        keyword_alias=keyword_alias,
    )

    if args.apply_labels:
        # The ONE write path. Build the current-label map from the same issue set the
        # router folded (fetcher when a view/--issues narrowed it, else a fresh fetch),
        # plan the idempotent add/remove diff, and apply ONLY under --apply-labels-write.
        do_write = args.apply_labels_write
        src_issues = (fetcher(workspace) if fetcher
                      else fetch_issues(workspace, limit=args.issue_limit))
        current_labels = {
            int(i.get("number") or 0): _label_names(i) for i in src_issues
        }
        routed_rows = [r for r in payload.get("issues", []) if r.get("number")]
        all_changes = plan_class_label_changes(routed_rows, current_labels)
        limit = args.label_change_limit
        changes, remaining = bound_class_label_changes(all_changes, limit)
        ensure_class_labels(workspace, apply=do_write)
        result = apply_class_label_changes(workspace, changes, apply=do_write)
        result["total_changes"] = len(all_changes)
        result["remaining"] = remaining
        result["change_limit"] = limit
        if args.json:
            print(json.dumps({"label_backfill": result, "changes": changes}, indent=2))
        else:
            print(render_label_plan(changes, applied=do_write, errors=result["errors"]))
        return 1 if result["errors"] else 0

    if args.md:
        date = run_text(["git", "log", "-1", "--format=%cs"], workspace)["stdout"].strip() or "unknown"
        Path(args.md).write_text(render_md(payload, date=date), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
