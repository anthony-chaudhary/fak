#!/usr/bin/env python3
"""PYTHON PARITY MIRROR of the Go per-issue launch profile.

This is the Python side of ``internal/dispatchtick/launchprofile.go`` (+ the tier
label grammar in ``internal/dispatchtick/tiertag.go`` and the tier ordering in
``internal/modelroute/tierpolicy.go``). It turns an issue's trusted tier labels into
the ``(model, reasoning-effort, ultracode)`` triple a headless worker is launched
with, so the Python dispatcher (``issue_resolve_dispatch.build_worker_command``)
selects the SAME tier profile the Go native dispatch tick does — a routine issue on
the cheap model, an ultra-hard one under exhaustive multi-agent orchestration.

WHY A MIRROR AND NOT AN IMPORT. Go and Python are two independent launch surfaces
for the same fleet. The Go leaf is the source of truth; this file MIRRORS its
default table and label grammar BY VALUE (the same by-value mirror idiom
``dispatchtick.tiertag`` uses for ``issuecontract``'s flag vocabulary, and that
``accounts_launch.go`` uses for ``UltracodeSettingsArg``). A golden test
(``tier_launch_test.py``) pins every mirrored value against the Go defaults so the
two can never silently drift.

CONSERVATIVE DEGRADE is load-bearing and matches Go exactly: an untagged or
ambiguous issue reports has_profile=False so the caller keeps today's seat default,
because forcing ultracode or xhigh onto unlabeled work would silently change real
token spend. An uplift only ever fires on an explicit, self-consistent tier signal.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# WORK-TIER ORDERING — mirror of internal/modelroute/tierpolicy.go.
# T0 is the MOST demanding (ultra-hard / high-risk); T2 is routine. Numerically
# T0 < T1 < T2, so "at least as demanding" is a <= comparison (meets_requirement).
# ---------------------------------------------------------------------------
TIER_T0 = 0  # ultra-hard / high-risk — the MOST demanding
TIER_T1 = 1  # normal implementation
TIER_T2 = 2  # routine / bounded / low-trust

# workTierTokenRE mirror: the leading 't' is MANDATORY so a stray "P1" (a priority,
# not a tier) never parses as a tier.
_WORK_TIER_TOKEN_RE = re.compile(r"t([0-9]+)")


def parse_work_tier(token: str) -> int | None:
    """Mirror of modelroute.ParseWorkTier: 't0'/'t1'/'t2' -> tier int; None for an
    absent or out-of-range token (e.g. 'T3', 'P1', ''). The caller decides whether
    None is a missing or an invalid tier."""
    m = _WORK_TIER_TOKEN_RE.search(token.lower())
    if m is None:
        return None
    return {"0": TIER_T0, "1": TIER_T1, "2": TIER_T2}.get(m.group(1))


def meets_requirement(capability: int, required: int) -> bool:
    """Mirror of modelroute.WorkTier.MeetsRequirement: capability must be at least
    as DEMANDING as the floor. Because T0<T1<T2 numerically but T0 is the most
    demanding, "at least as demanding" is ``capability <= required``."""
    return capability <= required


# ---------------------------------------------------------------------------
# WORKER MODEL IDS + KNOBS — mirror of launchprofile.go's const block.
# The REAL versioned ids the worker CLI accepts, never the bare "fable" alias
# (a headless worker 400-crashes on the alias — a fleet crash-loop we have paid
# for before).
# ---------------------------------------------------------------------------
WORKER_MODEL_OPUS = "claude-opus-4-8"
WORKER_MODEL_FABLE = "claude-fable-5"

# EffortXHigh: the strongest reasoning-effort knob short of ultracode, emitted via
# the worker CLI's --effort flag (there is no settings.json field).
EFFORT_XHIGH = "xhigh"

# UltracodeSettingsArg mirror: the Claude --settings payload that puts a worker in
# ultracode (xhigh per-message reasoning PLUS dynamic multi-agent workflow
# orchestration). Session-only, handed per-launch. Kept equal to the Go
# dispatchtick.UltracodeSettingsArg by the golden test rather than an import.
ULTRACODE_SETTINGS_ARG = '{"ultracode":true}'

# UltraLabel mirror: promotes an issue to the ultra-hard bucket (fable + ultracode).
# Self-sufficient — an issue carrying it is uplifted even without co-tagged tier
# labels, so explicit operator intent always wins.
ULTRA_LABEL = "tier/ultra"


@dataclass(frozen=True)
class LaunchProfile:
    """The resolved launch configuration for one worker. Effort and ultracode are
    mutually exclusive on emit (ultracode already implies xhigh), so a canonical
    profile sets exactly one of them. Mirror of dispatchtick.LaunchProfile."""

    model: str
    effort: str = ""
    ultracode: bool = False


# The four canonical profiles: {opus, fable} x {xhigh, ultracode}.
PROFILE_OPUS_XHIGH = LaunchProfile(model=WORKER_MODEL_OPUS, effort=EFFORT_XHIGH)
PROFILE_OPUS_ULTRACODE = LaunchProfile(model=WORKER_MODEL_OPUS, ultracode=True)
PROFILE_FABLE_XHIGH = LaunchProfile(model=WORKER_MODEL_FABLE, effort=EFFORT_XHIGH)
PROFILE_FABLE_ULTRACODE = LaunchProfile(model=WORKER_MODEL_FABLE, ultracode=True)

# The four tier buckets a profile maps from (mirror of dispatchtick.LaunchBucket).
BUCKET_ROUTINE = "routine"  # T2 — routine / bounded / low-trust
BUCKET_NORMAL = "normal"    # T1 — normal implementation
BUCKET_HARD = "hard"        # T0 — ultra-hard / high-risk
BUCKET_ULTRA = "ultra"      # T0 + UltraLabel — the very hardest


def default_tier_launch_table() -> dict[str, LaunchProfile]:
    """Mirror of dispatchtick.DefaultTierLaunchTable: routine runs on the cheap
    model at strong reasoning; normal/hard escalate the MODEL to opus (hard adds
    ultracode); ultra-hard trades the model back down to fable but turns on
    ultracode's exhaustive multi-agent orchestration."""
    return {
        BUCKET_ROUTINE: PROFILE_FABLE_XHIGH,
        BUCKET_NORMAL: PROFILE_OPUS_XHIGH,
        BUCKET_HARD: PROFILE_OPUS_ULTRACODE,
        BUCKET_ULTRA: PROFILE_FABLE_ULTRACODE,
    }


# ---------------------------------------------------------------------------
# ISSUE TIER TAGGING — mirror of internal/dispatchtick/tiertag.go.
# Parse the namespaced tier/T<N>-required|optimal grammar into a typed tier, with
# a closed-vocabulary flag list naming exactly why an issue stays conservative.
# ---------------------------------------------------------------------------

# Closed-vocabulary tag flags — mirror issuecontract's model_tier_<role>_* names.
TAG_FLAG_REQUIRED_MISSING = "model_tier_required_missing"
TAG_FLAG_OPTIMAL_MISSING = "model_tier_optimal_missing"
TAG_FLAG_REQUIRED_INVALID = "model_tier_required_invalid"
TAG_FLAG_OPTIMAL_INVALID = "model_tier_optimal_invalid"
TAG_FLAG_REQUIRED_CONFLICT = "model_tier_required_conflict"
TAG_FLAG_OPTIMAL_CONFLICT = "model_tier_optimal_conflict"
TAG_FLAG_CONTRADICTION = "model_tier_contradiction"

# tierLabelRE mirror: matches tier/T1-required or tier/t0-optimal after lower-casing.
# A priority label like priority/P1 does NOT match (the "Priority/P1 is not model
# tier T1" disambiguation).
_TIER_LABEL_RE = re.compile(r"^tier/(t[0-9]+)-(required|optimal)$")


@dataclass(frozen=True)
class IssueTier:
    """Per-issue tier metadata. has_tier distinguishes a tagged issue from an
    untagged one: a missing/ambiguous tier stays conservative (frontier-required)
    rather than silently inferring routine. Mirror of dispatchtick.IssueTier."""

    required: int = TIER_T0
    optimal: int = TIER_T0
    has_tier: bool = False


def _resolve_role_tier(
    labels: list[str], role: str, missing_flag: str, invalid_flag: str, conflict_flag: str
) -> tuple[int | None, bool, list[str]]:
    """Resolve the tier for one role (required|optimal): 0 matching labels ->
    missing; >=2 DISTINCT tier tokens -> conflict; exactly one -> parsed (a repeated
    identical tier is deduped, not a conflict). Mirror of tiertag.resolveRoleTier."""
    tokens: list[str] = []
    for label in labels:
        m = _TIER_LABEL_RE.match(label.strip().lower())
        if m is None or m.group(2) != role:
            continue
        tok = m.group(1)
        if tok not in tokens:  # dedupe: two labels naming the SAME tier are not a conflict
            tokens.append(tok)
    if len(tokens) == 0:
        return None, False, [missing_flag]
    if len(tokens) == 1:
        t = parse_work_tier(tokens[0])
        if t is None:
            return None, False, [invalid_flag]
        return t, True, []
    return None, False, [conflict_flag]


def issue_tier_from_labels(labels: list[str]) -> tuple[IssueTier, list[str]]:
    """Build an IssueTier from an issue's GitHub labels. Returns the typed tier plus
    a closed-vocabulary flag list: an empty flag list with has_tier=True means the
    tags are present, valid, and self-consistent; any flag means the issue stays
    conservative (has_tier=False). Mirror of tiertag.IssueTierFromLabels."""
    req, req_ok, req_flags = _resolve_role_tier(
        labels, "required", TAG_FLAG_REQUIRED_MISSING, TAG_FLAG_REQUIRED_INVALID, TAG_FLAG_REQUIRED_CONFLICT
    )
    opt, opt_ok, opt_flags = _resolve_role_tier(
        labels, "optimal", TAG_FLAG_OPTIMAL_MISSING, TAG_FLAG_OPTIMAL_INVALID, TAG_FLAG_OPTIMAL_CONFLICT
    )
    flags = list(req_flags) + list(opt_flags)

    # Either role unresolved -> conservative, carrying the flags that name why.
    if not req_ok or not opt_ok:
        return IssueTier(), flags

    # Both parsed. Optimal must be at least as demanding as the required floor. A
    # weaker optimal contradicts the floor and is not trusted.
    if not meets_requirement(opt, req):  # type: ignore[arg-type]
        return IssueTier(), flags + [TAG_FLAG_CONTRADICTION]

    return IssueTier(required=req, optimal=opt, has_tier=True), []  # type: ignore[arg-type]


def _has_ultra_label(labels: list[str]) -> bool:
    """Whether the issue carries the explicit ultra promotion label (case-insensitive,
    trimmed). Mirror of launchprofile.hasUltraLabel."""
    return any(label.strip().lower() == ULTRA_LABEL for label in labels)


def launch_bucket_for_issue(labels: list[str]) -> tuple[str | None, bool]:
    """Pick the bucket from an issue's labels. The ultra label is a self-sufficient,
    explicit signal and is checked FIRST (the tier vocabulary has no level beyond T0).
    Otherwise the bucket follows the parsed optimal tier. Returns ok=False for an
    untagged or ambiguous issue. Mirror of launchprofile.LaunchBucketForIssue."""
    if _has_ultra_label(labels):
        return BUCKET_ULTRA, True
    it, _ = issue_tier_from_labels(labels)
    if not it.has_tier:
        return None, False
    if it.optimal == TIER_T2:
        return BUCKET_ROUTINE, True
    if it.optimal == TIER_T1:
        return BUCKET_NORMAL, True
    return BUCKET_HARD, True  # TierT0 — optimal is validated >= required by issue_tier_from_labels


def launch_profile_for_issue(
    labels: list[str], table: dict[str, LaunchProfile] | None = None
) -> tuple[LaunchProfile | None, str | None, bool]:
    """Resolve the launch profile for an issue's labels against a tier->profile table
    (None => default_tier_launch_table). Returns has_profile=False for an untagged
    issue, or a bucket the supplied table leaves undefined (filled from the built-in
    default), so the caller falls back to today's seat-default launch. Mirror of
    launchprofile.LaunchProfileForIssue."""
    bucket, ok = launch_bucket_for_issue(labels)
    if not ok:
        return None, None, False
    if table is None:
        table = default_tier_launch_table()
    profile = table.get(bucket)  # type: ignore[arg-type]
    if profile is None:
        # Fill any gap (partial override table) from the built-in default so a
        # defined bucket always resolves to a profile.
        profile = default_tier_launch_table().get(bucket)  # type: ignore[arg-type]
        if profile is None:
            return None, bucket, False
    return profile, bucket, True


# ---------------------------------------------------------------------------
# THE OPT-IN GATE — mirror of dispatch_tick.go's dispatchTierLaunchEnabled /
# dispatchTierLaunchProfile. Default OFF: an unconfigured fleet keeps every worker
# on the seat-default model with no effort/ultracode uplift, byte-identical to
# before this seam. The uplift is Claude-only (opencode/codex pin their own seat
# model with -m and ignore both knobs).
# ---------------------------------------------------------------------------

# The env knob name, shared by NAME with the Go path so one operator toggle drives
# both launch surfaces.
TIER_LAUNCH_ENV = "FLEET_TIER_LAUNCH"

# The falsy grammar (mirror of dispatchTierLaunchEnabled): unset or any of these
# values is OFF. Anything else is ON.
_TIER_LAUNCH_OFF_VALUES = frozenset({"", "0", "off", "false", "no", "disable", "disabled"})


def tier_launch_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether the opt-in per-issue tier launch profile (FLEET_TIER_LAUNCH) is on.
    Default (unset / an off-ish value) is OFF. Mirror of dispatchTierLaunchEnabled."""
    src = os.environ if env is None else env
    if TIER_LAUNCH_ENV not in src:
        return False
    return src[TIER_LAUNCH_ENV].strip().lower() not in _TIER_LAUNCH_OFF_VALUES


def tier_launch_profile(
    backend: str, labels: list[str], env: dict[str, str] | None = None
) -> tuple[LaunchProfile | None, str | None]:
    """Resolve the opt-in per-issue launch profile for a target issue's labels, or
    (None, None) to leave the seat-default posture. Returns (None, None) when the
    FLEET_TIER_LAUNCH knob is off, the backend is not claude, or the issue carries
    no trusted tier (untagged / ambiguous). Mirror of dispatchTierLaunchProfile."""
    if backend != "claude" or not tier_launch_enabled(env):
        return None, None
    profile, bucket, ok = launch_profile_for_issue(labels)
    if not ok:
        return None, None
    return profile, bucket
