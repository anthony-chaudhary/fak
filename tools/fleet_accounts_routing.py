#!/usr/bin/env python3
"""Tier routing, wave slot allocation, seat-pool accounting, and CLI helpers for fleet_accounts.

Separated from ``tools/fleet_accounts.py`` so task classification, single-account
routing/resolution, multi-lane wave allocation, live `.account` seat-pool views,
and CLI formatting/argument parsing live in one cohesive module under the file-size gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable

from fleet_accounts_cap import (
    _as_int,
    _session_cap,
    account_tag,
    is_duplicate_identity,
    read_oauth_token,
    routable_worker,
)

HARD_TASK_HINT_RE = re.compile(
    r"\b("
    r"implement|fix|debug|refactor|review|test|ship|complete|build|edit|write|"
    r"modify|patch|investigate|research|search|browse|audit|design|architect|"
    r"merge|rebase|deploy|security|production|goal|plan|best\s+effort"
    r")\b",
    re.I,
)

LIGHT_TASK_PATTERNS = (
    re.compile(r"^(hi|hello|hey|ping|pong|thanks|thank you)[.!?\s]*$", re.I),
    re.compile(r"^say\s+[\w .,'\"-]{1,40}$", re.I),
    re.compile(r"^reply\s+with\s+(exactly\s+)?[\w .,'\"-]{1,50}$", re.I),
    re.compile(r"^(what('| i)?s\s+)?(the\s+)?(time|date)(\s+now|\s+today)?[?]?$", re.I),
    re.compile(r"^(pwd|whoami|date)$", re.I),
)

# Structured WORK-KIND vocabulary -- the channel a *caller that already knows the
# kind of work* uses to pin a tier, instead of hoping the prompt text trips the
# light/hard regexes above. A gardening/maintenance loop (curate-cluster,
# issue-triage, memory-compact, a doc/index sweep) is genuinely tier-2-appropriate
# even though its prompt would read as "hard" to HARD_TASK_HINT_RE ("audit",
# "review", "plan" all match). Engineering work is tier-1. These tokens are
# operator-stated facts (confidence 1.0), so they take precedence over the
# free-text heuristic -- but ONLY when explicitly supplied: an unstated kind stays
# "auto" and the regex heuristic (max-quality default) decides, so nothing is ever
# silently demoted.
GARDENING_WORK_KINDS = frozenset(
    {"gardening", "garden", "maintenance", "maint", "cleanup", "chore", "triage"})
ENGINEERING_WORK_KINDS = frozenset(
    {"engineering", "eng", "dev", "feature", "implementation"})

FAKLOCAL_TAG = "faklocal"

# Each live issue-resolution worker stamps the switcher account it leased into a
# `.account` sidecar next to its `.pid` (issue_resolve_dispatch.write_account_sidecar).
# That sidecar IS the seat lease: a live worker whose `.account` names a seat holds it.
# Kept as a literal (a stable on-disk contract) so this module need not import the
# heavier issue_resolve_dispatch to read its own seat bindings.
ACCOUNT_SIDECAR_SUFFIX = ".account"
SEAT_POOL_SCHEMA = "fleet-seat-pool/1"

# Operator-facing seat inventory (#1799): a coarser, dispatcher-vocabulary state on top
# of ``state`` (leased/free/blocked) that names WHY an unavailable seat is held, reusing
# the SAME closed vocabulary ``runtime_status`` already produces (``block_kind`` in
# auth/access/credit/usage, ``throttled``, ``reset``) instead of inventing new terms.
# "cooling" is the one state ``state`` does not distinguish: a usage-throttled seat that
# will free itself at a known ``reset`` time is a different operator action (wait) than a
# hard block (re-auth/top-up) or a depleted pool (no seat exists to free).
_HOLD_KIND_REASON = {
    "auth": "auth_failed",
    "access": "access_disabled",
    "credit": "credit_exhausted",
}


def classify_task(task_text: str = "", task_class: str = "auto",
                  policy: dict | None = None,
                  *, policy_fn: Callable[[], dict] | None = None) -> dict:
    """Classify a request for v1 model routing.

    Only trivial, short prompts are considered light. Everything ambiguous or
    dev-shaped is hard so max-quality remains the default.
    """
    pol = policy or (policy_fn() if policy_fn is not None else {})
    threshold = float(pol.get("routing", {}).get("light_confidence", 0.999))
    requested = (task_class or "auto").lower()
    if requested in ("light", "easy", "tier2", "t2", "2"):
        return {
            "class": "light",
            "confidence": 1.0,
            "reason": "operator requested light/tier2",
            "target_tier": 2,
            "light_threshold": threshold,
        }
    if requested in ("hard", "default", "tier1", "t1", "1"):
        return {
            "class": "hard",
            "confidence": 1.0,
            "reason": "operator requested hard/tier1",
            "target_tier": 1,
            "light_threshold": threshold,
        }
    if requested in ("tier3", "t3", "3"):
        return {
            "class": "tier3",
            "confidence": 1.0,
            "reason": "operator requested tier3",
            "target_tier": 3,
            "light_threshold": threshold,
        }
    # Structured work-kind: a caller that KNOWS this is gardening/maintenance (or
    # engineering) states it; that fact wins over the free-text heuristic. The
    # `class` is the kind itself so the routing decision is legible downstream.
    if requested in GARDENING_WORK_KINDS:
        return {
            "class": "gardening",
            "confidence": 1.0,
            "reason": f"operator stated work_kind={requested} (maintenance -> tier2)",
            "target_tier": 2,
            "light_threshold": threshold,
        }
    if requested in ENGINEERING_WORK_KINDS:
        return {
            "class": "engineering",
            "confidence": 1.0,
            "reason": f"operator stated work_kind={requested} (engineering -> tier1)",
            "target_tier": 1,
            "light_threshold": threshold,
        }

    text = re.sub(r"\s+", " ", (task_text or "").strip())
    if not text:
        return {
            "class": "hard",
            "confidence": 0.5,
            "reason": "no task text; defaulting to max-quality tier",
            "target_tier": 1,
            "light_threshold": threshold,
        }
    if len(text) <= 80 and not HARD_TASK_HINT_RE.search(text):
        if any(p.search(text) for p in LIGHT_TASK_PATTERNS):
            return {
                "class": "light",
                "confidence": threshold,
                "reason": "short trivial prompt matched v1 light-task allowlist",
                "target_tier": 2,
                "light_threshold": threshold,
            }
    return {
        "class": "hard",
        "confidence": 1.0 - threshold,
        "reason": "not a high-confidence trivial prompt; defaulting to max-quality tier",
        "target_tier": 1,
        "light_threshold": threshold,
    }


def _route_rank(row: dict) -> tuple:
    # Order:
    #   1. operator room bias  -- higher route_weight (more capacity) sorts FIRST;
    #   2. fewest LIVE sessions -- spread CONCURRENT load off any one account now;
    #   3. fewest ACTIVE sessions -- spread CUMULATIVE load (a weak room proxy);
    #   4. product, then tag   -- a stable, deterministic final tiebreak.
    # route_weight defaults to 0, so with no operator bias this is exactly the historical
    # (live, active, product, tag) balancing. NOTE: keys 2-3 count SESSIONS, not remaining
    # quota -- the fleet collects no quota headroom, so true "pick the account with the most
    # room left" is only expressible via route_weight (see _clean_profile).
    return (
        -int(row.get("route_weight") or 0),
        int(row.get("live_sessions") or 0),
        int(row.get("active_sessions") or 0),
        str(row.get("product") or ""),
        str(row.get("tag") or row.get("account") or ""),
    )


def _public_blocked(row: dict) -> dict:
    return {
        "tag": row.get("tag"),
        "account": row.get("account"),
        "product": row.get("product"),
        "model_tier": row.get("model_tier"),
        "model": row.get("model"),
        "reason": row.get("block_reason") or row.get("reason") or "blocked",
    }


def route_account(rows: list[dict], task_text: str = "", task_class: str = "auto",
                  *, allow_tier_fallback: bool = False,
                  strict_tier: bool = False,
                  product: str | None = None,
                  leases: list[dict] | None = None,
                  policy: dict | None = None,
                  policy_fn: Callable[[], dict] | None = None) -> dict:
    """Choose an account by task difficulty and model tier.

    ``task_class`` accepts the tier aliases (``t1``/``t2``/``t3``, ``hard``/``light``)
    AND the structured work-kind tokens (``gardening``/``maintenance``/...,
    ``engineering``/``dev``/...; see GARDENING_WORK_KINDS / ENGINEERING_WORK_KINDS).
    A work-kind is an operator-stated fact and wins over the free-text heuristic --
    that is how a maintenance loop pins tier 2 even though its prompt reads as
    "hard", while engineering work pins tier 1.

    V1 policy:
      - high-confidence trivial prompts (or a gardening work-kind) target tier 2;
      - everything else (incl. an engineering work-kind) targets tier 1;
      - tier 3 is reported but never auto-selected yet;
      - a tier-2 target may up-shift to tier 1 when no tier-2 account is free
        (preserving quality) unless ``strict_tier``; a tier-1 target only falls
        back to tier 2 when ``allow_tier_fallback``.

    Within the chosen tier, ties are broken by ``_route_rank``: an operator
    ``route_weight`` bias first (higher = more room -> preferred), then fewest live
    sessions, then fewest active sessions, then product/tag. The router has NO
    remaining-quota signal, so "prefer the account with more room left" is only
    expressible by setting the account's weight in the policy ``route_weights`` map.
    """
    pol = policy or (policy_fn() if policy_fn is not None else {})
    task = classify_task(task_text, task_class, pol, policy_fn=policy_fn)
    wanted_product = (product or "").lower()
    workers = [
        r for r in rows
        if routable_worker(r)  # excludes duplicate-identity dirs (same account as canonical)
        and (not wanted_product or str(r.get("product") or "").lower() == wanted_product)
    ]
    if not workers:
        return {
            "ok": False,
            "reason": "no worker accounts match product filter" if wanted_product else "no worker accounts",
            "task": task,
            "target_tier": task["target_tier"],
            "fallback_used": False,
            "account": None,
            "blocked_target_accounts": [],
        }

    lease_workers, _ = _lease_workers_by_pool(workers, list(leases or []))

    def _pool_has_free_slot(row: dict) -> bool:
        pool = _pool_key(row)
        return len(lease_workers.get(pool, [])) < max(1, _session_cap(row))

    available = [r for r in workers if r.get("available") and _pool_has_free_slot(r)]
    target = int(task["target_tier"])
    fallback_policy = str(pol.get("routing", {}).get("hard_tier1_fallback", "stop")).lower()
    effective_allow_fallback = allow_tier_fallback or fallback_policy in (
        "allow",
        "fallback",
        "tier2",
        "t2",
    )
    tier_order = [target]
    if target == 2 and not strict_tier:
        # Lightweight work may safely move up to tier 1 if GLM/tier-2 capacity
        # is unavailable; that preserves quality.
        tier_order.append(1)
    elif effective_allow_fallback:
        tier_order.append(2)

    for tier in tier_order:
        candidates = [r for r in available if _as_int(r.get("model_tier"), 3) == tier]
        if candidates:
            chosen = dict(sorted(candidates, key=_route_rank)[0])
            return {
                "ok": True,
                "reason": "selected target tier" if tier == target else "selected fallback tier",
                "task": task,
                "target_tier": target,
                "selected_tier": tier,
                "fallback_used": tier != target,
                "account": chosen,
                "blocked_target_accounts": [
                    _public_blocked(r) for r in workers
                    if _as_int(r.get("model_tier"), 3) == target and not r.get("available")
                ],
            }

    blocked = [
        _public_blocked(r) for r in workers
        if _as_int(r.get("model_tier"), 3) in set(tier_order) and not r.get("available")
    ]
    fallback_note = ""
    if target == 1 and not effective_allow_fallback:
        fallback_note = " (tier-1 fallback disabled)"
    elif strict_tier:
        fallback_note = " (exact tier requested)"
    if not any(_as_int(r.get("model_tier"), 3) == target for r in workers):
        fallback_note = " (no matching worker tier)"
    return {
        "ok": False,
        "reason": f"no available tier {target} account{fallback_note}",
        "task": task,
        "target_tier": target,
        "selected_tier": None,
        "fallback_used": False,
        "account": None,
        "blocked_target_accounts": blocked,
    }


def _flatten_resolved(acct: dict, *, ok: bool, reason: str,
                      selected_tier=None, target_tier=None,
                      fallback_used: bool = False,
                      block_reason: str = "",
                      token_fn: Callable[[str], str | None] = read_oauth_token) -> dict:
    """Project an account row into the flat resolve_account() return shape, stamping
    the long-lived oauth token so every front door reads ONE record."""
    config_dir = str(acct.get("dir") or "")
    return {
        "ok": ok,
        "reason": reason,
        "account": str(acct.get("account") or ""),
        "tag": str(acct.get("tag") or ""),
        "product": str(acct.get("product") or ""),
        "config_dir": config_dir,
        "oauth_token": token_fn(config_dir),
        "model": str(acct.get("model") or ""),
        "model_tier": acct.get("model_tier"),
        "selected_tier": selected_tier if selected_tier is not None else acct.get("model_tier"),
        "target_tier": target_tier,
        "fallback_used": bool(fallback_used),
        "block_reason": block_reason,
    }


def resolve_account(pin: str | None = None, *, task_text: str = "",
                    task_class: str = "auto", work_kind: str = "",
                    product: str | None = None,
                    allow_tier_fallback: bool = False,
                    strict_tier: bool = False,
                    faklocal_ok: bool = False,
                    home: str | None = None, policy: dict | None = None,
                    registry: dict | None = None,
                    config_home: str | None = None,
                    policy_fn: Callable[[], dict] | None = None,
                    roster_fn: Callable[..., list[dict]] | None = None,
                    token_fn: Callable[[str], str | None] = read_oauth_token) -> dict:
    """Resolve ONE fully-specified account for a dispatch -- the canonical front-door call.

    The single entry point every front door (launch_goal_detached.ps1, the dogfood
    launchers, issue_dispatch) routes through instead of stitching ``json``+``route``+a
    hand-rolled availability check + a separate ``.oauth-token`` read. Returns a FLAT dict
    ``{ok, reason, account, tag, product, config_dir, oauth_token, model, model_tier,
    selected_tier, target_tier, fallback_used, block_reason}`` -- ``config_dir`` is what to
    pin as CLAUDE_CONFIG_DIR and ``oauth_token`` is the long-lived setup token (or None ->
    drop the ambient one).
    """
    resolved_home = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~")) if home is None else home
    pol = policy or (policy_fn() if policy_fn is not None else {})

    # The dogfood isolated account: not a discovered worker, synthesized on demand.
    if faklocal_ok and pin and account_tag(pin) == FAKLOCAL_TAG:
        d = os.path.join(resolved_home, ".claude-faklocal")
        try:
            os.makedirs(os.path.join(d, "projects"), exist_ok=True)
        except OSError:
            pass
        return _flatten_resolved(
            {"account": ".claude-faklocal", "tag": FAKLOCAL_TAG, "product": "claude",
             "dir": d, "model": "local", "model_tier": 3},
            ok=True, reason="isolated dogfood faklocal account",
            selected_tier=3, target_tier=3, token_fn=token_fn)

    rows = roster_fn(resolved_home, pol, registry, config_home=config_home) if roster_fn is not None else []

    if pin:
        needle = pin.strip().lower()
        match = next(
            (r for r in rows
             if r.get("kind") == "worker"
             and (str(r.get("tag") or "").lower() == needle
                  or str(r.get("account") or "").lower() == needle)),
            None)
        if match is None:
            return _flatten_resolved(
                {}, ok=False, reason=f"account '{pin}' is not an offered worker",
                token_fn=token_fn)
        if not match.get("available") and not allow_tier_fallback:
            why = str(match.get("block_reason") or "blocked")
            return _flatten_resolved(
                match, ok=False,
                reason=f"account '{pin}' is blocked: {why}",
                block_reason=why, token_fn=token_fn)
        return _flatten_resolved(
            match, ok=True, reason="pinned account",
            selected_tier=match.get("model_tier"),
            target_tier=match.get("model_tier"), token_fn=token_fn)

    # Tier / work-kind routing among available accounts.
    cls = task_class
    strict = strict_tier
    wk = (work_kind or "").strip().lower()
    if wk in GARDENING_WORK_KINDS or wk in ENGINEERING_WORK_KINDS:
        cls, strict = wk, False
    route = route_account(rows, task_text, cls,
                          allow_tier_fallback=allow_tier_fallback,
                          strict_tier=strict, product=product, policy=pol,
                          policy_fn=policy_fn)
    if not route.get("ok"):
        return {
            "ok": False,
            "reason": str(route.get("reason") or "no available account"),
            "account": "", "tag": "", "product": "",
            "config_dir": "", "oauth_token": None,
            "model": "", "model_tier": None,
            "selected_tier": None, "target_tier": route.get("target_tier"),
            "fallback_used": False, "block_reason": "",
            "blocked_target_accounts": route.get("blocked_target_accounts", []),
        }
    return _flatten_resolved(
        route["account"], ok=True, reason=str(route.get("reason") or "routed"),
        selected_tier=route.get("selected_tier"),
        target_tier=route.get("target_tier"),
        fallback_used=bool(route.get("fallback_used")), token_fn=token_fn)


def _pool_key(row: dict) -> str:
    """The RATE-LIMIT POOL a worker dir draws on -- the unit a wave must hand out
    distinctly. Two Claude dirs logged into one Anthropic account share ONE usage
    pool (their accountUuid), so they are the SAME pool even with different dir
    names; that is the whole reason a wave keys on identity, not basename. A dir
    with no login (or a non-Claude product with no uuid) is its own pool, keyed by
    its dir basename."""
    uuid = str(row.get("account_uuid") or "")
    if uuid:
        return f"uuid:{uuid}"
    return f"dir:{row.get('account') or row.get('dir') or ''}"


def _wave_id_for(pools: list[str]) -> str:
    """A deterministic, content-addressed id for a wave: a short digest of its
    granted pools (sorted). Same membership -> same id, with no clock or random
    input, so an auditor reproduces it from the lanes alone. An empty wave
    (granted 0) gets the empty id."""
    if not pools:
        return ""
    digest = hashlib.blake2b(",".join(sorted(pools)).encode("utf-8"), digest_size=6)
    return "wave-" + digest.hexdigest()


def _wave_tier_order(target: int, strict: bool, allow_fallback: bool) -> list[int]:
    order = [target]
    if target == 2 and not strict:
        order.append(1)
    elif allow_fallback:
        order.append(2)
    return order


def allocate_wave(count: int, *, task_text: str = "", task_class: str = "auto",
                  work_kind: str = "", product: str | None = None,
                  allow_tier_fallback: bool = False, strict_tier: bool = False,
                  home: str | None = None, policy: dict | None = None,
                  registry: dict | None = None,
                  config_home: str | None = None,
                  leases: list[dict] | None = None,
                  wave_id: str | None = None,
                  policy_fn: Callable[[], dict] | None = None,
                  roster_fn: Callable[..., list[dict]] | None = None,
                  token_fn: Callable[[str], str | None] = read_oauth_token) -> dict:
    """Allocate up to ``count`` bounded account session slots for a parallel fan-out
    (an "ultracode" wave), balanced across the roster -- the primitive a wave needs
    that single-account ``resolve_account`` cannot provide.

    Returns a flat dict::

        {ok, requested, granted, shortfall, distinct_pools, size, wave_id,
         target_tier, reason, lanes: [...], blocked_target_accounts}
    """
    resolved_home = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~")) if home is None else home
    pol = policy or (policy_fn() if policy_fn is not None else {})
    n = max(0, int(count))

    cls, strict = task_class, strict_tier
    wk = (work_kind or "").strip().lower()
    if wk in GARDENING_WORK_KINDS or wk in ENGINEERING_WORK_KINDS:
        cls, strict = wk, False

    task = classify_task(task_text, cls, pol, policy_fn=policy_fn)
    target = int(task["target_tier"])

    rows = roster_fn(resolved_home, pol, registry, config_home=config_home) if roster_fn is not None else []
    wanted_product = (product or "").lower()
    workers = [
        r for r in rows
        if routable_worker(r)  # excludes duplicate-identity dirs (same pool as canonical)
        and (not wanted_product or str(r.get("product") or "").lower() == wanted_product)
    ]
    available = [r for r in workers if r.get("available")]

    fallback_policy = str(pol.get("routing", {}).get("hard_tier1_fallback", "stop")).lower()
    effective_allow_fallback = allow_tier_fallback or fallback_policy in (
        "allow", "fallback", "tier2", "t2")
    tier_order = _wave_tier_order(target, strict, effective_allow_fallback)

    lanes: list[dict] = []
    lease_workers, _ = _lease_workers_by_pool(workers, list(leases or []))
    pool_load: dict[str, int] = {
        pool: len(bound) for pool, bound in lease_workers.items()
    }
    for tier in tier_order:
        if len(lanes) >= n:
            break
        candidates: list[dict] = []
        seen_candidate_pools: set[str] = set()
        for r in sorted(
                (r for r in available if _as_int(r.get("model_tier"), 3) == tier),
                key=_route_rank):
            pool = _pool_key(r)
            if pool in seen_candidate_pools:
                continue
            seen_candidate_pools.add(pool)
            candidates.append(r)
        while len(lanes) < n:
            choices = [
                r for r in candidates
                if pool_load.get(_pool_key(r), 0) < max(1, _session_cap(r))
            ]
            if not choices:
                break
            r = min(choices, key=lambda row: (pool_load.get(_pool_key(row), 0),
                                              _route_rank(row)))
            pool = _pool_key(r)
            cap = max(1, _session_cap(r))
            slot = pool_load.get(pool, 0) + 1
            lane = _flatten_resolved(
                r, ok=True,
                reason="wave lane (target tier)" if tier == target else "wave lane (fallback tier)",
                selected_tier=r.get("model_tier"), target_tier=target,
                fallback_used=tier != target)
            lane["pool"] = pool
            lane["session_slot"] = slot
            lane["session_cap"] = cap
            lanes.append(lane)
            pool_load[pool] = slot

    granted = len(lanes)
    shortfall = max(0, n - granted)
    distinct = len({lane["pool"] for lane in lanes})
    wid = wave_id if wave_id is not None else _wave_id_for(
        [f"{lane['pool']}#{lane.get('session_slot', 1)}" for lane in lanes])
    for i, lane in enumerate(lanes):
        lane["rank"] = i
        lane["wave_id"] = wid
        lane["size"] = granted
    blocked = [
        _public_blocked(r) for r in workers
        if _as_int(r.get("model_tier"), 3) == target and not r.get("available")
    ]
    if granted == 0:
        reason = (f"no available account for a wave (target tier {target}"
                  + (f", product {wanted_product}" if wanted_product else "") + ")")
    elif shortfall:
        reason = (f"granted {granted} of {n} session slot(s) across {distinct} distinct pool(s); "
                  f"{shortfall} short (roster has no more available session slots "
                  "at the requested tiers)")
    else:
        reason = f"granted {granted} session slot(s) across {distinct} distinct pool(s)"
    return {
        "ok": granted > 0,
        "requested": n,
        "granted": granted,
        "shortfall": shortfall,
        "distinct_pools": distinct,
        "size": granted,
        "wave_id": wid,
        "target_tier": target,
        "reason": reason,
        "lanes": lanes,
        "blocked_target_accounts": blocked,
    }


def _seat_hold_reason(row: dict) -> tuple[str, str]:
    """(dispatch_state, hold_reason) for one seat row, derived from the SAME
    ``runtime_status`` fields (``block_kind``/``throttled``/``reset``/``weekly``/
    ``block_reason``) the roster and preflight already surface -- no new state names.

    dispatch_state is one of: available / cooling / unavailable (``leased``/``busy``
    is decided by the caller, which alone knows the live-worker binding). ``hold_reason``
    is "" when available, else a specific token/detail -- never the bare word
    "unavailable"."""
    if bool(row.get("available")):
        return "available", ""
    kind = str(row.get("block_kind") or "")
    if bool(row.get("throttled")) or kind == "usage":
        reset = row.get("reset")
        detail = f"cooldown_until={reset}" if reset else "rate_limited"
        weekly = row.get("weekly")
        if weekly:
            detail += f";weekly={weekly}"
        return "cooling", detail
    if kind in _HOLD_KIND_REASON:
        return "unavailable", _HOLD_KIND_REASON[kind]
    reason = str(row.get("block_reason") or "").strip()
    return "unavailable", reason or "no_capacity"


def _seat_cooldown(row: dict) -> dict | None:
    """Structured cooldown for a rate-limited / usage-throttled seat, so dispatch status
    explains throttle-related capacity loss from NAMED fields instead of re-parsing the
    packed ``hold_reason`` string (#1801). Derived from the SAME ``runtime_status`` fields
    ``_seat_hold_reason`` reads, so a seat is cooling here iff it is cooling there:
    ``reason`` is the block kind (``usage``) or ``rate_limited`` when none is known,
    ``since`` the cooldown start when the throttle ledger record stamped one
    (``throttled_since``), ``until`` the next-eligible reset, ``weekly`` the weekly window
    when present. Returns None when the seat is not cooling."""
    kind = str(row.get("block_kind") or "")
    if not (bool(row.get("throttled")) or kind == "usage"):
        return None
    cd = {
        "reason": kind or "rate_limited",
        "since": row.get("throttled_since"),
        "until": row.get("reset"),
    }
    weekly = row.get("weekly")
    if weekly:
        cd["weekly"] = weekly
    return cd


def _lease_matches_seat(lease: dict, row: dict) -> bool:
    """Does a live-worker lease record (parsed from its ``.account`` sidecar) bind to
    this seat row? Match on the account DIR first (the precise key the sidecar stamps),
    then fall back to the short TAG. A lease names the dir/tag the worker launched on;
    the seat carries the same on its roster row."""
    ldir = str(lease.get("dir") or "")
    rdir = str(row.get("dir") or "")
    if ldir and (
        (rdir and ldir == rdir)
        or os.path.basename(ldir.rstrip("/\\")) == str(row.get("account") or "")
    ):
        return True
    ltag = str(lease.get("tag") or "").lower()
    return bool(ltag) and ltag == str(row.get("tag") or "").lower()


def _pool_row_less(a: dict, b: dict) -> bool:
    if bool(a.get("available")) != bool(b.get("available")):
        return bool(a.get("available"))
    return _route_rank(a) < _route_rank(b)


def _unique_pool_rows(rows: list[dict]) -> list[dict]:
    by_pool: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        pool = _pool_key(row)
        if pool not in by_pool:
            by_pool[pool] = row
            order.append(pool)
            continue
        if _pool_row_less(row, by_pool[pool]):
            by_pool[pool] = row
    return [by_pool[pool] for pool in order]


def _lease_worker_label(lease: dict) -> str:
    return str(lease.get("worker") or lease.get("pid") or "?")


def _lease_workers_by_pool(rows: list[dict], leases: list[dict]) -> tuple[dict[str, list[str]], set[int]]:
    out: dict[str, list[str]] = {}
    matched: set[int] = set()
    for i, lease in enumerate(leases):
        for row in rows:
            if not _lease_matches_seat(lease, row):
                continue
            out.setdefault(_pool_key(row), []).append(_lease_worker_label(lease))
            matched.add(i)
            break
    return out, matched


def seat_pool(rows: list[dict], leases: list[dict] | None = None,
              *, product: str | None = None) -> dict:
    """The explicit multi-slot account pool: bounded routable worker session slots
    x tier, with the seat->worker binding for the live workers leasing them.

    A row is one distinct rate-limit POOL among the routable workers -- keyed by
    ``_pool_key`` (the Anthropic accountUuid for a logged-in Claude dir, else the dir
    basename), so two dirs on ONE Anthropic account are one pool, never two.
    """
    leases = list(leases or [])
    wanted = (product or "").lower()
    seats: list[dict] = []
    double_booked: list[dict] = []
    total = free = leased = blocked = 0
    pool_rows: list[dict] = []
    for row in rows:
        if not routable_worker(row):  # excludes duplicate-identity dirs -> one seat per pool
            continue
        if wanted and str(row.get("product") or "").lower() != wanted:
            continue
        pool_rows.append(row)
    lease_workers, matched = _lease_workers_by_pool(pool_rows, leases)
    for row in _unique_pool_rows(pool_rows):
        capacity = max(1, _session_cap(row))
        seat_workers = list(lease_workers.get(_pool_key(row), []))
        available = bool(row.get("available"))
        leased_slots = len(seat_workers)
        leased_capped = min(leased_slots, capacity)
        free_slots = 0
        if seat_workers:
            state = "leased"
            leased += leased_capped
            if available:
                free_slots = max(0, capacity - leased_capped)
                free += free_slots
            else:
                blocked += max(0, capacity - leased_capped)
        elif available:
            state = "free"
            free_slots = capacity
            free += free_slots
        else:
            state = "blocked"
            blocked += capacity
        total += capacity
        if seat_workers:
            dispatch_state, hold_reason = "busy", f"leased to {', '.join(seat_workers)}"
            cooldown = None
        else:
            dispatch_state, hold_reason = _seat_hold_reason(row)
            cooldown = _seat_cooldown(row) if dispatch_state == "cooling" else None
        seat = {
            "seat": _pool_key(row),
            "tag": row.get("tag"),
            "account": row.get("account"),
            "product": row.get("product"),
            "model": row.get("model"),
            "model_tier": row.get("model_tier"),
            "available": available,
            "state": state,
            "dispatch_state": dispatch_state,
            "hold_reason": hold_reason,
            "cooldown": cooldown,
            "session_cap": capacity,
            "leased_slots": leased_slots,
            "free_slots": free_slots,
            "workers": seat_workers,
        }
        seats.append(seat)
        if leased_slots > capacity:
            double_booked.append({"seat": seat["seat"], "tag": seat["tag"],
                                  "workers": seat_workers, "session_cap": capacity})
    unbound = [
        {"worker": str(ls.get("worker") or ls.get("pid") or "?"),
         "tag": ls.get("tag"), "dir": ls.get("dir")}
        for i, ls in enumerate(leases) if i not in matched
    ]
    seats.sort(key=lambda s: (s["state"] != "leased", s["state"] != "free",
                              str(s.get("product") or ""), str(s.get("tag") or "")))
    by_dispatch_state: dict[str, int] = {}
    for s in seats:
        ds = s["dispatch_state"]
        by_dispatch_state[ds] = by_dispatch_state.get(ds, 0) + 1
    return {
        "schema": SEAT_POOL_SCHEMA,
        "product": wanted or "all",
        "total_seats": total,
        "free_seats": free,
        "leased_seats": leased,
        "blocked_seats": blocked,
        "by_dispatch_state": by_dispatch_state,
        "depleted": free == 0,
        "double_booked": double_booked,
        "unbound_leases": unbound,
        "seats": seats,
    }


def live_seat_leases(runs_dir: str | None = None, *,
                     alive: set | None = None, probe=None) -> list[dict]:
    """Lease records for the LIVE issue-resolution workers: one ``{worker, pid, tag,
    dir}`` per worker whose ``.pid`` sidecar still passes the identity-gated liveness
    check, read from its sibling ``.account`` sidecar.

    This is the seat->worker BINDING source ``seat_pool`` consumes: a seat is leased iff
    a live worker's ``.account`` names it. Because the binding is derived from LIVE pids
    (the SAME ``dispatch_preflight`` liveness gate the spawn cap uses), a worker that
    exits frees its seat on the very next read -- its sidecar is no longer live -- with
    no separate release step. Best effort: a missing runs dir or unreadable sidecar
    contributes nothing. ``alive``/``probe`` are injectable for hermetic tests (the same
    shape ``dispatch_preflight.resolve_sidecar_pid_is_live`` accepts)."""
    try:
        import dispatch_preflight  # lazy: no import-time coupling (it shells out to us)
    except ImportError:
        return []
    if runs_dir is None:
        runs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            dispatch_preflight.RUNS_DIRNAME)
    rd = Path(runs_dir)
    if not rd.is_dir():
        return []
    leases: list[dict] = []
    for pid_file in sorted(rd.glob("resolve-*.pid")):
        if not dispatch_preflight.resolve_sidecar_pid_is_live(
                pid_file, alive=alive, probe=probe):
            continue
        rec: dict = {}
        try:
            parsed = json.loads(
                pid_file.with_suffix(ACCOUNT_SIDECAR_SUFFIX).read_text(encoding="utf-8"))
            rec = parsed if isinstance(parsed, dict) else {}
        except (OSError, ValueError):
            rec = {}
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        leases.append({"worker": pid_file.stem, "pid": pid,
                       "tag": str(rec.get("tag") or ""), "dir": str(rec.get("dir") or "")})
    return leases


def _cli_list(rows: list[dict], *, user: str = "", policy_path: str = "",
              policy_example_path: str = "") -> None:
    kinds = {"available": [], "blocked": [], "duplicate": [], "excluded": [], "non-account": []}
    for r in rows:
        if r["kind"] == "worker" and is_duplicate_identity(r):
            kinds["duplicate"].append(r)   # same Anthropic account as a canonical dir
        elif r["kind"] == "worker" and r.get("available"):
            kinds["available"].append(r)
        elif r["kind"] == "worker":
            kinds["blocked"].append(r)
        else:
            kinds.setdefault(r["kind"], []).append(r)
    products = sorted({r.get("product", "claude") for r in rows}) or ["claude"]
    # how many DISTINCT Anthropic accounts back the Claude worker dirs?
    claude_logins = {str(r.get("account_uuid")) for r in rows
                     if r.get("product") == "claude" and r.get("kind") == "worker"
                     and r.get("account_uuid")}
    claude_dirs = sum(1 for r in rows
                      if r.get("product") == "claude" and r.get("kind") == "worker")
    print(f"fleet accounts under {user}  ({len(rows)} dirs, products: "
          f"{'+'.join(products)})")
    if claude_dirs:
        print(f"identity: {claude_dirs} Claude worker dir(s) -> "
              f"{len(claude_logins)} distinct Anthropic account(s)"
              + (f"  ({len(kinds['duplicate'])} duplicate dir(s) not offered)"
                 if kinds["duplicate"] else ""))
    print()
    print(f"AVAILABLE (offered to switcher now): {len(kinds['available'])}")
    for r in kinds["available"]:
        detail = f"{r['active_sessions']} active, {r['live_sessions']} live"
        if r.get("usage_soon_reset"):
            detail += f"; cap resets {r['usage_soon_reset']}"
        tier = f"t{r.get('model_tier', '?')}"
        model = r.get("model") or ""
        print(f"  [{r.get('product','claude'):<8}] {r['tag']:<16} {r['account']:<28} {tier:<3} {model:<24} {detail}")
    if kinds["blocked"]:
        print(f"\nBLOCKED (real account, do not offer now): {len(kinds['blocked'])}")
        for r in kinds["blocked"]:
            tier = f"t{r.get('model_tier', '?')}"
            model = r.get("model") or ""
            print(f"  [{r.get('product','claude'):<8}] {r['tag']:<16} {r['account']:<28} {tier:<3} {model:<24} {r['block_reason']}")
    if kinds["duplicate"]:
        print(f"\nDUPLICATE IDENTITY (same Anthropic account as a canonical dir -- not offered): {len(kinds['duplicate'])}")
        for r in kinds["duplicate"]:
            peers = ", ".join(r.get("identity_peers") or [])
            print(f"  [{r.get('product','claude'):<8}] {r['tag']:<16} {r['account']:<28} login={r.get('login_email','')}  shares with: {peers}")
    if kinds["excluded"]:
        print(f"\nEXCLUDED (tombstoned): {len(kinds['excluded'])}")
        for r in kinds["excluded"]:
            print(f"  [{r.get('product','claude'):<8}] {r['tag']:<16} {r['account']:<28} {r['reason']}")
    if kinds["non-account"]:
        print(f"\nNON-ACCOUNT (ignored): {len(kinds['non-account'])}")
        for r in kinds["non-account"]:
            print(f"  [{r.get('product','claude'):<8}] {r['tag']:<16} {r['account']:<28} {r['reason']}")
    if policy_path and os.path.exists(policy_path):
        pol_src = policy_path
    elif policy_example_path and os.path.exists(policy_example_path):
        pol_src = policy_example_path + " (example; copy to _registry/ to customize)"
    else:
        pol_src = "(built-in defaults)"
    print(f"\npolicy: {pol_src}")


def _cli_seats(pool: dict) -> None:
    """Human view of the explicit seat pool + the seat->worker binding for live workers."""
    print(f"seat pool [{pool.get('product')}]: {pool.get('total_seats')} session slot(s)  "
          f"free={pool.get('free_seats')} leased={pool.get('leased_seats')} "
          f"blocked={pool.get('blocked_seats')}"
          + ("  DEPLETED" if pool.get("depleted") else ""))
    for s in pool.get("seats", []):
        workers = ", ".join(s.get("workers") or []) or "-"
        tier = f"t{s.get('model_tier', '?')}"
        hold = s.get("hold_reason") or ""
        suffix = f"  ({hold})" if hold else ""
        print(f"  [{str(s.get('dispatch_state') or ''):<11}] {str(s.get('tag') or ''):<16} "
              f"{str(s.get('account') or ''):<28} {tier:<3} "
              f"slots={s.get('leased_slots', 0)}/{s.get('session_cap', 1)} "
              f"free={s.get('free_slots', 0)} -> {workers}{suffix}")
    if pool.get("double_booked"):
        print("\nDOUBLE-BOOKED (one seat, >1 live worker -- INVARIANT VIOLATION):")
        for d in pool["double_booked"]:
            print(f"  {d.get('tag')}: {', '.join(d.get('workers') or [])}")
    if pool.get("unbound_leases"):
        print("\nUNBOUND LEASES (live worker on an account not in the pool):")
        for u in pool["unbound_leases"]:
            print(f"  {u.get('worker')}: tag={u.get('tag')} dir={u.get('dir')}")


def _arg_value(argv: list[str], names: tuple[str, ...], default: str = "") -> str:
    for i, arg in enumerate(argv):
        for name in names:
            if arg == name and i + 1 < len(argv):
                return argv[i + 1]
            prefix = name + "="
            if arg.startswith(prefix):
                return arg[len(prefix):]
    return default


def _has_arg(argv: list[str], names: tuple[str, ...]) -> bool:
    return any(arg in names for arg in argv)


def _tier_arg(argv: list[str]) -> tuple[str, bool]:
    for arg in argv:
        low = arg.lower()
        if low in ("-t1", "--t1"):
            return "t1", True
        if low in ("-t2", "--t2"):
            return "t2", True
        if low in ("-t3", "--t3"):
            return "t3", True
    raw = _arg_value(argv, ("-t", "--tier", "--task-tier"), "auto")
    return raw, raw.lower() not in ("", "auto")
