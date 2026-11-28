#!/usr/bin/env python3
r"""fleet_accounts -- COMPATIBILITY SHIM over the canonical Go account contract.

The source of truth for "what is an account, and is it offered?" is now the Go
package ``internal/fleetaccounts`` behind ``fak fleet-accounts``
(``roster|list|json|available|resolve|wave|seats|status``), byte-parity-proven
against this module by ``internal/fleetaccounts/testdata/parity_check.sh``
(#1415). The dispatch tick/wave path and the launch scripts
(``launch_goal_detached.ps1`` / ``launch_wave_detached.ps1``) already consume
the Go surface. This file stays behind, semantics frozen, for the consumers not
yet migrated -- the resume watchdog, ``fleet_status.ps1``, the control pane,
``dispatch_preflight`` / account top-up -- and for the folds the Go port does
not cover yet (the ACTIVE network ``probe`` and the relogin/top-up mutations).
Fix behavior in Go first; only mirror here if a Python consumer needs it.
pythongate tracking: this file remains in the grandfathered baseline until the
remaining consumers move and it is ported-and-deleted (the baseline only
shrinks on delete -- see ``internal/pythongate/doc.go``).

The fleet resume layer discovers accounts by globbing config dirs. Historically
every ``.claude*`` match with a ``projects/`` subdir was treated as an equal,
resumable account -- which swept in the break-glass *backup* account alongside
the real workers, and gave the operator no way to see the roster or exclude an
account by hand. opencode accounts (``~/.config/opencode*``) were not seen at
all. This module classifies config dirs from BOTH products into one ``kind``:

  worker       a real, offered account (claude default/gem8/...; opencode
               default/glm/...)
  excluded     tombstoned by policy -- present on disk but never offered as a
               resume target (the backup account, by default)
  non-account  not an account dir at all: no account marker
               (Claude: no ``projects/`` subdir; opencode: no ``opencode.json``)
               or a plain file (``.claude.json``)

Discovery roots (per product):
  Claude    ``<home>/.claude*``            -- ``home`` defaults to the user home
  opencode  ``<config_home>/opencode*``    -- ``config_home`` defaults to XDG
                                              config home (~/.config)

Policy lives in ``tools/_registry/accounts_policy.json`` (operator-editable) and
applies to BOTH products uniformly:

    {
      "exclude": ["backup"],        // substrings; tombstoned accounts
      "include_only": [],           // if non-empty, ONLY these tags are workers
      "notes": {"backup": "break-glass backup; never auto-resume"}
    }

``exclude`` substrings are matched against the bare account *tag*
(``.claude-<tag>-acct`` -> ``<tag>``; ``opencode-<tag>`` -> ``<tag>``;
``.claude`` / ``opencode`` -> ``default``) and the dir basename.
``include_only`` (when non-empty) is an allowlist. The built-in default already
excludes ``backup`` / ``breakglass`` so backup accounts are out of the box even
with no policy file present.

Runtime status (usage throttle / auth block / live sessions) is account-keyed
and therefore product-neutral: once the session-tracking layer records an
opencode session under its ``opencode`` basename, it flows through the same
``runtime_status`` path as a Claude session. With no recorded sessions an
opencode account simply reports available/healthy.

CLI:
    python tools/fleet_accounts.py list       # human roster table + live status
    python tools/fleet_accounts.py json       # machine roster + live status
    python tools/fleet_accounts.py available  # account dirs safe to offer now
    python tools/fleet_accounts.py route --task "say pong"  # tier-aware pick
    python tools/fleet_accounts.py resolve --work-kind engineering  # ONE flat record:
                                              # config_dir + oauth_token + tier (pin via
                                              # --account; dogfood via --faklocal-ok). The
                                              # single call every dispatch front door makes.
    python tools/fleet_accounts.py wave --count 20 --explain # allocate up to 20 bounded
                                              # account session slots for a parallel fan-out
                                              # (an ultracode wave); --explain prints the
                                              # distinct pool count vs the naive 1.
                                              # Omit --count for every session slot free now.
    python tools/fleet_accounts.py seats --product claude    # the EXPLICIT seat pool:
                                              # bounded account session slots x tier with the
                                              # seat->worker binding for live workers (from
                                              # each live worker's .account lease sidecar).
                                              # Claude worker accounts contribute 4 slots;
                                              # depleted == every slot leased. --json for machines.
"""
from __future__ import annotations

import copy
import datetime as dt
import glob
import json
import os
import re
import sys
from typing import Any

import fleet_regdir
import fleet_session_signals
from fleet_accounts_cap import (
    DEFAULT_ACCOUNT_SESSION_CAP as DEFAULT_ACCOUNT_SESSION_CAP,
    DEFAULT_CLAUDE_ACCOUNT_SESSION_CAP as DEFAULT_CLAUDE_ACCOUNT_SESSION_CAP,
    DEFAULT_CODEX_ACCOUNT_SESSION_CAP as DEFAULT_CODEX_ACCOUNT_SESSION_CAP,
    SESSIONS_PER_ACCOUNT_ENV as SESSIONS_PER_ACCOUNT_ENV,
    _DAILY_RESET_WINDOW as _DAILY_RESET_WINDOW,
    _OVERRIDE_STREAK as _OVERRIDE_STREAK,
    _WEEKLY_MAX_AGE as _WEEKLY_MAX_AGE,
    _CapObservation as _CapObservation,
    _CapState as _CapState,
    _account_cap_until as _account_cap_until,
    _age_min as _age_min,
    _as_int as _as_int,
    _cap_observation_from as _cap_observation_from,
    _claude_session_cap as _claude_session_cap,
    _derive_cap_observations as _derive_cap_observations,
    _disambiguate_cap as _disambiguate_cap,
    _normalize_throttle as _normalize_throttle,
    _now_utc as _now_utc,
    _parse_utc as _parse_utc,
    _probe_is_ok as _probe_is_ok,
    _reset_is_future as _reset_is_future,
    _reset_text as _reset_text,
    _row_seen_utc as _row_seen_utc,
    _session_cap as _session_cap,
    _weekly_reset_text as _weekly_reset_text,
    _weekly_throttle_is_active as _weekly_throttle_is_active,
    account_product as account_product,
    account_tag as account_tag,
    active_account_cap_throttles as _active_account_cap_throttles_impl,
    is_duplicate_identity as is_duplicate_identity,
    read_oauth_token as read_oauth_token,
    routable_worker as routable_worker,
    throttle_is_active as throttle_is_active,
)
from fleet_accounts_routing import (
    ACCOUNT_SIDECAR_SUFFIX as ACCOUNT_SIDECAR_SUFFIX,
    ENGINEERING_WORK_KINDS as ENGINEERING_WORK_KINDS,
    FAKLOCAL_TAG as FAKLOCAL_TAG,
    GARDENING_WORK_KINDS as GARDENING_WORK_KINDS,
    HARD_TASK_HINT_RE as HARD_TASK_HINT_RE,
    LIGHT_TASK_PATTERNS as LIGHT_TASK_PATTERNS,
    SEAT_POOL_SCHEMA as SEAT_POOL_SCHEMA,
    _HOLD_KIND_REASON as _HOLD_KIND_REASON,
    _arg_value as _arg_value,
    _cli_list as _cli_list_impl,
    _cli_seats as _cli_seats,
    _flatten_resolved as _flatten_resolved,
    _has_arg as _has_arg,
    _lease_matches_seat as _lease_matches_seat,
    _lease_worker_label as _lease_worker_label,
    _lease_workers_by_pool as _lease_workers_by_pool,
    _pool_key as _pool_key,
    _pool_row_less as _pool_row_less,
    _public_blocked as _public_blocked,
    _route_rank as _route_rank,
    _seat_cooldown as _seat_cooldown,
    _seat_hold_reason as _seat_hold_reason,
    _tier_arg as _tier_arg,
    _unique_pool_rows as _unique_pool_rows,
    _wave_id_for as _wave_id_for,
    allocate_wave as _allocate_wave_impl,
    classify_task as _classify_task_impl,
    live_seat_leases as live_seat_leases,
    resolve_account as _resolve_account_impl,
    route_account as _route_account_impl,
    seat_pool as seat_pool,
)

_UNSET: Any = object()

USER = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))
# opencode keeps its config under XDG_CONFIG_HOME (defaults to ~/.config), not
# the user home like Claude's .claude*. This is the root we glob for opencode
# account dirs (opencode, opencode-<tag>). Overridable for tests/hosts.
CONFIG_HOME = os.environ.get(
    "FLEET_CONFIG_HOME",
    os.environ.get("XDG_CONFIG_HOME") or os.path.join(USER, ".config"),
)
# A dir counts as an opencode *account* when it holds one of these config files
# (the opencode.json/jsonc is the switch seam, the way projects/ is for Claude).
OPENCODE_MARKER_FILES = ("opencode.json", "opencode.jsonc")
# The RUNTIME registry: $FLEET_REG_DIR when the fleet names it, else the host ladder
# (see fleet_regdir). It used to fall back straight to the clone-root tools/_registry,
# which is what let an env-unset run maintain a SECOND registry beside the watchdog's.
REG_DIR = fleet_regdir.reg_dir()
# The account POLICY is operator CONFIG, not runtime STATE, so it must NOT move with
# FLEET_REG_DIR. The watchdog (fleet_status.ps1 / fleet_resume_watchdog.ps1) redirects
# FLEET_REG_DIR to a host state dir (e.g. %LOCALAPPDATA%\Fleet\registry) so the live
# sessions.json lands off the repo -- but the policy deciding which accounts are workers
# is the operator-edited, docs-pinned file in the repo's tools/_registry/. Deriving
# POLICY_PATH from REG_DIR silently pointed the watchdog at a NON-EXISTENT LOCALAPPDATA
# policy, so load_policy() fell back to the committed EXAMPLE -- the operator's host-local
# exclusions (c10, jack-barker) were honored by the CLI but IGNORED by the watchdog, and
# the two surfaces drifted (one offered an account the other tombstoned). Pin the policy
# dir to the repo (override with FLEET_POLICY_DIR / FLEET_POLICY_PATH); only the runtime
# registry below follows FLEET_REG_DIR.
POLICY_DIR = os.environ.get(
    "FLEET_POLICY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry"),
)
POLICY_PATH = os.environ.get(
    "FLEET_POLICY_PATH", os.path.join(POLICY_DIR, "accounts_policy.json"))
# Committed template, used when the operator hasn't created a live policy file
# (the live one lives under the gitignored _registry/). Falls back to the
# built-in DEFAULT_POLICY if neither file is present.
POLICY_EXAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "accounts_policy.example.json")
REGISTRY_PATH = os.path.join(REG_DIR, "sessions.json")
ACCOUNTS_REGISTRY_PATH = os.environ.get(
    "FAK_ACCOUNTS_REGISTRY",
    os.path.join(USER, ".claude-accounts", "registry.json"),
)
RUNS_DIRNAME = ".dispatch-runs"
ACCOUNT_CAP_PREFIX = "account-cap-"
ACCOUNT_CAP_RUNS_DIR = os.environ.get(
    "FLEET_ACCOUNT_CAP_RUNS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), RUNS_DIRNAME),
)

# How fresh an active-probe ledger entry must be to override the registry's carried
# status. The probe LEDGER (account_probe's probe_ledger.jsonl) is a SEPARATE file from
# the watchdog's sessions.json -- a manual or watchdog `account_probe` run writes its
# OK/LIMIT verdict only to the ledger, and nothing folds that back into sessions.json. So
# without consulting the ledger here, runtime_status keeps reporting a stale carried
# throttle ("resets 11pm") for an account a probe just confirmed OK -- the day24 incident,
# where the live probe returned OK but the roster (and every consumer: the switcher AND
# resume_resolver's re-home target search) still saw it blocked, yielding PIN_BLOCKED with
# a healthy worker sitting right there. A probe is ground truth the moment it lands but
# decays: an OK from hours ago must NOT override a real current limit, so we honor a ledger
# verdict only within this window. Matches account_probe's own anti-spam interval scale.
PROBE_LEDGER_FRESH_MIN = float(os.environ.get("FLEET_PROBE_FRESH_MIN", "20"))

# The probe-COVERAGE budget: how old a seat's newest ledger row may be before the ledger stops
# counting as evidence about that seat at all. Distinct from PROBE_LEDGER_FRESH_MIN above,
# which asks a different question -- "does this VERDICT still override the registry" -- on a
# window sized for quota walls. This one asks "has this seat been MEASURED lately", and a seat
# probed OK two hours ago is past the first window and comfortably inside this one.
# accountprobe.SeatCoverageMaxAgeMin is the authority for the value and for why 1440 (24h) is
# a chosen rather than a derived number; keep the two in step.
SEAT_COVERAGE_MAX_AGE_MIN = 1440.0

NIM_DEEPSEEK_V4_PRO_MODEL = "deepseek-ai/deepseek-v4-pro"
NIM_KIMI_K26_MODEL = "moonshotai/kimi-k2.6"
NIM_GLM52_MODEL = "z-ai/glm-5.2"

# NVIDIA-hosted API-demo config homes. DeepSeek/Kimi are restricted by the
# 2026-08-11 outcome audit; GLM retains its prior classification pending audit.
NIM_CODING_SEAT_PROFILES = {
    "nim-deepseek-v4-pro": {
        "model_tier": 3,
        "model": NIM_DEEPSEEK_V4_PRO_MODEL,
        "agent": "opencode",
    },
    "nim-kimi-k26": {
        "model_tier": 3,
        "model": NIM_KIMI_K26_MODEL,
        "agent": "opencode",
    },
    "nim-glm52": {
        "model_tier": 1,
        "model": NIM_GLM52_MODEL,
        "agent": "opencode",
    },
}

DEFAULT_NIM_CODING_ROUTE_WEIGHTS = {
    "opencode:nim-deepseek-v4-pro": 0,
    "opencode:nim-kimi-k26": 0,
    "opencode:nim-glm52": 10,
}

# Built-in defaults, used when no policy file is present. Keep backup or
# break-glass accounts off the auto-resume roster until the operator opts in.
DEFAULT_POLICY = {
    "exclude": ["backup", "breakglass"],
    "include_only": [],
    "notes": {"backup": "break-glass backup account; never auto-resume"},
    # Optional operator overrides keyed by account basename, product:tag, tag,
    # or product. Built-in inference handles the v1 defaults: Claude accounts
    # are tier-1 Opus/xhigh unless obviously local, opencode config models are
    # read from opencode.json, GLM-5.2 is tier 2, and unknowns are tier 3.
    "account_profiles": {},
    # Per-account routing CAPACITY bias, keyed (like account_profiles) by account
    # basename, product:tag, tag, or product -> an int (default 0). HIGHER = more room
    # -> preferred EARLIER in the switcher's tie-break. This is a SEPARATE map from
    # account_profiles ON PURPOSE: the fleet has no remaining-quota telemetry (a probe
    # only returns OK/AUTH/LIMIT, never headroom), so the router otherwise balances
    # purely on live/active SESSION counts and cannot tell which account has more quota
    # left. route_weights is how an operator encodes that out-of-band knowledge without
    # touching an account's model-tier inference (a bare account_profiles entry WOULD
    # clobber the inferred tier and silently drop the account out of routing).
    # e.g. {"gem7": 10} to prefer the roomy account, {"gem8": -10} to push a near-capped one down.
    # The built-in NVIDIA NIM coding seats use this same map for their benchmark rank:
    # opencode-nim-deepseek-v4-pro > opencode-nim-kimi-k26 > opencode-nim-glm52.
    "route_weights": dict(DEFAULT_NIM_CODING_ROUTE_WEIGHTS),
    "routing": {
        "light_confidence": 0.999,
        "hard_tier1_fallback": "stop",
    },
}


def load_policy(path: str = POLICY_PATH) -> dict:
    """Load the operator policy, falling back to DEFAULT_POLICY. A malformed or
    absent file is treated as 'use defaults' -- policy must never crash the
    discovery the rest of the fleet depends on.

    Resolution order: the requested ``path`` (the live, gitignored
    _registry/accounts_policy.json), then the committed
    accounts_policy.example.json, then the built-in DEFAULT_POLICY. The first
    file that exists wins; built-in defaults always backstop missing keys."""
    pol = copy.deepcopy(DEFAULT_POLICY)
    src = path if os.path.exists(path) else (
        POLICY_EXAMPLE_PATH if path == POLICY_PATH and os.path.exists(POLICY_EXAMPLE_PATH)
        else path)
    try:
        with open(src, encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            for k in ("exclude", "include_only"):
                if isinstance(user.get(k), list):
                    pol[k] = [str(x) for x in user[k]]
            if isinstance(user.get("notes"), dict):
                pol["notes"].update({str(k): str(v) for k, v in user["notes"].items()})
            if isinstance(user.get("account_profiles"), dict):
                pol["account_profiles"].update({
                    str(k): v for k, v in user["account_profiles"].items()
                    if isinstance(v, dict)
                })
            if isinstance(user.get("route_weights"), dict):
                pol["route_weights"].update({
                    str(k): v for k, v in user["route_weights"].items()
                })
            if isinstance(user.get("routing"), dict):
                pol["routing"].update(dict(user["routing"]))
    except (OSError, ValueError):
        pass
    return pol


def _excluded_match(tag: str, account: str, exclude: list[str],
                    *identity_values: str) -> str | None:
    """Return the matching exclude substring (for the reason text), or None."""
    haystacks = [tag, account, *identity_values]
    for sub in exclude:
        if not sub:
            continue
        sl = sub.lower()
        if any(sl in str(value).lower() for value in haystacks if value):
            return sub
    return None


def _model_tier_from_name(model: object) -> int:
    """Small v1 model taxonomy.

    Tier 1 is the max-quality frontier set the operator named. Tier 2 is the
    lightweight-work set (GLM-5.2 and Gemini 3.5 Flash). Everything else is tier 3
    until explicitly classified later.
    """
    text = str(model or "").lower().replace("_", "-").replace(" ", "-")
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if "gpt-5.5" in text or "gpt55" in compact:
        return 1
    if "opus-4.6" in text or "opus46" in compact or text in ("opus", "claude-opus"):
        return 1
    if ("deepseek-v4-pro" in text or "deepseekv4pro" in compact
            or "kimi-k2.6" in text or "kimik26" in compact):
        return 1
    if "glm-5.2" in text or "glm52" in compact:
        return 2
    # Gemini 3.5 Flash — Google's fast/lightweight tier, served via GCP Vertex AI on
    # the OpenAI-compatible endpoint as `google/gemini-3.5-flash`. Lightweight, tier 2.
    if "gemini-3.5-flash" in text or "gemini35flash" in compact:
        return 2
    return 3


def _safe_opencode_models(acct_dir: str) -> dict[str, str]:
    """Read only model identifiers from opencode config files.

    Provider credentials and other options are deliberately ignored.
    """
    for marker in OPENCODE_MARKER_FILES:
        path = os.path.join(acct_dir, marker)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        out = {}
        for key in ("model", "small_model"):
            value = doc.get(key)
            if value:
                out[key] = str(value)
        return out
    return {}


def _clean_profile(raw: dict | None, *, source: str) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    model = str(raw.get("model") or "")
    profile = {
        "model_tier": _as_int(raw.get("model_tier", raw.get("tier")), 0),
        "model": model,
        "small_model": str(raw.get("small_model") or ""),
        "model_effort": str(raw.get("effort") or raw.get("model_effort") or ""),
        "agent": str(raw.get("agent") or ""),
        "profile_source": source,
    }
    if profile["model_tier"] not in (1, 2, 3):
        profile["model_tier"] = _model_tier_from_name(model)
    if profile["model_tier"] not in (1, 2, 3):
        profile["model_tier"] = 3
    return profile


def account_profile(row: dict, policy: dict | None = None) -> dict:
    """Return the model-routing profile for an account row.

    Operator policy can override by exact account, ``product:tag``, short tag, or
    product. Defaults are intentionally conservative: unknown models are tier 3.
    """
    pol = policy or load_policy()
    profiles = pol.get("account_profiles", {})
    product = str(row.get("product") or account_product(str(row.get("account") or "")))
    account = str(row.get("account") or "")
    tag = str(row.get("tag") or account_tag(account))
    for key in (account, f"{product}:{account}", f"{product}:{tag}", tag, product):
        if isinstance(profiles, dict) and key in profiles:
            return _clean_profile(profiles.get(key), source=f"policy:{key}")

    if product == "claude":
        localish = "local" in tag.lower() or "faklocal" in account.lower()
        if localish:
            return _clean_profile({
                "model_tier": 3,
                "model": "local",
                "agent": "claude",
            }, source="default:claude-local")
        return _clean_profile({
            "model_tier": 1,
            "model": "opus",
            "effort": "xhigh",
            "agent": "claude",
        }, source="default:claude-opus")

    if product == "opencode":
        models = _safe_opencode_models(str(row.get("dir") or ""))
        seat = NIM_CODING_SEAT_PROFILES.get(tag.lower())
        if seat:
            raw = dict(seat)
            if not raw.get("small_model"):
                raw["small_model"] = models.get("small_model", "")
            return _clean_profile(raw, source=f"default:nvidia-nim-coding:{tag}")
        model = models.get("model", "")
        tier = _model_tier_from_name(model)
        if tier == 3 and ("glm" in tag.lower() or "zai" in tag.lower()
                          or "glm" in account.lower() or "zai" in account.lower()):
            model = model or "zai-coding-plan/glm-5.2"
            tier = 2
        return _clean_profile({
            "model_tier": tier,
            "model": model,
            "small_model": models.get("small_model", ""),
            "agent": "opencode",
        }, source="default:opencode-config")

    return _clean_profile({"model_tier": 3, "agent": product}, source="default:unknown")


def account_route_weight(row: dict, policy: dict | None = None) -> int:
    """Operator capacity bias for an account's routing tie-break (default 0).

    Resolved from the policy ``route_weights`` map by the same key precedence as
    account_profiles (exact account, ``product:account``, ``product:tag``, short tag,
    product). HIGHER = more room -> the switcher prefers it EARLIER. Kept SEPARATE from
    account_profiles so setting a weight never disturbs an account's model-tier
    inference. The fleet has no remaining-quota signal, so this is the only channel for
    "prefer the account with more room left" (see ``_route_rank``)."""
    pol = policy or load_policy()
    weights = pol.get("route_weights", {})
    if not isinstance(weights, dict) or not weights:
        return 0
    product = str(row.get("product") or account_product(str(row.get("account") or "")))
    account = str(row.get("account") or "")
    tag = str(row.get("tag") or account_tag(account))
    for key in (account, f"{product}:{account}", f"{product}:{tag}", tag, product):
        if key in weights:
            return _as_int(weights.get(key), 0)
    return 0


def load_registry(path: str = REGISTRY_PATH) -> dict:
    """Best-effort read of the live session registry.

    The static roster must keep working when the watchdog has not produced a
    registry yet, so missing/malformed data becomes an empty dict.
    """
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def load_accounts_registry(path: str | None = None, *, home: str = USER) -> dict:
    """Best-effort read of the canonical ``fak accounts`` registry.

    ``fleet_accounts`` still discovers config dirs directly so it can scan live
    transcripts, but lifecycle truth lives in ``fak accounts``. A tombstoned seat
    with a remaining ``projects/`` dir must not re-enter Slack/routing.
    """
    p = path or os.environ.get(
        "FAK_ACCOUNTS_REGISTRY",
        os.path.join(home, ".claude-accounts", "registry.json"),
    )
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _registry_age_min(registry: dict) -> float | None:
    raw = registry.get("generated_utc")
    if not raw:
        return None
    try:
        ts = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return round((dt.datetime.now(dt.timezone.utc) - ts).total_seconds() / 60.0, 1)
    except ValueError:
        return None


def _normalize_auth(auth: dict | None) -> dict:
    out = {}
    for account, info in (auth or {}).items():
        if isinstance(info, dict):
            row = dict(info)
        else:
            row = {"block_reason": str(info) if info else "auth/login required"}
        reason_text = " ".join(
            str(row.get(k) or "") for k in ("last", "reason", "block_reason")
        )
        if reason_text.strip():
            row["block_kind"] = fleet_session_signals.auth_block_kind(reason_text)
            row["block_reason"] = fleet_session_signals.auth_block_reason(reason_text)
        else:
            row.setdefault("block_kind", "auth")
            row.setdefault("block_reason", "auth/login required")
        out[account] = row
    return out


# Map account_probe's closed STATUS set onto the (available, block_kind) shape runtime_status
# uses, so a ledger verdict rides the SAME fresh-probe override the _probe session row does.
_PROBE_STATUS_KIND = {"AUTH": "auth", "ACCESS": "access", "CREDIT": "credit", "LIMIT": "usage"}
_IDENTITY_KEYS = ("account_uuid", "login_email", "org_uuid")
_IDENTITY_ALIASES = {
    "account_uuid": ("account_uuid", "accountUuid"),
    "login_email": ("login_email", "emailAddress", "email"),
    "org_uuid": ("org_uuid", "organizationUuid"),
    "token_fp": ("token_fp", "tokenFP"),
}


def _identity_account_key(identity: dict) -> str:
    if not isinstance(identity, dict):
        return ""
    uuid = str(identity.get("account_uuid") or "").strip().lower()
    if uuid:
        return "uuid:" + uuid
    token_fp = str(identity.get("token_fp") or identity.get("tokenFP") or "").strip().lower()
    if token_fp:
        return "tok:" + token_fp
    return ""


def _path_key(path: str) -> str:
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(path))


def _accounts_registry_reason(home: dict, default: str) -> str:
    reason = str(home.get("tombstone_reason") or default)
    rehome = str(home.get("rehome_to") or "")
    if rehome and "rehome" not in reason.lower():
        reason += f"; rehome -> {rehome}"
    return reason


def _accounts_registry_index(registry: dict | None) -> dict:
    """Index active/tombstoned homes from ``fak accounts`` registry.json."""
    idx = {
        "active_names": set(),
        "active_dirs": set(),
        "tombstoned_names": {},
        "tombstoned_dirs": {},
        "tombstoned_identities": {},
    }
    if not isinstance(registry, dict):
        return idx
    for h in registry.get("homes") or []:
        if not isinstance(h, dict):
            continue
        status = str(h.get("status") or "active").strip().lower()
        name = str(h.get("name") or "").strip()
        dkey = _path_key(str(h.get("dir") or ""))
        identity_key = _identity_account_key(_account_identity_from(h))
        if status == "tombstoned":
            if name:
                idx["tombstoned_names"][name.lower()] = h
            if dkey:
                idx["tombstoned_dirs"][dkey] = h
            if identity_key:
                idx["tombstoned_identities"][identity_key] = h
        else:
            if name:
                idx["active_names"].add(name.lower())
            if dkey:
                idx["active_dirs"].add(dkey)
    return idx


def _accounts_registry_lookup_keys(row: dict) -> set[str]:
    account = str(row.get("account") or "")
    tag = str(row.get("tag") or account_tag(account))
    product = str(row.get("product") or account_product(account))
    keys = {k.lower() for k in (account, tag) if k}
    if product == "claude":
        if account == ".claude":
            keys.add("default")
        elif account.startswith(".claude-"):
            keys.add(account[len(".claude-"):].lower())
    return keys


def _accounts_registry_exclusion(row: dict, registry_index: dict | None) -> str:
    if not registry_index:
        return ""
    keys = _accounts_registry_lookup_keys(row)
    dkey = _path_key(str(row.get("dir") or ""))
    tomb_names = registry_index.get("tombstoned_names") or {}
    tomb_dirs = registry_index.get("tombstoned_dirs") or {}
    for key in keys:
        if key in tomb_names:
            return _accounts_registry_reason(
                tomb_names[key], "tombstoned in fak accounts registry")
    if dkey and dkey in tomb_dirs:
        return _accounts_registry_reason(
            tomb_dirs[dkey], "tombstoned in fak accounts registry")

    # Exact active registry entries stay valid even if an older tombstone shares the
    # same account identity. Unknown aliases with that retired identity are excluded.
    if keys & set(registry_index.get("active_names") or set()):
        return ""
    if dkey and dkey in (registry_index.get("active_dirs") or set()):
        return ""

    identity_key = _identity_account_key(_account_identity_from(row))
    tomb_identities = registry_index.get("tombstoned_identities") or {}
    if identity_key and identity_key in tomb_identities:
        return _accounts_registry_reason(
            tomb_identities[identity_key],
            "same account identity as tombstoned fak accounts registry seat",
        )
    return ""


# annotate_accounts calls runtime_status once per worker, so a naive per-call ledger read
# re-parses the whole probe_ledger.jsonl ~10x per roster render. Memoize the parsed snapshot
# keyed on the ledger file's (mtime, size): a render reads it once, and a new probe (which
# changes size) invalidates it immediately. Keeps runtime_status's signature unchanged.
_PROBE_LEDGER_CACHE: dict = {"key": None, "by_account": {}, "ages": {}, "obs": {}}


def _probe_ledger_snapshot():
    """(latest-entry-by-account, age-min-by-account) from account_probe's ledger, memoized on
    the file's mtime+size so one roster render parses it once. Empty on any read failure.

    The same key-gated parse also folds each account's FULL history into a _CapObservation
    (OK streak + blocked-episode start) under the cache's ``obs`` map, read via _cap_observation
    -- so the cap-disambiguation cycles cost no extra ledger read per roster row."""
    try:
        import account_probe  # lazy: account_probe imports fleet_accounts, not vice-versa
    except ImportError:
        return {}, {}
    path = account_probe.probe_ledger_path()
    try:
        st = os.stat(path)
        key = (st.st_mtime, st.st_size)
    except OSError:
        return {}, {}
    if _PROBE_LEDGER_CACHE["key"] != key:
        try:
            by_account = account_probe.last_probe_by_account()
            ages = {a: account_probe.recent_probe_age_min(a) for a in by_account}
            obs = _derive_cap_observations(account_probe.read_ledger())
        except Exception:
            by_account, ages, obs = {}, {}, {}
        _PROBE_LEDGER_CACHE.update(key=key, by_account=by_account, ages=ages, obs=obs)
    return _PROBE_LEDGER_CACHE["by_account"], _PROBE_LEDGER_CACHE["ages"]


def _cap_observation(account: str) -> "_CapObservation":
    """The ledger-derived _CapObservation for one account (zero observation when the ledger is
    absent/unreadable or the account has no history -- which keeps _disambiguate_cap on its
    legacy single-shot path)."""
    _probe_ledger_snapshot()  # refresh the memoized cache for the current ledger file
    return _PROBE_LEDGER_CACHE.get("obs", {}).get(account) or _CapObservation()


def _fresh_probe_from_ledger(account: str, fresh_min: float = PROBE_LEDGER_FRESH_MIN
                             ) -> dict | None:
    """The freshest active-probe verdict for ``account`` from account_probe's ledger, IF it
    is within ``fresh_min`` minutes -- else ``None``.

    This is the missing link between the prober and the roster: ``account_probe`` writes its
    OK/LIMIT/AUTH verdict to ``probe_ledger.jsonl``, a file distinct from the watchdog's
    ``sessions.json`` that ``runtime_status`` reads. Nothing folds the ledger back into the
    registry, so a fresh probe was invisible to every roster consumer (the day24 stale-throttle
    incident). Consulting it here lets a recent probe override a carried block for the switcher
    AND resume_resolver's re-home search alike, with one freshness gate so a stale OK can't
    mask a real current limit. Returns a record shaped for the fresh-probe branch."""
    by_account, ages = _probe_ledger_snapshot()
    entry = by_account.get(account)
    if not entry:
        return None
    age = ages.get(account)
    if age is None or age > fresh_min:
        return None
    status = str(entry.get("status") or "").upper()
    identity = _account_identity_from(entry)
    if status == "OK":
        return {"available": True, "age_min": age, **identity}
    if status in _PROBE_STATUS_KIND:
        kind = _PROBE_STATUS_KIND[status]
        reset = entry.get("reset")
        reason = entry.get("block_reason") or entry.get("reason")
        if not reason:
            reason = (f"usage limit; resets {reset}" if reset else "usage limit") \
                if kind == "usage" else f"{kind} block"
        # Carry the recorded cooldown START (`since`) alongside the reason and the
        # next-eligible windows (#1801): append_probe_ledger stamps `ts` when the block
        # was first observed, but the fresh-probe verdict used to drop it -- so a roster
        # consumer could say WHY a seat is cooling and WHEN it reopens, but not since when.
        # Surfacing it completes the recorded triad (start/reason/next-eligible) end-to-end.
        return {"available": False, "block_kind": kind, "block_reason": reason,
                "reset": reset, "weekly": entry.get("weekly"),
                "since": entry.get("ts"), "age_min": age,
                **identity}
    # Any other status (APIERR/TRANSPORT/unknown) is not a clean availability signal --
    # fall through to the registry's own status rather than inventing a verdict.
    return None


def _account_identity_from(info: dict | None) -> dict:
    if not isinstance(info, dict):
        return {}
    sources = [info]
    for nested_key in ("identity", "account_identity", "oauthAccount"):
        nested = info.get(nested_key)
        if isinstance(nested, dict):
            sources.append(nested)
    out = {}
    for key, aliases in _IDENTITY_ALIASES.items():
        for src in sources:
            for alias in aliases:
                value = src.get(alias)
                if value:
                    out[key] = str(value).strip().lower()
                    break
            if key in out:
                break
    return out


def _identity_match(left: dict, right: dict) -> bool | None:
    for key in _IDENTITY_KEYS:
        if left.get(key) and right.get(key):
            return left[key] == right[key]
    return None


def _registry_account_identity(registry: dict | None, account: str) -> dict:
    if not isinstance(registry, dict):
        return {}
    for row in registry.get("accounts") or []:
        if isinstance(row, dict) and row.get("account") == account:
            return _account_identity_from(row)
    return {}


def _current_config_identity(account: str) -> dict:
    try:
        return _account_identity_from(read_account_identity(os.path.join(USER, account)))
    except Exception:
        return {}


def _throttle_matches_current_identity(account: str, throttle_info: dict,
                                       registry: dict | None,
                                       acct_sessions: list[dict],
                                       probe_identity: dict | None = None) -> bool:
    throttle_identity = _account_identity_from(throttle_info)
    if not throttle_identity:
        return True
    candidates = [
        probe_identity or {},
        _current_config_identity(account),
        _registry_account_identity(registry, account),
    ]
    candidates.extend(
        _account_identity_from(s) for s in sorted(
            acct_sessions, key=lambda row: _age_min(row) if _age_min(row) is not None else 10**9
        )
    )
    for candidate in candidates:
        verdict = _identity_match(throttle_identity, candidate)
        if verdict is not None:
            return verdict
    return True


def _apply_throttle_status(status: dict, throttle_info: dict) -> dict:
    reset = throttle_info.get("reset")
    weekly = throttle_info.get("weekly")
    reason = f"usage limit; resets {reset}" if reset else "usage limit"
    if weekly:
        reason += f"; weekly {weekly}"
    status.update({
        "available": False,
        "blocked": True,
        "block_kind": "usage",
        "block_reason": reason,
        "reset": reset,
        "weekly": weekly,
        # Cooldown START, when the throttle ledger record carried one (#1801). The
        # throttle map is read whole via _normalize_throttle, so a `since` the recorder
        # stamped survives here; None when it did not, so this stays forward-compatible.
        "throttled_since": throttle_info.get("since"),
        "throttled": True,
    })
    return status


def _registry_blocks_derivable() -> bool:
    """Whether the registry dir THIS process reads can derive a seat BLOCK at all.

    Mirror of accountprobe.RegChoice.BlocksDerivable, graded the way statRegSite grades a
    site: a dir carries a derivable block verdict exactly when ``probe_ledger.jsonl`` is
    present in it. That file is the ONLY place a block can come from -- account_probe appends
    its OK/LIMIT/AUTH lines there and nothing folds them into the watchdog's sessions.json --
    so a dir holding sessions.json with no ledger beside it reports "nothing blocked" when it
    means "cannot tell". Reporting the first as the second is the shape of #5390.

    The dir is account_probe.probe_ledger_path()'s, i.e. exactly the one
    _probe_ledger_snapshot would then read, so the gate and the read never disagree about
    which registry the answer is about."""
    try:
        import account_probe  # lazy: account_probe imports fleet_accounts, not vice-versa
    except ImportError:
        return False
    try:
        return os.path.isfile(account_probe.probe_ledger_path())
    except OSError:
        return False


def _should_consult_probe_ledger(registry: dict | None, probe_ledger: bool | None) -> bool:
    """Whether the probe-ledger rung may run at all (mirror of Go shouldConsultProbeLedger).

    The predicate is DERIVABILITY: the registry dir this process would actually read must
    carry the probe ledger. It used to be ``bool(os.environ.get("FLEET_REG_DIR"))``, on the
    reasoning that the env var is what makes the Python writer and the Go reader agree on a
    dir. That reasoning does not survive #5390 -- naming a dir is not the same as the dir
    holding a ledger -- and the Go twin migrated off it in #5439; this is the Python half of
    the same fix, so the two surfaces answer one question the same way:

      - FLEET_REG_DIR names a ledger-bearing dir -> consulted, exactly as before.
      - FLEET_REG_DIR unset but the dir the host ladder resolves to (see fleet_regdir)
        carries the ledger -> consulted NOW, where before the rung never ran and a fresh
        probe verdict stayed invisible to the roster no matter how correctly it resolved.
      - FLEET_REG_DIR names a ledger-LESS dir -> NOT consulted, where before the rung ran
        over a ledger that was not there. Every read returned nothing, so the fold fell
        through to the carried block anyway; refusing to run says so instead of pretending
        to have looked. This is the one case whose verdict the change FLIPS.
      - Nothing anywhere -> not consulted, exactly as before, so a fresh checkout and CI keep
        the pure passive fold.

    The price is that the answer is a function of the filesystem rather than of one env var:
    a caller (or a test) wanting a definite verdict must arrange the dir, not just the
    variable. Repeated calls over an unchanged filesystem are stable -- no clock is consulted.

    The two rungs ABOVE the predicate are caller affordances the Go twin has no equivalent of
    (its callers cannot override the resolution): an explicit ``probe_ledger`` wins outright,
    and ``registry is None`` means the live front door already decided to read the live
    registry. Only the fall-through is the ported decision."""
    if probe_ledger is not None:
        return probe_ledger
    if registry is None:
        return True
    return _registry_blocks_derivable()


def _mark_usage_soon(status: dict, throttle_info: dict | None) -> None:
    """Carry a still-active DAILY usage cap onto ``status`` as an advisory.

    A fresh OK probe reopening a seat is correct for AVAILABILITY -- a healthy seat must not
    sit blocked behind a stale/near-expired carried cap (the day24 incident). But the reopen
    used to DROP the cap entirely, so a seat sitting at its daily limit and about to roll over
    showed as a plain ``serving`` row with no usage at all -- an operator could not see it was
    one request from the wall. Surface the still-future daily ``reset`` as advisory only:
    ``available``/``throttled`` are left untouched (the seat stays offered), a consumer/roster
    just gains a ``usage_soon_reset`` it can render as "serving, cap resets HH:MM". A weekly cap
    never reaches here (it holds the seat closed upstream), and an absent/already-expired cap
    contributes nothing -- so a normal serving row gains no key and its JSON is unchanged.
    """
    if not throttle_info or not throttle_is_active(throttle_info):
        return
    if _weekly_throttle_is_active(throttle_info):
        return
    reset = throttle_info.get("reset")
    if reset:
        status["usage_soon_reset"] = reset


def _seat_probe_unmeasured(account: str) -> bool:
    """Whether a BUSY probe ledger -- one carrying rows for at least one account -- holds no
    current evidence about THIS seat: no row at all, or a newest row past
    SEAT_COVERAGE_MAX_AGE_MIN (or one whose timestamp will not parse, which is the same thing
    here, since an undatable row is not evidence about now).

    Mirror of Go seatProbeUnmeasured, and the whole of #5391: on the host that filed it the
    ledger was present, derivable and busy -- ``opencode-*`` rows current to the minute --
    while several claude seats' newest rows were 8-9 days old. Every registry-level question
    answered "yes, blocks are derivable here", and none of that was evidence about those seats.
    They read available, and because a 403 burns no quota the headroom-weighted allocator
    PREFERRED the one seat whose org had disabled access.

    The "busy" precondition is deliberate and is why this does not re-open #5439's boundary. A
    ledger that has recorded NOTHING is already described by the registry-level judgement, and
    a seat-level downgrade there would only restate it. A ledger that has recorded rows for
    OTHER accounts is the case where the registry-level answer is affirmatively misleading:
    the prober demonstrably ran and demonstrably skipped this seat. Only the second moves.

    Costs no extra ledger read -- _probe_ledger_snapshot is the memoized parse the fresh-probe
    rung already paid for on this render."""
    by_account, ages = _probe_ledger_snapshot()
    if not by_account:
        return False
    if account not in by_account:
        return True
    age = ages.get(account)
    return age is None or age > SEAT_COVERAGE_MAX_AGE_MIN


def _mark_unknown_health(status: dict, blocks_derivable: bool,
                         seat_unmeasured: bool = False) -> dict:
    """Weaken an UNBLOCKED seat's ``status_source`` from the confident ``registry`` to
    ``registry-unknown`` when it is being published on probe evidence that does not exist
    (mirror of Go markUnknownHealth). Two disjoint absences reach the same verdict:

      - ``blocks_derivable`` false -- the registry itself cannot derive a block at all (see
        _registry_blocks_derivable).
      - ``seat_unmeasured`` true -- the registry CAN derive blocks and its ledger is busy, but
        that ledger holds no current row for this particular seat (see _seat_probe_unmeasured).
        #5391: "never probed" and "probed OK" must not both read as a proven-free seat just
        because the prober is healthy for some other account class.

    The second is the narrower and later of the two, and it is what keeps the first from being
    read as a sufficient test: a registry-wide grade cannot see a per-class coverage hole.

    Unknown-health is a THIRD state, and deliberately NOT a block. Converting absence into
    blocked would strand every seat on a host whose prober has not run -- and worse, it is
    self-sealing: the roster is what routes the work that runs the probe, so a block imposed
    for want of a probe forbids the very probe that would clear it. So the seat stays offered
    (``available`` is untouched) and only the CLAIM is weakened. A consumer that cannot
    tolerate an unproven seat now has a name to switch on; one that does not care keeps
    today's behavior, since every existing status_source consumer treats an unrecognized
    value exactly as it treats ``registry``.

    A BLOCKED seat keeps ``registry``: blocked is a positive derivation from the registry's
    own throttle/auth rows, not a statement about probe evidence, so its provenance is not in
    doubt. An empty registry keeps ``none``, which already says nothing was consulted."""
    if status.get("blocked") or status.get("status_source") != "registry":
        return status
    if blocks_derivable and not seat_unmeasured:
        return status
    status["status_source"] = "registry-unknown"
    return status


def runtime_status(account: str, registry: dict | None = None,
                   throttle: dict | None = None,
                   sessions: list[dict] | None = None,
                   probe_ledger: bool | None = None) -> dict:
    """Return the live availability status for one account basename.

    Static policy answers "is this a real worker account?". Runtime status answers
    "should the switcher offer it right now?". Usage limits and auth/credit blocks
    are account-wide blockers; completed/user-closed sessions are not.
    """
    reg = load_registry() if registry is None else registry
    throttle_map = _normalize_throttle(throttle if throttle is not None else reg.get("throttle"))
    auth_map = _normalize_auth(reg.get("auth") if isinstance(reg, dict) else {})
    generated_utc = reg.get("generated_utc") if isinstance(reg, dict) else None
    sess = sessions if sessions is not None else reg.get("sessions", [])
    acct_sessions = [s for s in (sess or []) if s.get("account") == account]

    # SINGLE SOURCE OF TRUTH: a fresh ACTIVE PROBE is the most authoritative signal -- it
    # literally hit the live account just now. If a probe says OK, it OVERRIDES a stale
    # carried throttle/auth (e.g. a usage limit that belonged to a DIFFERENT account the dir
    # was logged into before a re-login). The probe row is synthetic (project == "_probe").
    probe_rows = [s for s in acct_sessions if s.get("project") == "_probe"]
    fresh_probe_ok = any(
        str(s.get("probe_status") or "").upper() == "OK"
        or (s.get("disp") == "LIVE" and not s.get("probe_status"))
        for s in probe_rows
    ) if probe_rows else False
    fresh_probe_block = next(
        (s for s in probe_rows
         if str(s.get("probe_status") or "").upper() not in ("", "OK")), None)

    active = sum(1 for s in acct_sessions if s.get("disp") not in ("DONE", "USER_CLOSED"))
    live = sum(1 for s in acct_sessions if s.get("disp") == "LIVE")
    auth_blocked = [
        s for s in acct_sessions
        if s.get("action") == "BLOCKED_AUTH" or s.get("disp") == "INFRA_AUTH"
    ]
    latest_auth_age = min(
        (age for s in auth_blocked if (age := _age_min(s)) is not None),
        default=None,
    )
    latest_success_age = min(
        (
            age
            for s in acct_sessions
            if s.get("disp") in ("LIVE", "DONE") and (age := _age_min(s)) is not None
        ),
        default=None,
    )
    session_auth_current = bool(
        auth_blocked and (
            latest_success_age is None or
            latest_auth_age is None or
            latest_success_age > latest_auth_age
        )
    )
    latest_success_seen = max(
        (
            seen
            for s in acct_sessions
            if s.get("disp") in ("LIVE", "DONE")
            if (seen := _row_seen_utc(s, generated_utc)) is not None
        ),
        default=None,
    )
    auth_info = auth_map.get(account)
    auth_seen = _parse_utc(auth_info.get("seen_utc")) if auth_info else None
    known_auth_current = bool(
        auth_info and (
            latest_success_seen is None or
            auth_seen is None or
            latest_success_seen <= auth_seen
        )
    )
    auth_current = session_auth_current or known_auth_current

    status = {
        "available": True,
        "blocked": False,
        "block_kind": None,
        "block_reason": "",
        "reset": None,
        "weekly": None,
        "throttled": False,
        "active_sessions": active,
        "live_sessions": live,
        "auth_blocked_sessions": len(auth_blocked),
        "status_source": "registry" if reg else "none",
        "registry_age_min": _registry_age_min(reg) if reg else None,
    }
    thr = throttle_map.get(account)

    # A fresh probe is authoritative. OK -> available now (clears stale carried blocks);
    # a fresh probe BLOCK -> that block, with the probe's own (account-correct) reason,
    # never a stale dir-keyed throttle from a previous login. The exception is a carried
    # weekly cap whose reset is still active for the same account identity: an OK probe
    # during the probe-freshness window must not reopen that seat before the weekly reset.
    if fresh_probe_ok:
        probe_identity = {}
        for row in probe_rows:
            if str(row.get("probe_status") or "").upper() == "OK" \
                    or (row.get("disp") == "LIVE" and not row.get("probe_status")):
                probe_identity = _account_identity_from(row)
                break
        if thr and _weekly_throttle_is_active(thr) \
                and _throttle_matches_current_identity(
                    account, thr, reg, acct_sessions, probe_identity):
            return _apply_throttle_status(status, thr)
        _mark_usage_soon(status, thr)
        status["status_source"] = "probe"
        return status
    if fresh_probe_block is not None:
        kind = {"AUTH": "auth", "ACCESS": "access", "CREDIT": "credit",
                "LIMIT": "usage"}.get(
                    str(fresh_probe_block.get("probe_status")).upper(), "auth")
        status.update({
            "available": False, "blocked": True, "block_kind": kind,
            "block_reason": str(fresh_probe_block.get("reason")
                                or fresh_probe_block.get("last") or "blocked"),
            "reset": fresh_probe_block.get("throttle_reset"),
            "weekly": fresh_probe_block.get("throttle_weekly"),
            "throttled": kind == "usage",
            "status_source": "probe",
        })
        return status

    # No synthetic _probe session row in the registry -> consult the active-probe LEDGER
    # directly. account_probe writes its verdict only there, NOT into sessions.json, so a
    # fresh manual/watchdog probe would otherwise be invisible here and the carried throttle
    # below would win (the day24 incident: probe OK, roster still "resets 11pm"). A ledger
    # verdict within PROBE_LEDGER_FRESH_MIN is treated as the same authoritative fresh probe.
    #
    # The cap-disambiguation cycles (aging + probe-override) fold a _CapObservation drawn from
    # the SAME probe ledger, so gate it on the same switch: with the ledger unconsulted (or no
    # carried throttle) cap_obs is the zero observation and _disambiguate_cap stays on its
    # legacy single-shot path at both seams below. Mirrors computeRuntimeStatus in Go.
    consult_ledger = _should_consult_probe_ledger(registry, probe_ledger)
    cap_obs = _cap_observation(account) if (consult_ledger and thr) else _CapObservation()
    if consult_ledger:
        led = _fresh_probe_from_ledger(account)
        if led is not None:
            if led.get("available"):
                # A fresh OK must not reopen a still-active WEEKLY cap for the seat's CURRENT
                # login; a >= _OVERRIDE_STREAK run of OKs past a passed daily reset overturns a
                # stale/unparseable weekly it has outgrown (folded by _disambiguate_cap). With no
                # streak the observation is inert and this equals _weekly_throttle_is_active(thr).
                if thr and _disambiguate_cap(thr, cap_obs).weekly_active \
                        and _throttle_matches_current_identity(
                            account, thr, reg, acct_sessions, led):
                    return _apply_throttle_status(status, thr)
                _mark_usage_soon(status, thr)
                status["status_source"] = "probe-ledger"
                status["probe_age_min"] = led.get("age_min")
                return status
            kind = led.get("block_kind") or "auth"
            status.update({
                "available": False, "blocked": True, "block_kind": kind,
                "block_reason": led.get("block_reason") or "blocked",
                "reset": led.get("reset"), "weekly": led.get("weekly"),
                "throttled": kind == "usage",
                "status_source": "probe-ledger", "probe_age_min": led.get("age_min"),
            })
            return status

    # Carried-throttle fallback (no fresh probe verdict). The aging valve lives here: a seat
    # blocked past _WEEKLY_MAX_AGE with a stale/unparseable weekly and no live daily leg has
    # outlived any real weekly window, so _disambiguate_cap releases it via the derived episode
    # start. Absent that history (or a young/parseable-future cap) this equals throttle_is_active.
    if thr and _disambiguate_cap(thr, cap_obs).active:
        return _apply_throttle_status(status, thr)

    if auth_current:
        last = " ".join(str(s.get("last") or s.get("reason") or "") for s in auth_blocked)
        if known_auth_current and not session_auth_current:
            kind = str(auth_info.get("block_kind") or "auth")
            reason = str(auth_info.get("block_reason") or fleet_session_signals.auth_block_reason(""))
        else:
            kind = fleet_session_signals.auth_block_kind(last)
            reason = fleet_session_signals.auth_block_reason(last)
        status.update({
            "available": False,
            "blocked": True,
            "block_kind": kind,
            "block_reason": reason,
        })
    # Nothing above found a block. If the registry could not have derived one (#5439) -- or if
    # the busy ledger it derives them from has no current row for THIS seat (#5391) -- say so
    # rather than publishing a seat as proven-free on evidence that was never available. The
    # seat grade is read only when it can still change the answer, so a blocked seat (whose
    # provenance is not in doubt) and a ledger-less registry (already answered) skip it.
    seat_unmeasured = False
    if consult_ledger and not status.get("blocked") \
            and status.get("status_source") == "registry":
        seat_unmeasured = _seat_probe_unmeasured(account)
    return _mark_unknown_health(status, consult_ledger, seat_unmeasured)


def annotate_accounts(rows: list[dict], registry: dict | None = None,
                      throttle: dict | None = None,
                      sessions: list[dict] | None = None,
                      probe_ledger: bool | None = None,
                      cap_runs_dir: str | os.PathLike[str] | None = None,
                      seat_leases: list[dict] | None = None) -> list[dict]:
    """Attach live availability fields to discover_accounts() rows.

    ``seat_leases`` (as ``live_seat_leases`` returns) reflects live DISPATCHED opencode
    workers into the roster: the watchdog session scan reads Claude transcripts, so an
    opencode/glm worker -- which writes an opencode transcript the scan does not fold --
    would otherwise show 0 live_sessions even while actively resolving a docs issue. When
    provided (the live front door passes it), a leased opencode seat becomes visible the
    same way a claude worker is. ``None`` (every hermetic caller) leaves counts untouched."""
    raw_reg = load_registry() if registry is None else registry
    reg = copy.deepcopy(raw_reg) if isinstance(raw_reg, dict) else {}
    effective_probe_ledger = True if registry is None and probe_ledger is None else probe_ledger
    effective_throttle = _normalize_throttle(
        throttle if throttle is not None
        else (reg.get("throttle") if isinstance(reg, dict) else None)
    )
    effective_throttle.update(
        active_account_cap_throttles(rows, runs_dir=cap_runs_dir))
    if effective_throttle:
        reg["throttle"] = effective_throttle
    out = []
    for row in rows:
        r = dict(row)
        if r.get("kind") == "worker":
            r.update(runtime_status(
                r["account"], reg, sessions=sessions,
                probe_ledger=effective_probe_ledger))
        else:
            r.update({
                "available": False,
                "blocked": False,
                "block_kind": None,
                "block_reason": r.get("reason", ""),
                "reset": None,
                "weekly": None,
                "throttled": False,
                "active_sessions": 0,
                "live_sessions": 0,
                "auth_blocked_sessions": 0,
                "status_source": "static",
                "registry_age_min": None,
            })
        out.append(r)
    _fold_dispatch_leases(out, seat_leases)
    _reconcile_identity_peer_availability(out)
    out.sort(key=lambda r: (r.get("product", ""),
                            r["kind"] != "worker",
                            not r.get("available", False),
                            r["tag"]))
    return out


def _fold_dispatch_leases(rows: list[dict], leases: list[dict] | None) -> None:
    """Bump opencode worker rows' live/active counts from live dispatch-lease sidecars.

    Scoped to opencode seats ON PURPOSE: claude workers already surface through the
    watchdog session registry, so this only fills the gap the registry cannot -- a
    dispatched opencode/glm worker whose transcript the session scan does not fold. Matches
    a lease to a row by config dir (precise), falling back to tag; bumps a count only UPWARD
    and never touches availability/blocks, so it can add capacity visibility, never hide a
    block. Best effort: a malformed lease contributes nothing."""
    if not leases:
        return
    from collections import Counter
    by_dir: Counter = Counter()
    by_tag: Counter = Counter()
    for lease in leases:
        raw_dir = str(lease.get("dir") or "")
        if raw_dir:
            by_dir[os.path.normcase(os.path.normpath(raw_dir))] += 1
        tag = str(lease.get("tag") or "")
        if tag:
            by_tag[tag] += 1
    for r in rows:
        if r.get("kind") != "worker":
            continue
        if str(r.get("product") or "claude").lower() != "opencode":
            continue
        rdir = str(r.get("dir") or "")
        n = by_dir.get(os.path.normcase(os.path.normpath(rdir)), 0) if rdir else 0
        if not n:
            n = by_tag.get(str(r.get("tag") or ""), 0)
        if n and int(r.get("live_sessions") or 0) < n:
            r["live_sessions"] = n
            r["active_sessions"] = max(int(r.get("active_sessions") or 0), n)
            r["dispatch_leases"] = n  # provenance: count came from .pid/.account sidecars


def _reconcile_identity_peer_availability(rows: list[dict]) -> None:
    """Promote the canonical dir when a same-account duplicate has fresher success.

    ``fak accounts`` and identity reconciliation deliberately collapse several dirs onto
    one Anthropic account bucket. Runtime status is still keyed by dir basename, so a stale
    auth block on the canonical dir can hide the whole bucket even while a duplicate/default
    dir is actively serving it. Only auth-shaped blocks are repaired here; usage/access/credit
    walls remain account-wide blocks.
    """
    by_uuid: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("product") != "claude" or r.get("kind") != "worker":
            continue
        uuid = str(r.get("account_uuid") or "").strip()
        if not uuid:
            continue
        by_uuid.setdefault(uuid, []).append(r)
    for group in by_uuid.values():
        if len(group) < 2:
            continue
        canonical = next((r for r in group if r.get("identity_role") == "canonical"), None)
        if not canonical or canonical.get("available"):
            continue
        kind = str(canonical.get("block_kind") or "").lower()
        if kind and kind != "auth":
            continue
        peers = [r for r in group if r is not canonical and r.get("available")]
        if not peers:
            continue
        peer = sorted(
            peers,
            key=lambda r: (
                -int(r.get("live_sessions") or 0),
                -int(r.get("active_sessions") or 0),
                str(r.get("tag") or ""),
            ),
        )[0]
        canonical.update({
            "available": True,
            "blocked": False,
            "block_kind": None,
            "block_reason": "",
            "reset": None,
            "weekly": None,
            "throttled": False,
            "status_source": "identity-peer",
            "active_sessions": max(
                int(canonical.get("active_sessions") or 0),
                int(peer.get("active_sessions") or 0),
            ),
            "live_sessions": max(
                int(canonical.get("live_sessions") or 0),
                int(peer.get("live_sessions") or 0),
            ),
            "auth_blocked_sessions": int(canonical.get("auth_blocked_sessions") or 0)
            + int(peer.get("auth_blocked_sessions") or 0),
        })


def read_account_identity(acct_dir: str) -> dict:
    """Read the logged-in Anthropic identity from a Claude config dir's .claude.json.

    The single source of truth for WHO a dir is actually logged in as -- the field the
    dir-name-keyed roster historically ignored, which let three differently-named dirs
    silently resolve to one account. Returns
    ``{account_uuid, login_email, org_uuid, org_type, plan}`` (empty strings when not
    logged in / unreadable). Never raises -- discovery must not crash on a malformed or
    huge config (a heavily-used account's .claude.json can be ~40 KB). Reads only the small
    ``oauthAccount`` identity fields; credentials/tokens are never touched.
    """
    out = {"account_uuid": "", "login_email": "", "org_uuid": "",
           "org_type": "", "plan": ""}
    path = os.path.join(acct_dir, ".claude.json")
    try:
        # cap the read so a pathological file can't stall discovery; oauthAccount sits
        # near the top of the doc, so a partial parse fallback recovers it if needed.
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return out
    if not isinstance(doc, dict):
        return out
    oa = doc.get("oauthAccount")
    if isinstance(oa, dict):
        out["account_uuid"] = str(oa.get("accountUuid") or "")
        out["login_email"] = str(oa.get("emailAddress") or "")
        out["org_uuid"] = str(oa.get("organizationUuid") or "")
        out["org_type"] = str(oa.get("organizationType") or "")
        out["plan"] = str(oa.get("organizationType") or oa.get("seatTier") or "")
    return out


def _classify_row(acct_dir: str, product: str, account: str, pol: dict,
                  accounts_index: dict | None = None) -> dict:
    """Apply policy + structure checks to one discovered dir, returning a row.

    Shared by the Claude and opencode discovery passes so the worker/excluded/
    non-account logic stays identical across products. ``product`` ("claude" |
    "opencode") only changes how an *account* dir is recognized (a ``projects/``
    subdir vs an ``opencode.json`` config file) -- the caller has already decided
    that before invoking this.
    """
    tag = account_tag(account)
    notes = pol.get("notes", {})
    note = notes.get(tag, "")
    if not os.path.isdir(acct_dir):
        return {"dir": acct_dir, "product": product, "account": account, "tag": tag,
                "kind": "non-account", "reason": "not a directory", "notes": note}
    # Intrinsic tombstone: a dir name carrying the `.DELETED` marker (the suffix the
    # account-decommission path stamps, e.g. `.claude-smith-netra.DELETED-2026-06-26`)
    # is decommissioned regardless of policy and must NEVER be offered to the switcher
    # -- otherwise the spawn gate routes a worker onto a dead login and burns the launch.
    # This is checked before the policy-substring exclude so it can't be missed by a
    # roster that only lists live-account name fragments.
    if ".deleted" in account.lower():
        return {"dir": acct_dir, "product": product, "account": account, "tag": tag,
                "kind": "excluded", "reason": "tombstoned (.DELETED marker)", "notes": note}
    # Intrinsic dogfood exclusion: the `.claude-faklocal*` homes are the fak-kernel
    # dogfood dirs -- synthesized on demand by `resolve --faklocal-ok`, never an enrolled
    # account. They hold no login and only point Claude Code at a locally served model, so
    # a credential-less `needs_login` row for one only clutters the switcher roster. Keep
    # them off it the way a .DELETED marker does; the dogfood resolve path bypasses
    # discovery entirely, so nothing that depends on it regresses.
    if account.lower().startswith(".claude-faklocal"):
        return {"dir": acct_dir, "product": product, "account": account, "tag": tag,
                "kind": "excluded",
                "reason": "fak-kernel dogfood home (synthesized on demand; not a roster account)",
                "notes": note}
    identity = read_account_identity(acct_dir) if product == "claude" else {}
    registry_reason = _accounts_registry_exclusion(
        {"dir": acct_dir, "product": product, "account": account, "tag": tag,
         "identity": identity, **identity},
        accounts_index,
    )
    if registry_reason:
        return {"dir": acct_dir, "product": product, "account": account, "tag": tag,
                "kind": "excluded", "reason": registry_reason, "notes": note}
    hit = _excluded_match(tag, account, pol.get("exclude", []),
                          str(identity.get("login_email", "")))
    if hit:
        why = note or notes.get(hit, "") or f"excluded by policy (matches '{hit}')"
        return {"dir": acct_dir, "product": product, "account": account, "tag": tag,
                "kind": "excluded", "reason": why, "notes": note}
    include_only = [t for t in pol.get("include_only", []) if t]
    if include_only and not any(t.lower() in tag.lower() for t in include_only):
        return {"dir": acct_dir, "product": product, "account": account, "tag": tag,
                "kind": "excluded", "reason": "not in include_only allowlist", "notes": note}
    label = "real offered opencode account" if product == "opencode" else "real offered account"
    row = {"dir": acct_dir, "product": product, "account": account, "tag": tag,
           "kind": "worker", "reason": label, "notes": note}
    row.update(account_profile(row, pol))
    # Routing capacity bias is orthogonal to the model profile, so it is stamped
    # separately (a route_weights entry must never clobber model-tier inference).
    row["route_weight"] = account_route_weight(row, pol)
    if product == "claude":
        # stamp the logged-in identity so the roster can see WHO a dir really is, not
        # just what it's named -- the seam the duplicate-identity reconciliation rides.
        row.update(identity)
    return row


def _discover_claude(home: str, pol: dict,
                     accounts_index: dict | None = None) -> list[dict]:
    """Glob ``<home>/.claude*`` -- Claude config dirs under the user home."""
    rows: list[dict] = []
    for acct_dir in glob.glob(os.path.join(home, ".claude*")):
        account = os.path.basename(acct_dir)
        tag = account_tag(account)
        note = pol.get("notes", {}).get(tag, "")
        # non-account: a plain file (.claude.json[.backup]) or a dir with no projects/
        if not os.path.isdir(acct_dir):
            rows.append({"dir": acct_dir, "product": "claude", "account": account, "tag": tag,
                         "kind": "non-account", "reason": "not a directory", "notes": note})
            continue
        if not os.path.isdir(os.path.join(acct_dir, "projects")):
            rows.append({"dir": acct_dir, "product": "claude", "account": account, "tag": tag,
                         "kind": "non-account", "reason": "no projects/ subdir", "notes": note})
            continue
        rows.append(_classify_row(acct_dir, "claude", account, pol, accounts_index))
    return rows


def _discover_opencode(config_home: str, pol: dict,
                       accounts_index: dict | None = None) -> list[dict]:
    """Glob ``<config_home>/opencode*`` -- opencode config dirs under XDG home.

    A dir is an opencode *account* when it holds an ``opencode.json`` /
    ``opencode.jsonc`` (the config file is the account switch seam, analogous to
    the ``projects/`` subdir for Claude). The data lives separately under
    ``~/.local/share/opencode`` and is not what makes an account an account.
    """
    rows: list[dict] = []
    for acct_dir in glob.glob(os.path.join(config_home, "opencode*")):
        account = os.path.basename(acct_dir)
        tag = account_tag(account)
        note = pol.get("notes", {}).get(tag, "")
        if not os.path.isdir(acct_dir):
            rows.append({"dir": acct_dir, "product": "opencode", "account": account, "tag": tag,
                         "kind": "non-account", "reason": "not a directory", "notes": note})
            continue
        if not any(os.path.isfile(os.path.join(acct_dir, m))
                   for m in OPENCODE_MARKER_FILES):
            rows.append({"dir": acct_dir, "product": "opencode", "account": account, "tag": tag,
                         "kind": "non-account", "reason": "no opencode.json config", "notes": note})
            continue
        rows.append(_classify_row(acct_dir, "opencode", account, pol, accounts_index))
    return rows


def _dir_recency(acct_dir: str) -> float:
    """Newest mtime under the dir's projects/ -- a cheap 'last actually used' proxy
    used to pick which of several same-identity dirs is the canonical (live) one."""
    proj = os.path.join(acct_dir, "projects")
    newest = 0.0
    try:
        for root, _dirs, files in os.walk(proj):
            for fn in files:
                if fn.endswith(".jsonl"):
                    try:
                        m = os.path.getmtime(os.path.join(root, fn))
                    except OSError:
                        continue
                    if m > newest:
                        newest = m
    except OSError:
        pass
    return newest


def _reconcile_identities(rows: list[dict]) -> list[dict]:
    """Detect Claude worker dirs that share ONE logged-in Anthropic account.

    The roster historically keyed an account purely on dir-name, so N dirs logged into
    the same account looked like N independent workers -- which broke throttle accounting
    (one limited account presented as several healthy ones) and triplicated a single
    blocker. This stamps each Claude worker with:
      account_uuid     the logged-in identity (already stamped at classify time)
      identity_role    "unique" | "canonical" | "duplicate" | "no-login"
      identity_peers   other dir tags sharing this account_uuid
      tag_login_match  False when the dir's tag clearly disagrees with the login email
    Duplicates stay VISIBLE but callers exclude them from routing/availability counts.
    Only Claude workers carry an oauth identity; opencode/others are left untouched.
    """
    claude_workers = [r for r in rows
                      if r.get("kind") == "worker" and r.get("product") == "claude"]
    by_uuid: dict[str, list[dict]] = {}
    for r in claude_workers:
        uuid = str(r.get("account_uuid") or "")
        if uuid:
            by_uuid.setdefault(uuid, []).append(r)

    # pass 1: tag<->login agreement for EVERY worker first, so the canonical tie-break
    # (which reads peers' tag_login_match) sees it set on all group members.
    for r in claude_workers:
        email = str(r.get("login_email") or "")
        tag = str(r.get("tag") or "")
        # the tag's short name should appear in the login email
        # (e.g. tag 'gem8' in 'jack...'? no -> mismatch). Empty login can't be judged.
        r["tag_login_match"] = bool(email) and (
            tag.lower() in email.lower()
            or any(part and part in tag.lower() for part in email.split("@")[0].lower().split(".")))

    # pass 2: role per worker (canonical / duplicate / unique / no-login)
    for r in claude_workers:
        uuid = str(r.get("account_uuid") or "")
        email = str(r.get("login_email") or "")
        if not uuid:
            r["identity_role"] = "no-login"
            r["identity_peers"] = []
            continue
        group = by_uuid.get(uuid, [r])
        peers = [g for g in group if g is not r]
        r["identity_peers"] = sorted(str(g.get("tag")) for g in peers)
        if len(group) == 1:
            r["identity_role"] = "unique"
        else:
            # canonical pick, best first:
            #   1. a dir whose TAG matches its login wins (the purpose-named dir, e.g. the
            #      'gem5' dir holding gem5@, beats a generic 'default' dir that merely
            #      shares the login -- 'default' may legitimately hold any account, so it
            #      must never steal canonical from a name-matched dir);
            #   2. then the most-recently-active dir.
            def _canon_key(g):
                gtag = str(g.get("tag") or "")
                return (1 if g.get("tag_login_match") else 0,
                        0 if gtag == "default" else 1,   # 'default' yields to a named dir
                        _dir_recency(str(g.get("dir") or "")))
            canonical = max(group, key=_canon_key)
            r["identity_role"] = "canonical" if r is canonical else "duplicate"
            if r["identity_role"] == "duplicate":
                r["reason"] = (f"duplicate identity: same Anthropic account as "
                               f"{canonical.get('tag')} ({email or uuid[:8]})")
    return rows


def discover_accounts(home: str = USER, policy: dict | None = None,
                      *, config_home: str | None = None) -> list[dict]:
    """Classify every account config dir, across both products.

    Claude dirs come from ``<home>/.claude*`` (the user home); opencode dirs
    come from ``<config_home>/opencode*`` (XDG config home, default ~/.config).

    Returns a list of dicts (sorted by product, then kind, then tag) with keys:
      dir       absolute path of the account directory
      product   ``claude`` | ``opencode``
      account   the dir basename (e.g. ``.claude-gem8-acct``, ``opencode-glm``)
      tag       normalized short name (e.g. ``gem8``, ``default``)
      kind      one of "worker" | "excluded" | "non-account"
      reason    one-line human explanation of the classification
      notes     operator note for this tag, if any

    Claude worker rows additionally carry the logged-in IDENTITY:
      account_uuid     the Anthropic accountUuid the dir is logged into ("" if none)
      login_email      the logged-in email ("" if not logged in)
      identity_role    "unique" | "canonical" | "duplicate" | "no-login" -- "duplicate"
                       means another dir shares this exact account (do not double-count)
      identity_peers   tags of other dirs sharing this account_uuid
      tag_login_match  False when the dir tag clearly disagrees with the login email
    """
    pol = policy or load_policy()
    ch = config_home or CONFIG_HOME
    accounts_index = _accounts_registry_index(load_accounts_registry(home=home))
    rows = _discover_claude(home, pol, accounts_index) + _discover_opencode(
        ch, pol, accounts_index)
    # reconcile WHO each dir is logged into: collapse N dirs on one Anthropic account to
    # one canonical worker + duplicates, so the roster stops presenting one throttled
    # account as several independent healthy workers.
    rows = _reconcile_identities(rows)
    rows.sort(key=lambda r: (r.get("product", ""), r["kind"] != "worker", r["tag"]))
    return rows


def worker_accounts(home: str = USER, policy: dict | None = None,
                    *, config_home: str | None = None) -> list[dict]:
    """Just the offered (worker) accounts, across both products."""
    return [r for r in discover_accounts(home, policy, config_home=config_home)
            if r["kind"] == "worker"]


def annotated_roster(home: str = USER, policy: dict | None = None,
                     registry: dict | None = None,
                     throttle: dict | None = None,
                     sessions: list[dict] | None = None,
                     *, config_home: str | None = None) -> list[dict]:
    """The full roster with live availability attached -- the canonical "give me the
    live accounts" call.

    Consumers historically spelled this out as
    ``annotate_accounts(discover_accounts(...), registry=load_registry())`` in five
    places (account_probe, claude_agent_chat, fleet_sessions x2, resume_resolver), each
    a slightly different copy. This is the one helper they all route through, so a change
    to discovery/annotation reaches every front door at once. ``registry`` defaults to
    the live session registry (``load_registry()``); pass an explicit dict (or the
    ``throttle``/``sessions`` overrides) when a caller already has the data in hand."""
    consult_probe_ledger = registry is None
    reg = load_registry() if registry is None else registry
    # On the live front door (no explicit registry), reflect live DISPATCHED opencode
    # workers from their .pid/.account sidecars so a docs-lane worker is visible in the
    # roster. Hermetic callers pass an explicit registry and skip the filesystem read.
    seat_leases = live_seat_leases() if consult_probe_ledger else None
    return annotate_accounts(
        discover_accounts(home, policy, config_home=config_home),
        reg, throttle, sessions, probe_ledger=consult_probe_ledger,
        seat_leases=seat_leases)


def available_accounts(home: str = USER, policy: dict | None = None,
                       registry: dict | None = None,
                       throttle: dict | None = None,
                       sessions: list[dict] | None = None,
                       *, config_home: str | None = None) -> list[dict]:
    """Worker accounts that are safe for a switcher to offer right now.

    Duplicate-identity dirs are excluded: they resolve to the same Anthropic account as
    their canonical sibling, so offering them would double-count one account's capacity.
    """
    rows = annotated_roster(home, policy, registry, throttle, sessions,
                            config_home=config_home)
    return [r for r in rows if routable_worker(r) and r["available"]]


def is_worker(account: str, home: str = USER, policy: dict | None = None) -> bool:
    """True iff ``account`` (a config-dir basename, any product) classifies as a worker.

    Product-neutral: works for ``.claude-*`` and ``opencode-*`` basenames alike.
    Cheap, standalone check the session tools call per-account in their scan loop;
    it does not require the account dir to currently exist on disk."""
    if ".deleted" in account.lower():
        return False
    pol = policy or load_policy()
    tag = account_tag(account)
    product = account_product(account)
    acct_dir = os.path.join(CONFIG_HOME if product == "opencode" else home, account)
    identity = read_account_identity(acct_dir) if product == "claude" else {}
    if _accounts_registry_exclusion(
        {"dir": acct_dir, "product": product, "account": account, "tag": tag,
         "identity": identity, **identity},
        _accounts_registry_index(load_accounts_registry(home=home)),
    ):
        return False
    if _excluded_match(tag, account, pol.get("exclude", [])):
        return False
    include_only = [t for t in pol.get("include_only", []) if t]
    if include_only and not any(t.lower() in tag.lower() for t in include_only):
        return False
    return True


def active_account_cap_throttles(
    rows: list[dict],
    *,
    runs_dir: str | os.PathLike[str] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, dict]:
    return _active_account_cap_throttles_impl(
        rows, runs_dir=runs_dir or ACCOUNT_CAP_RUNS_DIR, now=now
    )


def classify_task(task_text: str = "", task_class: str = "auto",
                  policy: dict | None = None) -> dict:
    return _classify_task_impl(task_text, task_class, policy, policy_fn=load_policy)


def route_account(rows: list[dict], task_text: str = "", task_class: str = "auto",
                  *, allow_tier_fallback: bool = False,
                  strict_tier: bool = False,
                  product: str | None = None,
                  leases: list[dict] | None = None,
                  policy: dict | None = None) -> dict:
    return _route_account_impl(
        rows, task_text, task_class,
        allow_tier_fallback=allow_tier_fallback,
        strict_tier=strict_tier, product=product, leases=leases,
        policy=policy, policy_fn=load_policy,
    )


def resolve_account(pin: str | None = None, *, task_text: str = "",
                    task_class: str = "auto", work_kind: str = "",
                    product: str | None = None,
                    allow_tier_fallback: bool = False,
                    strict_tier: bool = False,
                    faklocal_ok: bool = False,
                    home: Any = _UNSET, policy: dict | None = None,
                    registry: dict | None = None,
                    config_home: str | None = None) -> dict:
    return _resolve_account_impl(
        pin, task_text=task_text, task_class=task_class, work_kind=work_kind,
        product=product, allow_tier_fallback=allow_tier_fallback,
        strict_tier=strict_tier, faklocal_ok=faklocal_ok,
        home=USER if home is _UNSET else home, policy=policy,
        registry=registry, config_home=config_home,
        policy_fn=load_policy, roster_fn=annotated_roster,
        token_fn=read_oauth_token,
    )


def allocate_wave(count: int, *, task_text: str = "", task_class: str = "auto",
                  work_kind: str = "", product: str | None = None,
                  allow_tier_fallback: bool = False, strict_tier: bool = False,
                  home: Any = _UNSET, policy: dict | None = None,
                  registry: dict | None = None,
                  config_home: str | None = None,
                  leases: list[dict] | None = None,
                  wave_id: str | None = None) -> dict:
    return _allocate_wave_impl(
        count, task_text=task_text, task_class=task_class, work_kind=work_kind,
        product=product, allow_tier_fallback=allow_tier_fallback,
        strict_tier=strict_tier, home=USER if home is _UNSET else home,
        policy=policy, registry=registry, config_home=config_home,
        leases=leases, wave_id=wave_id,
        policy_fn=load_policy, roster_fn=annotated_roster,
        token_fn=read_oauth_token,
    )


def _cli_list(rows: list[dict]) -> None:
    _cli_list_impl(
        rows, user=USER, policy_path=POLICY_PATH,
        policy_example_path=POLICY_EXAMPLE_PATH,
    )


def main(argv: list[str]) -> int:
    mode = next((a for a in argv if not a.startswith("-")), "list")
    rows = annotate_accounts(discover_accounts())
    if mode == "json":
        print(json.dumps({"home": USER, "policy_path": POLICY_PATH,
                          "policy_exists": os.path.exists(POLICY_PATH),
                          "registry_path": REGISTRY_PATH,
                          "registry_exists": os.path.exists(REGISTRY_PATH),
                          "available_accounts": [
                              r for r in rows if routable_worker(r) and r["available"]
                          ],
                          "accounts": rows}, indent=1))
    elif mode == "available":
        for r in rows:
            if routable_worker(r) and r["available"]:
                print(r["account"])
    elif mode == "route":
        tier, strict = _tier_arg(argv)
        # A stated --work-kind/--kind wins over --tier and is passed as the task_class
        # (so the gardening/engineering class + reason survive). Gardening is NON-strict
        # so it up-shifts rather than stalls when no tier-2 account is free.
        kind = _arg_value(argv, ("--work-kind", "--kind"), "").strip().lower()
        if kind in GARDENING_WORK_KINDS or kind in ENGINEERING_WORK_KINDS:
            tier, strict = kind, False
        task = _arg_value(argv, ("--task", "--goal", "--prompt"), "")
        product = _arg_value(argv, ("--product",), "")
        doc = route_account(
            rows,
            task,
            tier,
            allow_tier_fallback=_has_arg(argv, ("--allow-tier-fallback",)),
            strict_tier=strict,
            product=product or None,
            leases=live_seat_leases(),
        )
        print(json.dumps(doc, indent=1))
        return 0 if doc.get("ok") else 1
    elif mode == "resolve":
        # The single front-door call: pin OR tier/work-kind route -> ONE flat record with
        # config_dir + oauth_token + tier. Same arg grammar as `route`, plus --account (pin)
        # and --faklocal-ok (synthesize the dogfood .claude-faklocal dir).
        tier, strict = _tier_arg(argv)
        kind = _arg_value(argv, ("--work-kind", "--kind"), "").strip().lower()
        task = _arg_value(argv, ("--task", "--goal", "--prompt"), "")
        product = _arg_value(argv, ("--product",), "")
        pin = _arg_value(argv, ("--account",), "")
        doc = resolve_account(
            pin or None,
            task_text=task,
            task_class=tier,
            work_kind=kind,
            product=product or None,
            allow_tier_fallback=_has_arg(argv, ("--allow-tier-fallback",)),
            strict_tier=strict,
            faklocal_ok=_has_arg(argv, ("--faklocal-ok",)),
        )
        print(json.dumps(doc, indent=1))
        return 0 if doc.get("ok") else 1
    elif mode == "wave":
        # Allocate N available account session slots for a parallel fan-out (an
        # ultracode wave). Same arg grammar as `resolve`, plus --count/-n N (omit ->
        # every slot free now). --explain reduces the output to the headroom witness:
        # distinct_pools granted vs the naive 1.
        tier, strict = _tier_arg(argv)
        kind = _arg_value(argv, ("--work-kind", "--kind"), "").strip().lower()
        task = _arg_value(argv, ("--task", "--goal", "--prompt"), "")
        product = _arg_value(argv, ("--product",), "")
        count_raw = _arg_value(argv, ("--count", "-n"), "")
        explicit = bool(count_raw)
        count = _as_int(count_raw, 0) if explicit else 10 ** 6
        doc = allocate_wave(
            count,
            task_text=task,
            task_class=tier,
            work_kind=kind,
            product=product or None,
            allow_tier_fallback=_has_arg(argv, ("--allow-tier-fallback",)),
            strict_tier=strict,
            leases=live_seat_leases(),
        )
        if not explicit:
            # No count asked => report "all available session slots now", not a shortfall
            # against the 10**6 sentinel.
            doc["requested"] = doc["granted"]
            doc["shortfall"] = 0
            doc["reason"] = (f"all {doc['granted']} available session slot(s) across "
                             f"{doc['distinct_pools']} distinct pool(s)")
        if _has_arg(argv, ("--explain",)):
            doc = {
                "ok": doc["ok"],
                "requested": doc["requested"],
                "granted": doc["granted"],
                "shortfall": doc["shortfall"],
                "distinct_pools": doc["distinct_pools"],
                "target_tier": doc["target_tier"],
                "naive_pools": 1 if doc["granted"] else 0,
                "headroom_multiplier": doc["distinct_pools"],
                "reason": doc["reason"],
                "lane_tags": [lane.get("tag") for lane in doc["lanes"]],
                "lane_pools": [lane.get("pool") for lane in doc["lanes"]],
            }
        print(json.dumps(doc, indent=1))
        return 0 if doc.get("ok") else 1
    elif mode == "seats":
        # The explicit account session-slot pool, with the seat->worker binding for live
        # workers. Reads each live worker's `.account` sidecar so a depleted pool is
        # visible and an exited worker frees its slot on the next read. --product scopes
        # the pool; --json for machines.
        product = _arg_value(argv, ("--product",), "").strip().lower() or None
        pool = seat_pool(rows, live_seat_leases(), product=product)
        if _has_arg(argv, ("--json",)):
            print(json.dumps(pool, indent=1))
        else:
            _cli_seats(pool)
        return 0
    elif mode == "probe":
        # ACTIVE probe: get ground truth now (vs the passive roster above), and show
        # where the live verdict CORRECTS the stale roster. Delegates to account_probe
        # so the corrections table has one implementation.
        import account_probe  # local import; only paid for on `probe`
        passthrough = [a for a in argv if a != "probe"]
        return account_probe.main(passthrough)
    else:
        _cli_list(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
