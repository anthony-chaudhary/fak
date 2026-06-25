#!/usr/bin/env python3
r"""fleet_accounts -- the single source of truth for "what is an account,
and is it offered?", across BOTH product families: Claude Code and opencode.

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
    python tools/fleet_accounts.py wave --count 8 --explain  # allocate up to 8 DISTINCT
                                              # available pools for a parallel fan-out (an
                                              # ultracode wave); --explain prints the headroom
                                              # multiplier (distinct_pools vs the naive 1).
                                              # Omit --count for every distinct pool free now.
"""
from __future__ import annotations

import copy
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None

import fleet_session_signals

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
REG_DIR = os.environ.get(
    "FLEET_REG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry"),
)
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
    "route_weights": {},
    "routing": {
        "light_confidence": 0.999,
        "hard_tier1_fallback": "stop",
    },
}


def account_product(account: str) -> str:
    """Classify a discovered dir basename to its product family.

    ``.claude*``  -> ``claude`` (the default config under the user home)
    ``opencode*`` -> ``opencode`` (the config under ~/.config)
    Anything else defaults to ``claude`` so historical call sites keep working.
    """
    a = account.lower()
    if a.startswith("opencode"):
        return "opencode"
    return "claude"


def account_tag(account: str) -> str:
    """Normalize a config-dir basename to its short tag, matching the
    convention used across the resume layer (fleet_sessions.py / the watchdog).

    Per product:
      Claude:   ``.claude-gem8-acct`` -> ``gem8``; ``.claude`` -> ``default``
      opencode: ``opencode-glm``       -> ``glm``;  ``opencode``  -> ``default``

    The trailing ``-acct`` org suffix (Claude convention) is stripped if present.
    """
    product = account_product(account)
    if product == "opencode":
        tag = account.replace("opencode-", "").replace("opencode", "")
    else:
        tag = account.replace(".claude-", "").replace(".claude", "")
    if tag.endswith("-acct"):
        tag = tag[: -len("-acct")]
    return tag or "default"


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


def _excluded_match(tag: str, account: str, exclude: list[str]) -> str | None:
    """Return the matching exclude substring (for the reason text), or None."""
    for sub in exclude:
        if not sub:
            continue
        if sub.lower() in tag.lower() or sub.lower() in account.lower():
            return sub
    return None


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _model_tier_from_name(model: object) -> int:
    """Small v1 model taxonomy.

    Tier 1 is the max-quality frontier set the operator named. Tier 2 is GLM-5.2
    for obvious lightweight work. Everything else is tier 3 until explicitly
    classified later.
    """
    text = str(model or "").lower().replace("_", "-").replace(" ", "-")
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if "gpt-5.5" in text or "gpt55" in compact:
        return 1
    if "opus-4.6" in text or "opus46" in compact or text in ("opus", "claude-opus"):
        return 1
    if "glm-5.2" in text or "glm52" in compact:
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


def _normalize_throttle(throttle: dict | None) -> dict:
    out = {}
    for account, info in (throttle or {}).items():
        if isinstance(info, dict):
            out[account] = dict(info)
        else:
            out[account] = {"reset": info}
    return out


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


def _age_min(row: dict) -> float | None:
    raw = row.get("age_min")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_utc(raw: object) -> dt.datetime | None:
    if not raw:
        return None
    try:
        ts = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _row_seen_utc(row: dict, generated_utc: object = None) -> dt.datetime | None:
    seen = _parse_utc(row.get("seen_utc"))
    if seen is not None:
        return seen
    age = _age_min(row)
    generated = _parse_utc(generated_utc)
    if age is None or generated is None:
        return None
    return generated - dt.timedelta(minutes=age)


def _reset_text(info: dict | str | None) -> str | None:
    if isinstance(info, dict):
        reset = info.get("reset")
    else:
        reset = info
    return str(reset) if reset else None


# Claude's daily usage limit resets on a rolling window of at most ~5 hours, so a
# BARE reset time (no date, e.g. "3pm") is always close to NOW -- a few hours ahead
# at most. The next occurrence of that clock time is therefore the right anchor only
# when it lands inside this bound; a bare time already PAST today has reset, it does
# not mean "tomorrow". This is the slack we allow before declaring a passed bare time
# expired (kept a little above 5h so a clock-skewed or just-observed boundary still
# reads as future rather than flapping to expired).
_DAILY_RESET_WINDOW = dt.timedelta(hours=6)


def _reset_is_future(reset: str | None, now: dt.datetime | None = None) -> bool | None:
    """Best-effort parser for Claude's reset strings.

    Returns True for a reset that is still in the future, False for an expired
    parsed reset, and None when the format is unknown.

    Two shapes occur in the wild:
      * DATED weekly resets -- "Jun 25, 1pm" -- anchored to this year (rolled to
        next year only when the parse lands implausibly far in the past).
      * BARE daily resets -- "3pm", "12:30am" -- a clock time with no date. These
        belong to the ~5h rolling daily window, so the answer is the NEAREST
        occurrence around now, bounded by ``_DAILY_RESET_WINDOW``: today's
        occurrence if it is still ahead; otherwise tomorrow's ONLY if that is
        within the window; a bare time already past beyond the window has reset
        (expired). The previous heuristic rolled any pre-6am time a full day
        forward whenever observed after noon, so an already-passed "12:30am"
        reset (~13h ago) was mis-read as "tomorrow 12:30am" and the account
        stayed falsely throttled -- the resume-resolver then re-homed off a
        healthy account. (#resume-resolver false throttle)
    """
    if not reset:
        return None
    zone = None
    if "America/Los_Angeles" in reset and ZoneInfo is not None:
        zone = ZoneInfo("America/Los_Angeles")
    now = now or dt.datetime.now(zone or dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=zone or dt.timezone.utc)

    raw = re.sub(r"\s*\([^)]*$", "", reset).strip()
    raw = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    raw = re.sub(r"\s+", " ", raw).lower()
    formats = ("%b %d, %I:%M%p", "%b %d, %I%p", "%I:%M%p", "%I%p")
    for fmt in formats:
        candidate_raw = raw
        candidate_fmt = fmt
        if "%b" in fmt:
            candidate_raw = f"{raw} {now.year}"
            candidate_fmt = f"{fmt} %Y"
        try:
            parsed = dt.datetime.strptime(candidate_raw, candidate_fmt)
        except ValueError:
            continue
        if "%b" in fmt:
            candidate = parsed.replace(year=now.year, tzinfo=now.tzinfo)
            if candidate < now and (now - candidate).days > 180:
                candidate = candidate.replace(year=now.year + 1)
            return candidate > now
        # Bare time: anchor to the nearest occurrence within the daily window.
        candidate = parsed.replace(year=now.year, month=now.month, day=now.day,
                                   tzinfo=now.tzinfo)
        if candidate > now:
            # today's occurrence is still ahead -- the live reset.
            return True
        # today's occurrence has passed; tomorrow's counts as future ONLY if it
        # falls inside the rolling daily window. Otherwise the limit has reset.
        tomorrow = candidate + dt.timedelta(days=1)
        if tomorrow - now <= _DAILY_RESET_WINDOW:
            return True
        return False
    return None


def throttle_is_active(info: dict | str | None) -> bool:
    reset_state = _reset_is_future(_reset_text(info))
    if reset_state is False:
        return False
    return True


def runtime_status(account: str, registry: dict | None = None,
                   throttle: dict | None = None,
                   sessions: list[dict] | None = None) -> dict:
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

    # A fresh probe is authoritative. OK -> available now (clears stale carried blocks);
    # a fresh probe BLOCK -> that block, with the probe's own (account-correct) reason,
    # never a stale dir-keyed throttle from a previous login.
    if fresh_probe_ok:
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

    thr = throttle_map.get(account)
    if thr and throttle_is_active(thr):
        reset = thr.get("reset")
        weekly = thr.get("weekly")
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
            "throttled": True,
        })
        return status

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
    return status


def annotate_accounts(rows: list[dict], registry: dict | None = None,
                      throttle: dict | None = None,
                      sessions: list[dict] | None = None) -> list[dict]:
    """Attach live availability fields to discover_accounts() rows."""
    out = []
    for row in rows:
        r = dict(row)
        if r.get("kind") == "worker":
            r.update(runtime_status(r["account"], registry, throttle, sessions))
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
    out.sort(key=lambda r: (r.get("product", ""),
                            r["kind"] != "worker",
                            not r.get("available", False),
                            r["tag"]))
    return out


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


def _classify_row(acct_dir: str, product: str, account: str, pol: dict) -> dict:
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
    hit = _excluded_match(tag, account, pol.get("exclude", []))
    if hit:
        why = note or f"excluded by policy (matches '{hit}')"
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
        row.update(read_account_identity(acct_dir))
    return row


def _discover_claude(home: str, pol: dict) -> list[dict]:
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
        rows.append(_classify_row(acct_dir, "claude", account, pol))
    return rows


def _discover_opencode(config_home: str, pol: dict) -> list[dict]:
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
        rows.append(_classify_row(acct_dir, "opencode", account, pol))
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
    rows = _discover_claude(home, pol) + _discover_opencode(ch, pol)
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


def is_duplicate_identity(row: dict) -> bool:
    """True for a worker dir that is a non-canonical copy of another dir's account.

    Routing to a duplicate is identical to routing to its canonical (same Anthropic
    account, same limits), so offering it would double-count one account's capacity --
    the exact bug where one throttled account looked like three healthy workers."""
    return str(row.get("identity_role") or "") == "duplicate"


def routable_worker(row: dict) -> bool:
    """A worker the switcher may actually offer: a real worker that is not a duplicate
    of another dir's identity."""
    return row.get("kind") == "worker" and not is_duplicate_identity(row)


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
    reg = load_registry() if registry is None else registry
    return annotate_accounts(
        discover_accounts(home, policy, config_home=config_home),
        reg, throttle, sessions)


def read_oauth_token(account_dir: str) -> str | None:
    """Return the account's long-lived setup token, or None.

    The single implementation of the credential rule the dispatch front doors share:
    prefer the dir's ``.oauth-token`` (a long-lived setup token) over the interactive
    ``.credentials.json``, which EXPIRES -- a worker pinned only by CLAUDE_CONFIG_DIR can
    false-fail on turn 1 when the interactive creds have lapsed but the setup token is
    still good. Returns the stripped token string when ``<account_dir>/.oauth-token``
    exists and is readable, else None so the caller can DROP any ambient
    CLAUDE_CODE_OAUTH_TOKEN (never bleed a sibling account's token into this worker).
    Pure read -- the caller decides whether to set or pop the env var.

    Previously duplicated in launch_goal_detached.ps1 and issue_dispatch.py."""
    if not account_dir:
        return None
    try:
        with open(os.path.join(account_dir, ".oauth-token"), encoding="utf-8") as f:
            tok = f.read().strip()
    except OSError:
        return None
    return tok or None


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


def classify_task(task_text: str = "", task_class: str = "auto",
                  policy: dict | None = None) -> dict:
    """Classify a request for v1 model routing.

    Only trivial, short prompts are considered light. Everything ambiguous or
    dev-shaped is hard so max-quality remains the default.
    """
    pol = policy or load_policy()
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
                  policy: dict | None = None) -> dict:
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
    pol = policy or load_policy()
    task = classify_task(task_text, task_class, pol)
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

    available = [r for r in workers if r.get("available")]
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


FAKLOCAL_TAG = "faklocal"


def _flatten_resolved(acct: dict, *, ok: bool, reason: str,
                      selected_tier=None, target_tier=None,
                      fallback_used: bool = False,
                      block_reason: str = "") -> dict:
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
        "oauth_token": read_oauth_token(config_dir),
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
                    home: str = USER, policy: dict | None = None,
                    registry: dict | None = None,
                    config_home: str | None = None) -> dict:
    """Resolve ONE fully-specified account for a dispatch -- the canonical front-door call.

    The single entry point every front door (launch_goal_detached.ps1, the dogfood
    launchers, issue_dispatch) routes through instead of stitching ``json``+``route``+a
    hand-rolled availability check + a separate ``.oauth-token`` read. Returns a FLAT dict
    ``{ok, reason, account, tag, product, config_dir, oauth_token, model, model_tier,
    selected_tier, target_tier, fallback_used, block_reason}`` -- ``config_dir`` is what to
    pin as CLAUDE_CONFIG_DIR and ``oauth_token`` is the long-lived setup token (or None ->
    drop the ambient one).

    Three request shapes:
      * ``pin`` set: pick the worker dir whose tag/account matches ``pin`` and validate it
        is available (refuse with ok=False unless ``allow_tier_fallback``). The dogfood
        ``faklocal`` default is special-cased when ``faklocal_ok``: it synthesizes the
        isolated ``<home>/.claude-faklocal`` dir (creating ``projects/``) so the launchers
        stop hand-rolling it.
      * otherwise: delegate to route_account() (tier/work-kind routing among AVAILABLE
        accounts), then flatten its pick + attach the token.

    A work_kind (gardening/engineering) is an operator-stated fact and wins over the
    free-text heuristic, exactly as route_account documents."""
    pol = policy or load_policy()

    # The dogfood isolated account: not a discovered worker, synthesized on demand.
    if faklocal_ok and pin and account_tag(pin) == FAKLOCAL_TAG:
        d = os.path.join(home, ".claude-faklocal")
        try:
            os.makedirs(os.path.join(d, "projects"), exist_ok=True)
        except OSError:
            pass
        return _flatten_resolved(
            {"account": ".claude-faklocal", "tag": FAKLOCAL_TAG, "product": "claude",
             "dir": d, "model": "local", "model_tier": 3},
            ok=True, reason="isolated dogfood faklocal account",
            selected_tier=3, target_tier=3)

    rows = annotated_roster(home, pol, registry, config_home=config_home)

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
                {}, ok=False, reason=f"account '{pin}' is not an offered worker")
        if not match.get("available") and not allow_tier_fallback:
            why = str(match.get("block_reason") or "blocked")
            return _flatten_resolved(
                match, ok=False,
                reason=f"account '{pin}' is blocked: {why}",
                block_reason=why)
        return _flatten_resolved(
            match, ok=True, reason="pinned account",
            selected_tier=match.get("model_tier"),
            target_tier=match.get("model_tier"))

    # Tier / work-kind routing among available accounts.
    cls = task_class
    strict = strict_tier
    wk = (work_kind or "").strip().lower()
    if wk in GARDENING_WORK_KINDS or wk in ENGINEERING_WORK_KINDS:
        cls, strict = wk, False
    route = route_account(rows, task_text, cls,
                          allow_tier_fallback=allow_tier_fallback,
                          strict_tier=strict, product=product, policy=pol)
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
        fallback_used=bool(route.get("fallback_used")))


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


def allocate_wave(count: int, *, task_text: str = "", task_class: str = "auto",
                  work_kind: str = "", product: str | None = None,
                  allow_tier_fallback: bool = False, strict_tier: bool = False,
                  home: str = USER, policy: dict | None = None,
                  registry: dict | None = None,
                  config_home: str | None = None,
                  wave_id: str | None = None) -> dict:
    """Allocate up to ``count`` DISTINCT available accounts for a parallel fan-out
    (an "ultracode" wave), balanced across the roster -- the primitive a wave needs
    that single-account ``resolve_account`` cannot provide.

    Why this exists. ``resolve_account`` picks the best ONE account, ranked by
    fewest live sessions. A fan-out that calls it N times in a BURST gets the SAME
    account N times, because no session has registered yet to move the live-load
    tie-break -- so all N workers pile onto ONE rate-limit pool and the fan-out
    silently SERIALIZES (witnessed: 3 calls -> the same tag thrice while 3 distinct
    pools sat free). ``allocate_wave`` hands out DISTINCT pools in one call, so N
    lanes draw on N independent per-account usage limits.

    "Distinct" is by RATE-LIMIT POOL (``_pool_key``: the Anthropic ``accountUuid``
    for a logged-in Claude dir, else the dir basename), never the dir name -- two
    dirs on one account are one pool and must not both be handed out (that is the
    same double-count the duplicate-identity reconciliation already guards, applied
    to allocation). Duplicate-identity dirs are excluded up front via
    ``routable_worker``.

    Filling order mirrors ``route_account``: the target tier first (best quality),
    each tier's pools ordered by ``_route_rank`` (operator room bias, then fewest
    live, then fewest active). A tier-2 wave up-shifts into tier 1 to fill remaining
    lanes unless ``strict_tier``; a tier-1 wave only spills into tier 2 when
    ``allow_tier_fallback``. Every lane carries its own ``selected_tier`` and
    ``fallback_used`` so a mixed-tier wave is legible.

    Honest under-fill: if the roster can only safely back K < count distinct pools
    right now, ``granted == K`` and ``shortfall == count - K`` -- never a silent
    duplicate that would re-collapse two lanes onto one pool. ``distinct_pools`` (==
    granted) is the wave's true concurrency multiplier vs the naive 1.

    Rank-stamped membership (the typed group). Each granted lane carries its
    ``rank`` in ``[0, granted)``, the shared ``wave_id`` (a deterministic, content-
    addressed digest of the granted pools -- override via the ``wave_id`` arg), and
    the wave ``size`` (== granted). This is the Python counterpart to
    ``internal/comm``'s ``Membership`` value and borrows ``MPI_Comm_spawn_multiple``'s
    typed-array-with-counts shape: the lanes are the typed array, ``granted``/``size``
    the counts, and ``shortfall`` the honest maxprocs/error under-fill. A caller that
    spawns the wave stamps these onto each child (env + sidecars) so the group is a
    legible thing an auditor enumerates from the filesystem ("wave W: ranks 0..K-1,
    K=granted, shortfall=S") without trusting any worker's self-report.

    NOT a collective. The stamps LABEL an allocation; they do not make the wave a
    communicator -- no barrier, no all-to-all, no gather across ranks. A wave stays
    N independent detached workers whose only shared fabric is git + the
    ``dos arbitrate`` lease. The honest under-fill is the spawn_multiple
    maxprocs/error array, not a guaranteed group size.

    Returns a flat dict::

        {ok, requested, granted, shortfall, distinct_pools, size, wave_id,
         target_tier, reason, lanes: [<resolve record + 'pool'/'rank'/'wave_id'/
         'size'>...], blocked_target_accounts}

    Each ``lanes`` entry is the same flat shape ``resolve_account`` returns (so a
    caller pins ``config_dir`` as CLAUDE_CONFIG_DIR and serves on ``oauth_token``),
    plus a ``pool`` field carrying its ``_pool_key`` for distinctness auditing and
    the ``rank``/``wave_id``/``size`` membership stamp."""
    pol = policy or load_policy()
    n = max(0, int(count))

    # Fold a stated work-kind into the routing class exactly as resolve_account does:
    # an operator-stated kind (gardening/engineering) wins over the free-text heuristic.
    cls, strict = task_class, strict_tier
    wk = (work_kind or "").strip().lower()
    if wk in GARDENING_WORK_KINDS or wk in ENGINEERING_WORK_KINDS:
        cls, strict = wk, False

    task = classify_task(task_text, cls, pol)
    target = int(task["target_tier"])

    rows = annotated_roster(home, pol, registry, config_home=config_home)
    wanted_product = (product or "").lower()
    workers = [
        r for r in rows
        if routable_worker(r)  # excludes duplicate-identity dirs (same pool as canonical)
        and (not wanted_product or str(r.get("product") or "").lower() == wanted_product)
    ]
    available = [r for r in workers if r.get("available")]

    # Tier fill order: target first, then the one permitted fallback tier -- identical
    # policy to route_account, so a wave and a single resolve agree on what "tier 1 with
    # fallback" means.
    fallback_policy = str(pol.get("routing", {}).get("hard_tier1_fallback", "stop")).lower()
    effective_allow_fallback = allow_tier_fallback or fallback_policy in (
        "allow", "fallback", "tier2", "t2")
    tier_order = [target]
    if target == 2 and not strict:
        tier_order.append(1)
    elif effective_allow_fallback:
        tier_order.append(2)

    lanes: list[dict] = []
    seen_pools: set[str] = set()
    for tier in tier_order:
        if len(lanes) >= n:
            break
        candidates = sorted(
            (r for r in available if _as_int(r.get("model_tier"), 3) == tier),
            key=_route_rank)
        for r in candidates:
            if len(lanes) >= n:
                break
            pool = _pool_key(r)
            if pool in seen_pools:
                continue  # a second dir on a pool already taken -> would re-serialize
            seen_pools.add(pool)
            lane = _flatten_resolved(
                r, ok=True,
                reason="wave lane (target tier)" if tier == target else "wave lane (fallback tier)",
                selected_tier=r.get("model_tier"), target_tier=target,
                fallback_used=tier != target)
            lane["pool"] = pool
            lanes.append(lane)

    granted = len(lanes)
    shortfall = max(0, n - granted)
    # Stamp the rank-stamped membership now that the group size is known: each lane
    # gets its rank in [0, granted), the shared (content-addressed) wave id, and the
    # wave size. This is the typed-group identity a spawner carries onto each child.
    wid = wave_id if wave_id is not None else _wave_id_for([lane["pool"] for lane in lanes])
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
        reason = (f"granted {granted} of {n} distinct pools; {shortfall} short "
                  f"(roster has no more distinct available pools at the requested tiers)")
    else:
        reason = f"granted {granted} distinct pools"
    return {
        "ok": granted > 0,
        "requested": n,
        "granted": granted,
        "shortfall": shortfall,
        "distinct_pools": granted,
        "size": granted,
        "wave_id": wid,
        "target_tier": target,
        "reason": reason,
        "lanes": lanes,
        "blocked_target_accounts": blocked,
    }


def is_worker(account: str, home: str = USER, policy: dict | None = None) -> bool:
    """True iff ``account`` (a config-dir basename, any product) classifies as a worker.

    Product-neutral: works for ``.claude-*`` and ``opencode-*`` basenames alike.
    Cheap, standalone check the session tools call per-account in their scan loop;
    it does not require the account dir to currently exist on disk."""
    pol = policy or load_policy()
    tag = account_tag(account)
    if _excluded_match(tag, account, pol.get("exclude", [])):
        return False
    include_only = [t for t in pol.get("include_only", []) if t]
    if include_only and not any(t.lower() in tag.lower() for t in include_only):
        return False
    return True


def _cli_list(rows: list[dict]) -> None:
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
    print(f"fleet accounts under {USER}  ({len(rows)} dirs, products: "
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
    if os.path.exists(POLICY_PATH):
        pol_src = POLICY_PATH
    elif os.path.exists(POLICY_EXAMPLE_PATH):
        pol_src = POLICY_EXAMPLE_PATH + " (example; copy to _registry/ to customize)"
    else:
        pol_src = "(built-in defaults)"
    print(f"\npolicy: {pol_src}")


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
        # Allocate N DISTINCT available pools for a parallel fan-out (an ultracode wave).
        # Same arg grammar as `resolve`, plus --count/-n N (omit -> every distinct pool
        # free now). --explain reduces the output to the headroom witness: distinct_pools
        # granted vs the naive 1, i.e. the fan-out's true concurrency multiplier.
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
        )
        if not explicit:
            # No count asked => report "all distinct pools available now", not a shortfall
            # against the 10**6 sentinel.
            doc["requested"] = doc["granted"]
            doc["shortfall"] = 0
            doc["reason"] = f"all {doc['granted']} distinct available pool(s)"
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
