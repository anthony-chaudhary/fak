#!/usr/bin/env python3
"""Cap-window, reset parsing, and account identity/slot predicates for fleet_accounts.

Separated from ``tools/fleet_accounts.py`` so pure throttle/reset parsing, ledger-backed
cap disambiguation (``_disambiguate_cap``), and account-cap sidecar folding live in one
cohesive, dependency-free module under the file-size gate.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None

RUNS_DIRNAME = ".dispatch-runs"
ACCOUNT_CAP_PREFIX = "account-cap-"

DEFAULT_CLAUDE_ACCOUNT_SESSION_CAP = 4
DEFAULT_CODEX_ACCOUNT_SESSION_CAP = 20
DEFAULT_ACCOUNT_SESSION_CAP = 1

# One-knob-one-way override for the Claude per-account session budget, read the SAME
# way by Go (fleetaccounts.AccountSessionCap, via fleetaccounts.SessionsPerAccountEnv)
# so the two languages never drift. Lets an operator widen/narrow account capacity per
# host without a rebuild; a non-positive or unparseable value keeps the default.
SESSIONS_PER_ACCOUNT_ENV = "FAK_SESSIONS_PER_ACCOUNT"

# Claude's daily usage limit resets on a rolling window of at most ~5 hours, so a
# BARE reset time (no date, e.g. "3pm") is always close to NOW -- a few hours ahead
# at most. The next occurrence of that clock time is therefore the right anchor only
# when it lands inside this bound; a bare time already PAST today has reset, it does
# not mean "tomorrow". This is the slack we allow before declaring a passed bare time
# expired (kept a little above 5h so a clock-skewed or just-observed boundary still
# reads as future rather than flapping to expired).
_DAILY_RESET_WINDOW = dt.timedelta(hours=6)

# --- Cap-disambiguation core (mirror of internal/fleetaccounts/capstate.go) -------------
# throttle_is_active / _weekly_throttle_is_active are thin VIEWS over _disambiguate_cap: one
# place folds the daily+weekly reset states into a _CapState, then layers two ledger-driven
# cycles on top of the single-shot rule -- an AGING valve (a weekly episode older than
# _WEEKLY_MAX_AGE with no live daily leg has outlived any real window) and a probe-OVERRIDE
# (a run of >= _OVERRIDE_STREAK fresh OKs past a passed daily reset overturns a stale weekly
# the seat has demonstrably outgrown). Both cycles read a _CapObservation derived from the
# probe ledger; with the ZERO observation the two views reduce exactly to the legacy
# predicates, so a caller that passes no observation is behaviorally unchanged. Mirrors the Go
# split so the two ports stay in lockstep (fix behavior in Go first -- see fleet_accounts.py).
_WEEKLY_MAX_AGE = dt.timedelta(days=7)
_OVERRIDE_STREAK = 2


def _default_cap_runs_dir() -> str:
    return os.environ.get(
        "FLEET_ACCOUNT_CAP_RUNS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), RUNS_DIRNAME),
    )


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


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _claude_session_cap() -> int:
    raw = os.environ.get(SESSIONS_PER_ACCOUNT_ENV)
    if raw is not None:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            return n
    return DEFAULT_CLAUDE_ACCOUNT_SESSION_CAP


def _session_cap(row: dict) -> int:
    """Bounded concurrent worker sessions one routable account pool may back.

    Claude Code accounts can carry several independent worker sessions before the
    account itself is the limiter. Keep the cap explicit and conservative; the
    per-spawn host/lease guards still lower real launches when the machine or file
    trees cannot carry this many sessions. The Claude cap honors the
    FAK_SESSIONS_PER_ACCOUNT override (see _claude_session_cap).
    """
    product = str(row.get("product") or "claude").lower()
    if product == "claude":
        return _claude_session_cap()
    if product == "codex":
        return DEFAULT_CODEX_ACCOUNT_SESSION_CAP
    return DEFAULT_ACCOUNT_SESSION_CAP


def is_duplicate_identity(row: dict) -> bool:
    """True for a worker dir that is a non-canonical copy of another dir's account.

    Routing to a duplicate is identical to routing to its canonical (same Anthropic
    account, same limits), so offering it would double-count one account's capacity --
    the exact bug where one throttled account looked like three healthy workers.
    """
    return str(row.get("identity_role") or "") == "duplicate"


def routable_worker(row: dict) -> bool:
    """A worker the switcher may actually offer: a real worker that is not a duplicate
    of another dir's identity."""
    return row.get("kind") == "worker" and not is_duplicate_identity(row)


def read_oauth_token(account_dir: str) -> str | None:
    """Return the account's long-lived setup token, or None.

    This is the optional setup-token reader, not the default dispatch credential
    rule. The default worker path pins ``CLAUDE_CONFIG_DIR`` and clears any ambient
    ``CLAUDE_CODE_OAUTH_TOKEN`` so it matches account_probe's health check and never
    bleeds a sibling account's token into the worker. A caller that deliberately opts
    into setup-token auth can use this pure read, then decide whether to set the env var.

    Previously duplicated in launch_goal_detached.ps1 and issue_dispatch.py.
    """
    if not account_dir:
        return None
    try:
        with open(os.path.join(account_dir, ".oauth-token"), encoding="utf-8") as f:
            tok = f.read().strip()
    except OSError:
        return None
    return tok or None


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


def _weekly_reset_text(info: dict | str | None) -> str | None:
    if isinstance(info, dict):
        weekly = info.get("weekly")
        return str(weekly) if weekly else None
    return None


def _reset_is_future(reset: str | None, now: dt.datetime | None = None) -> bool | None:
    """Best-effort parser for Claude's reset strings.

    Returns True for a reset that is still in the future, False for an expired
    parsed reset, and None when the format is unknown.

    Two shapes occur in the wild:
      * DATED weekly resets -- "Jun 25, 1pm" or "Mon Jun 25 at 1pm" --
        anchored to this year. A yearless month/day that already passed this
        year is expired; it is not rolled forward to next year, because stale
        throttle records would otherwise re-block seats months after the real
        reset.
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
    raw = re.sub(r"^(mon|tue|wed|thu|fri|sat|sun)\s+", "", raw)
    formats = (
        "%b %d at %I:%M%p",
        "%b %d at %I%p",
        "%b %d, %I:%M%p",
        "%b %d, %I%p",
        "%I:%M%p",
        "%I%p",
    )
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


class _CapObservation:
    """Ledger-derived signals the cap cycles consume (mirror of Go CapObservation).

    first_seen: start of the current contiguous blocked episode (None unless the seat is
    currently blocked). ok_streak: run of consecutive OK verdicts at the ledger tail."""
    __slots__ = ("first_seen", "ok_streak")

    def __init__(self, first_seen: dt.datetime | None = None, ok_streak: int = 0) -> None:
        self.first_seen = first_seen
        self.ok_streak = ok_streak


class _CapState:
    """Folded verdict for one carried throttle (mirror of Go CapState)."""
    __slots__ = ("active", "weekly_active", "reset", "weekly", "aged_out", "overridden_by")

    def __init__(self) -> None:
        self.active = False
        self.weekly_active = False
        self.reset: str | None = None
        self.weekly: str | None = None
        self.aged_out = False
        self.overridden_by = 0


def _now_utc(now: dt.datetime | None) -> dt.datetime:
    """A tz-aware UTC instant for the aging comparison, consistent with the ledger's
    _parse_utc timestamps. Mirrors the single ``now`` Go threads through DisambiguateCap."""
    if now is None:
        return dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(dt.timezone.utc)


def _disambiguate_cap(info: dict | str | None,
                      obs: "_CapObservation | None" = None,
                      now: dt.datetime | None = None,
                      weekly_max_age: dt.timedelta = _WEEKLY_MAX_AGE,
                      override_streak: int = _OVERRIDE_STREAK) -> "_CapState":
    obs = obs or _CapObservation()
    st = _CapState()
    st.reset = _reset_text(info)
    st.weekly = _weekly_reset_text(info)

    weekly_present = bool(st.weekly)
    weekly_future = _reset_is_future(st.weekly, now)      # True / False / None
    weekly_provably_past = weekly_present and weekly_future is False
    reset_future = _reset_is_future(st.reset, now)
    reset_provably_past = reset_future is False

    # Base single-shot rule -- identical to the legacy throttle_is_active / weekly predicate.
    if weekly_present and not weekly_provably_past:
        st.active = True
    else:
        st.active = not reset_provably_past
    st.weekly_active = weekly_present and not weekly_provably_past and st.active

    # AGING valve: a live weekly episode older than weekly_max_age with no provably-future
    # daily leg has outlived any real weekly window -- release it. An unknown/absent daily no
    # longer walls the seat; only a provably-future daily still holds it (as a plain daily cap).
    if st.weekly_active and obs.first_seen is not None:
        if _now_utc(now) - obs.first_seen >= weekly_max_age:
            st.aged_out = True
            st.weekly_active = False
            st.active = reset_future is True

    # Probe-OVERRIDE: a run of fresh OKs past a passed daily reset overturns a stale/unparseable
    # weekly the seat has demonstrably outgrown. Distinct from aging -- keyed on recovery
    # evidence (the OK streak), not elapsed time -- and never fires once aged_out already
    # released the seat.
    if (st.weekly_active and not st.aged_out
            and obs.ok_streak >= override_streak and reset_provably_past):
        st.overridden_by = obs.ok_streak
        st.weekly_active = False
        st.active = False

    return st


def throttle_is_active(info: dict | str | None,
                       now: dt.datetime | None = None) -> bool:
    return _disambiguate_cap(info, None, now).active


def _weekly_throttle_is_active(info: dict | str | None,
                               now: dt.datetime | None = None) -> bool:
    return _disambiguate_cap(info, None, now).weekly_active


def _probe_is_ok(status: object) -> bool:
    """Whether a ledger status string is the clean-availability OK verdict (mirror of Go
    probeIsOK)."""
    return str(status or "").strip().upper() == "OK"


def _cap_observation_from(entries: list[dict]) -> "_CapObservation":
    """Distill one account's append-ordered ledger entries into a _CapObservation.

    Mirror of internal/fleetaccounts/capobs.go deriveCapObservation: OK streak is the run of
    consecutive OK verdicts at the tail; first_seen is the start of the trailing blocked
    episode and is set only while the seat is CURRENTLY blocked (the tail is non-OK) -- a tail
    of OKs means no live episode, so aging stays dormant and the override is the live cycle."""
    obs = _CapObservation()
    if not entries:
        return obs
    streak = 0
    for e in reversed(entries):
        if not _probe_is_ok(e.get("status")):
            break
        streak += 1
    obs.ok_streak = streak
    if not _probe_is_ok(entries[-1].get("status")):
        episode_start = entries[-1]
        for e in reversed(entries):
            if _probe_is_ok(e.get("status")):
                break
            episode_start = e
        obs.first_seen = _parse_utc(episode_start.get("ts"))
    return obs


def _derive_cap_observations(entries: list[dict]) -> dict:
    """account basename -> _CapObservation, from the whole append-ordered ledger."""
    by_acct: dict[str, list[dict]] = {}
    for e in entries:
        acct = e.get("account")
        if acct:
            by_acct.setdefault(acct, []).append(e)
    return {a: _cap_observation_from(rows) for a, rows in by_acct.items()}


def _normalize_throttle(throttle: dict | None) -> dict:
    out = {}
    for account, info in (throttle or {}).items():
        if isinstance(info, dict):
            out[account] = dict(info)
        else:
            out[account] = {"reset": info}
    return out


def _account_cap_until(state: dict) -> dt.datetime | None:
    until = _parse_utc(state.get("until") if isinstance(state, dict) else None)
    if until is None:
        return None
    return until.astimezone(dt.timezone.utc)


def active_account_cap_throttles(
    rows: list[dict],
    *,
    runs_dir: str | os.PathLike[str] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, dict]:
    """Return throttle entries derived from active account-cap sidecars.

    issue_resolve_dispatch.py writes ``.dispatch-runs/account-cap-<product>.json``
    after a live worker proves a quota wall. The switcher routes by account
    basename, while the sidecar stores product + short tag, so this folds active
    holds through the discovered roster before ``runtime_status()`` runs. Missing,
    malformed, expired, or unmatchable sidecars contribute nothing (fail-open).
    """
    by_product_tag: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.get("kind") != "worker":
            continue
        product = str(row.get("product") or account_product(str(row.get("account") or ""))).lower()
        account = str(row.get("account") or "")
        tag = str(row.get("tag") or account_tag(account)).lower()
        if product and account and tag:
            by_product_tag[(product, tag)] = account
            by_product_tag[(product, account.lower())] = account

    rd = Path(runs_dir or _default_cap_runs_dir())
    if not rd.is_dir():
        return {}
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    else:
        now = now.astimezone(dt.timezone.utc)

    out: dict[str, dict] = {}
    for path in sorted(rd.glob(f"{ACCOUNT_CAP_PREFIX}*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(state, dict):
            continue
        until = _account_cap_until(state)
        if until is None or until <= now:
            continue
        product = str(
            state.get("product")
            or path.stem[len(ACCOUNT_CAP_PREFIX):]
        ).lower()
        tag = str(state.get("account") or "").lower()
        account = by_product_tag.get((product, tag))
        if not account:
            continue
        reset = str(state.get("until") or "")
        kind = str(state.get("kind") or "session").lower()
        throttle = {
            "reset": reset,
            "since": state.get("detected"),
            "source": "account-cap-sidecar",
            "cap_kind": kind,
            "reset_text": state.get("reset_text") or "",
        }
        if kind == "weekly":
            throttle["weekly"] = reset
        out[account] = throttle
    return out
