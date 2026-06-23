#!/usr/bin/env python3
"""Shared transcript signal patterns for the fleet session/account tools."""
from __future__ import annotations

import re

# `<when>` can itself contain a parenthesized timezone, e.g.
# "12:10am (America/Los_Angeles)". Capture the time and an optional trailing
# "(...)" group as a unit, then stop before banner junk. The terminator accepts
# a sentence-final period (". " / end) so both the daily and the weekly window
# of a Claude throttle banner parse even when each ends in ".".
LIMIT_RE = re.compile(
    r"limit\s*[·:|.\-]?\s*resets?\s+([^()\"\n]+?(?:\([^()\n]*\))?)\s*"
    r"(?:[\"\n<]|$|\.(?:\s|$))",
    re.I,
)

AUTH_RE = re.compile(
    r"Login interrupted|please run /login|authentication_error|"
    r"invalid x-api-key|invalid authentication credentials|"
    r"API Error:\s*401|HTTP\s*401|401\s+(?:authentication required|unauthorized)|"
    r"OAuth token has expired|credit balance is too low|"
    r"organization has disabled Claude subscription access|"
    r"Use an Anthropic API key instead",
    re.I,
)

ACCESS_WALL_RE = re.compile(
    r"organization has disabled Claude subscription access|"
    r"Claude subscription access .*disabled|"
    r"Use an Anthropic API key instead|"
    r"ask your admin to enable access",
    re.I,
)

LOGIN_REQUIRED_RE = re.compile(
    r"Login interrupted|please run /login|authentication_error|"
    r"invalid x-api-key|invalid authentication credentials|"
    r"API Error:\s*401|HTTP\s*401|401\s+(?:authentication required|unauthorized)|"
    r"OAuth token has expired|Not logged in",
    re.I,
)

# Transport/server errors only. Auth and quota signals are checked separately.
APIERR_RE = re.compile(
    r"isApiErrorMessage|API Error|overloaded_error|\boverloaded\b|"
    r"\b429\b|\b529\b|\b503\b|fetch failed|ECONNRESET|ETIMEDOUT|"
    r"socket hang up|Internal Server Error|service unavailable|"
    r"connection error|network error",
    re.I,
)


def limit_resets(text: str) -> dict:
    """All usage-limit reset windows in a Claude throttle banner.

    Claude's banner can carry a short (hourly / daily) window AND a weekly one
    in the same message. Each ``limit ... resets <when>`` occurrence is
    classified by the ~24 chars preceding it: a ``week`` hint makes it the
    weekly window, otherwise it is the short window. Returns a dict with
    optional ``daily`` and ``weekly`` keys (raw reset strings, timezone suffix
    preserved); empty dict when no reset is present.
    """
    text = text or ""
    daily = weekly = None
    for m in LIMIT_RE.finditer(text):
        when = m.group(1).strip()
        prefix = text[max(0, m.start() - 24): m.start()].lower()
        if "week" in prefix:
            if weekly is None:
                weekly = when
        elif daily is None:
            daily = when
    out: dict = {}
    if daily:
        out["daily"] = daily
    if weekly:
        out["weekly"] = weekly
    return out


def limit_reset(text: str) -> str | None:
    """The primary (blocking) reset window.

    Prefers the short (hourly/daily) window so existing throttle detection
    keeps its semantics, but falls back to the weekly window when the banner
    only carries a weekly one -- a weekly cap still blocks the account, so the
    absence of a short window must not read as 'not throttled'.
    """
    windows = limit_resets(text)
    return windows.get("daily") or windows.get("weekly")


def weekly_reset(text: str) -> str | None:
    """Just the weekly reset window, or None."""
    return limit_resets(text).get("weekly")


_RESET_TIME_RE = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*([ap])m\b", re.I)
# IANA tz -> fixed UTC offset hours. The fleet's banners only ever name the US
# Pacific zone; a small explicit table avoids a tzdata dependency on the stdlib-only
# tool surface. PDT (DST, Mar-Nov) is UTC-7; the fleet runs year-round on Pacific.
_TZ_OFFSET = {
    "america/los_angeles": -7, "america/denver": -6, "america/chicago": -5,
    "america/new_york": -4, "utc": 0,
}


def _reset_tz_offset(when: str) -> int:
    m = re.search(r"\(([^)]+)\)", when or "")
    if m:
        return _TZ_OFFSET.get(m.group(1).strip().lower(), -7)
    return -7  # default to Pacific -- the only zone the banners use


def reset_passed(when: str, now_utc=None, anchor_utc=None) -> bool | None:
    """Has a usage-limit reset window already elapsed?

    ``when`` is a raw reset string from :func:`limit_reset`, e.g.
    ``"6am (America/Los_Angeles)"`` or ``"7:10am (America/Los_Angeles)"``. A reset
    banner names the NEXT occurrence of that wall-clock time, so this resolves the
    window against the banner's own time (``anchor_utc`` -- the transcript's last
    timestamp, when known) and reports whether ``now_utc`` is at/after it.

    Returns True (resumable now), False (still capped), or None (unparseable -- the
    caller should treat it conservatively as not-yet-passed). Pure + injectable so
    it unit-tests without a clock; production passes ``now_utc=datetime.now(UTC)``.

    This is the primitive the manifest-bound watcher lacked: without a past/future
    verdict on the reset window, a reset-cleared session is indistinguishable from a
    still-capped one, so a whole reset-passed cohort stays invisible until someone
    eyeballs the clock.
    """
    from datetime import datetime, timezone, timedelta
    m = _RESET_TIME_RE.search(when or "")
    if not m:
        return None
    hour = int(m.group(1)) % 12
    if m.group(3).lower() == "p":
        hour += 12
    minute = int(m.group(2) or 0)
    off = _reset_tz_offset(when)
    tz = timezone(timedelta(hours=off))
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    anchor = anchor_utc or now_utc
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    # the reset is the FIRST occurrence of (hour:minute) in tz at/after the anchor
    a_local = anchor.astimezone(tz)
    reset_local = a_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset_local < a_local:
        reset_local += timedelta(days=1)
    return now_utc >= reset_local.astimezone(timezone.utc)


def is_auth_error(text: str) -> bool:
    return bool(AUTH_RE.search(text or ""))


def is_api_error(text: str) -> bool:
    text = text or ""
    return bool(APIERR_RE.search(text)) and not is_auth_error(text)


def auth_block_kind(text: str) -> str:
    text = text or ""
    lower = text.lower()
    if "credit balance is too low" in lower:
        return "credit"
    if ACCESS_WALL_RE.search(text):
        return "access"
    return "auth"


def auth_block_reason(text: str) -> str:
    kind = auth_block_kind(text)
    if kind == "credit":
        return "credit balance too low"
    if kind == "access":
        return "Claude subscription access disabled"
    return "auth/login required"


def needs_login_prompt(text: str) -> bool:
    """True only for blockers a human login/credential refresh can plausibly fix."""
    return auth_block_kind(text) == "auth" and bool(LOGIN_REQUIRED_RE.search(text or ""))
