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
