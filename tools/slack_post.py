#!/usr/bin/env python3
r"""slack_post — the ONE zero-dependency Slack poster for fak's Python fleet tools.

The Go side resolves every Slack surface through ``internal/slackenv`` (the process
environment, then a gitignored ``.env.slack.local`` walked up from the working
directory) and posts through ``internal/scoreboard`` / ``fak slack send``. The
dispatch loop, its watchdogs, and the fleet-status views are Python — portable
mac/linux/windows ticks with no Go build on the hot path — so they need a sibling
poster that follows the SAME resolution order without a binary on PATH. This is that
sibling: pure stdlib (``urllib``), the channel/token resolved exactly like
``slackenv``, and a ``--dry-run`` that prints the resolution so an operator can see
WHICH token/channel a tick would use before any live post.

Resolution (mirrors the ``dispatch`` surface in ``cmd/fak/slack.go``):

    token   : FAK_DISPATCH_TOKEN  -> FAK_SCOREBOARD_TOKEN   (env then .env.slack.local)
    channel : --channel / explicit -> FAK_DISPATCH_CHANNEL  (env then .env.slack.local)
              -> the caller's default ("" => channel REQUIRED, like the Go surface)

    python tools/slack_post.py --text "dispatch is green" --channel C0ABC123
    echo "hi" | python tools/slack_post.py --text -                 # channel from env
    python tools/slack_post.py --text "x" --dry-run                 # show resolution, send nothing
    python tools/slack_post.py --text "x" --code                    # wrap text in a ``` block

It holds NO token and NO channel id in source — only HOW to read the key an operator
set, never WHICH value (the same contract as ``internal/slackenv``). Importers call
``send(...)``; the watchdogs gate it behind their own ``--slack`` opt-in so a tick
never posts unless the operator asked it to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-slack-post/1"

# The gitignored file every Slack surface reads its token/channel from — the same
# constant internal/slackenv exports (slackenv.EnvFileName). Kept byte-identical so a
# value an operator sets is visible to BOTH the Go and the Python poster.
ENV_FILE_NAME = ".env.slack.local"
# How many directories file_value ascends looking for ENV_FILE_NAME, mirroring
# slackenv.maxWalkUp: deep enough for a monorepo checkout, bounded so it never climbs
# to the filesystem root scanning unrelated parents.
MAX_WALK_UP = 6

# Default API base; overridable for tests/proxying via --api-base or send(api_base=...).
DEFAULT_API_BASE = "https://slack.com/api/"

# The dispatch surface's keys (the default importer target). A caller that posts on a
# different surface passes its own keys, exactly like the Go slackSurface registry.
DISPATCH_TOKEN_KEY = "FAK_DISPATCH_TOKEN"
DISPATCH_CHANNEL_KEY = "FAK_DISPATCH_CHANNEL"
SCOREBOARD_TOKEN_KEY = "FAK_SCOREBOARD_TOKEN"  # the shared workspace bot token


# ----- resolution (the Python sibling of internal/slackenv) ------------------

def _scan_file(path: Path, key: str) -> tuple[str, bool]:
    """Return (value, True) for the first ``key=...`` line in the file, else ("", False).

    Tolerates an optional ``export `` prefix and trims surrounding whitespace — the
    same grammar slackenv.scanFile accepts. A present-but-blank value (``KEY=``) returns
    ("", True) so it ends the walk just like a non-blank match: an operator who blanks a
    key in a near checkout is not overridden by a value set further up the tree.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", False
    prefix = key + "="
    for raw in text.split("\n"):
        ln = raw.strip()
        if ln.startswith("export "):
            ln = ln[len("export "):].strip()
        if ln.startswith(prefix):
            return ln[len(prefix):].strip(), True
    return "", False


def file_value(key: str, start: Path | None = None) -> str:
    """Walk up from ``start`` (default: cwd) returning the first ``.env.slack.local``
    value for ``key``. "" when the file is absent at every level, the key is nowhere, or
    the matched value is blank — callers treat "" as unset and fall back. Mirrors
    slackenv.fileValueFrom exactly (including: a file that exists but lacks the key does
    NOT stop the walk)."""
    dir_ = (start or Path.cwd()).resolve()
    for _ in range(MAX_WALK_UP):
        value, found = _scan_file(dir_ / ENV_FILE_NAME, key)
        if found:
            return value
        parent = dir_.parent
        if parent == dir_:
            break  # reached the filesystem root
        dir_ = parent
    return ""


def lookup(key: str, start: Path | None = None) -> tuple[str, str]:
    """Resolve ``key`` from the environment first, then ``.env.slack.local``.

    Returns (value, source) where source is "env" | "file" | "unset". The env-then-file
    half every Go surface's ResolveToken/ResolveChannel applies; surface-specific
    fallbacks layer on top in the resolver functions below.
    """
    env = os.environ.get(key, "").strip()
    if env:
        return env, "env"
    fv = file_value(key, start)
    if fv:
        return fv, "file"
    return "", "unset"


def resolve_token(
    token_key: str = DISPATCH_TOKEN_KEY,
    fallback_key: str = SCOREBOARD_TOKEN_KEY,
    *,
    start: Path | None = None,
) -> tuple[str, str]:
    """Resolve a surface's bot token: its own key (env then file), then the shared
    scoreboard token. Returns (value, source); source records which path won so a
    --dry-run report is self-explaining (the same fall-back slack.go's surface.token
    applies)."""
    val, src = lookup(token_key, start)
    if val:
        return val, f"{src}:{token_key}"
    if fallback_key and fallback_key != token_key:
        fval, fsrc = lookup(fallback_key, start)
        if fval:
            return fval, f"scoreboard-fallback ({fsrc}:{fallback_key})"
    return "", "unset"


def resolve_channel(
    explicit: str = "",
    channel_key: str = DISPATCH_CHANNEL_KEY,
    default: str = "",
    *,
    start: Path | None = None,
) -> tuple[str, str]:
    """Resolve a target channel: an explicit ``--channel`` first, then the surface's
    channel key (env then file), then the caller's built-in default. "" => channel
    REQUIRED (the Go surface's behavior when no default is registered)."""
    if explicit:
        return explicit, "explicit"
    val, src = lookup(channel_key, start)
    if val:
        return val, f"{src}:{channel_key}"
    if default:
        return default, "built-in default"
    return "", "unset"


def redact_token(token: str) -> str:
    """Show only that a token is present plus its last 4 chars, never the secret
    (same redaction as cmd/fak/slack.go's redactToken)."""
    if not token:
        return "(unset)"
    if len(token) <= 4:
        return "****"
    return "****" + token[-4:]


# ----- transport -------------------------------------------------------------

# A transport is (url, body_bytes, headers, timeout) -> (http_status, body_text). It is
# injected in tests so the fold/resolution is exercised with zero network.
Transport = Callable[[str, bytes, "dict[str, str]", float], "tuple[int, str]"]


def _urllib_transport(url: str, body: bytes, headers: dict[str, str],
                      timeout: float) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 (https only)
            return getattr(resp, "status", 200), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # a non-2xx still carries a Slack JSON body
        return exc.code, exc.read().decode("utf-8", "replace")


def post(
    text: str,
    *,
    channel: str,
    token: str,
    api_base: str = DEFAULT_API_BASE,
    thread_ts: str | None = None,
    timeout: float = 30.0,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """POST one chat.postMessage. Pure given an injected ``transport``; never raises —
    a transport/Slack failure IS the answer the caller wants to log. Returns
    {ok, channel, ts?, error?}."""
    base = (api_base or DEFAULT_API_BASE)
    if not base.endswith("/"):
        base += "/"
    url = base + "chat.postMessage"
    payload: dict[str, Any] = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    tx = transport or _urllib_transport
    try:
        status, raw = tx(url, body, headers, timeout)
    except Exception as exc:  # noqa: BLE001 — any transport error becomes a verdict
        return {"ok": False, "channel": channel, "error": f"transport: {exc}"}
    try:
        doc = json.loads(raw)
    except ValueError:
        return {"ok": False, "channel": channel,
                "error": f"non-JSON response (http {status}): {raw[:200]}"}
    if not isinstance(doc, dict) or not doc.get("ok"):
        err = (doc.get("error") if isinstance(doc, dict) else None) or f"http {status}"
        return {"ok": False, "channel": channel, "error": str(err)}
    return {"ok": True, "channel": doc.get("channel", channel), "ts": doc.get("ts")}


def wrap_code(text: str) -> str:
    """Wrap text in a Slack code fence so a box-drawn status card keeps its alignment
    (Slack renders ``` blocks in a monospace font)."""
    return "```\n" + text.rstrip("\n") + "\n```"


# Level glyphs for one-line watchdog events (a respawn, an auth wall, a failed canary).
# The watchdogs post a single actionable line, not a card, so a glyph + bold title reads
# at a glance in the channel and in a mobile push preview.
EVENT_GLYPH = {"info": "🟢", "resume": "♻️", "warn": "⚠️", "crit": "🔴"}


def event_text(title: str, detail: str = "", *, level: str = "info") -> str:
    """Format one actionable watchdog event: ``{glyph} *title* — detail``. Pure; the
    watchdogs share it so a respawn/auth-wall/failed-canary line reads the same way
    whichever loop emitted it."""
    glyph = EVENT_GLYPH.get(level, "•")
    line = f"{glyph} *{title}*"
    if detail:
        line += f" — {detail}"
    return line


def event(title: str, detail: str = "", *, level: str = "info",
          **send_kwargs: Any) -> dict[str, Any]:
    """Post one actionable watchdog event (a respawn, an auth wall, a failed canary).

    A thin wrapper over ``send`` that formats a single glyph-prefixed line — the
    watchdogs call this on a STATE CHANGE only (never every tick), so the channel
    carries exactly the events an operator would act on. All ``send`` keyword args
    (channel, dry_run, transport, …) pass through; returns the same typed verdict."""
    return send(event_text(title, detail, level=level), **send_kwargs)


def send(
    text: str,
    *,
    channel: str = "",
    token_key: str = DISPATCH_TOKEN_KEY,
    token_fallback: str = SCOREBOARD_TOKEN_KEY,
    channel_key: str = DISPATCH_CHANNEL_KEY,
    default_channel: str = "",
    code: bool = False,
    dry_run: bool = False,
    api_base: str = "",
    thread_ts: str | None = None,
    timeout: float = 30.0,
    transport: Transport | None = None,
    start: Path | None = None,
) -> dict[str, Any]:
    """Resolve token + channel like the Go ``dispatch`` surface and post ``text``.

    This is the importer entry point (dispatch_status / fleet_top / the watchdogs call
    it). It NEVER raises and always returns a typed verdict an operator can log:

        {schema, posted, dry_run, channel, channel_source, token_set, token_source,
         ts?, error?, skipped?}

    ``skipped`` is set (and ``posted`` False) when a precondition is missing — no channel
    resolved or no token resolved — so a misconfigured tick degrades to a clear "did
    nothing because X" rather than a crash. ``dry_run`` resolves everything and reports
    what it WOULD send without a network call.
    """
    tok, tok_src = resolve_token(token_key, token_fallback, start=start)
    chan, chan_src = resolve_channel(channel, channel_key, default_channel, start=start)
    body = wrap_code(text) if code else text

    base = {
        "schema": SCHEMA,
        "posted": False,
        "dry_run": dry_run,
        "channel": chan,
        "channel_source": chan_src,
        "token_set": bool(tok),
        "token_source": tok_src,
        "ts": None,
        "error": None,
        "skipped": None,
    }

    if not chan:
        base["skipped"] = "no channel resolved (set --channel or " + channel_key + ")"
        return base
    if not tok:
        base["skipped"] = ("no bot token resolved (set " + token_key + " or "
                           + token_fallback + ")")
        return base
    if dry_run:
        base["skipped"] = "dry-run"
        return base

    result = post(body, channel=chan, token=tok, api_base=(api_base or DEFAULT_API_BASE),
                  thread_ts=thread_ts, timeout=timeout, transport=transport)
    base["posted"] = bool(result.get("ok"))
    base["ts"] = result.get("ts")
    if not result.get("ok"):
        base["error"] = result.get("error")
    if result.get("channel"):
        base["channel"] = result["channel"]
    return base


# ----- CLI -------------------------------------------------------------------

def _render(result: dict[str, Any]) -> str:
    if result.get("posted"):
        return f"slack_post: posted to {result['channel']} (ts={result.get('ts')})"
    if result.get("dry_run"):
        return (f"slack_post (dry-run): would post to "
                f"{result['channel'] or '(unset)'} [{result['channel_source']}]; "
                f"token {redact_token('x' if result['token_set'] else '')} "
                f"[{result['token_source']}]")
    if result.get("skipped"):
        return f"slack_post: skipped — {result['skipped']}"
    return f"slack_post: FAILED — {result.get('error')}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Zero-dependency Slack poster for fak's Python fleet tools "
                    "(token/channel resolved like internal/slackenv).")
    ap.add_argument("--text", required=True,
                    help="message text; pass - to read the message from stdin")
    ap.add_argument("--channel", default="",
                    help="target channel id (e.g. C0ABC123); default: $" + DISPATCH_CHANNEL_KEY)
    ap.add_argument("--surface-token-key", default=DISPATCH_TOKEN_KEY,
                    help="env/file key for this surface's bot token")
    ap.add_argument("--surface-channel-key", default=DISPATCH_CHANNEL_KEY,
                    help="env/file key for this surface's channel id")
    ap.add_argument("--default-channel", default="",
                    help="built-in channel default when nothing else resolves")
    ap.add_argument("--code", action="store_true",
                    help="wrap the text in a ``` block (keeps box-drawn cards aligned)")
    ap.add_argument("--api-base", default="",
                    help="override the Slack API base URL (for testing/proxying)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve token/channel and report what WOULD be sent; send nothing")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    msg = args.text
    if msg == "-":
        msg = sys.stdin.read().strip()
    if not msg:
        print("slack_post: --text is required (or pipe a message via --text -)",
              file=sys.stderr)
        return 2

    result = send(
        msg,
        channel=args.channel,
        token_key=args.surface_token_key,
        channel_key=args.surface_channel_key,
        default_channel=args.default_channel,
        code=args.code,
        dry_run=args.dry_run,
        api_base=args.api_base,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_render(result))
    # Exit 0 on a real post or a dry-run; 1 when a live send was attempted and failed
    # or was skipped for a missing precondition (so a scheduled tick's LastResult flags
    # a misconfiguration, never a silent no-op).
    if result.get("posted") or result.get("dry_run"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
