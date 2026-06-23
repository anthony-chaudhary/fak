#!/usr/bin/env python3
r"""opencode_adapter — the opencode transcript adapter for the fleet-resilience layer.

The resilience layer (``fleet_sessions.py`` + the two watchdogs) answers
"what stopped, why, and how to resume" from Claude ``.jsonl`` transcripts.
Its disposition table is the contract every classifier must honor:

    LIVE / DONE / DEAD_MIDTOOL / DEAD_KILLED / USER_CLOSED
    / STOPPED_LIMIT / STOPPED_APIERR / INFRA_AUTH / PARKED_WAIT / STOPPED_QUIET

rolled up into the five categories INFRA / AGENT / USER / HANGING / LIVE.

opencode sessions are a DIFFERENT schema — a sqlite db / JSON export with
its own message/part/finish shape, and ``opencode run --session <id>`` for
resume (no ``CLAUDE_CONFIG_DIR`` scoping). A mixed Claude+opencode fleet
that classified opencode workers with the Claude signal-extractor would
flag a finished worker as DEAD or miss a real crash. This module adapts the
*signal extraction* for opencode while keeping the invariant table intact:
it imports the canonical table from ``fleet_sessions`` so the two
classifiers can never silently drift apart.

The "talk-about-vs-is" pitfall applies doubly here: an opencode transcript
whose assistant text DISCUSSES "rate limit" must NOT classify as throttled.
Every disposition below is decided from STRUCTURED fields only — the
message ``role``, the assistant ``finish`` reason, the per-tool ``status``,
and the structured ``error`` object's ``name``/``type``. Free-text bodies
feed the human-readable ``last`` field and nothing else.

Normalized session shape this adapter consumes (the I/O layer that reads
``opencode db`` / the export JSON normalizes into this; see
``from_storage`` for the storage-message mapping)::

    {
      "session": "ses_abc123",        # opencode session id
      "updated_epoch": 1700000000.0,  # last activity, for liveness (None => unknown)
      "directory": "/repo",           # the worker cwd (maps to the Claude `cwd`)
      "autonomous": True,             # optional explicit flag; else inferred from
                                      #   the first user message's directive
      "messages": [
        {"role": "user", "text": "...", "finish": None, "error": None, "tools": []},
        {"role": "assistant", "text": "...",
         "finish": "stop"|"length"|"aborted"|"error"|None,
         "error": None | {"name": "ProviderRateLimitError", "message": "..."},
         "tools": [{"name": "bash", "status": "completed"|"running"|"error"|"pending"}]},
        ...
      ]
    }

Used by the resume-watchdog to emit ``opencode run --session <id>`` and to
key the resume-once ledger on ``backend + session id`` so a Claude session
and an opencode session can never collide on a bare uuid.
"""
import os
import re
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling helper
import fleet_sessions  # noqa: E402  -- the canonical invariant table lives here

BACKEND = "opencode"

# The invariant table is OWNED by fleet_sessions; we import it so a Claude
# classifier and the opencode classifier share one disposition->category map.
# If fleet_sessions adds/renames a disposition this adapter inherits it.
CATEGORY = fleet_sessions.CATEGORY
DEAD = fleet_sessions.DEAD
STOPLIKE = fleet_sessions.STOPLIKE
LIVE_MIN = fleet_sessions.LIVE_MIN
RESUME_PROMPT = fleet_sessions.RESUME_PROMPT

# Structured error-name classification. We match against the error OBJECT's
# `name`/`type` field (a structured classifier emitted by opencode / the
# provider SDK), never the prose message body -- that is the whole point of
# anchoring to the structured stop event. Substrings are lowercased before test.
_RATE_NAMES = ("ratelimit", "rate_limit", "overloaded", "429",
               "too_many_requests", "toomanyrequests", "quota", "usage_limit",
               "usagelimit")
_AUTH_NAMES = ("auth", "unauthorized", "401", "403", "forbidden", "credit",
               "billing", "payment", "subscription", "login")

# finish reasons that mean the assistant turn STRUCTURALLY ended (not a crash).
_DONE_FINISH = ("stop", "end_turn", "length", "max_tokens", "content_filter")
# finish reasons / part kinds that mean a human aborted the turn.
_ABORT_FINISH = ("aborted", "abort", "cancel", "cancelled", "canceled", "interrupted")
# tool statuses that mean the call never returned a terminal result -> mid-tool death.
_PENDING_TOOL = ("running", "pending", "in_progress", "started")

# autonomy: a dispatched/headless opencode worker (vs an interactive chat). Mirrors
# fleet_sessions.AUTON_RE -- matched against the FIRST user directive only.
_AUTON_RE = re.compile(r"</?command-name>|/(goal|loop|dispatch|next-up|fanout)\b|"
                       r"resolve github issue|keep working until|autonomous-loop", re.I)


def _err_kind(error):
    """Map a structured error object to one of 'rate'/'auth'/'api'/None.

    Reads only ``name``/``type`` (the structured discriminator). A bare error
    with an unrecognized name is still a real stop -> 'api' (transient infra)."""
    if not isinstance(error, dict):
        return None
    tag = (str(error.get("name") or "") + " " + str(error.get("type") or "")).lower()
    if not tag.strip():
        return "api"
    if any(s in tag for s in _RATE_NAMES):
        return "rate"
    if any(s in tag for s in _AUTH_NAMES):
        return "auth"
    return "api"


def _last_assistant(messages):
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m
    return None


def _has_pending_tool(msg):
    for t in (msg.get("tools") or []):
        if isinstance(t, dict) and str(t.get("status") or "").lower() in _PENDING_TOOL:
            return True
    return False


def _autonomous(session):
    if isinstance(session.get("autonomous"), bool):
        return session["autonomous"]            # explicit flag wins
    for m in (session.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return bool(_AUTON_RE.search(str(m.get("text") or "")))  # first user directive only
    return False


def _age_min(updated_epoch, now=None):
    if updated_epoch is None:
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    seen = dt.datetime.fromtimestamp(float(updated_epoch), dt.timezone.utc)
    return (now - seen).total_seconds() / 60.0


def classify_session(session, now=None):
    """Classify a normalized opencode session into the shared disposition shape.

    Returns the same row keys a Claude ``classify`` row carries (disp / category
    / cause / reason / autonomous / session / cwd / last / age_min / backend), so
    the merge / decide / rollup pipeline can treat both backends uniformly.

    Disposition is decided from STRUCTURED signals in this precedence (mirroring
    ``fleet_sessions.classify`` so the two never disagree on the same situation):
      structured error: rate -> STOPPED_LIMIT, auth -> INFRA_AUTH, other -> STOPPED_APIERR
      then live (recent activity), user-abort, mid-tool, clean finish, killed, quiet.
    """
    messages = session.get("messages") or []
    la = _last_assistant(messages)
    last_msg = messages[-1] if messages else None
    age = _age_min(session.get("updated_epoch"), now)
    err_kind = _err_kind(la.get("error")) if la else None
    finish = str((la or {}).get("finish") or "").lower()
    # `error` finish with no structured error object still routes as an API error.
    if not err_kind and finish == "error":
        err_kind = "api"

    if err_kind == "rate":
        disp, reason = "STOPPED_LIMIT", "opencode worker hit a provider rate/usage limit"
    elif err_kind == "auth":
        disp, reason = "INFRA_AUTH", "opencode worker stopped on an auth/credit wall (needs re-login)"
    elif err_kind == "api":
        disp, reason = "STOPPED_APIERR", "opencode worker stopped on an API/transport error (transient)"
    elif age is not None and age <= LIVE_MIN:
        disp, reason = "LIVE", "appended within %g min" % LIVE_MIN
    elif finish in _ABORT_FINISH:
        disp, reason = "USER_CLOSED", "opencode turn aborted (user intentionally stopped it)"
    elif la is not None and _has_pending_tool(la):
        pend = next(t.get("name") for t in la["tools"]
                    if str(t.get("status") or "").lower() in _PENDING_TOOL)
        disp, reason = "DEAD_MIDTOOL", "died mid tool (%s) with no terminal result" % pend
    elif finish in _DONE_FINISH:
        disp, reason = "DONE", "last assistant turn ended cleanly (finish=%s)" % finish
    elif la is not None:
        # an assistant message exists but carries no finish reason and no pending
        # tool: the process was killed before the turn closed out.
        disp, reason = "DEAD_KILLED", "assistant turn never closed (no finish reason)"
    else:
        disp, reason = "STOPPED_QUIET", "quiet; no completion/crash/close signal"

    category, cause = CATEGORY.get(disp, ("HANGING", "unknown"))
    last_txt = ""
    if isinstance(last_msg, dict):
        last_txt = str(last_msg.get("text") or "")
    return {
        "backend": BACKEND,
        "disp": disp, "category": category, "cause": cause, "reason": reason,
        "session": session.get("session") or "",
        "cwd": session.get("directory"),
        "autonomous": _autonomous(session),
        "supervised": False,
        "age_min": round(age, 1) if age is not None else None,
        "last": last_txt[:200].replace("\n", " "),
    }


def ledger_key(backend, session):
    """Resume-once ledger key, scoped by backend so a Claude uuid and an opencode
    session id can never collide on a bare id in a mixed fleet."""
    return "%s:%s" % (backend or "claude", session or "")


def resume_cmd(session, prompt=None):
    """Operator-runnable opencode resume command.

    opencode resumes by SESSION ID with ``opencode run --session <id>`` (the
    account is pinned by XDG_CONFIG_HOME upstream, not by a per-command flag --
    unlike Claude's ``CLAUDE_CONFIG_DIR``). The resume message is the same
    re-establish-your-goal prompt the Claude path uses."""
    sid = session.get("session") if isinstance(session, dict) else session
    msg = prompt if prompt is not None else RESUME_PROMPT
    return ("opencode run --session %s --dangerously-skip-permissions %s"
            % (sid, _shell_quote(msg)))


def _shell_quote(s):
    """Single-quote a positional arg for a POSIX-ish shell; opencode workers are
    launched cross-platform but the resume line is operator-pasted, so keep it
    simple and unambiguous rather than platform-perfect."""
    return "'" + str(s).replace("'", "'\\''") + "'"


def plan_entry(row):
    """A resume-plan record in the same shape ``fleet_sessions.plan_entry`` emits,
    tagged ``backend: "opencode"`` and carrying the opencode resume command +
    backend-scoped ledger key. The watchdog consumes Claude and opencode plan
    entries from one plan without conflation: it branches on ``backend``."""
    sid = row.get("session", "")
    return {
        "backend": BACKEND,
        "ledger_key": ledger_key(BACKEND, sid),
        "session": sid,
        "cwd": row.get("cwd"),
        "disp": row.get("disp"),
        "resume_cmd": resume_cmd(row) if row.get("disp") in STOPLIKE else None,
        # opencode has no CLAUDE_CONFIG_DIR; these stay None so a mixed-fleet
        # rollup never tries to copy an opencode transcript across config dirs.
        "config_dir": None,
        "resume_account": None,
    }


def from_storage(info, messages, now=None):
    """Normalize an opencode storage session (``info`` + raw ``messages``) into the
    shape ``classify_session`` consumes, then classify it.

    opencode storage carries a session ``info`` record (id, directory, time) and a
    list of message records whose ``parts`` mix text, tool calls (each with a
    ``state.status``), plus a per-message ``finish``/``error``. We fold each
    message's parts down to (text, tools[]) and keep the structured finish/error
    fields verbatim -- the classifier reads those, never the prose."""
    info = info or {}
    norm_msgs = []
    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        texts, tools = [], []
        for p in (m.get("parts") or []):
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text":
                texts.append(str(p.get("text") or ""))
            elif ptype in ("tool", "tool-invocation", "tool_use"):
                state = p.get("state") if isinstance(p.get("state"), dict) else {}
                tools.append({
                    "name": p.get("tool") or p.get("name") or "tool",
                    "status": state.get("status") or p.get("status") or "completed",
                })
        norm_msgs.append({
            "role": m.get("role"),
            "text": " ".join(t for t in texts if t),
            "finish": m.get("finish") or m.get("finishReason"),
            "error": m.get("error"),
            "tools": tools,
        })
    time = info.get("time") if isinstance(info.get("time"), dict) else {}
    updated = info.get("updated_epoch")
    if updated is None:
        # opencode stores ms-epoch under time.updated/completed; fall back to that.
        raw = time.get("updated") or time.get("completed") or time.get("created")
        if raw is not None:
            updated = float(raw) / 1000.0 if float(raw) > 1e12 else float(raw)
    session = {
        "session": info.get("id") or info.get("session"),
        "updated_epoch": updated,
        "directory": info.get("directory") or info.get("cwd"),
        "autonomous": info.get("autonomous"),
        "messages": norm_msgs,
    }
    return classify_session(session, now=now)


if __name__ == "__main__":
    # Tiny self-demo: classify an export JSON passed on argv (or stdin) and print
    # the row + the resume command. Keeps the module runnable for an operator.
    import json
    raw = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1], encoding="utf-8").read()
    data = json.loads(raw)
    if "messages" in data and "info" not in data:
        row = classify_session(data)
    else:
        row = from_storage(data.get("info"), data.get("messages"))
    print(json.dumps({"row": row, "resume_cmd": resume_cmd(row),
                      "ledger_key": ledger_key(BACKEND, row["session"])}, indent=1))
