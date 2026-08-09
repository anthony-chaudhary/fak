#!/usr/bin/env python3
r"""fleet_sessions â€” the cross-account "what stopped, why, and how to resume" index.

The problem this kills: with N Claude Code accounts under ``<home>/.claude*``,
when a headless worker stops you otherwise have to GUESS which account owns it,
whether it really stopped or is just idle, why it stopped, and whether its
account is rate-limited. Resuming under the wrong account fails with
"No conversation found with session ID". This tool answers all of that
deterministically from the on-disk transcripts â€” no guessing.

Signals (transcript format v2.1.x):
  throttle  : a `<synthetic>` assistant message ".. limit . resets <when>"
              => the OWNING ACCOUNT is rate-limited until <when>.
  mid-tool  : last meaningful record is an assistant tool_use with no
              following tool_result  => the process died mid-work.
  interrupt : last text has "Login interrupted" / "[Request interrupted by user".
  parked    : last assistant text says it is awaiting a background task.
  done      : last assistant text reads as a wrap-up.
  live      : transcript appended within LIVE_MIN minutes.

Modes:
  summary  (default)  compact operator table, grouped by disposition
  json                full machine payload
  resume              ready-to-run, account-correct resume commands for
                      genuinely-stopped sessions on NON-throttled accounts

Usage:  python fleet_sessions.py [summary|json|resume] [--window H] [--max-age MIN]
"""
import os
import sys
import json
import glob
import re
import hashlib
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling helper
import fleet_version  # noqa: E402
import fleet_regdir  # noqa: E402  -- the host's one registry dir (never a second one)
import fleet_accounts  # noqa: E402  -- account-policy layer (worker/excluded/non-account)
import fleet_session_signals  # noqa: E402

USER = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))
NOW = dt.datetime.now(dt.timezone.utc)
LIVE_MIN = 4.0
# read_tail's window. A transcript at or below it is read WHOLE, so "the model never
# emitted a real assistant turn" (NEVER_STARTED) is authoritative rather than a tail
# artifact -- a session that truly never started is a few KB, far under this.
NEVER_STARTED_MAX_BYTES = 512 * 1024
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DONE_RE = re.compile(r"\b(complete|completed|shipped|pushed|committed|delivered|"
                     r"lease released|all checks|all set|terminated cleanly|"
                     r"goal is met|witness (?:holds|is met)|this completes)\b", re.I)
PARK_RE = re.compile(r"still running|awaiting|wait for the|will notify me|harness will|"
                     r"notify me when|waiting for the (?:exit|completion|result)|holding", re.I)
# autonomy: a session that WANTS to keep going on its own (vs. an interactive Q&A).
AUTON_RE = re.compile(r"<command-name>/(goal|loop|dispatch|next-up|fanout)\b|"
                      r"Stop hook is now active|/loop\b|ScheduleWakeup|"
                      r"autonomous-loop|keep working until", re.I)
# supervised: owned by the job-fleet supervisor -> leave it to run_supervise_loop, don't --resume.
SUPERVISED_RE = re.compile(r"JOB_SUPERVISED_WORKER|supervisor-spawned|/dispatch-loop\b", re.I)
GOALCLEAR_RE = re.compile(r"goal (?:condition )?(?:met|satisfied|cleared)|hook (?:auto-)?clear", re.I)
# Wrapper text that is NOT the session's real task instruction -- the harness injects
# these ahead of (or around) the operator's actual first message. The first head record
# whose text is none of these, once stripped, is the task identity used for dedup.
# ``<local-command-`` (not just ``<local-command-stdout>``) so EVERY local-command
# wrapper block is skipped. The harness opens a slash-command session with a
# ``<local-command-caveat>Caveat: ...`` record whose text is IDENTICAL across every such
# session; if that boilerplate is read as the first instruction, _task_sig collapses all
# slash-command sessions to ONE signature and the dedup pre-pass DEFERs distinct /goal
# workers as "duplicates" of one. Skipping the whole local-command-* family lets the walk
# reach the real /goal directive. (#fleet-sessions: caveat-wrapper false-dedup collapse)
_WRAPPER_RE = re.compile(
    r"^\s*(?:Caveat:|<system-reminder|<command-name>|<command-message>|"
    r"<command-args>|<local-command-|<user-memory|Codebase and user instructions)",
    re.I)
# Slash commands whose ARGUMENT defines the session's task. /effort, /model, etc. only
# CONFIGURE the session and carry fleet-identical args ("ultracode"), so capturing their
# command-args would re-collapse distinct workers; only these task-defining commands
# contribute their payload to the signature.
_TASK_CMD_RE = re.compile(r"<command-name>\s*/(goal|loop|dispatch|fanout|next-up)\b", re.I)
# A re-homed transcript opens with this exact synthetic resume prompt (see RESUME_PROMPT
# below). It is identical across every re-home, so it must NOT be treated as a task
# instruction -- otherwise every re-homed session in a project collapses to one signature.
_RESUME_PROMPT_PREFIX = "Resume where you left off"


def _first_instruction(head_records):
    """The session's real first task instruction, for the dedup signature.

    Walk the head records in order; return the first user/system text that is a
    genuine instruction -- skipping harness wrappers (caveat / system-reminder /
    command-* / local-command-* / memory blocks) and the fixed resume prompt a
    re-home injects. A ``/goal``/``/loop`` directive's ARGUMENT is the truest task
    identity, so when a TASK-DEFINING command (/goal,/loop,/dispatch,/fanout,/next-up)
    carries a command-args payload it is preferred; a config command's args (e.g.
    /effort ultracode) are ignored, since they are identical fleet-wide and would
    re-collapse distinct workers. Returns a normalized, whitespace-collapsed string."""
    cmd_args = None
    for ho in head_records:
        if ho.get("type") not in ("user", "system"):
            continue
        mc = (ho.get("message") or {}).get("content", ho.get("content", ""))
        txt = mc if isinstance(mc, str) else text_of(mc)
        if not txt or not txt.strip():
            continue
        # capture a /goal|/loop|... argument payload (the truest task identity); a config
        # command like /effort carries fleet-identical args, so only task-defining commands
        # contribute -- otherwise every /effort+/goal worker re-collapses to one signature.
        if _TASK_CMD_RE.search(txt):
            m = re.search(r"<command-args>(.*?)</command-args>", txt, re.S | re.I)
            if m and m.group(1).strip():
                cmd_args = " ".join(m.group(1).split())
        stripped = txt.strip()
        if _WRAPPER_RE.match(stripped):
            continue
        if stripped.startswith(_RESUME_PROMPT_PREFIX):
            continue
        return " ".join((cmd_args + " " + stripped).split()) if cmd_args else " ".join(stripped.split())
    return cmd_args or ""


def _task_sig(project, cwd, instruction):
    """Stable 16-hex signature of a task identity. Same (project, cwd, instruction)
    across different sids => the same recurring task => dedup candidates."""
    if not instruction:
        return ""
    raw = f"{project}\0{cwd}\0{instruction[:400]}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:16]


# disposition -> (category, cause). category buckets: INFRA / AGENT / USER / HANGING / LIVE.
CATEGORY = {
    "LIVE":          ("LIVE",    "live"),
    "DONE":          ("AGENT",   "completed"),
    "DEAD_MIDTOOL":  ("AGENT",   "crash_mid_tool"),
    "DEAD_KILLED":   ("AGENT",   "killed_mid_turn"),
    "USER_CLOSED":   ("USER",    "user_stopped"),
    "STOPPED_LIMIT": ("INFRA",   "rate_limit"),
    "STOPPED_APIERR":("INFRA",   "api_error"),
    "STOPPED_CONTEXT_EXHAUSTED": ("INFRA", "context_exhausted"),
    "INFRA_AUTH":    ("INFRA",   "auth"),
    "INFRA_ORG_DISABLED": ("INFRA", "org_disabled"),
    "NEVER_STARTED": ("AGENT",   "never_started"),
    "PARKED_WAIT":   ("HANGING", "parked_on_task"),
    "STOPPED_QUIET": ("HANGING", "ambiguous_quiet"),
}

def args_get(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

MODE = next((a for a in sys.argv[1:] if not a.startswith("-")), "summary")
# Default scan window. The registry WRITERS (dispatch preflight, resume watchdog)
# invoke `registry` with no --window, so this default is what they scan. classify()
# tails every transcript touched inside the window; under a crash-loop that produces
# thousands of STOPPED_APIERR corpses/hour, a 10h window made the preflight scan
# 22k+ files (6GB+) and blow past its 120s timeout -> a spurious REFUSE_NO_ACCOUNT.
# 3h keeps every LIVE (<=4min) and recently-stopped/resumable session while cutting
# the scan ~3x. Widen with --window or FLEET_WINDOW_H when triaging older sessions.
WINDOW_H = float(args_get("--window", os.environ.get("FLEET_WINDOW_H", "3")))
MAX_AGE = float(args_get("--max-age", "1e9"))
# Active probing: --probe[=blocked|stale|all|none]. Default none keeps the fast passive
# path untouched. A bare --probe means "blocked". Probe rows are appended to the scanned
# rows BEFORE the merge pipeline, so a fresh OK probe clears a stale carry-forward blocker
# and a fresh AUTH/LIMIT probe sets one -- without anyone running a real session.
def _probe_selector():
    for a in sys.argv:
        if a.startswith("--probe="):
            val = a.split("=", 1)[1].strip().lower()
            return val if val in ("blocked", "stale", "all", "none") else "blocked"
    if "--probe" in sys.argv:
        # bare "--probe" means blocked; "--probe <selector>" reads the next token only
        # when it names a known selector (otherwise the next token is some other arg).
        i = sys.argv.index("--probe")
        nxt = sys.argv[i + 1].strip().lower() if i + 1 < len(sys.argv) else ""
        return nxt if nxt in ("blocked", "stale", "all", "none") else "blocked"
    return "none"

PROBE_SELECTOR = _probe_selector()
# Anti-spam floor: skip an account probed within the last N minutes (0 = no floor).
PROBE_MIN_INTERVAL = float(args_get("--min-interval-min", "0"))

def read_tail(path, tail_bytes=512 * 1024):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []

def read_head(path, n=40):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in range(n):
                ln = f.readline()
                if not ln:
                    break
                out.append(ln)
    except OSError:
        pass
    return out

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    out.append(c if isinstance(c, str) else text_of(c))
        return " ".join(x for x in out if x)
    return ""

def last_tooluse(content):
    if isinstance(content, list):
        for b in reversed(content):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                return b.get("name")
    return None

def classify(path):
    st = os.stat(path)
    mtime = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
    age = (NOW - mtime).total_seconds() / 60.0
    cwd = git = sid = None
    throttle = None
    throttle_weekly = None  # weekly reset window, when the banner carries one
    pending = None          # an assistant tool_use still awaiting its tool_result
    last = None             # summary of the last meaningful user/assistant record
    last_ts = ""            # that record's OWN ISO-8601 timestamp (#3459 newest-copy key)
    last_kind = None        # nature of that final record (drives DONE vs DEAD vs USER_CLOSED)
    saw_assistant = False   # did the MODEL ever emit a real (non-synthetic) assistant turn?
    for ln in read_tail(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except json.JSONDecodeError:
            continue
        cwd = o.get("cwd", cwd)
        git = o.get("gitBranch", git)
        sid = o.get("sessionId", sid)
        if o.get("type") not in ("user", "assistant"):
            continue
        m = o.get("message") or {}
        c = m.get("content")
        txt = text_of(c)
        if m.get("model") == "<synthetic>":
            windows = fleet_session_signals.limit_resets(txt)
            primary = windows.get("daily") or windows.get("weekly")
            if primary:
                throttle = primary
                throttle_weekly = windows.get("weekly")
        last = {"role": m.get("role", o.get("type")), "txt": txt,
                "syn": m.get("model") == "<synthetic>", "stop": m.get("stop_reason")}
        # The TURN's own timestamp, not the file's mtime -- the key that ranks copies of
        # one session id in _copy_rank (#3459). Records missing it keep the prior value.
        last_ts = str(o.get("timestamp") or "") or last_ts
        if o.get("type") == "assistant":
            if m.get("model") != "<synthetic>":
                saw_assistant = True     # a genuine model turn (not a harness banner)
            n = last_tooluse(c)
            if n:
                pending = n
                last_kind = "assistant_tooluse"
            elif m.get("stop_reason") == "end_turn":
                last_kind = "assistant_end"      # model voluntarily ended its turn => finished
            else:
                last_kind = "assistant_text"
        else:  # user
            is_tr = isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
            if is_tr:
                pending = None
            if re.search(r"\[Request interrupted by user", txt):
                last_kind = "user_interrupt"     # user pressed Esc/Ctrl-C
            elif re.search(r"Login interrupted", txt) or \
                    re.search(r"<command-name>/(quit|exit|clear|logout|login)\b", txt):
                last_kind = "user_close"          # user issued a close/login command
            elif is_tr:
                last_kind = "user_toolresult"     # a tool answer with no following assistant
            else:
                last_kind = "user_text"
    lt = (last or {}).get("txt", "") or ""
    # disposition. Infra failures (rate-limit/API/auth) are separated from agent
    # failures (crash vs finished), from user stops, from hanging sessions.
    #   DONE        = finished cleanly (model ended its turn)        -> never resume   [AGENT]
    #   USER_CLOSED = user intentionally interrupted/closed it       -> never resume   [USER]
    #   DEAD_*      = crashed/killed mid-work                        -> resume if auton [AGENT]
    #   STOPPED_LIMIT / STOPPED_APIERR / INFRA_AUTH                  -> infra, retry    [INFRA]
    #   STOPPED_CONTEXT_EXHAUSTED = operator-stop/context wall       -> fresh continuation, NEVER resume-in-place [INFRA]
    #   PARKED_WAIT / STOPPED_QUIET                                  -> hanging/orphan  [HANGING]
    throttle_current = bool(throttle and last and last.get("syn"))
    if throttle_current:
        disp, reason = "STOPPED_LIMIT", "hit account session limit; resets %s" % throttle
    elif fleet_session_signals.is_auth_error(lt):
        kind = fleet_session_signals.auth_block_kind(lt)
        if kind == "access":
            # Org/admin disabled subscription access for THIS account. A /login on the
            # same account can't clear it -- but the transcript is intact and portable,
            # so re-homing it onto a different, non-org-disabled account WITH usage DOES
            # recover the work. Split it out from INFRA_AUTH so decide() can route it
            # through the same re-home machinery the rate-limit path uses, instead of
            # dead-ending at BLOCKED_AUTH.
            disp = "INFRA_ORG_DISABLED"
            reason = "stopped on a Claude subscription/access wall (re-home to a usable account; /login won't fix the owner)"
        elif kind == "credit":
            disp, reason = "INFRA_AUTH", "stopped on account credit/billing state"
        else:
            disp, reason = "INFRA_AUTH", "stopped on an auth/login requirement (needs re-login)"
    elif fleet_session_signals.is_operator_stop(lt) and last_kind != "assistant_end":
        # #3458: the gateway's operator-stop / BUDGET_CONTEXT_EXHAUSTED wall. It rides
        # in on the "API Error:" prefix but is TERMINAL for this transcript: a raw
        # `claude --resume` reloads the exhausted context and is refused with the same
        # 409 forever (the amnesia loop). Checked BEFORE is_api_error so it can never
        # fold into the transient STOPPED_APIERR retry path.
        disp, reason = ("STOPPED_CONTEXT_EXHAUSTED",
                        "stopped on an operator-stop/context-exhausted wall "
                        "(needs a fresh continuation, not a resume)")
    elif fleet_session_signals.is_api_error(lt) and last_kind != "assistant_end":
        disp, reason = "STOPPED_APIERR", "stopped on an API/transport error (transient infra)"
    elif age <= LIVE_MIN:
        disp, reason = "LIVE", "appended within %g min" % LIVE_MIN
    elif last_kind in ("user_interrupt", "user_close"):
        disp, reason = "USER_CLOSED", "ended on %s (user intentionally stopped it)" % last_kind
    elif pending:
        disp, reason = "DEAD_MIDTOOL", "died mid tool_use (%s) with no tool_result" % pending
    elif not saw_assistant and st.st_size <= NEVER_STARTED_MAX_BYTES:
        # Launched (a goal/prompt was written) but the model never emitted a single
        # real assistant turn, and the session is past LIVE_MIN. This is a launch/dispatch
        # non-start, NOT a mid-work hang: on 2026-07-09 it was 98/114 "HANGING" rows, 57 of
        # them born in one 3-minute dispatch wave. Split out so the HANGING headline stops
        # conflating never-started, cleanly-parked, and genuinely-hung. Resumable (there is
        # no partial work to lose) -> decide() re-launches an autonomous one.
        disp, reason = "NEVER_STARTED", "launched but produced no assistant turn (never started)"
    elif PARK_RE.search(lt):
        disp, reason = "PARKED_WAIT", "parked awaiting a background task"
    elif last_kind == "assistant_end" or (last and last.get("role") == "assistant" and DONE_RE.search(lt)):
        disp, reason = "DONE", "last assistant turn ended cleanly (stop_reason=end_turn / wrap-up)"
    elif last_kind == "user_toolresult":
        disp, reason = "DEAD_KILLED", "killed after a tool_result before the next assistant turn"
    elif last and last.get("role") == "assistant":
        disp, reason = "DONE", "ended on an assistant message"
    else:
        disp, reason = "STOPPED_QUIET", "quiet; no completion/crash/close signal"
    category, cause = CATEGORY.get(disp, ("HANGING", "unknown"))
    # autonomy / ownership â€” parse the HEAD RECORDS (the session's own directive),
    # not a content blob. A session is autonomous only if it was actually launched
    # with /goal|/loop|/dispatch|/fanout|/next-up, or a Stop-hook goal was installed,
    # or it carries the supervised-worker marker. This avoids flagging interactive
    # sessions that merely DISCUSS goals/loops.
    autonomous = supervised = False
    head_records = []          # parsed head objects, for the dedup task-signature
    for hl in read_head(path, 30):
        hl = hl.strip()
        if not hl:
            continue
        try:
            ho = json.loads(hl)
        except json.JSONDecodeError:
            continue
        head_records.append(ho)
        htxt = ""
        if ho.get("type") in ("user", "system"):
            mc = (ho.get("message") or {}).get("content", ho.get("content", ""))
            htxt = mc if isinstance(mc, str) else text_of(mc)
        if AUTON_RE.search(htxt):
            autonomous = True
        if SUPERVISED_RE.search(htxt):
            supervised = True
            autonomous = True
    autonomous = autonomous or supervised
    # task signature: the dedup identity of a recurring autonomous task. Computed for
    # autonomous rows (the only ones we ever auto-resume / dedup) AND for LIVE/DONE rows
    # so a live/done sibling can serve as task COVER for its duplicates. Never for plain
    # interactive sessions -- their similar-looking prompts must not be deduped.
    task_sig = ""
    if autonomous or disp in ("LIVE", "DONE"):
        task_sig = _task_sig(os.path.basename(os.path.dirname(path)), cwd or "",
                             _first_instruction(head_records))
    return {"disp": disp, "category": category, "cause": cause, "reason": reason,
            "last_kind": last_kind, "age_min": round(age, 1),
            "seen_utc": mtime.isoformat(), "last_ts": last_ts,
            "throttle_reset": throttle if throttle_current else None,
            "throttle_weekly": throttle_weekly if throttle_current else None,
            "throttle_seen": throttle,
            "throttle_current": throttle_current,
            "pending_tool": pending,
            # the literal HTTP/transport code from the terminal banner (429/529/401/...),
            # so "last reported status" is a real field rather than buried in `last` prose.
            "http_status": fleet_session_signals.http_status(lt),
            # who started this session: an autonomous /goal|/loop|supervised worker is
            # "agent"-initiated; everything else is an interactive ("user") session.
            "initiated_by": "agent" if autonomous else "user",
            "session": sid or os.path.splitext(os.path.basename(path))[0],
            "cwd": cwd, "git": git, "last": lt[:200].replace("\n", " "), "path": path,
            "autonomous": autonomous, "supervised": supervised,
            "task_sig": task_sig, "records": st.st_size}

ACCT_POLICY = fleet_accounts.load_policy()

def _account_still_worker(acct):
    return fleet_accounts.is_worker(acct, USER, ACCT_POLICY)


def active_rearmed_session_ids(reg_dir):
    """Return UUIDs whose latest attempt-budget event is a rearm."""
    path = os.path.join(reg_dir, "resume_ledger.jsonl")
    active = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                sid = str(rec.get("session") or "").strip()
                if not sid:
                    continue
                phase = str(rec.get("phase") or rec.get("outcome") or "").strip().lower()
                action = str(rec.get("action") or "").strip().lower()
                if phase == "rearm":
                    active[sid] = True
                elif phase in ("launched", "settled", "operator_settled", "consolidated") or action.startswith("consolidate"):
                    active.pop(sid, None)
    except OSError:
        return set()
    return set(active)


def scan():
    rows, throttle = [], {}
    rearmed = active_rearmed_session_ids(REG_DIR)
    for acct_dir in glob.glob(os.path.join(USER, ".claude*")):
        acct = os.path.basename(acct_dir)
        proj = os.path.join(acct_dir, "projects")
        if not os.path.isdir(proj):
            continue
        # account policy: skip tombstoned/excluded accounts (e.g. the backup
        # account) so they never produce rows, resume commands, or plan entries.
        if not _account_still_worker(acct):
            continue
        for path in glob.glob(os.path.join(proj, "*", "*.jsonl")):
            base = os.path.splitext(os.path.basename(path))[0]
            if not UUID_RE.match(base) or os.path.basename(os.path.dirname(path)).startswith("wf_"):
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            age = (NOW - dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)).total_seconds() / 60.0
            # #4157: rearm is a targeted reconsideration request. It overrides
            # the mtime scan window for exactly this UUID without widening the
            # expensive whole-fleet window. A later launch/settle consumes it.
            if (age > WINDOW_H * 60 or age > MAX_AGE) and base not in rearmed:
                continue
            r = classify(path)
            r["account"] = acct
            r["project"] = os.path.basename(os.path.dirname(path))
            rows.append(r)
            if r["throttle_reset"] and r["disp"] == "STOPPED_LIMIT" and fleet_accounts.throttle_is_active(r["throttle_reset"]):
                cur = throttle.get(acct)
                if not cur or r["age_min"] < cur["age_min"]:
                    entry = {"reset": r["throttle_reset"], "age_min": r["age_min"]}
                    if r.get("throttle_weekly"):
                        entry["weekly"] = r["throttle_weekly"]
                    throttle[acct] = entry
    rows.sort(key=lambda r: r["age_min"])
    return rows, throttle

def merge_known_throttle(throttle, rows):
    """Carry forward cached account limits whose reset has not expired yet."""
    newest = {}
    for r in sorted(rows, key=lambda x: x["age_min"]):
        newest.setdefault(r["account"], r)
    cleared = {
        acct for acct, r in newest.items()
        if r.get("disp") == "LIVE" and not r.get("throttle_current")
    }
    prev = fleet_accounts.load_registry().get("throttle", {}) or {}
    merged = {}
    for source in (prev, throttle):
        for acct, info in source.items():
            if not _account_still_worker(acct):
                continue
            if acct in cleared:
                continue
            if not fleet_accounts.throttle_is_active(info):
                continue
            merged[acct] = info if isinstance(info, dict) else {"reset": info}
    return merged

def _parse_utc(raw):
    if not raw:
        return None
    try:
        ts = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)

def _row_seen_utc(row):
    seen = _parse_utc(row.get("seen_utc"))
    if seen is not None:
        return seen
    age = fleet_accounts._age_min(row)
    if age is None:
        return None
    return NOW - dt.timedelta(minutes=age)

def _auth_info_seen_utc(info, generated_utc=None):
    if not isinstance(info, dict):
        return None
    seen = _parse_utc(info.get("seen_utc"))
    if seen is not None:
        return seen
    age = fleet_accounts._age_min(info)
    generated = _parse_utc(generated_utc)
    if age is None or generated is None:
        return None
    return generated - dt.timedelta(minutes=age)

def _auth_info_from_row(row):
    last = str(row.get("last") or row.get("reason") or "")
    seen = _row_seen_utc(row) or NOW
    return {
        "block_kind": fleet_session_signals.auth_block_kind(last),
        "block_reason": fleet_session_signals.auth_block_reason(last),
        "seen_utc": seen.isoformat(),
        "age_min": row.get("age_min"),
        "session": row.get("session"),
        "project": row.get("project"),
        "last": last[:200],
    }

def _normalize_auth_info(info):
    row = dict(info) if isinstance(info, dict) else {
        "block_kind": "auth",
        "block_reason": str(info) if info else "auth/login required",
    }
    reason_text = " ".join(
        str(row.get(k) or "") for k in ("last", "reason", "block_reason")
    )
    if reason_text.strip():
        row["block_kind"] = fleet_session_signals.auth_block_kind(reason_text)
        row["block_reason"] = fleet_session_signals.auth_block_reason(reason_text)
    else:
        row.setdefault("block_kind", "auth")
        row.setdefault("block_reason", "auth/login required")
    return row

def merge_known_auth(rows):
    """Carry forward account auth blockers until a newer successful turn clears them."""
    latest_success = {}
    current_auth = {}
    for r in rows:
        acct = r.get("account")
        seen = _row_seen_utc(r)
        if not acct or seen is None:
            continue
        if r.get("disp") in ("LIVE", "DONE"):
            if acct not in latest_success or seen > latest_success[acct]:
                latest_success[acct] = seen
        if r.get("disp") == "INFRA_AUTH":
            cur = current_auth.get(acct)
            if cur is None or seen > (_auth_info_seen_utc(cur) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)):
                current_auth[acct] = _auth_info_from_row(r)

    prev_reg = fleet_accounts.load_registry()
    prev = prev_reg.get("auth", {}) or {}
    prev_generated = prev_reg.get("generated_utc")
    merged = {}
    for source in (prev, current_auth):
        for acct, info in source.items():
            if not _account_still_worker(acct):
                continue
            row = _normalize_auth_info(info)
            seen = _auth_info_seen_utc(row, prev_generated)
            success_seen = latest_success.get(acct)
            if success_seen is not None and seen is not None and success_seen > seen:
                continue
            merged[acct] = row
    return merged

DEAD = {"DEAD_MIDTOOL", "DEAD_KILLED"}              # crashed/killed mid-work -> resumable
# Infra dispositions: the SERVER interrupted the session (rate-limit / transient API
# error / org-disabled), it did NOT finish. #1353: these are safe to auto-resume
# REGARDLESS of autonomy -- an interactive chat walled mid-conversation by a transient
# 529 was interrupted, not abandoned, so the autonomy gate over-reaches if it strands it.
# (USER_CLOSED / DONE stay excluded -- those ARE intentional human stops; INFRA_AUTH stays
# excluded -- it needs a human re-login, not a resume.)
INFRA_RESUMABLE = ("STOPPED_LIMIT", "STOPPED_APIERR", "INFRA_ORG_DISABLED")
# #3458: STOPPED_CONTEXT_EXHAUSTED is deliberately in NEITHER set: not INFRA_RESUMABLE
# (a resume re-hits the same 409 wall) and not STOPLIKE (so decide() never stamps a
# resume_cmd for it -- the recovery is a FRESH continuation, not `claude --resume`).
STOPLIKE = DEAD | set(INFRA_RESUMABLE) | {"STOPPED_QUIET", "INFRA_AUTH",
                   "USER_CLOSED", "PARKED_WAIT", "NEVER_STARTED"}
# $FLEET_REG_DIR when the fleet names it, else the host ladder (fleet_regdir). This module
# WRITES sessions.json / decisions.log / transitions.log / resume_plan.json, so its old
# clone-root fallback is what physically forked a host: a hand-run `fleet_sessions.py
# registry` published a second, ledger-less roster beside the watchdog's live one.
REG_DIR = fleet_regdir.reg_dir()
RESUME_PROMPT = ("Resume where you left off; re-establish any /goal or /loop "
                 "and continue toward it.")

def config_dir(acct):
    return os.path.join(USER, acct)

def _verdict_freshness(account, rows, auth):
    """How fresh is THIS account's verdict, and where did it come from?

    Returns (source, age_min):
      "probe"   a synthetic probe row fed this refresh (ground truth, age~0)
      "passive" a real transcript row inside the window is the evidence
      "carried" no fresh row -- the verdict is a carry-forward from the auth/throttle
                map (the stale-latch case the observability exists to surface)
    The age is minutes since the newest evidence backing the verdict.
    """
    acct_rows = [r for r in rows if r.get("account") == account]
    probe_rows = [r for r in acct_rows if r.get("project") == "_probe"]
    if probe_rows:
        age = min((float(r.get("age_min") or 0.0) for r in probe_rows), default=0.0)
        return "probe", round(age, 1)
    session_rows = [r for r in acct_rows if r.get("project") != "_probe"]
    if session_rows:
        age = min((float(r.get("age_min") or 0.0) for r in session_rows
                   if r.get("age_min") is not None), default=None)
        if age is not None:
            return "passive", round(age, 1)
    # no fresh row -> the verdict rides a carried auth/throttle entry; age it from seen_utc
    info = (auth or {}).get(account)
    seen = fleet_accounts._parse_utc(info.get("seen_utc")) if isinstance(info, dict) else None
    if seen is not None:
        return "carried", round((NOW - seen).total_seconds() / 60.0, 1)
    return "carried", None


def account_availability(throttle, rows, auth=None):
    """Per worker account: is it safe for the switcher to offer right now?"""
    registry = {"generated_utc": NOW.isoformat(), "auth": auth or {}}
    annotated = fleet_accounts.annotated_roster(
        USER, ACCT_POLICY, registry=registry, throttle=throttle, sessions=rows)
    out = []
    for a in annotated:
        if a["kind"] != "worker":
            continue
        source, age = _verdict_freshness(a["account"], rows, auth)
        out.append({
            "account": a["account"], "tag": a["tag"],
            "config_dir": config_dir(a["account"]),
            "available": a["available"],
            "blocked": a["blocked"],
            "block_kind": a["block_kind"],
            "block_reason": a["block_reason"],
            "throttled": a["throttled"],
            "reset": a["reset"],
            "weekly": a.get("weekly"),
            "active_sessions": a["active_sessions"],
            "live_sessions": a["live_sessions"],
            "auth_blocked_sessions": a["auth_blocked_sessions"],
            # freshness: makes a stale carried-forward verdict visibly stale, the single
            # field whose absence let seven accounts silently latch as blocked.
            "verdict_source": source,
            "verdict_age_min": age,
        })
    out.sort(key=lambda a: (not a["available"], a["tag"]))
    return out

def resume_cmd(r):
    """Operator-runnable resume command, account-correct.

    For a re-homed session the command first copies the transcript out of the
    throttled owner's config dir into the healthy target account's config dir
    (claude --resume is CLAUDE_CONFIG_DIR + cwd scoped, so the conversation must
    physically live under the account it resumes on), then resumes there."""
    cfg = r.get("resume_config_dir") or config_dir(r["account"])
    prefix = ""
    if r.get("rehomed"):
        src = r.get("source_config_dir") or config_dir(r["account"])
        proj, sid = r.get("project", ""), r["session"]
        src_file = os.path.join(src, "projects", proj, sid + ".jsonl")
        dst_dir = os.path.join(cfg, "projects", proj)
        prefix = (f"New-Item -ItemType Directory -Force -Path '{dst_dir}' | Out-Null; "
                  f"Copy-Item '{src_file}' '{os.path.join(dst_dir, sid + '.jsonl')}' -Force; ")
    return (f"{prefix}$env:CLAUDE_CONFIG_DIR='{cfg}'; "
            f"claude --resume {r['session']} -p '{RESUME_PROMPT}' --dangerously-skip-permissions")

# How many sessions one account may be assigned IN A SINGLE re-home pass before
# it is considered "full" and dropped from the candidate pool. A re-home adds a
# fresh autonomous `claude --resume` to the target, and an account that is already
# running near its session ceiling will itself hit the usage limit the moment the
# burst lands -- which is exactly the 32->1 stampede that wedged every resume onto
# one account. The cap is the in-pass admission ceiling: assigned + already-live
# must stay under it. Override with FAK_REHOME_CAP for hosts with fatter accounts.
REHOME_CAP = int(os.environ.get("FAK_REHOME_CAP", "4"))

# After this many IN-PLACE auto-resumes that did not stick, a session that keeps dying
# on its OWN account is re-homed onto a different healthy seat instead of being re-pinned
# to the owner -- the "retrying on the same account isn't working, use another one"
# escalation (#1342/#1345/#1859). The owner-throttled/org-disabled paths already re-home
# on the first pass; THIS is the healthy-owner-but-repeatedly-crashing path they missed.
# 0 disables escalation (always in-place). Override with FAK_RESUME_ESCALATE_AFTER.
RESUME_ESCALATE_AFTER = int(os.environ.get("FAK_RESUME_ESCALATE_AFTER", "2"))

# Owner-loaded spread kill-switch (mirrors internal/resume/rehome LoadSpreadEnv, the
# Go source of truth): safe-by-default ON; FAK_LOAD_REHOME=0/false/off restores the
# historical resume-in-place-whenever-the-owner-is-healthy behavior.
LOAD_REHOME = os.environ.get("FAK_LOAD_REHOME", "").strip().lower() not in ("0", "false", "off", "no")


def _owner_live_load(availability, account):
    """The owner's live-session count in the availability snapshot, plus whether it
    has reached REHOME_CAP. An owner absent from the snapshot reads as not loaded --
    the spread fires only on positive load evidence, never on a missing row.
    Mirrors internal/resume/rehome ownerLiveLoad."""
    for a in (availability or []):
        if a.get("account") == account:
            live = int(a.get("live_sessions") or 0)
            return live, REHOME_CAP > 0 and live >= REHOME_CAP
    return 0, False


def _has_positive_evidence(a) -> bool:
    """The launch-boundary admission predicate (#619): True iff an account's health
    verdict rests on POSITIVE evidence it is serving right now -- a fresh probe
    (verdict_source 'probe') or a real session row inside the window ('passive') --
    rather than a stale 'carried' verdict or the absence-of-evidence default ('none').
    account_availability stamps exactly one verdict_source per account, so this is the
    single field load-routing consults. Load is only ever admitted onto accounts that
    pass this gate; a carried verdict must be re-probed before it can take a workload.
    That is what makes routing deterministic: the SAME evidence yields the SAME
    decision regardless of which pass or tool asks -- a carried 'available' that
    flip-flops with whether the pass happened to probe can no longer admit load."""
    return str(a.get("verdict_source") or "none") in ("probe", "passive")


def rehome_targets(availability, exclude_account, assigned=None, cap=None):
    """Available Claude worker accounts a throttled session can move to, least
    loaded first. opencode accounts are excluded: a Claude transcript can only
    resume under another Claude config dir, not an opencode one.

    This is the SHARED re-home ranking seam. Three consumers depend on it: the
    in-module admission gate :func:`_admissible_targets` (which filters this
    ranking down to positive-evidence targets) and, across the module boundary,
    :mod:`resume_resolver` (the interactive ``c --resume`` path, with its own
    owner-probe discipline). Because an out-of-module caller relies on it, it is
    public; ``_rehome_targets`` remains as a back-compat alias.

    ``assigned`` is the per-account count this pass has ALREADY re-homed onto each
    target (account basename -> n). It is folded into the load so a burst of
    throttled sessions spreads across healthy accounts instead of all picking the
    same momentary least-loaded one: the snapshot's live/active counts are static
    within a pass, so without this every caller computes the identical winner and
    stampedes it. An account whose (already-live + just-assigned) load reaches
    ``cap`` drops out of the pool entirely -- better to DEFER_THROTTLED and
    wait for a reset than to pile a session onto an account that will limit-wall.

    ``cap`` defaults to ``REHOME_CAP`` (the fleet burst-spread ceiling). A caller
    routing a SINGLE interactive resume -- not a burst -- can pass a larger cap (or
    ``math.inf``) to get the same least-loaded ranking without the stampede gate, so
    a lone resume is not stranded on PIN_BLOCKED when the only healthy account is
    merely over the fleet cap (the day24 incident: available, but live=7 >= cap=4)."""
    assigned = assigned or {}
    cap = REHOME_CAP if cap is None else cap
    cands = []
    for a in (availability or []):
        acct = a.get("account", "")
        if (not a.get("available")
                or acct == exclude_account
                or not str(acct).startswith(".claude")):
            continue
        base_load = int(a.get("live_sessions") or 0)
        if base_load + assigned.get(acct, 0) >= cap:
            continue                       # already at this pass's admission ceiling
        cands.append(a)
    # Rank by load (live + in-pass assigned) first, but break ties by PROVEN health:
    # an account with a fresh positive verdict sorts ahead of one whose `available`
    # is merely the absence-of-evidence default. account_availability stamps
    # verdict_source as one of probe (a live probe just hit it), passive (a real
    # session row inside the window proves it alive), or carried (a stale verdict
    # carried forward with no fresh evidence). probe/passive are genuine positive
    # evidence; carried/none are not -- so probe/passive sort ahead of carried/none.
    # This is RANKING only; the hard launch-boundary admission rule (#619) -- refusing
    # to route load onto a carried verdict at all -- is enforced by _admissible_targets,
    # which decide() consults. rehome_targets stays a pure ranker so resume_resolver
    # (a separate consumer with its own owner-probe discipline) keeps its behavior.
    def _unproven(a):
        return 0 if _has_positive_evidence(a) else 1
    cands.sort(key=lambda a: (int(a.get("live_sessions") or 0) + assigned.get(a.get("account", ""), 0),
                              _unproven(a),
                              int(a.get("active_sessions") or 0),
                              str(a.get("tag") or a.get("account") or "")))
    return cands


# Back-compat alias: this ranker was private (`_rehome_targets`) before its
# cross-module consumers (resume_resolver, its tests) made it a de-facto public
# seam. The name was promoted to `rehome_targets`; the underscore alias keeps
# every existing caller -- in this module, in resume_resolver, and in the tests
# -- working unchanged.
_rehome_targets = rehome_targets


def _admissible_targets(availability, exclude_account, assigned=None):
    """Re-home targets that pass the kernel's launch-boundary admission rule (#619).

    The load-bearing rule: NEVER admit a real workload onto a CARRIED / absence-of-
    evidence verdict. rehome_targets ranks the available candidates; this gate then
    drops every one whose health verdict is not POSITIVE evidence it is serving right
    now (verdict_source probe | passive). A carried "available" -- the day24 incident,
    where the account read available@22:17, throttled@22:19, available@22:20 purely on
    whether that pass probed -- can no longer take load: it must be re-probed first.
    decide() consults THIS, not the raw ranker, so the same evidence yields the same
    routing decision on every pass; with no admissible target it DEFERs and waits for
    a probe, which is the deterministic outcome the issue requires."""
    return [t for t in rehome_targets(availability, exclude_account, assigned)
            if _has_positive_evidence(t)]

def _owner_available(availability, account):
    """True when this session's CURRENT owner account is ADMISSIBLE for an in-place
    resume in the freshness-stamped snapshot. Used to detect a STALE STOPPED_LIMIT
    banner: a transcript copied off a once-throttled owner carries that owner's old
    "limit reached" line, so decide() can read STOPPED_LIMIT for a session whose owner
    has since cleared. The static counterpart of resume_resolver's carried-throttle
    re-probe -- account_availability already folds a fresh probe row into `available`,
    so a proven-available owner means the limit lifted: resume in place, don't re-home.
    Admissible (#619) = available AND backed by positive evidence (probe/passive):
    resuming a workload onto the owner IS a launch, so this in-place path obeys the
    same launch-boundary rule as re-home target selection. A carried "available" owner
    -- the exact passive-banner-vs-probe flip from the day24 incident -- is NOT trusted
    to take load in place; it falls through to re-home/defer until a fresh probe
    confirms it. Conservative when the snapshot is absent (availability is None) ->
    False, which preserves the pre-existing re-home/defer behavior."""
    for a in (availability or []):
        if a.get("account") == account:
            return bool(a.get("available")) and _has_positive_evidence(a)
    return False

def _ledger_blocked_sids(reg_dir=None):
    """Sids the resume ledger shows are permanently blocked -- they hit the attempt
    cap or an unrecoverable (auth) wall on their last launch. Read so the dedup
    primary election can SKIP them: a poisoned primary must not bury its duplicates
    forever. Mirrors the watchdog's per-sid gate, but read-only and ledger-derived,
    so this classifier needs no import of the watchdog.

    A sid is blocked when it has >= MAX_ATTEMPTS launch rows, OR a row marking a
    manual/operator settle, OR its last recorded outcome is 'unrecoverable'. Absent
    or unreadable ledger => empty set (fail-open: dedup still elects a primary)."""
    reg_dir = reg_dir or REG_DIR
    max_attempts = int(os.environ.get("FAK_MAX_ATTEMPTS", "8"))
    path = os.path.join(reg_dir, "resume_ledger.jsonl")
    launches: dict[str, int] = {}
    blocked: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                sid = rec.get("session")
                if not sid:
                    continue
                # Re-arm marker (2026-07-10): reclaims a sid that burned its attempt budget on a
                # KNOWN-transient infra fault (e.g. the managed-cache-1h-TTL 400 wave, #2178) rather
                # than a real defect. Processed in append order, so it zeroes the launches accrued
                # BEFORE it and lifts a prior block, letting the dedup primary election pick the sid
                # again; a later manual_override/unrecoverable row re-blocks (last write wins). Mirrors
                # the .ps1 launch gate so the planner and the launcher agree on eligibility.
                if rec.get("phase") == "rearm" or rec.get("outcome") == "rearm":
                    launches[sid] = 0
                    blocked.discard(sid)
                    continue
                if rec.get("manual_override") or str(rec.get("action", "")).startswith("consolidate"):
                    blocked.add(sid)
                if rec.get("outcome") == "unrecoverable":
                    blocked.add(sid)
                if rec.get("phase") in ("launched", "resumed") or rec.get("cause"):
                    launches[sid] = launches.get(sid, 0) + 1
    except OSError:
        return set()
    for sid, n in launches.items():
        if n >= max_attempts:
            blocked.add(sid)
    return blocked


def _ledger_inplace_attempts(reg_dir=None):
    """Per-sid count of prior IN-PLACE (non-rehomed) auto-resume launches from the durable
    ledger. decide() consults this to ESCALATE a session that keeps dying on its OWN
    account: after RESUME_ESCALATE_AFTER in-place attempts that did not stick, the next
    resume is re-homed onto a different healthy seat rather than re-pinning the owner
    (#1342/#1345/#1859). A re-homed launch (rehomed=True) is NOT counted -- once a session
    has moved seats its in-place streak is over, so it stays escalated instead of
    ping-ponging back to the failing owner. Absent/unreadable ledger => empty (fail-open:
    no escalation, plain in-place resume exactly as before)."""
    reg_dir = reg_dir or REG_DIR
    path = os.path.join(reg_dir, "resume_ledger.jsonl")
    counts: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                sid = rec.get("session")
                if not sid or rec.get("rehomed"):
                    continue
                # Re-arm marker (#2178 fix): a reclaimed sid restarts its in-place streak from 0, so
                # it resumes in place once more before re-home escalation rather than immediately
                # ping-ponging seats on reclaim. Zeroes the streak accrued before the marker.
                if rec.get("phase") == "rearm" or rec.get("outcome") == "rearm":
                    counts[sid] = 0
                    continue
                if rec.get("phase") in ("launched", "resumed") or rec.get("cause"):
                    counts[sid] = counts.get(sid, 0) + 1
    except OSError:
        return {}
    return counts


def _relaunch_badge(n):
    """Compact per-session suffix for the operator table (#3553): a row auto-resumed N>0
    times IN PLACE on its own seat carries ``relaunch×N`` so a repeat-crasher is visible
    without cross-referencing the ledger. 0 attempts renders nothing -- no badge spam on
    the common (never-relaunched) case."""
    return f"  relaunch×{n}" if n > 0 else ""


def _resumable_disp(r):
    """Whether a row's disposition is one the resume path might auto-resume (so it is a
    candidate to be a dedup primary or a deferred duplicate). LIVE/DONE are NOT resumable
    but participate as task COVER -- handled separately in _dedup_defer.

    #1353: infra dispositions (the server interrupted the session) are resumable
    REGARDLESS of autonomy; the agent-crash (DEAD) path keeps the autonomy gate so an
    interactive chat the human walked away from after a crash is not relaunched."""
    if r["disp"] in INFRA_RESUMABLE:
        return True
    # NEVER_STARTED joins the autonomy-gated resumable set: an autonomous goal that never
    # produced a turn has no partial work, so re-launching it is the lowest-risk resume.
    return bool(r.get("autonomous")) and (r["disp"] in DEAD or r["disp"] == "NEVER_STARTED")


def _copy_rank(r):
    """Rank one COPY of a session id against its siblings; the highest tuple wins.

    Ordered last-turn timestamp, then transcript size, then file mtime -- deliberately
    the same rule as the Go source of truth (internal/resume/sweep.Classify: "superset =
    latest last-ts, then most records (NOT file mtime)"). mtime-first is the exact trap
    #3459 was filed on: when a dead driver appends a synthetic "API Error" banner to a
    stale PREFIX of a session, that copy's mtime jumps to now WITHOUT gaining a single
    real turn -- so the freshest FILE can be the least-progressed TRANSCRIPT. ISO-8601
    UTC timestamps compare correctly as plain strings, so no parsing is needed."""
    return (str(r.get("last_ts") or ""),
            int(r.get("records") or 0),
            str(r.get("seen_utc") or ""))


def _newest_by_sid(rows):
    """{session id: the ONE row that speaks for it} -- the newest copy of each sid.

    A re-home writes the same sid under another config dir, so one session routinely
    exists as ~20 files across the ``.claude-*`` account dirs on this host and scan()
    classifies EVERY one of them independently."""
    best: dict[str, dict] = {}
    for r in rows:
        sid = r.get("session", "")
        cur = best.get(sid)
        if cur is None or _copy_rank(r) > _copy_rank(cur):
            best[sid] = r
    return best


def _live_resume_sids():
    """Session ids a `claude --resume <sid>` process is driving on this host RIGHT NOW.

    Delegates to resume_sweep's census (one process-enumeration implementation, not a
    fork) and is imported LAZILY so the registry refresh never pays for it -- or dies on
    it -- outside decide(). FAIL-OPEN: any failure yields an EMPTY set, leaving the gate
    inert, because an unreadable process table must never strand a genuinely-crashed
    session. That census is Windows-only today; the cross-platform belt is the
    driver-side WatchdogSkipLive guard in internal/resume/watchdog.go, which refuses to
    LAUNCH a live sid whatever the plan says."""
    try:
        import resume_sweep  # noqa: PLC0415 -- lazy: only paid for on the decide() path
        return {str(s) for s in resume_sweep.live_resume_sids()}
    except Exception:
        return set()


def _live_or_stale_copy_gate(rows, live_sids=None):
    """Pre-pass: decide WHICH copy of a session id may reach the decision ladder at all,
    and drop every copy of a session that is demonstrably alive (#3459).

    The harm this closes: ``resume_plan.json`` queued 94aea02a as STOPPED_APIERR --
    crashed -- while a live `claude` process was advancing a NEWER copy of that same
    UUID under another account dir (its terminal turn was an ordinary Write tool_use).
    Running the watchdog `--live` would have fired a second driver onto a live
    transcript. scan() classifies every on-disk copy independently, so the stale copy's
    synthetic "API Error" tail became the plan's verdict for the whole session.

    Two gates, both keyed on the session id, both stamped BEFORE decide()'s per-row loop
    so a gated row never re-homes, never consumes a REHOME_CAP slot, and never reaches
    plan_entry:
      (1) LIVENESS -- a running `claude --resume <sid>` driver (process census), or any
          copy of the sid that classified LIVE, means the session is already being
          advanced. NO copy of it may be queued, whatever disposition the bytes on disk
          carry. The copy that is itself LIVE is left alone: decide()'s own SKIP_LIVE
          already names it correctly.
      (2) NEWEST-COPY -- of the remaining copies only the newest (_copy_rank: last-turn
          timestamp, then size, then mtime) speaks for the session, so the plan
          classifies the terminal turn of the newest transcript rather than an arbitrary
          older one. Every older copy is stamped SKIP_STALE_COPY here.

    live_sids is injectable so the decision stays hermetic under test; None means "no
    census was taken" (inert), NOT "nothing is live"."""
    live = {str(s) for s in (live_sids or ())}
    live |= {r.get("session", "") for r in rows if r.get("disp") == "LIVE"}
    live.discard("")
    newest = _newest_by_sid(rows)
    for r in rows:
        sid = r.get("session", "")
        if r.get("disp") == "LIVE":
            continue                                   # decide() stamps this one SKIP_LIVE
        if sid and sid in live:
            r["action"] = "SKIP_LIVE_DRIVER"
        elif r is not newest.get(sid):
            r["action"] = "SKIP_STALE_COPY"
    return rows


# Actions stamped by a pre-pass BEFORE decide()'s ladder runs. A row carrying one is
# already decided: it skips the ladder entirely (so it can never re-home or consume an
# `assigned` cap slot) and, being none of them AUTO_RESUME, never reaches the plan.
PRE_STAMPED = ("DEFER_DUPLICATE_TASK", "SKIP_LIVE_DRIVER", "SKIP_STALE_COPY")


def _dedup_defer(rows, reg_dir=None):
    """Pre-pass: when several crashed sessions are the SAME recurring autonomous task
    (same task_sig), resume only ONE and DEFER the rest. Stamps the losers'
    ``action = "DEFER_DUPLICATE_TASK"`` BEFORE decide()'s per-row loop, so a deferred
    duplicate never reaches _rehome_targets and never consumes a REHOME_CAP slot.

    Steps, per the design:
      (a) collapse same-sid rows (a re-home writes the same sid under two config
          dirs) to the newest copy, so a re-homed primary is not its own phantom dup;
      (b) group by task_sig, INCLUDING LIVE/DONE rows (cover check);
      (c) if any member is LIVE or DONE the task is already covered -> defer every
          resumable member, elect no primary;
      (d) else elect a deterministic primary (records desc, seen_utc desc, sid asc),
          excluding ledger-blocked sids, and defer the rest."""
    blocked = _ledger_blocked_sids(reg_dir)
    # (a) collapse same-sid -> newest (by seen_utc, then records) so dedup keys on the
    #     live copy of a re-homed session, not a stale origin copy.
    by_sid: dict[str, dict] = {}
    for r in rows:
        sid = r.get("session", "")
        cur = by_sid.get(sid)
        if cur is None or (r.get("seen_utc", ""), r.get("records", 0)) > (
                cur.get("seen_utc", ""), cur.get("records", 0)):
            by_sid[sid] = r
    # (b) group by task_sig (skip empty sigs -- interactive / no-instruction rows)
    groups: dict[str, list] = {}
    for r in by_sid.values():
        sig = r.get("task_sig") or ""
        if sig:
            groups.setdefault(sig, []).append(r)
    for sig, members in groups.items():
        resumable = [r for r in members if _resumable_disp(r)]
        if len(resumable) <= 1:
            continue                                   # nothing to dedup
        covered = any(r["disp"] in ("LIVE", "DONE") for r in members)
        if covered:
            for r in resumable:                        # (c) live/done sibling covers it
                r["action"] = "DEFER_DUPLICATE_TASK"
            continue
        # (d) elect primary: drop ledger-blocked sids (a poisoned primary must not bury
        #     its duplicates), then rank most-progressed first. seen_utc is descending
        #     (newer wins) and sid ascending as deterministic, time/random-free ties.
        eligible = [r for r in resumable if r.get("session") not in blocked] or resumable
        winner = min(eligible, key=lambda r: (
            -int(r.get("records") or 0),
            _desc(r.get("seen_utc", "")),
            r.get("session", "")))
        for r in resumable:
            if r is not winner:
                r["action"] = "DEFER_DUPLICATE_TASK"
    return rows


def _desc(s):
    """Sort helper: make a string compare DESCENDING inside an ascending key tuple.
    seen_utc is an ISO timestamp where NEWER should win, so invert its byte order."""
    return tuple(-ord(c) for c in (s or ""))


def _resume_inplace_or_escalate(r, availability, assigned, inplace_counts):
    """Stamp a resumable IN-PLACE row (an agent crash or a transient API error whose OWN
    account is healthy) AUTO_RESUME. Normally that resumes on the owner. But if this
    session has already been resumed in place >= RESUME_ESCALATE_AFTER times and an
    admissible seat on ANOTHER account exists, re-home it there instead of re-pinning the
    owner -- so a session that keeps dying on one seat MOVES to a fresh one rather than
    looping on the same failing account (#1342/#1345/#1859). The re-home reuses the exact
    machinery the throttle path uses (_admissible_targets ranks least-loaded proven-healthy
    seats excluding the owner; the watchdog copies the transcript across on rehomed=True),
    and folds the target into ``assigned`` so a burst spreads instead of stampeding one
    seat. Falls back to plain in-place resume when escalation is disabled, the attempt
    threshold is not met, or no other healthy seat is admissible.

    Owner-loaded spread (mirrors internal/resume/rehome loadSpreadDecision, the Go
    source of truth): BEFORE the escalation ladder, an owner already carrying
    REHOME_CAP live sessions is the seat most likely to limit-wall next (the july7
    429 pile-up shape), so the resume re-homes to a strictly less-loaded admissible
    seat when one exists. Fires only on positive load evidence; FAK_LOAD_REHOME=0
    disables; with no freer seat the in-place resume stands."""
    r["action"] = "AUTO_RESUME"
    if LOAD_REHOME:
        live, overloaded = _owner_live_load(availability, r["account"])
        if overloaded:
            targets = _admissible_targets(availability, r["account"], assigned)
            if targets:
                tgt = targets[0]
                tgt_load = int(tgt.get("live_sessions") or 0) + assigned.get(tgt.get("account", ""), 0)
                if tgt_load < live:
                    r["rehomed"] = True
                    r["resume_account"] = tgt["account"]
                    r["resume_config_dir"] = tgt.get("config_dir") or config_dir(tgt["account"])
                    assigned[tgt["account"]] = assigned.get(tgt["account"], 0) + 1
                    return
    if RESUME_ESCALATE_AFTER <= 0:
        return
    if inplace_counts.get(r["session"], 0) < RESUME_ESCALATE_AFTER:
        return
    targets = _admissible_targets(availability, r["account"], assigned)
    if not targets:
        return
    tgt = targets[0]
    r["rehomed"] = True
    r["resume_account"] = tgt["account"]
    r["resume_config_dir"] = tgt.get("config_dir") or config_dir(tgt["account"])
    assigned[tgt["account"]] = assigned.get(tgt["account"], 0) + 1


def decide(rows, throttle, availability=None, live_sids=None):
    """Stamp each row with a deterministic action + an account-correct resume command.
    Only AUTONOMOUS, genuinely-DEAD (crashed/killed) sessions are auto-resumable.
    The two look-alikes are held back explicitly so they are never resumed and the
    reason is logged:
      DONE        -> the agent finished; resuming would redo finished work.
      USER_CLOSED -> the user intentionally interrupted/closed it; honor that.

    Rate-limit handling: a resumable autonomous session whose OWNING account is
    throttled is RE-HOMED onto a healthy account (AUTO_RESUME + rehomed=True with
    a resume_config_dir pointing at the target) instead of being parked until the
    owner's limit resets -- which for a weekly cap can be days. Re-home only fires
    when a healthy Claude worker account actually exists; otherwise the session
    falls back to DEFER_THROTTLED and waits, exactly as before.

    Re-home spread: the availability snapshot is static within one pass, so a burst
    of throttled sessions would all pick the same momentary least-loaded target and
    stampede it. ``assigned`` tracks how many this pass has already routed to each
    target and is fed back into ``_rehome_targets`` so the load it sees reflects the
    in-flight decisions -- spreading the burst across healthy accounts and capping
    each at REHOME_CAP so none is pushed over its own session limit.

    Dedup: identical repeating tasks (same task_sig across sids) are collapsed by the
    ``_dedup_defer`` pre-pass to ONE primary; the rest are stamped DEFER_DUPLICATE_TASK
    here and skip the decision ladder entirely, so they never consume a re-home slot.

    Liveness / newest-copy (#3459): the ``_live_or_stale_copy_gate`` pre-pass runs FIRST
    and settles WHICH copy of a session id may reach the ladder at all -- a sid with a
    live `claude --resume` driver is dropped whole, and of the rest only the newest copy
    speaks for the session. Without it the ladder classified whichever copy scan() found,
    so a stale copy's synthetic "API Error" tail queued a session that was alive.
    ``live_sids`` is the caller's process census (main() takes it); None leaves the
    process half INERT (fail-open) so a hermetic caller decides on the on-disk evidence
    alone -- the disp=="LIVE" half of the gate still applies."""
    _live_or_stale_copy_gate(rows, live_sids)  # pre-pass: gate live sids + stale copies
    _dedup_defer(rows)                       # pre-pass: stamp duplicate-task losers
    assigned: dict[str, int] = {}
    # prior in-place resume counts per sid -> escalate a repeat-crasher to another seat
    inplace_counts = _ledger_inplace_attempts()
    for r in rows:
        cwd_ok = bool(r["cwd"]) and os.path.isdir(r["cwd"])
        # resume target defaults to the owning account; re-home overrides it below
        r["source_config_dir"] = config_dir(r["account"])
        r["resume_account"] = r["account"]
        r["resume_config_dir"] = config_dir(r["account"])
        r["rehomed"] = False
        if r.get("action") in PRE_STAMPED:
            # already decided by a pre-pass -- a duplicate task the primary covers, a sid
            # a live driver is already advancing, or an older copy of a sid the newest one
            # speaks for. Skip the ladder so this never re-homes, never consumes an
            # `assigned` cap slot, and (being no AUTO_RESUME) never reaches the plan.
            r["resume_cmd"] = None
            continue
        if "pytest" in r["project"] or not cwd_ok:
            r["action"] = "SKIP_EPHEMERAL"          # cwd gone / pytest temp
        elif r["disp"] == "LIVE":
            r["action"] = "SKIP_LIVE"
        elif r["disp"] == "DONE":
            r["action"] = "SKIP_DONE"               # finished cleanly â€” do NOT resume
        elif r["disp"] == "USER_CLOSED":
            r["action"] = "SKIP_USER_CLOSED"        # user stopped it on purpose â€” honor it
        elif r["disp"] == "PARKED_WAIT":
            r["action"] = "SKIP_PARKED"
        elif r["supervised"]:
            r["action"] = "SUPERVISED"              # run_supervise_loop owns it
        elif r["disp"] == "STOPPED_CONTEXT_EXHAUSTED":
            # #3458: the operator-stop/BUDGET_CONTEXT_EXHAUSTED wall is bound to the
            # TRANSCRIPT (its context is exhausted), not to the seat -- resuming the
            # same transcript in place OR re-homed re-hits the identical 409, so it
            # must never enter _resume_inplace_or_escalate / AUTO_RESUME. Escalate to
            # a fresh continuation (the gateway's restart_fresh_session directive)
            # instead. Placed before the throttle branch: a throttled owner does not
            # make this transcript any more resumable.
            r["action"] = "ESCALATE_FRESH_CONTINUATION"
        elif r["disp"] == "STOPPED_LIMIT" or r["account"] in throttle:
            # Owning account is rate-limited. Re-home an autonomous, resumable
            # session to a healthy account rather than waiting for the reset.
            #
            # Stale-banner guard (mirrors resume_resolver's carried-throttle re-probe,
            # #621): a STOPPED_LIMIT disp can be a stale "limit reached" line carried
            # inside a transcript copied off a once-throttled owner. When a row enters
            # this branch ONLY on that disp -- its CURRENT owner is NOT in the throttle
            # map -- and the owner reads available in the freshness-stamped snapshot, the
            # limit has cleared: resume IN PLACE on the healthy owner instead of re-homing
            # it away. Re-homing a healthy owner is what stranded 5/15 sessions in the
            # 2026-06-24 incident; the throttle map stays authoritative, so an owner that
            # is genuinely throttled still re-homes.
            if (r["account"] not in throttle and r["disp"] == "STOPPED_LIMIT"
                    and _owner_available(availability, r["account"])):
                r["action"] = "AUTO_RESUME"         # INFRA: stale limit banner -> resume in place (#1353: any autonomy)
            else:
                # #1353: STOPPED_LIMIT / STOPPED_APIERR resume regardless of autonomy
                # (server-interrupted); the DEAD agent-crash path keeps the autonomy gate.
                resumable = _resumable_disp(r) and (
                    r["disp"] in DEAD or r["disp"] in ("STOPPED_LIMIT", "STOPPED_APIERR"))
                # #619: only PROVEN-healthy (probe/passive) targets may take load; a
                # carried "available" is refused here -> defer until a fresh probe.
                targets = _admissible_targets(availability, r["account"], assigned) if resumable else []
                if targets:
                    tgt = targets[0]
                    r["action"] = "AUTO_RESUME"     # INFRA: rate limit -> move to healthy acct
                    r["rehomed"] = True
                    r["resume_account"] = tgt["account"]
                    r["resume_config_dir"] = tgt.get("config_dir") or config_dir(tgt["account"])
                    assigned[tgt["account"]] = assigned.get(tgt["account"], 0) + 1
                else:
                    r["action"] = "DEFER_THROTTLED" # no healthy account -> wait for reset
        elif r["disp"] == "INFRA_ORG_DISABLED":
            # Owner account's org/subscription access is disabled. /login can't fix the
            # owner -- but the transcript is portable, so re-home an autonomous session
            # onto a healthy, non-org-disabled account WITH usage (the same machinery as
            # the rate-limit path; _rehome_targets already excludes blocked accounts).
            resumable = _resumable_disp(r)  # #1353: org-disabled resumes regardless of autonomy
            # #619: same launch-boundary gate -- never route an org-disabled session's
            # workload onto a carried/absence-of-evidence target; require positive evidence.
            targets = _admissible_targets(availability, r["account"], assigned) if resumable else []
            if targets:
                tgt = targets[0]
                r["action"] = "AUTO_RESUME"         # INFRA: org-disabled -> move to a usable acct
                r["rehomed"] = True
                r["resume_account"] = tgt["account"]
                r["resume_config_dir"] = tgt.get("config_dir") or config_dir(tgt["account"])
                assigned[tgt["account"]] = assigned.get(tgt["account"], 0) + 1
            else:
                # No healthy usage-bearing seat. Distinct from BLOCKED_AUTH: re-login on
                # the owner won't help, so this waits for a seat to free up, not a human.
                r["action"] = "DEFER_NO_USAGE"
        elif r["disp"] == "INFRA_AUTH":
            # A "Not logged in" tail usually means the SEAT needs a human /login --
            # resume can't fix that, so it blocks. But when the owner seat reads
            # ADMISSIBLE in the freshness-stamped snapshot (#619 positive evidence),
            # the seat itself contradicts the banner: the failure is session-local --
            # a frozen banner tail from before a re-login, or a guard-gateway child
            # whose recorded auth wiring died with its parent (2026-07-02: cbdc1e5d
            # answered every in-place resume with the banner while its owner probed
            # pong; the same transcript re-homed onto another seat resumed cleanly).
            # Route those through the in-place ladder: retry on the proven owner
            # (covers the frozen-tail-after-relogin case), escalate to another seat
            # after RESUME_ESCALATE_AFTER in-place attempts (covers the dead-binding
            # case). A seat WITHOUT positive evidence keeps the human-re-login block.
            if _owner_available(availability, r["account"]):
                _resume_inplace_or_escalate(r, availability, assigned, inplace_counts)
            else:
                r["action"] = "BLOCKED_AUTH"        # INFRA: needs human re-login; resume won't help
        elif r["disp"] == "STOPPED_APIERR":
            # INFRA: transient API error -> retry (#1353: any autonomy; server interrupted it).
            # Repeated in-place retries that don't stick escalate to another seat.
            _resume_inplace_or_escalate(r, availability, assigned, inplace_counts)
        elif r["disp"] in DEAD and r["autonomous"]:
            # AGENT crash, autonomous -> resume; escalate to another seat if it keeps dying here.
            _resume_inplace_or_escalate(r, availability, assigned, inplace_counts)
        elif r["disp"] == "NEVER_STARTED" and r["autonomous"]:
            # Launched autonomous goal that never produced a turn -> re-launch it (no partial
            # work to lose). Same escalation machinery as a crash: if it keeps failing to
            # start on this seat it moves to a fresh one after RESUME_ESCALATE_AFTER tries.
            _resume_inplace_or_escalate(r, availability, assigned, inplace_counts)
        elif r["disp"] in DEAD or r["disp"] in ("STOPPED_QUIET", "NEVER_STARTED"):
            r["action"] = "SURFACE"                 # agent crash / quiet stop / non-auton non-start -> human
        else:
            r["action"] = "SKIP"
        r["resume_cmd"] = resume_cmd(r) if r["disp"] in STOPLIKE else None
    return rows

def _log_decisions(rows):
    """Persist WHY each session was treated as completed / dead / user-closed.
    - decisions.log : full human-readable current-state snapshot (overwritten).
    - transitions.log : append-only audit trail of disposition CHANGES across runs."""
    snap = os.path.join(REG_DIR, "decisions.log")
    cat_counts = {}
    for r in rows:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    with open(snap, "w", encoding="utf-8") as f:
        f.write(f"# fleet session decisions @ {NOW.isoformat()}  ({len(rows)} sessions)\n")
        f.write("# categories: " + "  ".join(f"{k}={v}" for k, v in sorted(cat_counts.items())) + "\n")
        f.write("# age    category project                    disp/action            cause / reason  [sid]\n")
        for r in sorted(rows, key=lambda r: (r["category"], r["age_min"])):
            tag = r["account"].replace(".claude-", "").replace(".claude", "default")
            f.write(f"{r['age_min']:>7}m {r['category']:<8} {r['project']:<26} "
                    f"{r['disp']:<14} {r['action']:<16} {r['cause']:<16} {r['reason']}  "
                    f"[{tag}/{r['session'][:8]}]\n")
    # transitions vs previous snapshot
    prev_path = os.path.join(REG_DIR, "_prev_disp.json")
    prev = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, ValueError):
            prev = {}
    cur = {r["session"]: r["disp"] for r in rows}
    trans = os.path.join(REG_DIR, "transitions.log")
    by_sid = {r["session"]: r for r in rows}
    with open(trans, "a", encoding="utf-8") as f:
        for sid, d in cur.items():
            old = prev.get(sid)
            if old and old != d:
                r = by_sid[sid]
                tag = r["account"].replace(".claude-", "").replace(".claude", "default")
                f.write(f"{NOW.isoformat()}  [{r['category']:<7}] {sid[:8]}  {tag}/{r['project']}  "
                        f"{old} -> {d}  [{r['action']}]  {r['reason']}\n")
    with open(prev_path, "w", encoding="utf-8") as f:
        json.dump(cur, f)

def plan_entry(r):
    """One AUTO_RESUME plan record for the watchdog.

    Carries both the source (where the transcript lives now) and the resume
    target (where it should run). They differ only for a re-homed session;
    the watchdog copies the transcript across before resuming when rehomed."""
    return {"account": r["account"], "config_dir": config_dir(r["account"]),
            "source_config_dir": r["source_config_dir"],
            "resume_account": r["resume_account"],
            "resume_config_dir": r["resume_config_dir"],
            "rehomed": r["rehomed"],
            "session": r["session"], "cwd": r["cwd"], "project": r["project"],
            "disp": r["disp"], "resume_cmd": r["resume_cmd"]}

# ---- observability: storm + per-account health summary (registry `summary` block) ----
# A first-class, pre-aggregated view of the registry so every downstream consumer
# (dispatch preflight, the Prometheus/Grafana sink, operators) sees the crash-loop
# STORM and per-account health WITHOUT re-deriving it from thousands of raw rows â€”
# the exact blindness that let a bloated 23k-row registry read as "saturated" when the
# fleet was actually 0/24 with every account rate-limited. The recency buckets
# (15/30/60m) turn a slow whole-window count into a rate the dashboard graphs over time.
STORM_BUCKETS_MIN = (15, 30, 60)

def _recent_age(age_min):
    """Normalize a row's age for recency bucketing: None -> not countable; a small
    negative (clock skew on a just-written transcript) -> 0.0 (as fresh as possible)."""
    if age_min is None:
        return None
    a = float(age_min)
    return 0.0 if a < 0 else a

def session_storm_summary(session_rows, throttle):
    """Pre-aggregate the registry into {counts_by_disp, storm, accounts}. Pure over its
    inputs so both write_registry and the operator summary share one computation."""
    throttle = throttle or {}
    counts_by_disp = {}
    apierr = {w: 0 for w in STORM_BUCKETS_MIN}
    neverstart = {w: 0 for w in STORM_BUCKETS_MIN}   # #3553: launch-storm (many seats, no turn)
    total = {w: 0 for w in STORM_BUCKETS_MIN}
    accts = {}
    for r in session_rows:
        d = r.get("disp") or "?"
        counts_by_disp[d] = counts_by_disp.get(d, 0) + 1
        a = _recent_age(r.get("age_min"))
        is_err = d == "STOPPED_APIERR"
        is_neverstart = d == "NEVER_STARTED"
        if a is not None:
            for w in STORM_BUCKETS_MIN:
                if a <= w:
                    total[w] += 1
                    if is_err:
                        apierr[w] += 1
                    if is_neverstart:
                        neverstart[w] += 1
        acc = r.get("account")
        if acc:
            h = accts.setdefault(acc, {"newest_disp": None, "newest_age_min": None,
                                       "apierr_30m": 0, "live": 0})
            if a is not None and (h["newest_age_min"] is None or a < h["newest_age_min"]):
                h["newest_age_min"], h["newest_disp"] = a, d
            if is_err and a is not None and a <= 30:
                h["apierr_30m"] += 1
            if d == "LIVE":
                h["live"] += 1
    storm = {f"apierr_{w}m": apierr[w] for w in STORM_BUCKETS_MIN}
    storm.update({f"total_{w}m": total[w] for w in STORM_BUCKETS_MIN})
    storm.update({f"neverstart_{w}m": neverstart[w] for w in STORM_BUCKETS_MIN})
    storm["apierr_per_min_30m"] = round(apierr[30] / 30.0, 3)
    storm["apierr_frac_30m"] = round(apierr[30] / total[30], 3) if total[30] else 0.0
    storm["neverstart_per_min_30m"] = round(neverstart[30] / 30.0, 3)
    for acc, h in accts.items():
        t = throttle.get(acc)
        h["throttled"] = bool(t)
        h["throttle_reset"] = (t.get("reset") if isinstance(t, dict) else t) if t else None
    return {"counts_by_disp": counts_by_disp, "storm": storm, "accounts": accts}

def write_registry(rows, throttle, auth, probes=None):
    if not os.path.isdir(REG_DIR):
        os.makedirs(REG_DIR, exist_ok=True)
    # synthetic probe rows (project == "_probe") feed the mergers but are NOT real
    # sessions -- keep them out of the sessions list so the operator view stays honest.
    session_rows = [r for r in rows if r.get("project") != "_probe"]
    reg = {"schema": "fleet-sessions/3", "app_version": fleet_version.app_version(), "generated_utc": NOW.isoformat(),
           "window_h": WINDOW_H, "throttle": throttle, "auth": auth,
           "accounts": account_availability(throttle, rows, auth),
            "sessions": [{**{k: r[k] for k in ("account", "project", "session", "cwd", "git",
                          "category", "cause", "disp", "reason", "action", "autonomous",
                          "supervised", "age_min", "seen_utc", "throttle_reset",
                          "throttle_weekly", "resume_cmd", "rehomed", "resume_account", "last")},
                          "task_sig": r.get("task_sig", ""),
                          "http_status": r.get("http_status"),
                          "initiated_by": r.get("initiated_by", "user")}
                        for r in session_rows]}
    reg["summary"] = session_storm_summary(session_rows, throttle)
    if probes:
        reg["probes"] = probes  # raw active-probe verdicts (evidence for the operator/UI)
    sessions_path = os.path.join(REG_DIR, "sessions.json")
    plan = [plan_entry(r) for r in rows if r["action"] == "AUTO_RESUME"]
    plan_path = os.path.join(REG_DIR, "resume_plan.json")
    with open(sessions_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=1)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({"app_version": fleet_version.app_version(), "generated_utc": NOW.isoformat(), "plan": plan}, f, indent=1)
    _log_decisions(rows)
    return sessions_path, plan_path, len(plan)

def print_accounts(throttle, rows, auth):
    """Operator-facing 'which accounts are available right now' block. Shown in
    summary/resume so operator and switcher see the same dynamic blockers."""
    accts = account_availability(throttle, rows, auth)
    avail = [a for a in accts if a["available"]]
    blocked = [a for a in accts if not a["available"]]
    print("ACCOUNTS AVAILABLE NOW (worker, not blocked):")
    print("  " + (", ".join(a["tag"] for a in avail) if avail else "(none - all blocked)"))
    if blocked:
        print("BLOCKED: " + ", ".join(f"{a['tag']} ({a['block_reason']})" for a in blocked))
    print()

def run_probes(rows, selector):
    """Active-probe selected accounts and return (probe_rows, verdicts).

    Probe rows are in classify()'s row shape (via account_probe.verdict_to_row) so the
    existing mergers consume them unchanged. The roster used to pick targets is annotated
    with the throttle/auth derived from THIS scan's rows (not just the stale registry), so
    "blocked" reflects current passive evidence before we spend a probe.
    """
    if selector in ("none", ""):
        return [], []
    try:
        import account_probe  # local import: only paid for when probing
    except ImportError:
        return [], []
    # annotate the roster with the current scan's throttle/auth so target selection is fresh
    throttle = merge_known_throttle({}, rows)
    auth = merge_known_auth(rows)
    registry = {"generated_utc": NOW.isoformat(), "auth": auth, "throttle": throttle}
    annotated = fleet_accounts.annotated_roster(USER, ACCT_POLICY, registry=registry,
                                                throttle=throttle, sessions=rows)
    targets = account_probe.select_targets(annotated, selector=selector,
                                           skip_active_throttle=True,
                                           min_interval_min=PROBE_MIN_INTERVAL,
                                           reg_dir_path=REG_DIR)
    if not targets:
        return [], []
    verdicts = account_probe.probe_accounts(targets)
    # record each probe in the per-account ledger (prev_status -> flip detection), the
    # audit trail the status card reads for RECENT PROBE FLIPS.
    try:
        account_probe.append_probe_ledger(verdicts, REG_DIR)
    except OSError:
        pass
    probe_rows = []
    for v in verdicts:
        pr = account_probe.verdict_to_row(v)
        pr["account"] = v.get("account")
        probe_rows.append(pr)
    return probe_rows, verdicts


def registry_summary(sp, pp, nsess, n, probe_verdicts):
    """The machine-readable summary emitted as the LAST stdout line of `registry`
    mode. The Go dispatch tick (cmd/fak/dispatch_tick.go dispatchRunJSON) parses
    this helper's combined output with lastJSONObject(); without a trailing JSON
    object it reports a SUCCESSFUL refresh -- the two files were already written --
    as "no JSON object in helper output" -> registry_refresh.ok=false, a
    false-failure that masks the real refresh state on every tick and misleads
    diagnosis. dispatchRefreshRegistry recomputes ok from _error, so a run that
    reaches here (no exception) is reported ok=true. Kept pure + module-level so
    the cross-language contract shape is unit-testable without a full fleet scan."""
    return {"ok": True, "mode": "registry", "sessions": nsess,
            "auto_resume": n, "probed": len(probe_verdicts),
            "wrote": [str(sp), str(pp)]}


def main():
    rows, throttle = scan()
    probe_rows, probe_verdicts = run_probes(rows, PROBE_SELECTOR)
    if probe_rows:
        rows = rows + probe_rows
    throttle = merge_known_throttle(throttle, rows)
    auth = merge_known_auth(rows)
    availability = account_availability(throttle, rows, auth)
    # #3459: hand decide() this host's live-driver census. Without it the liveness gate
    # only sees sessions whose OWN copy classified LIVE -- and the copy that queued
    # 94aea02a was a stale one under another account dir, so no on-disk row carried that
    # evidence. _live_resume_sids() is the process-table half and fails open to empty.
    decide(rows, throttle, availability, live_sids=_live_resume_sids())
    if MODE == "json":
        print(json.dumps({"app_version": fleet_version.app_version(), "now": NOW.isoformat(),
                          "throttle": throttle, "auth": auth,
                          "accounts": account_availability(throttle, rows, auth),
                          "probes": probe_verdicts,
                          "rows": rows}, indent=1))
        return
    if MODE == "registry":
        sp, pp, n = write_registry(rows, throttle, auth, probes=probe_verdicts)
        nsess = len([r for r in rows if r.get("project") != "_probe"])
        probed = f", {len(probe_verdicts)} probed" if probe_verdicts else ""
        print(f"wrote {sp} ({nsess} sessions{probed})")
        print(f"wrote {pp} ({n} AUTO_RESUME)")
        # Machine-readable summary as the LAST stdout line -- see registry_summary().
        print(json.dumps(registry_summary(sp, pp, nsess, n, probe_verdicts)))
        return
    if MODE == "plan":  # machine-readable AUTO_RESUME set for the watchdog
        plan = [plan_entry(r) for r in rows if r["action"] == "AUTO_RESUME"]
        print(json.dumps({"app_version": fleet_version.app_version(), "generated_utc": NOW.isoformat(), "plan": plan}, indent=1))
        return
    if MODE == "resume":
        auto = [r for r in rows if r["action"] == "AUTO_RESUME"]
        surf = [r for r in rows if r["action"] == "SURFACE"]
        print_accounts(throttle, rows, auth)
        print("# AUTO-RESUMABLE (autonomous, dead, account available) â€” safe to run:\n")
        for r in auto:
            print(f"# [{r['disp']}] {r['project']} ({r['git']})  age={r['age_min']}m")
            print(r["resume_cmd"])
            print()
        if not auto:
            print("# (none right now)\n")
        print("# SURFACE â€” stopped but interactive; resume only if you mean to:\n")
        for r in surf:
            print(f"# [{r['disp']}] {r['project']} age={r['age_min']}m  -- {r['last'][:70]}")
            print(r["resume_cmd"])
            print()
        return
    # summary (uncapped within window; explicit truncation note)
    print(f"fleet_sessions @ {NOW.strftime('%Y-%m-%d %H:%M')}Z   window={WINDOW_H}h   {len(rows)} sessions\n")
    print_accounts(throttle, rows, auth)
    if throttle:
        print("THROTTLED ACCOUNTS (resume will instantly re-die):")
        for a, t in throttle.items():
            line = f"  {a:<30} resets {t['reset']}"
            weekly = t.get("weekly")
            if weekly:
                line += f"  | weekly {weekly}"
            print(line)
        print()
    order = ["STOPPED_LIMIT", "STOPPED_APIERR", "STOPPED_CONTEXT_EXHAUSTED",
             "INFRA_AUTH", "INFRA_ORG_DISABLED",
             "DEAD_MIDTOOL", "DEAD_KILLED", "NEVER_STARTED",
             "USER_CLOSED", "STOPPED_QUIET", "PARKED_WAIT", "LIVE", "DONE"]
    counts, acts, cats = {}, {}, {}
    for r in rows:
        counts[r["disp"]] = counts.get(r["disp"], 0) + 1
        acts[r["action"]] = acts.get(r["action"], 0) + 1
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    order += [d for d in counts if d not in order]   # any unforeseen disp still shown
    catorder = ["INFRA", "AGENT", "USER", "HANGING", "LIVE"]
    n_agent = sum(1 for r in rows if r.get("initiated_by") == "agent")
    print("category: " + "  ".join(f"{k}={cats[k]}" for k in catorder if cats.get(k)))
    print("disp:     " + "  ".join(f"{k}={counts.get(k,0)}" for k in order if counts.get(k)))
    print("action:   " + "  ".join(f"{k}={acts[k]}" for k in sorted(acts)))
    # who started these sessions: agent-driven (/goal,/loop,supervised) vs interactive.
    print(f"initiated: agent={n_agent}  user={len(rows) - n_agent}")
    # storm line: the crash-loop signal at a glance (recency-bucketed, so a spike is
    # visible without reading the per-disp table). Mirrors the registry `summary.storm`.
    st = session_storm_summary([r for r in rows if r.get("project") != "_probe"], throttle)["storm"]
    flag = "  <-- STORM" if st["apierr_per_min_30m"] >= 1.0 else ""
    print(f"storm:     apierr 15m={st['apierr_15m']} 30m={st['apierr_30m']} 60m={st['apierr_60m']}  "
          f"rate={st['apierr_per_min_30m']}/min  frac30m={st['apierr_frac_30m']}{flag}")
    # never-start burst line (#3553): a launch-storm -- many seats launched, none produced an
    # assistant turn -- is invisible on the apierr storm line; bucket NEVER_STARTED by the same
    # recency windows so it is legible at a glance, on its own line.
    nsflag = "  <-- NEVER-START BURST" if st["neverstart_per_min_30m"] >= 1.0 else ""
    print(f"neverstart:      15m={st['neverstart_15m']} 30m={st['neverstart_30m']} 60m={st['neverstart_60m']}  "
          f"rate={st['neverstart_per_min_30m']}/min{nsflag}")
    print()
    inplace_counts = _ledger_inplace_attempts()   # per-sid prior in-place relaunch count (#3553)
    CAP = 40
    for disp in order:
        grp = [r for r in rows if r["disp"] == disp]
        if not grp:
            continue
        print(f"== {disp}: {len(grp)} ==")
        for r in grp[:CAP]:
            thr = "  [THROTTLED]" if r["account"] in throttle else ""
            mark = {"AUTO_RESUME": " *AUTO", "SURFACE": " surface", "SUPERVISED": " (sup)",
                    "ESCALATE_FRESH_CONTINUATION": " fresh-cont",
                    "DEFER_THROTTLED": " defer", "DEFER_NO_USAGE": " defer-no-usage",
                    "DEFER_DUPLICATE_TASK": " dup-task", "BLOCKED_AUTH": " blocked-auth",
                    "SKIP_DONE": " done",
                    "SKIP_USER_CLOSED": " user-closed"}.get(r["action"], "")
            if r.get("rehomed"):
                rtag = r["resume_account"].replace(".claude-", "").replace(".claude", "default")
                mark += f" -> {rtag}"
            tag = r["account"].replace(".claude-", "").replace(".claude", "default")
            who = "A" if r.get("initiated_by") == "agent" else "u"   # agent- vs user-initiated
            code = f" [{r['http_status']}]" if r.get("http_status") else ""
            relaunch = _relaunch_badge(inplace_counts.get(r["session"], 0))   # #3553 relaunch×N
            print(f"  {r['age_min']:>6}m {who} {tag:<18} {r['project']:<26} {r['session'][:8]}{mark}{thr}{code}{relaunch}")
        if len(grp) > CAP:
            print(f"  ... +{len(grp)-CAP} more")
        print()

if __name__ == "__main__":
    main()
