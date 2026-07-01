#!/usr/bin/env python3
"""fleet_resume_watchdog.py -- macOS/Linux port of fleet_resume_watchdog.ps1.

The cross-account resume layer for ALL autonomous Claude sessions on this host
(not just the supervisor's job workers). Designed to be run on a ~5-minute cron
schedule. Safe to run by hand any time.

Each tick:
  1. EXTRACT-IN-ADVANCE: refresh the on-disk session registry
     (tools/_registry/sessions.json) and the AUTO_RESUME plan
     (tools/_registry/resume_plan.json) via fleet_sessions.py.
  2. Resume each AUTO_RESUME session ONCE EVER, under its owning account's
     CLAUDE_CONFIG_DIR. "Once ever" is enforced by a durable ledger
     (tools/_registry/resume_ledger.jsonl): a session that dies again after
     being auto-resumed is left for a human, never re-resumed in a loop.
  3. Notify (macOS Notification Center via osascript + notifications.log) when
     an account needs a human re-login (BLOCKED_AUTH) -- once per session.

Safety rails (faithful to the .ps1):
  * DRY-RUN by default. Set FAK_LIVE=1 (or pass --live) to actually resume.
  * Interactive sessions are SURFACE (never auto-resumed); supervisor workers are
    SUPERVISED (left to run_supervise_loop); throttled accounts are deferred --
    all decided upstream by fleet_sessions.py when it writes the plan.
  * RESUME ONCE: ledger-gated, survives state-file loss.
  * Per-tick launch cap.

Usage:
  python3 fleet_resume_watchdog.py            # dry-run: log what it WOULD resume
  python3 fleet_resume_watchdog.py --live      # actually resume (once per session)
  FAK_LIVE=1 python3 fleet_resume_watchdog.py  # same, for cron
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DIR = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fleet_session_signals  # noqa: E402  -- shared limit/auth banner detection
import memory_cotravel  # noqa: E402  -- carry the slug-scoped memory store on re-home


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


LIVE = ("--live" in sys.argv) or _env_flag("FAK_LIVE")
# Post actionable events (a real resume, an account that needs re-login) to Slack when
# --slack is passed OR the cron opt-in FAK_DISPATCH_SLACK=1 is set. Dry-run resumes
# never toast, so Slack only ever carries real resumes + auth walls (no dry-run noise).
SLACK = ("--slack" in sys.argv) or _env_flag("FAK_DISPATCH_SLACK")
SLACK_DRY = "--slack-dry-run" in sys.argv
WINDOW_H = float(os.environ.get("FAK_WINDOW_H", "6"))
MAX_PER_TICK = int(os.environ.get("FAK_MAX_PER_TICK", "4"))
# How many times a session may be auto-resumed before the gate gives up and leaves
# it for a human. The original gate burned a session on the FIRST launch regardless
# of outcome, so a resume that died 2s later on a usage-limit wall was permanently
# stranded (only a manual `consolidate-resume-throttle-strand` ledger override
# revived it). Now a resume that fails RECOVERABLY (limit wall / transient) stays
# eligible up to this many attempts; a CLEAN finish or an unrecoverable auth wall
# still burns it once. Override with FAK_MAX_ATTEMPTS.
MAX_ATTEMPTS = int(os.environ.get("FAK_MAX_ATTEMPTS", "8"))
# Seconds to wait BETWEEN spawning each resume in a single tick. Without spacing the
# launcher fires all MAX_PER_TICK resumes within the same second, and a burst that big
# trips the server-side "temporarily limiting requests (not your usage limit)" 529 --
# every freshly-resumed session gets one turn out, then strands on that transient wall
# (observed: a cap-2/cap-4 burst stranded its whole batch on identical same-second
# errors). Pacing the spawns lets the shared rate budget refill between launches so a
# burst resumes cleanly instead of self-congesting. 0 restores the old all-at-once
# behavior; the default is deliberately conservative. Override with FAK_LAUNCH_SPACING_SEC.
LAUNCH_SPACING_SEC = float(os.environ.get("FAK_LAUNCH_SPACING_SEC", "8"))
# The id of the session this watchdog is running inside (set by the Claude Code
# harness). Used to refuse self-resume -- a live operator session can briefly look
# like a stopped autonomous worker. Empty when run outside a Claude session (cron).
SELF_SID = (os.environ.get("CLAUDE_CODE_SESSION_ID") or "").strip()
PYTHON = sys.executable or "python3"
# Source the claude binary from the fleet-wide convention (FLEET_CLAUDE_EXE), like
# account_probe.py and the rest of the fleet; FAK_CLAUDE_EXE stays a back-compat
# fallback. Resolve `claude`/`claude.exe` on PATH before the hardcoded default -- the
# old bare `~/.local/bin/claude` had no `.exe`, so it pointed at a non-existent path
# on Windows (the primary platform).
CLAUDE_EXE = (
    os.environ.get("FLEET_CLAUDE_EXE")
    or os.environ.get("FAK_CLAUDE_EXE")
    or shutil.which("claude")
    or shutil.which("claude.exe")
    or os.path.expanduser("~/.local/bin/claude")
)
# The fak binary used for the per-source concurrency gate (`fak resume admit`). Resolved
# on PATH before a bare default; None if fak is unavailable, in which case the gate
# fails OPEN (never strands the watchdog on a missing binary).
FAK_EXE = (
    os.environ.get("FLEET_FAK_EXE")
    or os.environ.get("FAK_EXE")
    or shutil.which("fak")
    or shutil.which("fak.exe")
)
LOG_DIR = os.environ.get("FAK_WATCHDOG_LOG_DIR", os.path.join(HERE, "_watchdog"))
# Honor FLEET_REG_DIR exactly as fleet_sessions.py does (same tools/_registry default),
# so the watchdog reads the plan/ledger/sessions from the dir the refresh child WRITES.
# The .ps1 pins this with `$env:FLEET_REG_DIR = $regDir`; without it an ambient
# FLEET_REG_DIR (set by fleet_control_pane.py or an operator) makes fleet_sessions.py
# write to $FLEET_REG_DIR while this watchdog reads HERE/_registry -> stale/empty plan
# (silent no-op) and a split resume-once ledger (latent double-resume).
REG_DIR = os.environ.get("FLEET_REG_DIR", os.path.join(HERE, "_registry"))
# Active-probe parity with the .ps1 (-Probe auto): on a live tick, re-probe blocked
# accounts so a silently-recovered one re-enters the pool instead of riding a stale
# carried verdict. auto -> blocked on --live, none on dry-run (keeps dry-run
# side-effect-free); override with FAK_PROBE=blocked|stale|all|none.
PROBE_MODE = os.environ.get("FAK_PROBE", "auto").strip().lower()
PROBE_MIN_INTERVAL_MIN = int(os.environ.get("FAK_PROBE_MIN_INTERVAL_MIN", "20"))


def resolve_probe_mode(setting: str, live: bool) -> str:
    """auto -> blocked on a live tick, none on dry-run; else the explicit setting.

    Gating to the live tick keeps the default dry-run side-effect-free (no probe
    spend), matching fleet_resume_watchdog.ps1's -Probe auto behavior."""
    if setting == "auto":
        return "blocked" if live else "none"
    return setting


RESUME_PROMPT = (
    "Resume where you left off; re-establish any /goal or /loop and continue toward it."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_admit_gate() -> tuple[bool, str]:
    """Ask the host-wide per-source concurrency gate whether this box may take one more
    live resume RIGHT NOW, across ALL accounts (#1341/#1344). Returns (admit, reason).

    The decision lives in the audited Go leaf `fak resume admit`: it counts the LIVE
    `claude --resume` processes on the host (the dimension the server-side 529 burst wall
    keys on -- which FAK_MAX_PER_TICK and the per-account rehome cap never measured) plus
    the recent launch rate from the shared ledger, and exits 3 to DEFER. Fronting the
    spawn with it means the safety rail lives INSIDE the launcher (the #617 lesson: a rail
    in bypassable tooling gets bypassed), read by one primitive instead of re-derived here.

    Fails OPEN: if fak is unavailable or the call errors for any reason other than a clean
    exit-3 DEFER, admit -- a broken gate must never strand the whole watchdog. The existing
    FAK_MAX_PER_TICK / spacing / once-gate remain as the fallback bound."""
    if not FAK_EXE:
        return True, "no-fak-binary"
    try:
        r = subprocess.run(
            [FAK_EXE, "resume", "admit", "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:  # missing binary raced away, timeout, OS error -> fail open
        return True, f"gate-error:{exc}"
    if r.returncode == 3:
        return False, (r.stdout or r.stderr or "SOURCE_DEFER").strip()
    return True, "admitted"


def note(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"{now_iso()}  {msg}"
    with open(os.path.join(LOG_DIR, "resume_watchdog.log"), "a") as fh:
        fh.write(line + "\n")
    print(line)


def post_slack_event(title: str, message: str, level: str = "info", *,
                     enabled: bool, dry_run: bool = False, transport=None):
    """Post one actionable resume event (a real resume, an auth wall) to Slack via
    tools/slack_post when ``enabled``. Best-effort and gated, exactly like the macOS
    toast — never raises, returns the slack_post verdict or None when disabled. ``enabled``
    is a parameter (not a read of the import-time flag) so it is testable in isolation."""
    if not enabled:
        return None
    try:
        import slack_post  # sibling module in tools/
    except Exception as exc:  # noqa: BLE001
        return {"posted": False, "error": f"slack_post unavailable: {exc}"}
    lvl = "warn" if level == "warn" else "resume"
    try:
        return slack_post.event(title, message, level=lvl, dry_run=dry_run,
                                transport=transport)
    except Exception as exc:  # noqa: BLE001 — a Slack failure must never kill a tick
        return {"posted": False, "error": str(exc)}


def toast(title: str, message: str, level: str = "info") -> None:
    """macOS Notification Center + a durable notifications.log + Slack (all best-effort).

    This is the watchdog's single "tell the operator something happened" seam, so routing
    Slack through here means every real resume and every auth wall lands in the channel
    without sprinkling post calls across main()."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "notifications.log"), "a") as fh:
        fh.write(f"{now_iso()}  [{level}] {title} -- {message}\n")
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification {json.dumps(message)} with title {json.dumps(title)}',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        pass
    post_slack_event(title, message, level, enabled=SLACK, dry_run=SLACK_DRY)


def load_json(path: str, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def rehome_transcript(src_cfg: str, dst_cfg: str, project: str, sid: str,
                      dest_projects: list[str] | None = None) -> bool:
    """Copy a session's transcript (and its sidecar subagents/workflows dir) from
    the throttled owner's config dir into the healthy target account's config dir.

    `claude --resume <sid>` is CLAUDE_CONFIG_DIR + cwd scoped: it only finds the
    conversation under <config>/projects/<sanitized-cwd>/<sid>.jsonl. So to resume
    on a different account the transcript must physically live there first. Returns
    False (caller skips the resume) when the source transcript is missing.

    ``dest_projects`` lets the caller land the copy under ADDITIONAL project slugs
    beyond the owner's original ``project``. This is the cross-directory resume fix:
    a session created under ``C--work-fak`` is stored under that slug, but when an
    operator runs ``c --resume <sid>`` from a DIFFERENT cwd (e.g. C:\\work\\slack-helpers,
    slug ``C--work-slack-helpers``), ``claude --resume`` looks under the NEW cwd's slug
    and 404s. The interactive resolver passes the launching cwd's slug here so the copy
    also lands where claude will actually look. The headless watchdog passes nothing and
    keeps the owner-slug-only behavior."""
    src = os.path.join(src_cfg, "projects", project, sid + ".jsonl")
    if not os.path.isfile(src):
        return False
    side = os.path.join(src_cfg, "projects", project, sid)
    # The owner's original slug PLUS any caller-supplied destination slugs, de-duped.
    slugs = [project]
    for p in (dest_projects or []):
        if p and p not in slugs:
            slugs.append(p)
    copied_any = False
    for slug in slugs:
        dst_dir = os.path.join(dst_cfg, "projects", slug)
        dst = os.path.join(dst_dir, sid + ".jsonl")
        # Skip the no-op self-copy: when mirroring WITHIN the owner account
        # (src_cfg == dst_cfg) the owner's own slug resolves dst == src, and
        # shutil.copy2 of an open file onto ITSELF raises WinError 32 on Windows.
        if os.path.abspath(dst) == os.path.abspath(src):
            copied_any = True
            continue
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied_any = True
        except OSError:
            # A live process may hold the transcript (Windows mandatory locks). A
            # failed copy must never crash the resolver -- it falls back to a plain
            # pin, the launcher's fail-open contract. Other slugs may still succeed.
            continue
        if os.path.isdir(side):
            try:
                shutil.copytree(side, os.path.join(dst_dir, sid), dirs_exist_ok=True)
            except Exception:
                pass
        # Carry the slug-scoped agent-memory store too, so the resumed session is not
        # amnesic on the target account. The SOURCE memory is the owner's original slug
        # (``project``); the DEST is this destination ``slug`` (== project for the owner
        # slug, or the launching cwd slug for a cross-dir resume). Gated by
        # FAK_MEMORY_COTRAVEL (default ``shadow``: decide + ledger, copy nothing) and
        # the FAK_MEMORY_MERGE strategy (default ``additive``: never clobber a dest
        # memory). Fail-open: any error inside is swallowed there, never crashing here.
        try:
            memory_cotravel.cotravel_memory(src_cfg, dst_cfg, project, sid, dst_slug=slug)
        except Exception:
            pass
    return copied_any


def _newest_transcript(sid: str) -> str | None:
    """The most-recently-modified transcript for this session across ALL account
    dirs (a re-home writes a fresh copy under the target). Mirrors resume_watch."""
    import glob
    home = os.path.expanduser("~")
    pats = [p for p in glob.glob(os.path.join(home, ".claude*", "projects", "*", sid + ".jsonl"))
            if os.path.isfile(p)]
    return max(pats, key=os.path.getmtime) if pats else None


def _record_text(rec: dict) -> str:
    """Text of one transcript record's message content (str or block-list)."""
    msg = rec.get("message") if isinstance(rec, dict) else None
    content = msg.get("content") if isinstance(msg, dict) else (
        rec.get("content") if isinstance(rec, dict) else None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _terminal_text(path: str) -> str:
    """Text of the transcript's TERMINAL user/assistant message record -- the last
    real turn, ignoring trailing control/metadata records (mode/permission-mode/
    last-prompt etc.). Classification must read only this: a usage-limit / 529
    banner that sits 5 turns back is NOT the session's current outcome (a later
    clean turn supersedes it), so concatenating the tail would misread a recovered,
    cleanly-finished session as still-walled."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return ""
    for ln in reversed(lines):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if (rec.get("type") in ("user", "assistant")
                or (isinstance(rec.get("message"), dict)
                    and rec["message"].get("role") in ("user", "assistant"))):
            return _record_text(rec)
    return ""


def last_resume_outcome(sid: str) -> str:
    """How did this session's LAST (resumed) turn end? Read from the transcript's
    TERMINAL turn -- ground truth, never a self-report, and never an earlier turn a
    later one superseded. Returns one of:
      'recoverable' -- ended on a usage-limit wall (resumable after the reset) or a
                        transient API error; another attempt is warranted.
      'unrecoverable' -- an auth/login/credit/access wall; a re-resume can't fix it.
      'progressed'  -- a normal/clean turn; the resume took. Burn it (resume-once).
      'unknown'     -- no transcript / unreadable; treat as progressed (conservative,
                        matches the old burn-once behavior so we never loop blindly).
    """
    tpath = _newest_transcript(sid)
    if not tpath:
        return "unknown"
    text = _terminal_text(tpath)
    if not text:
        return "unknown"
    if fleet_session_signals.is_auth_error(text) or fleet_session_signals.needs_login_prompt(text):
        return "unrecoverable"
    if fleet_session_signals.limit_reset(text):
        return "recoverable"
    low = text.lower()
    if "overloaded" in low or "529" in text or ("api error" in low and "rate" in low):
        return "recoverable"
    return "progressed"


def resume_blocked(sid: str, history: list[dict]) -> tuple[bool, str]:
    """Outcome-aware once-gate. ``history`` is the prior ledger entries for ``sid``
    (oldest first). Decides whether a NEW resume is blocked:

      * no history            -> NOT blocked (first resume).
      * any 'consolidate'/manual override or a confirmed clean finish -> blocked
        (an operator settled it, or it genuinely completed).
      * attempts >= MAX_ATTEMPTS -> blocked (give up, leave for a human).
      * else look at how the LAST attempt actually ended: 'recoverable' (limit /
        transient) -> NOT blocked (try again); 'unrecoverable' (auth) or
        'progressed' (clean) -> blocked.

    This replaces "any ledger row for the sid => never again" with "blocked unless
    the last attempt failed recoverably and we are under the attempt cap" -- so a
    resume that immediately hit a usage-limit wall is retried past the reset instead
    of being permanently stranded."""
    if not history:
        return False, ""
    # A manual override entry (operator-settled) is authoritative -- honor it.
    if any(str(h.get("action", "")).startswith("consolidate") or h.get("manual_override")
           for h in history):
        return True, "operator-settled (manual ledger override)"
    auto = [h for h in history if h.get("cause") or h.get("phase") in ("launched", "resumed")
            or h.get("action") in (None, "", "auto-resume")]
    attempts = len(auto) or len(history)
    if attempts >= MAX_ATTEMPTS:
        return True, f"attempt cap reached ({attempts}/{MAX_ATTEMPTS})"
    outcome = last_resume_outcome(sid)
    if outcome == "recoverable":
        return False, f"last resume failed recoverably ({outcome}); attempt {attempts + 1}/{MAX_ATTEMPTS}"
    if outcome == "unrecoverable":
        return True, "last resume hit an auth/login wall -- re-resume can't fix it"
    # progressed / unknown -> the resume took (or we can't prove it didn't): burn once.
    return True, "already resumed once (resume took)"


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. refresh registry + plan (extract in advance). On a live tick, also ACTIVELY
    # probe blocked accounts so a silently-recovered account (re-login / access
    # re-enabled / throttle expired) re-enters the pool instead of riding a stale
    # carried verdict -- parity with fleet_resume_watchdog.ps1.
    probe_mode = resolve_probe_mode(PROBE_MODE, LIVE)
    reg_argv = [PYTHON, os.path.join(HERE, "fleet_sessions.py"), "registry",
                "--window", str(WINDOW_H)]
    if probe_mode != "none":
        reg_argv += ["--probe", probe_mode, "--min-interval-min", str(PROBE_MIN_INTERVAL_MIN)]
    # Pin FLEET_REG_DIR so the child writes where we read; pass FLEET_CLAUDE_EXE (when it
    # resolves to a real file) so the probe spends its tiny `say pong` on the SAME binary
    # the resume uses, mirroring fleet_resume_watchdog.ps1.
    refresh_env = {**os.environ, "FLEET_REG_DIR": REG_DIR}
    if CLAUDE_EXE and os.path.exists(CLAUDE_EXE):
        refresh_env["FLEET_CLAUDE_EXE"] = CLAUDE_EXE
    subprocess.run(
        reg_argv,
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=refresh_env,
    )
    plan = (load_json(os.path.join(REG_DIR, "resume_plan.json"), {}) or {}).get("plan", []) or []
    mode = "LIVE" if LIVE else "DRY-RUN"
    note(f"TICK {mode} plan={len(plan)} window={WINDOW_H}h cap={MAX_PER_TICK}")

    # defense-in-depth: the set of account dir-basenames policy still treats as
    # workers. fleet_sessions.py already excludes non-workers when it writes the
    # plan, but a stale plan file could predate the policy -- re-check here too.
    worker_accts = set()
    try:
        acct_doc = json.loads(
            subprocess.run(
                [PYTHON, os.path.join(HERE, "fleet_accounts.py"), "json"],
                check=False, capture_output=True, text=True,
            ).stdout
            or "{}"
        )
        for a in acct_doc.get("accounts", []):
            if a.get("kind") == "worker":
                worker_accts.add(a.get("account"))
    except Exception:
        pass

    # durable resume ledger -- grouped per session so the gate can reason about the
    # OUTCOME and attempt count of prior resumes, not merely their existence.
    ledger_path = os.path.join(REG_DIR, "resume_ledger.jsonl")
    history: dict[str, list[dict]] = {}
    if os.path.exists(ledger_path):
        with open(ledger_path) as fh:
            for ln in fh:
                try:
                    rec0 = json.loads(ln)
                    history.setdefault(rec0["session"], []).append(rec0)
                except Exception:
                    pass

    launched = 0
    for p in plan:
        if launched >= MAX_PER_TICK:
            note(f"  per-tick cap reached ({MAX_PER_TICK})")
            break
        sid = p.get("session", "")
        sid8 = sid[:8]
        acct = (p.get("account", "") or "").replace(".claude-", "").replace(".claude", "") or "default"
        # Never resume the session this watchdog is running INSIDE. A live operator
        # session (e.g. one driving a /goal) can momentarily carry a STOPPED_APIERR
        # marker from a transient 529 mid-conversation; if it is also `autonomous`
        # the classifier flags it AUTO_RESUME, and a self-resume races two `claude`
        # processes on the same transcript. CLAUDE_CODE_SESSION_ID is the running
        # session's own id -- skip it unconditionally.
        if SELF_SID and sid == SELF_SID:
            note(f"  SKIP {sid8} -- this is the live session running the watchdog (self-resume guard)")
            continue
        if worker_accts and p.get("account") not in worker_accts:
            note(f"  SKIP {sid8} -- account {p.get('account')} is not an offered worker (policy/tombstoned)")
            continue
        blocked, why = resume_blocked(sid, history.get(sid, []))
        if blocked:
            note(f"  SKIP {sid8} -- {why}")
            continue
        if not LIVE:
            note(f"  WOULD RESUME {sid8} acct={acct} proj={p.get('project')}")
            continue

        # Host-wide per-source concurrency gate (#1341/#1344): before spawning, ask the
        # audited Go leaf whether the BOX may take one more live resume across ALL
        # accounts. A DEFER here bounds the standing concurrency the per-source 529 burst
        # wall keys on -- the thing FAK_MAX_PER_TICK (a per-tick count) and the per-account
        # rehome cap never measured. Record a phase="deferred" ledger row so this DEFER is
        # NOT counted as launch pressure by the next gate check, then skip this session
        # (it stays eligible next tick). Fails open (see source_admit_gate).
        admit_ok, admit_reason = source_admit_gate()
        if not admit_ok:
            note(f"  DEFER {sid8} acct={acct} -- per-source gate: {admit_reason}")
            with open(ledger_path, "a") as fh:
                fh.write(json.dumps({
                    "ts": now_iso(), "session": sid, "account": p.get("account"),
                    "resume_account": p.get("resume_account"),
                    "phase": "deferred", "cause": "source_concurrency_gate",
                    "reason": admit_reason,
                }) + "\n")
            continue

        env = dict(os.environ)
        resume_cfg = p.get("resume_config_dir") or p.get("config_dir", "")
        # re-home: copy the transcript into the target account first, else
        # `claude --resume` won't find it under the new CLAUDE_CONFIG_DIR.
        if p.get("rehomed"):
            src_cfg = p.get("source_config_dir") or p.get("config_dir", "")
            if not rehome_transcript(src_cfg, resume_cfg, p.get("project", ""), sid):
                note(f"  SKIP {sid8} -- re-home source transcript missing")
                continue
            note(f"  RE-HOME {sid8} {p.get('account')} -> {p.get('resume_account')} "
                 f"(transcript copied; resuming on healthy account)")
        env["CLAUDE_CONFIG_DIR"] = resume_cfg
        env.pop("JOB_SUPERVISED_WORKER", None)
        out = os.path.join(LOG_DIR, f"resume-{sid8}-{int(time.time())}.log")
        wd = p.get("cwd") if p.get("cwd") and os.path.isdir(p.get("cwd")) else FLEET_DIR
        spawn_kw = {}
        if os.name == "nt":
            # start_new_session is POSIX-only (silently ignored on Windows). Give the
            # resumed session a HIDDEN console (CREATE_NO_WINDOW) so neither it nor the
            # git/gh/fak/shell tools it spawns flashes a visible console window.
            spawn_kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        else:
            spawn_kw["start_new_session"] = True
        with open(out, "ab") as so, open(out + ".err", "ab") as se:
            proc = subprocess.Popen(
                [CLAUDE_EXE, "--resume", sid, "-p", RESUME_PROMPT,
                 "--dangerously-skip-permissions"],
                cwd=wd, env=env, stdout=so, stderr=se,
                **spawn_kw,
            )
        # Record the launch BEFORE anything else -- a crash can't double-LAUNCH in
        # this tick. But the gate keys on OUTCOME, not mere presence: phase="launched"
        # marks an attempt whose result is unknown until the next tick reads the
        # transcript (resume_blocked -> last_resume_outcome). A resume that dies
        # recoverably (limit/transient) stays eligible up to MAX_ATTEMPTS instead of
        # being burned on launch; a clean finish / auth wall blocks it as before.
        attempt = len([h for h in history.get(sid, []) if h.get("phase") in ("launched", "resumed")
                       or h.get("cause")]) + 1
        rec = {"ts": now_iso(), "session": sid, "account": p.get("account"),
               "resume_account": p.get("resume_account"), "rehomed": bool(p.get("rehomed")),
               "project": p.get("project"), "pid": proc.pid, "cause": p.get("disp"),
               "phase": "launched", "attempt": attempt}
        with open(ledger_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        history.setdefault(sid, []).append(rec)
        launched += 1
        note(f"  RESUMED {sid8} acct={acct} pid={proc.pid} "
             f"(attempt {attempt}/{MAX_ATTEMPTS}; re-eligible only if it fails recoverably)")
        toast("Resumed dead session", f"{sid8}  ({acct} / {p.get('project')})", "info")
        # Pace the next spawn so a burst does not slam the shared rate budget and trip a
        # transient 529 that strands the whole batch. Skipped after the final launch of
        # the tick (nothing follows) and when spacing is disabled (FAK_LAUNCH_SPACING_SEC=0).
        if LAUNCH_SPACING_SEC > 0 and launched < MAX_PER_TICK:
            time.sleep(LAUNCH_SPACING_SEC)

    # 2. alert on true login-blocked accounts -- once per account blocker.
    notified_path = os.path.join(REG_DIR, "_notified.json")
    notified = load_json(notified_path, {}) or {}
    registry = load_json(os.path.join(REG_DIR, "sessions.json"), {}) or {}
    changed = False
    for a in (registry.get("accounts", []) or []):
        if not a.get("blocked") or a.get("throttled") or a.get("block_kind") != "auth":
            continue
        key = f"auth-account:{a.get('account')}:{a.get('block_reason')}"
        if key in notified:
            continue
        acct = a.get("tag") or (a.get("account", "") or "").replace(".claude-", "").replace(".claude", "")
        reason = a.get("block_reason") or "auth/login required"
        sessions = int(a.get("auth_blocked_sessions") or 0)
        session_text = f" / {sessions} stopped session(s)" if sessions else ""
        toast("Account needs re-login", f"{acct} : {reason}{session_text}", "warn")
        note(f"  ALERT auth-blocked acct={acct} reason={reason} (notified)")
        notified[key] = True
        changed = True
    if changed:
        with open(notified_path, "w") as fh:
            json.dump(notified, fh)

    note(f"  done: launched={launched} sessions_in_ledger={len(history)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
