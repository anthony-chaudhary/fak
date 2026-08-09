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
  * BOUNDED ARTIFACTS (#3497): the tick/notification logs rotate at a size cap,
    per-resume log pairs expire after a retention window, and the resume ledger
    compacts past its window -- all inline at the write/read sites (this runs
    unattended under cron ~288 ticks/day, so any unbounded append is a slow
    disk leak and an ever-growing per-tick re-parse). No reaper process.

Managed-cache posture (#2178; on-by-default 2026-07-10): a resumed child is fronted with its
own `fak guard --managed-cache on --` by default (best-effort managed cache everywhere), so the
resume wave names the SAME cache posture as `fak accounts launch` / the dispatch worker. Set
FAK_GUARD_API_KEY_ENV=<var> to also bill an Anthropic API key; FAK_MANAGED_CACHE=off opts out,
and FAK_MANAGED_CACHE=auto restores the bare `claude --resume` (guard's own billing-gated auto).

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
import fleet_regdir  # noqa: E402  -- the host's one registry dir (never a second one)
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
# Continuous-drain of the resume backlog (#3587, gen/next -- DEFAULT OFF). The watchdog runs on
# a ~5-min cron, so the tick-quantized default caps recovery at FAK_MAX_PER_TICK launches per
# tick: a worker that dies just after a tick waits ~the full tick period before ANY resume, and
# a deep backlog (hundreds of dead workers at 100x) drains only MAX_PER_TICK/tick while seats sit
# free -- resume LATENCY is quantized by the cron, not by real capacity. When FAK_DRAIN_CONTINUOUS
# =1 on a LIVE tick, one tick keeps draining the plan PAST FAK_MAX_PER_TICK: the source governor
# (`fak resume admit` -- host-wide per-source concurrency + launch spacing from the shared ledger)
# plus LAUNCH_SPACING_SEC become the ONLY rate limiter, so latency collapses toward the governor's
# spacing floor instead of the tick period. Two hard safety rails keep this from becoming a storm:
#   (1) a governor DEFER ENDS the drain for the tick -- the box is saturated, so never spin the
#       rest of the plan onto capped seats (and never emit a deferred-row storm); and
#   (2) if the governor is UNAVAILABLE (fail-open: missing fak / gate error), the drain REVERTS to
#       the tick-quantized FAK_MAX_PER_TICK -- without an enforcing rate limiter a continuous drain
#       must NOT outrun per-seat safety.
# FAK_DRAIN_MAX is a per-tick backstop on total launches (a bounded loop guard, not a rate limiter:
# the governor bounds the rate long before this in a real box). Default OFF keeps the exact
# tick-quantized behavior until promotion evidence lands (gen/next: gated until dogfooded).
DRAIN_CONTINUOUS = _env_flag("FAK_DRAIN_CONTINUOUS")
DRAIN_MAX = int(os.environ.get("FAK_DRAIN_MAX", "500"))
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
# Resolve FLEET_REG_DIR exactly as fleet_sessions.py does (now the shared fleet_regdir
# ladder), so the watchdog reads the plan/ledger/sessions from the dir the refresh child
# WRITES. The .ps1 pins this with `$env:FLEET_REG_DIR = $regDir`; without it an ambient
# FLEET_REG_DIR (set by fleet_control_pane.py or an operator) makes fleet_sessions.py
# write to $FLEET_REG_DIR while this watchdog reads elsewhere -> stale/empty plan
# (silent no-op) and a split resume-once ledger (latent double-resume). Sharing ONE
# resolver is what closes the unpinned half of that: both sides now walk the same rungs.
REG_DIR = fleet_regdir.reg_dir()
# Active-probe parity with the .ps1 (-Probe auto): on a live tick, re-probe STALE
# accounts (blocked OR idle with no live-session evidence) so a silently-recovered one
# re-enters the pool — and an idle seat that quietly hit its limit leaves it — instead
# of riding a stale verdict. auto -> stale on --live, none on dry-run (keeps dry-run
# side-effect-free); override with FAK_PROBE=blocked|stale|all|none.
PROBE_MODE = os.environ.get("FAK_PROBE", "auto").strip().lower()
PROBE_MIN_INTERVAL_MIN = int(os.environ.get("FAK_PROBE_MIN_INTERVAL_MIN", "20"))
# Retention bounds (#3497). Every artifact this watchdog appends to carries an
# inline bound at the site that writes (or re-reads) it -- never a separate
# reaper. 0 or negative disables the corresponding bound.
#   * resume_watchdog.log / notifications.log rotate once past the size cap
#     (one .1 generation kept, so the on-disk total is <= ~2x the cap each).
#   * resume-<sid8>-<epoch>.log/.err pairs expire after LOG_RETAIN_DAYS,
#     pruned at the launch site (the only place that creates a pair).
#   * resume_ledger.jsonl compacts rows older than LEDGER_RETAIN_DAYS once the
#     file passes LEDGER_COMPACT_BYTES, which also bounds the whole-ledger
#     re-parse every tick performs to build the resume-once history.
LOG_RETAIN_DAYS = float(os.environ.get("FAK_RESUME_LOG_RETAIN_DAYS", "14"))
LOG_MAX_BYTES = int(os.environ.get("FAK_WATCHDOG_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
LEDGER_RETAIN_DAYS = float(os.environ.get("FAK_RESUME_LEDGER_RETAIN_DAYS", "30"))
LEDGER_COMPACT_BYTES = int(os.environ.get("FAK_RESUME_LEDGER_COMPACT_BYTES", str(512 * 1024)))


def resolve_probe_mode(setting: str, live: bool) -> str:
    """auto -> stale on a live tick, none on dry-run; else the explicit setting.

    Gating to the live tick keeps the default dry-run side-effect-free (no probe
    spend), matching fleet_resume_watchdog.ps1's -Probe auto behavior. 'stale'
    (blocked OR idle with no live-session evidence) rather than 'blocked': a passive
    available verdict only proves the seat was serving at its LAST activity, so an
    idle seat that hit its session limit after going quiet still reads available and
    the planner will re-home a crashed session onto its wall (observed 2026-07-06)."""
    if setting == "auto":
        return "stale" if live else "none"
    return setting


def tick_launch_cap(live: bool) -> int:
    """Max resumes a single tick may SPAWN. Default: FAK_MAX_PER_TICK (tick-quantized recovery,
    the pre-#3587 behavior). Continuous-drain (#3587): on a LIVE tick with FAK_DRAIN_CONTINUOUS=1
    the per-tick COUNT is lifted to FAK_DRAIN_MAX -- a bounded backstop, not a rate limiter -- so
    the source governor + LAUNCH_SPACING_SEC alone bound the launch rate and recovery latency stops
    being quantized by the ~5-min cron. Dry-run always keeps FAK_MAX_PER_TICK (side-effect-free,
    and the drain isn't exercised without a live launch). DRAIN_MAX floors at MAX_PER_TICK so a
    misconfigured tiny backstop can never make continuous-drain resume FEWER than the baseline."""
    if live and DRAIN_CONTINUOUS:
        return max(MAX_PER_TICK, DRAIN_MAX)
    return MAX_PER_TICK


RESUME_PROMPT = (
    "Resume where you left off; re-establish any /goal or /loop and continue toward it."
)
NON_LAUNCH_PHASES = {"deferred", "considered", "skipped", "gate_fail_open"}

# Fleet-wide managed-cache posture, mirrored from the Go launchers' guardCachePostureArgs
# (cmd/fak/guard_cache_posture.go) so a resume-wave child names the posture IDENTICALLY to
# `fak accounts launch` / `fak codex` / the dispatch worker (#2178). Since b2926823 a resumed
# child deliberately bypasses the parent's guard gateway and spawns `claude --resume` directly
# -- which fixed the whole-wave-dies-with-the-parent crash but left the child carrying NO cache
# posture at all, not even guard's own passive-with-banner. These two env knobs are the missing
# override surface: when the operator configures one, the child is fronted with its OWN
# `fak guard <posture> --`, so the posture comes from the child's own guard invocation on its
# own CLAUDE_CONFIG_DIR seat -- never a wire inherited from this watchdog (the env-strip at the
# spawn site still holds). On-by-default (2026-07-10): an UNSET knob now resolves to on, so the
# unconfigured fleet ALSO fronts the resume with `fak guard --managed-cache on --` (when fak is
# resolvable); an EXPLICIT FAK_MANAGED_CACHE=auto restores the bare `claude --resume` above.
FLEET_MANAGED_CACHE_ENV = "FAK_MANAGED_CACHE"
FLEET_GUARD_API_KEY_ENV_ENV = "FAK_GUARD_API_KEY_ENV"


def managed_cache_posture_args() -> tuple[list[str], str | None]:
    """Shape the `fak guard` managed-cache flags from the fleet env knobs, mirroring the Go
    guardCachePostureArgs / normalizeManagedCacheMode exactly: --api-key-env then
    --managed-cache, in that stable order. On-by-default (2026-07-10): an UNSET
    FAK_MANAGED_CACHE now normalizes to `on`, so an unconfigured fleet fronts the resume with
    --managed-cache on (best-effort managed cache everywhere) rather than staying byte-identical
    to the bare launch. Only an EXPLICIT `auto` emits NO --managed-cache (guard keeps its own
    billing-gated auto); `off` is the express opt-out. Returns (args, warning). A malformed
    FAK_MANAGED_CACHE returns ([], warning) rather than raising: this is a headless worker, so
    one bad env var must warn-and-continue passive instead of stranding the WHOLE resume wave
    -- the warn-and-continue half of guard_cache_posture.go's caller-decides contract (an
    interactive front-end aborts on the error; a headless worker does not)."""
    raw = os.environ.get(FLEET_MANAGED_CACHE_ENV, "").strip().lower()
    if raw == "":
        mode = "on"  # unset => on (operator policy 2026-07-10: best-effort managed cache everywhere)
    elif raw in ("auto", "on", "off"):
        mode = raw
    else:
        return [], (f"{FLEET_MANAGED_CACHE_ENV}={raw!r}: unknown managed-cache mode "
                    "(auto|on|off) -- ignoring; resuming passive")
    api_key_env = os.environ.get(FLEET_GUARD_API_KEY_ENV_ENV, "").strip()
    args: list[str] = []
    if api_key_env:
        args += ["--api-key-env", api_key_env]
    if mode != "auto":
        args += ["--managed-cache", mode]
    return args, None


def resume_child_argv(sid: str, posture_args: list[str]) -> list[str]:
    """The argv that resumes one dead session. On-by-default (2026-07-10): with fak resolvable
    the unconfigured fleet now FRONTS the child with `fak guard --managed-cache on --`. The bare
    `claude --resume ... -p <prompt> --dangerously-skip-permissions` fallback is kept for two
    cases only: fak is not on PATH (posture_args non-empty but FAK_EXE unset), or the operator
    set FAK_MANAGED_CACHE=auto (posture_args empty). When a posture is emitted AND fak is
    resolvable, FRONT the child with its OWN `fak guard <posture> --`: the resumed child then
    binds its own in-process gateway on its own seat, prints its own posture banner, and
    reaches the ACTIVE 1h-TTL upgrade when API-key-billed -- inheriting no wire from this
    watchdog. Guard auto-detects claude -> --provider anthropic, so the argv matches the Go
    launchers' shape (posture flags BEFORE `--`, agent after)."""
    child = [CLAUDE_EXE, "--resume", sid, "-p", RESUME_PROMPT,
             "--dangerously-skip-permissions"]
    if posture_args and FAK_EXE:
        return [FAK_EXE, "guard", *posture_args, "--", *child]
    return child


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


def record_gate_fail_open(ledger_path: str, reason: str) -> dict:
    """Append the durable, SESSION-LESS gate_fail_open warning row (#2173) and return it.

    Written when the source governor could not answer (missing fak binary / gate error)
    and the launcher proceeded fail-open. Session-less so every retry-accounting reader
    (which keys rows by ``session``) ignores it, and ``gate_fail_open`` is in
    NON_LAUNCH_PHASES so it never counts as launch pressure — it exists purely so an
    operator/status surface can see the host ran without the source-concurrency rail."""
    rec = {"ts": now_iso(), "phase": "gate_fail_open",
           "cause": "source_governor_unavailable", "reason": reason,
           "fak_exe": FAK_EXE or "", "launcher": "fleet_resume_watchdog.py"}
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def is_resume_attempt_record(h: dict) -> bool:
    """Whether a ledger record represents an actual spawned resume attempt.

    Source-gate DEFER rows carry a cause for observability, but they are not attempts and
    must not burn the retry cap.
    """
    phase = h.get("phase")
    if phase == "rearm" or h.get("outcome") == "rearm":
        return False  # a re-arm reclaim marker (#2178), not a spawned resume attempt
    if phase in NON_LAUNCH_PHASES:
        return False
    if phase in ("launched", "resumed"):
        return True
    if h.get("cause"):
        return True
    return h.get("action") in (None, "", "auto-resume")


def _rotate_log(path: str, max_bytes: int | None = None) -> bool:
    """Size-capped rotation at the append site (#3497): once ``path`` reaches the
    cap, shift it to ``path + '.1'`` (replacing the previous generation) so the
    next append starts a fresh file. One kept generation bounds the on-disk total
    at ~2x the cap while preserving the most recent history for an operator.
    Best-effort: a concurrent holder (Windows lock) or a vanished file must never
    crash a tick, so every OS error reads as 'did not rotate'."""
    limit = LOG_MAX_BYTES if max_bytes is None else max_bytes
    if limit <= 0:
        return False
    try:
        if os.path.getsize(path) < limit:
            return False
        os.replace(path, path + ".1")
        return True
    except OSError:
        return False


def prune_resume_logs(log_dir: str, retain_days: float | None = None,
                      now: float | None = None) -> int:
    """Expire per-resume ``resume-<sid8>-<epoch>.log`` / ``.log.err`` pairs older
    than the retention window. Called at the launch site -- the only place that
    CREATES a pair -- so the bound lives with the write and a quiet fleet does no
    work. Only the per-resume pair prefix/suffix shape is touched (never the tick
    log or notifications.log, which rotate instead). A file held open by a live
    child (Windows mandatory lock) survives via the per-file skip; those are
    recent anyway. Returns the number of files removed."""
    days = LOG_RETAIN_DAYS if retain_days is None else retain_days
    if days <= 0:
        return 0
    cutoff = (time.time() if now is None else now) - days * 86400.0
    removed = 0
    try:
        entries = list(os.scandir(log_dir))
    except OSError:
        return 0
    for e in entries:
        if not (e.name.startswith("resume-")
                and (e.name.endswith(".log") or e.name.endswith(".log.err"))):
            continue
        try:
            if e.is_file() and e.stat().st_mtime < cutoff:
                os.unlink(e.path)
                removed += 1
        except OSError:
            continue
    return removed


def compact_ledger(ledger_path: str, retain_days: float | None = None,
                   compact_bytes: int | None = None, now: float | None = None) -> int:
    """Time-windowed compaction of the resume-once ledger, applied at the read
    site so the whole-file re-parse every tick performs stays bounded (#3497).

    The ledger is the once-gate's memory, so a row is dropped ONLY when it can no
    longer influence a decision: the AUTO_RESUME plan only ever names sessions
    with activity inside the recent fleet window (hours), so a row older than the
    retention window (default 30 days) gates nothing that can still be planned.
    Two conservative keeps: operator-settled rows (consolidate/manual_override)
    are authoritative forever and never dropped, and a row whose ts is missing or
    unparsable is kept (never guess a row into the void). Compaction triggers
    only once the file passes the size threshold, so the common tick pays one
    getsize(); the rewrite is atomic (tmp + os.replace) so a crash mid-compact
    leaves the old ledger intact. Returns the number of rows dropped."""
    days = LEDGER_RETAIN_DAYS if retain_days is None else retain_days
    limit = LEDGER_COMPACT_BYTES if compact_bytes is None else compact_bytes
    if days <= 0 or limit <= 0:
        return 0
    try:
        if os.path.getsize(ledger_path) < limit:
            return 0
    except OSError:
        return 0
    cutoff = (time.time() if now is None else now) - days * 86400.0
    kept: list[str] = []
    dropped = 0
    try:
        with open(ledger_path, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                if not ln.endswith("\n"):
                    ln += "\n"
                try:
                    rec = json.loads(ln)
                    if (str(rec.get("action", "")).startswith("consolidate")
                            or rec.get("manual_override")):
                        kept.append(ln)
                        continue
                    ts = datetime.strptime(rec.get("ts", ""), "%Y-%m-%dT%H:%M:%SZ")
                    if ts.replace(tzinfo=timezone.utc).timestamp() >= cutoff:
                        kept.append(ln)
                    else:
                        dropped += 1
                except Exception:
                    kept.append(ln)
    except OSError:
        return 0
    if not dropped:
        return 0
    tmp = ledger_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(kept)
        os.replace(tmp, ledger_path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return 0
    return dropped


def note(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "resume_watchdog.log")
    _rotate_log(path)
    line = f"{now_iso()}  {msg}"
    with open(path, "a") as fh:
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
    notif_path = os.path.join(LOG_DIR, "notifications.log")
    _rotate_log(notif_path)
    with open(notif_path, "a") as fh:
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
    # Re-arm marker (#2178): a reclaim row zeroes the attempt budget accrued before it and lifts
    # any earlier operator/auth settle -- so consider only the history AFTER the last rearm. This
    # keeps the .py gate in parity with the .ps1 launch gate and the fleet_sessions.py planner.
    for i in range(len(history) - 1, -1, -1):
        h = history[i]
        if h.get("phase") == "rearm" or h.get("outcome") == "rearm":
            history = history[i + 1:]
            break
    if not history:
        return False, ""
    # A manual override entry (operator-settled) is authoritative -- honor it.
    if any(str(h.get("action", "")).startswith("consolidate") or h.get("manual_override")
           for h in history):
        return True, "operator-settled (manual ledger override)"
    auto = [h for h in history if is_resume_attempt_record(h)]
    attempts = len(auto)
    if attempts == 0 and all(h.get("phase") not in NON_LAUNCH_PHASES for h in history):
        attempts = len(history)
    if attempts == 0:
        return False, ""
    if attempts >= MAX_ATTEMPTS:
        return True, f"attempt cap reached ({attempts}/{MAX_ATTEMPTS})"
    outcome = last_resume_outcome(sid)
    if outcome == "recoverable":
        return False, f"last resume failed recoverably ({outcome}); attempt {attempts + 1}/{MAX_ATTEMPTS}"
    if outcome == "unrecoverable":
        return True, "last resume hit an auth/login wall -- re-resume can't fix it"
    # progressed / unknown -> the resume took (or we can't prove it didn't): burn once.
    return True, "already resumed once (resume took)"


def refresh_registry_plan() -> list[dict]:
    """Refresh the registry in a child, then read back the resume plan it wrote.

    On a live tick also ACTIVELY probe blocked accounts so a silently-recovered
    account (re-login / access re-enabled / throttle expired) re-enters the pool
    instead of riding a stale carried verdict -- parity with the .ps1 watchdog.
    """
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
    return (load_json(os.path.join(REG_DIR, "resume_plan.json"), {}) or {}).get("plan", []) or []


def offered_worker_accounts() -> set:
    """Account dir-basenames policy still treats as workers.

    Defense in depth: fleet_sessions.py already excludes non-workers when it writes
    the plan, but a stale plan file could predate the policy -- re-check here too.
    """
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
    return worker_accts


def load_resume_history(ledger_path: str) -> dict[str, list[dict]]:
    """Compact then read the durable resume ledger, grouped per session.

    Grouping is what lets the gate reason about the OUTCOME and attempt count of
    prior resumes rather than merely their existence.
    """
    compacted = compact_ledger(ledger_path)
    if compacted:
        note(f"  ledger compacted: dropped {compacted} row(s) older than "
             f"{LEDGER_RETAIN_DAYS:g}d (resume-once history bounded)")
    history: dict[str, list[dict]] = {}
    if os.path.exists(ledger_path):
        with open(ledger_path) as fh:
            for ln in fh:
                try:
                    rec0 = json.loads(ln)
                    history.setdefault(rec0["session"], []).append(rec0)
                except Exception:
                    pass
    return history


def tick_resume_posture() -> list[str]:
    """Resolve the managed-cache posture ONCE per tick (the env is tick-constant).

    A configured posture fronts each resumed child with its own `fak guard` (#2178
    parity); the default leaves the bare `claude --resume` untouched. A posture that
    cannot be applied (fak not on PATH) falls back to a direct launch LOUDLY rather
    than silently dropping it.
    """
    resume_posture_args, posture_warn = managed_cache_posture_args()
    if posture_warn:
        note(f"  WARN managed-cache: {posture_warn}")
    if resume_posture_args and not FAK_EXE:
        note("  WARN managed-cache posture configured but `fak` is not on PATH -- "
             "resuming children directly (passive, no posture banner)")
        return []
    if resume_posture_args:
        note("  managed-cache posture -> fronting resumed children with "
             f"`fak guard {' '.join(resume_posture_args)} --`")
    return resume_posture_args


def plan_skip_reason(p: dict, sid: str, worker_accts: set, history: dict) -> str:
    """Why this plan entry must not be resumed this tick, or "" if it may be.

    Never resume the session this watchdog is running INSIDE: a live operator session
    can momentarily carry a STOPPED_APIERR marker from a transient 529 mid-conversation,
    and a self-resume would race two `claude` processes on the same transcript.
    """
    if SELF_SID and sid == SELF_SID:
        return "this is the live session running the watchdog (self-resume guard)"
    if worker_accts and p.get("account") not in worker_accts:
        return f"account {p.get('account')} is not an offered worker (policy/tombstoned)"
    blocked, why = resume_blocked(sid, history.get(sid, []))
    if blocked:
        return why
    return ""


def record_deferred_row(ledger_path: str, p: dict, sid: str, admit_reason: str) -> None:
    """Record a phase="deferred" ledger row so a DEFER is not counted as launch pressure."""
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps({
            "ts": now_iso(), "session": sid, "account": p.get("account"),
            "resume_account": p.get("resume_account"),
            "phase": "deferred", "cause": "source_concurrency_gate",
            "reason": admit_reason,
        }) + "\n")


def resume_child_env(resume_cfg: str) -> dict:
    """Environment for a resumed child: its own seat, and none of the parent's API wiring.

    A tick run from INSIDE a guarded/Claude session (an operator's manual FAK_LIVE=1
    run) carries that session's model-API wiring: ANTHROPIC_API_KEY plus
    ANTHROPIC_BASE_URL point at the parent's loopback fak-guard gateway, and env auth
    takes precedence over the seat's OAuth login. A child inheriting them routes every
    request through the parent's proxy (wrong seat) and dies with the parent -- the
    whole-wave-crashes-at-one-instant signature (2026-07-01). Scheduled-task ticks never
    carry these, so stripping them is a no-op there.
    """
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = resume_cfg
    env.pop("JOB_SUPERVISED_WORKER", None)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
              "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION"):
        env.pop(k, None)
    return env


def spawn_resume_child(argv: list[str], out_path: str, wd: str, env: dict):
    """Spawn the detached resume, logging to `out_path`.

    start_new_session is POSIX-only (silently ignored on Windows), so on Windows give
    the resumed session a HIDDEN console (CREATE_NO_WINDOW) -- otherwise it and every
    git/gh/fak/shell tool it spawns flashes a visible console window.
    """
    spawn_kw = {}
    if os.name == "nt":
        spawn_kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    else:
        spawn_kw["start_new_session"] = True
    with open(out_path, "ab") as so, open(out_path + ".err", "ab") as se:
        return subprocess.Popen(argv, cwd=wd, env=env, stdout=so, stderr=se, **spawn_kw)


def record_launch_row(ledger_path: str, p: dict, sid: str, proc, history: dict) -> int:
    """Record the launch BEFORE anything else and return this session's attempt number.

    A crash can't double-LAUNCH in this tick. But the gate keys on OUTCOME, not mere
    presence: phase="launched" marks an attempt whose result is unknown until the next
    tick reads the transcript. A resume that dies recoverably (limit/transient) stays
    eligible up to MAX_ATTEMPTS instead of being burned on launch; a clean finish /
    auth wall blocks it as before.
    """
    attempt = len([h for h in history.get(sid, []) if is_resume_attempt_record(h)]) + 1
    rec = {"ts": now_iso(), "session": sid, "account": p.get("account"),
           "resume_account": p.get("resume_account"), "rehomed": bool(p.get("rehomed")),
           "project": p.get("project"), "pid": proc.pid, "cause": p.get("disp"),
           "phase": "launched", "attempt": attempt}
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    history.setdefault(sid, []).append(rec)
    return attempt


def alert_auth_blocked_accounts() -> None:
    """Toast once per true login-blocked account blocker (throttles are not auth walls)."""
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


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. refresh registry + plan (extract in advance).
    plan = refresh_registry_plan()
    mode = "LIVE" if LIVE else "DRY-RUN"
    # Effective per-tick launch cap. Continuous-drain (#3587) lifts it to the DRAIN_MAX backstop on
    # a live tick so the source governor + spacing (not the cron) bound recovery; a fail-open
    # governor reverts it to MAX_PER_TICK mid-tick (see the launch loop below).
    drain_cap = tick_launch_cap(LIVE)
    drain_note = f" drain=continuous(<={drain_cap})" if (LIVE and DRAIN_CONTINUOUS) else ""
    note(f"TICK {mode} plan={len(plan)} window={WINDOW_H}h cap={MAX_PER_TICK}{drain_note}")

    worker_accts = offered_worker_accounts()
    ledger_path = os.path.join(REG_DIR, "resume_ledger.jsonl")
    history = load_resume_history(ledger_path)

    launched = 0
    gate_fail_open_warned = False
    resume_posture_args = tick_resume_posture()
    for idx, p in enumerate(plan):
        if launched >= drain_cap:
            note(f"  per-tick cap reached ({drain_cap})")
            break
        sid = p.get("session", "")
        sid8 = sid[:8]
        acct = (p.get("account", "") or "").replace(".claude-", "").replace(".claude", "") or "default"
        skip = plan_skip_reason(p, sid, worker_accts, history)
        if skip:
            note(f"  SKIP {sid8} -- {skip}")
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
        governor_unavailable = admit_ok and admit_reason != "admitted"
        if governor_unavailable and not gate_fail_open_warned:
            # The gate answered WITHOUT a governor verdict (missing fak binary / gate
            # error): still fail open — a broken rail must not strand recovery — but
            # loudly (#2173). One durable session-less warning row per tick (invisible
            # to retry accounting, `gate_fail_open` is a non-launch phase) plus a toast,
            # so a host running without the source-concurrency rail is operator-visible.
            gate_fail_open_warned = True
            note(f"  WARN source governor UNAVAILABLE ({admit_reason}) -- failing OPEN; "
                 "only the per-tick cap/spacing bound launches this tick")
            record_gate_fail_open(ledger_path, admit_reason)
            toast("Resume source governor OFFLINE",
                  f"{admit_reason} -- live resumes are fail-open (no host-wide rail)", "warn")
        if governor_unavailable and DRAIN_CONTINUOUS and drain_cap > MAX_PER_TICK:
            # Continuous-drain safety rail (#3587): a fail-open governor cannot bound a storm, so
            # WITHOUT an enforcing rate limiter the drain must not run past the tick-quantized cap.
            # Revert to FAK_MAX_PER_TICK for the rest of this tick (idempotent -- only lowers once).
            drain_cap = MAX_PER_TICK
            note("  drain: source governor UNAVAILABLE -> reverting to per-tick cap "
                 f"({MAX_PER_TICK}) this tick (no continuous drain without the rate limiter)")
            if launched >= drain_cap:
                note(f"  per-tick cap reached ({drain_cap})")
                break
        if not admit_ok:
            note(f"  DEFER {sid8} acct={acct} -- per-source gate: {admit_reason}")
            record_deferred_row(ledger_path, p, sid, admit_reason)
            # Continuous-drain (#3587): the governor is host-wide, so a DEFER means the box is
            # saturated -- END the drain this tick rather than spinning the rest of the plan into a
            # deferred-row storm onto capped seats. The tick-quantized default keeps the old
            # per-session skip (each remaining entry re-checked; bounded anyway by MAX_PER_TICK).
            if DRAIN_CONTINUOUS:
                note("  drain: source governor DEFER -> box saturated, ending drain this tick")
                break
            continue

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
        env = resume_child_env(resume_cfg)
        pruned = prune_resume_logs(LOG_DIR)
        if pruned:
            note(f"  pruned {pruned} expired resume log(s) (>{LOG_RETAIN_DAYS:g}d)")
        out = os.path.join(LOG_DIR, f"resume-{sid8}-{int(time.time())}.log")
        wd = p.get("cwd") if p.get("cwd") and os.path.isdir(p.get("cwd")) else FLEET_DIR
        proc = spawn_resume_child(resume_child_argv(sid, resume_posture_args), out, wd, env)
        attempt = record_launch_row(ledger_path, p, sid, proc, history)
        launched += 1
        note(f"  RESUMED {sid8} acct={acct} pid={proc.pid} "
             f"(attempt {attempt}/{MAX_ATTEMPTS}; re-eligible only if it fails recoverably)")
        toast("Resumed dead session", f"{sid8}  ({acct} / {p.get('project')})", "info")
        # Pace the next spawn so a burst does not slam the shared rate budget and trip a
        # transient 529 that strands the whole batch. Skipped when spacing is disabled
        # (FAK_LAUNCH_SPACING_SEC=0), at the drain cap (nothing more launches this tick), and
        # after the final plan entry (nothing follows -- no trailing dead time before the tick ends,
        # which matters under continuous-drain where drain_cap >> len(plan)).
        if LAUNCH_SPACING_SEC > 0 and launched < drain_cap and idx < len(plan) - 1:
            time.sleep(LAUNCH_SPACING_SEC)

    # 2. alert on true login-blocked accounts -- once per account blocker.
    alert_auth_blocked_accounts()

    note(f"  done: launched={launched} sessions_in_ledger={len(history)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
