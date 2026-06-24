#!/usr/bin/env python3
r"""One guarded, switcher-routed dispatch TICK that spawns an ISSUE-RESOLUTION
worker — the arm that moves the open-issue counter on a plan-empty repo.

``issue_dispatch.py`` spawns the generic ``/dos-kernel:dos-dispatch-loop``
worker, which resolves units from the *plan portfolio*. This public repo ships
no ``PLAN-*.md`` (``PLAN_SURFACE_EMPTY``), so that worker has no work surface and
closes nothing — the backlog lives in GitHub *issues*. This tick is the
issue-driven sibling: it picks ONE concrete open issue on the busiest lane,
renders a scoped resolution prompt (``issue_worker_prompt.py`` — with the
``#N``-in-subject rule that lets the closure auditor witness the resulting
commit), and launches one detached ``claude -p "<prompt>"`` worker to land it.

It shares every safety primitive with ``issue_dispatch.py`` (imported, not
re-implemented): the per-tick registry refresh (route off current account
evidence), the ``dispatch_preflight`` DoS gate (host clean ∧ account free ∧ live
< cap), the switcher-pinned account env, and the detached spawn. DRY-RUN BY
DEFAULT — prints the issue, account, command, and prompt path. ``--live`` spawns.

In-flight de-dup: it skips an issue that already has a live resolution worker (a
dispatch log naming ``#N`` whose process is still alive) so two ticks never storm
the same issue.

    python tools/issue_resolve_dispatch.py                 # plan one tick (dry-run)
    python tools/issue_resolve_dispatch.py --live          # spawn one issue worker
    python tools/issue_resolve_dispatch.py --lane gateway --live
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_dispatch  # noqa: E402  (refresh_registry/preflight/worker_env/spawn_detached)
import issue_worker_prompt  # noqa: E402  (render the per-issue resolution prompt)
import dispatch_worker  # noqa: E402  (child_env for the opencode backend)

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")

# Worker backends this tick can launch:
#   claude   = opus (t1) -- the reference path, the established quota pool.
#   opencode = glm-5.2 via the zai-coding-plan accounts (t2) -- a SEPARATE quota
#              pool that sits idle. Routing a lane (e.g. docs, where glm is proven)
#              to opencode relieves the opus weekly-quota throughput ceiling.
BACKENDS = ("claude", "opencode")
_BACKEND_PRODUCT = {"claude": "claude", "opencode": "opencode"}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable or "python"


def lane_issue_numbers(root: Path, explicit_lane: str | None,
                       exclude: set[str] | None = None) -> dict[str, Any]:
    """Pick the lane (busiest, or explicit) and return its OPEN issue numbers,
    most-recent first. Reuses the same router fold issue_dispatch.pick_lane uses,
    but keeps the per-issue numbers (which pick_lane discards). ``exclude`` drops
    lanes from the busiest-pick (e.g. an opus task excludes 'docs' so the glm task
    owns it) -- ignored when an explicit lane is named."""
    router = issue_dispatch.run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
        root, timeout=130)
    lanes = router.get("lanes") or {}
    nums_by_lane: dict[str, list[int]] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        nums: list[int] = []
        for it in (iss or []):
            n = it.get("number") if isinstance(it, dict) else it
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
        nums_by_lane[ln] = sorted(nums, reverse=True)
    exclude = exclude or set()
    if explicit_lane:
        chosen = explicit_lane
    else:
        eligible = {k: v for k, v in nums_by_lane.items() if k not in exclude}
        chosen = max(eligible, key=lambda k: len(eligible[k])) if eligible else None
    return {"lane": chosen, "numbers": nums_by_lane.get(chosen or "", []),
            "by_lane_count": {k: len(v) for k, v in nums_by_lane.items()},
            "excluded_lanes": sorted(exclude),
            "router_error": router.get("_error")}


def live_resolution_issues(runs_dir: Path) -> set[int]:
    """Issue numbers that already have a LIVE resolution worker — read from the
    dispatch logs (``resolve-<N>-<stamp>.log``) whose pid is still alive. Best
    effort: a log without a recoverable pid is treated as not-live."""
    live: set[int] = set()
    if not runs_dir.is_dir():
        return live
    try:
        import psutil  # type: ignore
        alive = {p.pid for p in psutil.process_iter()}
    except ImportError:
        alive = set()
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        pid_file = log.with_suffix(".pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            if not alive or pid in alive:
                live.add(int(m.group(1)))
    return live


def recently_attempted_issues(runs_dir: Path, *, cooldown_min: int,
                              now_ts: float | None = None) -> set[int]:
    """Issue numbers attempted within the last ``cooldown_min`` minutes — read from
    the mtime of their ``resolve-<N>-*.log``. This is the anti-churn gate: a hard
    issue (e.g. a mislabeled epic) that a worker could not land must NOT be re-picked
    every tick — re-dispatching it re-storms a known drain while the rest of the
    lane's backlog goes untouched. After the cooldown it becomes eligible again (the
    repo may have changed, or a fresh worker may get further). 0 disables the gate."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        try:
            if log.stat().st_mtime >= horizon:
                recent.add(int(m.group(1)))
        except OSError:
            continue
    return recent


def pick_target_issue(numbers: list[int], skip: set[int]) -> int | None:
    """The first lane issue not in ``skip`` (live workers ∪ recently-attempted)."""
    for n in numbers:
        if n not in skip:
            return n
    return None


def build_worker_command(backend: str, prompt: str, model: str | None) -> list[str]:
    """The argv for one detached issue-resolution worker, per backend. Both forms
    run the SAME issue-resolution prompt (with its ``#N``-in-subject rule), so the
    resulting commit is witnessable by the closure auditor regardless of backend."""
    if backend == "claude":
        return ["claude", "-p", "--permission-mode", "bypassPermissions", prompt]
    if backend == "opencode":
        cmd = ["opencode", "run", "--dangerously-skip-permissions"]
        if model:
            cmd += ["-m", model]  # pin the exact model so the run is reproducible/traced
        cmd.append(prompt)
        return cmd
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def _opencode_config_home(account_dir: str, runs_dir: Path) -> str:
    """opencode loads its config from ``$XDG_CONFIG_HOME/opencode``. The switcher's
    account dir is either the canonical ``<config_home>/opencode`` (the default
    account) or a sibling ``<config_home>/opencode-<tag>``. For the canonical dir the
    parent IS the XDG home; for a sibling we mint a tiny pin dir whose ``opencode``
    entry is a junction to the account, so opencode loads exactly that account -- we
    never silently mis-attribute a run to the wrong account."""
    p = Path(account_dir)
    if p.name == "opencode":
        return str(p.parent)
    pin = runs_dir / ".opencode-pins" / p.name
    link = pin / "opencode"
    pin.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        try:
            os.symlink(account_dir, link, target_is_directory=True)
        except OSError:
            # Windows without symlink privilege: a directory junction needs none.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), account_dir],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    if not link.exists():
        raise RuntimeError(f"could not pin opencode account dir {account_dir!r}")
    return str(pin)


def opencode_worker_env(account_dir: str | None, lane: str, workspace: Path,
                        runs_dir: Path) -> dict[str, str]:
    """Child env for an opencode/glm worker: the account pinned via XDG_CONFIG_HOME
    (NOT CLAUDE_CONFIG_DIR / oauth token, which are claude-only), plus the same
    self-describing dispatch vars the claude path stamps."""
    env = dispatch_worker.child_env(lane, "opencode", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["XDG_CONFIG_HOME"] = _opencode_config_home(account_dir, runs_dir)
    return env


# Windows process-creation flags for a detached worker.
#   DETACHED_PROCESS  — the child gets NO console at all.
#   CREATE_NO_WINDOW  — the child gets a console, but it is HIDDEN (no window).
# Every detached worker — batch shim OR native exe — gets CREATE_NO_WINDOW, never
# DETACHED_PROCESS, for TWO reasons:
#   1. A .cmd/.bat launcher (opencode.CMD, the npm shim) is run BY cmd.exe, which
#      REQUIRES a console: under DETACHED_PROCESS the batch wrapper dies at the
#      "Terminate batch job (Y/N)?" prompt producing ZERO output — every dispatched
#      glm/opencode worker was DOA (0-byte log) until it got a (hidden) console.
#   2. A native worker (claude.exe) given DETACHED_PROCESS has no console, so every
#      console tool it spawns — git, gh, fak, the shell — pops its OWN visible
#      window: the "random popup windows" seen during a dispatched run. CREATE_NO_WINDOW
#      gives the worker one HIDDEN console the whole tool subtree inherits, so it
#      still outlives this dispatcher but never flashes a window. (Same conclusion as
#      claude_agent_chat.detached_creationflags.)
_DETACHED_PROCESS = 0x00000008  # retained to document the rejected alternative
_CREATE_NO_WINDOW = 0x08000000


def win_creationflags(exe: str) -> int:
    """The Windows creationflags for spawning ``exe`` detached: ALWAYS a HIDDEN
    console (CREATE_NO_WINDOW). A .cmd/.bat shim needs a console to live; a native
    exe needs one so its console grandchildren don't each pop a visible window.
    ``exe`` is accepted for call-site stability but no longer branches the result."""
    del exe  # both batch shims and native exes take the hidden-console path
    return _CREATE_NO_WINDOW


def spawn_issue_worker(command: list[str], env: dict[str, str], cwd: Path,
                       log_dir: Path, issue: int, lane: str,
                       backend: str) -> dict[str, Any]:
    """Launch a detached worker (claude or opencode) on one issue; record pid.

    The log keeps the backend-neutral ``resolve-<N>-<stamp>.log`` name so the close
    arm, silent-worker scan, and in-flight de-dup all see it uniformly."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"resolve-{issue}-{stamp}.log"
    exe = shutil.which(command[0]) or command[0]
    argv = [exe, *command[1:]]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = win_creationflags(exe)
    else:
        kwargs["start_new_session"] = True
    fh = open(out_log, "w", encoding="utf-8")
    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    (out_log.with_suffix(".pid")).write_text(str(proc.pid), encoding="utf-8")
    # Per-worker backend sidecar: makes each run's backend traceable from disk,
    # independent of the (multi-task, last-writer) tick record.
    (out_log.with_suffix(".backend")).write_text(backend, encoding="utf-8")
    return {"pid": proc.pid, "log": str(out_log), "issue": issue, "lane": lane,
            "backend": backend}


# --- Weekly-cap gate -------------------------------------------------------
# A worker spawned on a quota-exhausted (but still logged-in) account dies
# immediately with a ~65-byte banner, e.g.:
#   You've hit your weekly limit · resets Jun 25, 1pm (America/Los_Angeles)
#   You've hit your weekly limit · resets 4am (America/Los_Angeles)
# The spawn preflight cannot see this — it checks host/headroom/account-logged-in,
# NOT remaining quota — so absent this gate every tick re-spawns a doomed worker
# for the entire (multi-day) reset window: it ships nothing and floods the logs.
# These helpers read the banner back from recent worker logs and persist a hold so
# the dispatcher stops spawning until the quota resets, turning a silent spin into
# an honest WEEKLY_CAPPED hold. Everything here is FAIL-OPEN: any error resolves to
# "not capped", so the gate can only ever ADD a refusal, never wedge the loop.

_CAP_BANNER_RE = re.compile(r"hit your[\w\s]*limit", re.IGNORECASE)
_CAP_RESET_RE = re.compile(r"resets\s+(.+?)\s*\(America/Los_Angeles\)", re.IGNORECASE)
_CAP_RESET_FALLBACK_RE = re.compile(r"resets\s+([^\r\n]+)", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    start=1)}
# America/Los_Angeles is PDT (UTC-7) over the summer reset windows this gate sees;
# a PT wall-clock time + this offset == UTC. (A ±1h DST error only nudges the hold
# edge and self-corrects on the next probe — fine for a throttle.)
_PT_TO_UTC = dt.timedelta(hours=7)


def _backend_of_log(log: Path) -> str:
    """The backend that produced a worker log, from its ``.backend`` sidecar;
    legacy logs without one are the original ``claude`` backend."""
    try:
        return log.with_suffix(".backend").read_text(encoding="utf-8").strip() or "claude"
    except OSError:
        return "claude"


def _scan_recent_cap_banner(runs_dir: Path, *, product: str, lookback_min: int,
                            now_ts: float) -> dict[str, Any] | None:
    """The most-recent worker log (this backend, mtime within ``lookback_min``,
    tiny) whose body is the quota-limit banner → ``{reset_text, evidence_log}``,
    else None. The size cap (a real worker log is large; the banner is ~65 bytes)
    plus the specific phrase keep a prose log that merely mentions "limit" from
    false-matching."""
    if not runs_dir.is_dir():
        return None
    horizon = now_ts - lookback_min * 60
    best: tuple[float, str, str] | None = None
    for log in runs_dir.glob("resolve-*.log"):
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon or st.st_size > 512:
            continue
        if _backend_of_log(log) != product:
            continue
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not _CAP_BANNER_RE.search(text):
            continue
        m = _CAP_RESET_RE.search(text) or _CAP_RESET_FALLBACK_RE.search(text)
        reset_text = m.group(1).strip() if m else ""
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, reset_text, log.name)
    if best is None:
        return None
    return {"reset_text": best[1], "evidence_log": best[2]}


def _parse_reset_to_utc(reset_text: str, now_utc: dt.datetime) -> dt.datetime | None:
    """Best-effort parse of a banner reset clause ('Jun 25, 1pm', '4am',
    '11:30pm') as America/Los_Angeles wall-clock → a naive-UTC datetime, clamped to
    (now, now+8d]. None when no time-of-day is present (caller falls back to a short
    cooldown)."""
    if not reset_text:
        return None
    t = reset_text.lower()
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
    if not tm:
        return None
    hour = int(tm.group(1)) % 12 + (12 if tm.group(3) == "pm" else 0)
    minute = int(tm.group(2) or 0)
    mo = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})", t)
    if mo:
        month, day = _MONTHS[mo.group(1)], int(mo.group(2))
        try:
            cand = dt.datetime(now_utc.year, month, day, hour, minute) + _PT_TO_UTC
        except ValueError:
            return None
        if cand < now_utc - dt.timedelta(days=1):
            try:
                cand = dt.datetime(now_utc.year + 1, month, day, hour, minute) + _PT_TO_UTC
            except ValueError:
                return None
    else:
        now_pt = now_utc - _PT_TO_UTC
        cand_pt = now_pt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if cand_pt <= now_pt:
            cand_pt += dt.timedelta(days=1)
        cand = cand_pt + _PT_TO_UTC
    if cand <= now_utc:
        return None
    return min(cand, now_utc + dt.timedelta(days=8))


def _iso_to_utc(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat((s or "").replace("Z", ""))
    except (ValueError, AttributeError):
        return None


def check_weekly_cap(runs_dir: Path, *, product: str, account_tag: str | None,
                     now_ts: float | None = None, lookback_min: int = 45,
                     fallback_min: int = 60) -> dict[str, Any]:
    """Is ``account_tag`` quota-capped right now? Detects the limit banner from
    recent worker logs and persists a hold in ``account-cap-<product>.json`` so it
    survives the ticks AFTER spawns stop (no fresh banner is produced once we stop
    launching doomed workers). Returns ``{"capped": bool, ...}``. FAIL-OPEN: any
    exception → ``{"capped": False}``."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)  # naive UTC
        state_path = runs_dir / f"account-cap-{product}.json"
        hit = _scan_recent_cap_banner(runs_dir, product=product,
                                      lookback_min=lookback_min, now_ts=now_ts)
        if hit:
            until = (_parse_reset_to_utc(hit["reset_text"], now_utc)
                     or now_utc + dt.timedelta(minutes=fallback_min))
            state = {"product": product, "account": account_tag,
                     "reset_text": hit["reset_text"], "evidence_log": hit["evidence_log"],
                     "detected": now_utc.isoformat() + "Z", "until": until.isoformat() + "Z"}
            try:
                runs_dir.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except OSError:
                pass
            return {"capped": True, "until": state["until"],
                    "reset_text": hit["reset_text"], "source": "banner"}
        # No fresh banner: honor a persisted, unexpired hold for THIS account.
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {"capped": False}
            until = _iso_to_utc(state.get("until") or "")
            if state.get("account") == account_tag and until and now_utc < until:
                return {"capped": True, "until": state.get("until"),
                        "reset_text": state.get("reset_text", ""), "source": "state"}
            if until and now_utc >= until:
                try:
                    state_path.unlink()
                except OSError:
                    pass
        return {"capped": False}
    except Exception:
        return {"capped": False}


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
             live: bool, refresh: bool = True, cooldown_min: int = 120,
             backend: str = "claude",
             exclude_lanes: set[str] | None = None) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    product = _BACKEND_PRODUCT[backend]
    reg = issue_dispatch.refresh_registry(root) if refresh else {"skipped": True}
    pre = issue_dispatch.preflight(root, max_workers=max_workers, work_kind=work_kind,
                                   product=product)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}
    runs_dir = root / RUNS_DIRNAME

    # Weekly-cap gate — BEFORE the lane router's gh work. A logged-in account can
    # still be quota-exhausted; the preflight returns SPAWN_OK regardless. Without
    # this, every tick spawns a worker that instantly dies on the limit banner,
    # ships nothing, and floods the logs for the whole reset window. Read that
    # banner back from recent worker logs and HOLD until the stated reset instead.
    # Fail-open (check_weekly_cap can only return capped on positive evidence).
    cap = (check_weekly_cap(runs_dir, product=product, account_tag=acct.get("tag"))
           if pre_ok else {"capped": False})
    if pre_ok and cap.get("capped"):
        return {
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": reg,
            "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                          "cap": pre.get("cap"), "live": pre.get("live")},
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
            "weekly_cap": cap, "ok": False, "action": "weekly_capped",
            "verdict": "WEEKLY_CAPPED",
            "reason": (f"account '{acct.get('tag')}' is weekly-capped (resets "
                       f"{cap.get('reset_text') or '?'}); holding spawn until "
                       f"{cap.get('until')} — re-arm is automatic at the reset"),
        }

    pick = lane_issue_numbers(root, lane, exclude=exclude_lanes)
    chosen_lane = pick.get("lane")
    live_issues = live_resolution_issues(runs_dir)
    cooled = recently_attempted_issues(runs_dir, cooldown_min=cooldown_min)
    # Skip both a live worker's issue AND a recently-attempted one (anti-churn), so
    # the picker advances down the lane instead of re-storming one un-landable issue.
    skip = live_issues | cooled
    target = pick_target_issue(pick.get("numbers") or [], skip)

    payload: dict[str, Any] = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "registry_refresh": reg,
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen_lane, "lane_issue_count": len(pick.get("numbers") or []),
        "cooled_recently": sorted(cooled), "target_issue": target,
        "already_live": sorted(live_issues),
    }

    if not pre_ok:
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return payload
    if not chosen_lane:
        payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                        "reason": "no lane has open issues (router empty/error)"})
        return payload
    if target is None:
        payload.update({"ok": False, "action": "no_issue", "verdict": "NO_ISSUE",
                        "reason": (f"every open issue on lane '{chosen_lane}' is "
                                   f"either live ({sorted(live_issues)}) or in "
                                   f"cooldown ({sorted(cooled)}) — nothing fresh to "
                                   f"dispatch this tick")})
        return payload

    rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
    payload["prompt_chars"] = rec.get("prompt_chars")
    payload["issue_title"] = rec.get("title")
    model = acct.get("model") if backend == "opencode" else None
    preview_prompt = f"<resolve #{target} prompt, {rec.get('prompt_chars')} chars>"
    payload["command"] = build_worker_command(backend, preview_prompt, model)

    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 {backend} issue-resolution worker on "
                                   f"#{target} (lane '{chosen_lane}') under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')}, "
                                   f"{acct.get('model') or backend})")})
        return payload

    if backend == "claude":
        env = issue_dispatch.worker_env(acct.get("dir"), chosen_lane, root)
    else:
        env = opencode_worker_env(acct.get("dir"), chosen_lane, root, runs_dir)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    command = build_worker_command(backend, rec["prompt"], model)
    spawned = spawn_issue_worker(command, env, root, runs_dir, target, chosen_lane, backend)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned {backend} issue-resolution worker pid "
                               f"{spawned['pid']} on #{target} (lane '{chosen_lane}') "
                               f"under '{acct.get('tag')}' ({acct.get('model') or backend})")})
    _record(runs_dir, payload)
    return payload


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    # Scope the tick record by backend so concurrent tasks (e.g. the opus
    # FleetIssueDispatch and the glm FleetIssueDispatchGlm) don't clobber each
    # other's trace; also keep the legacy unscoped file (last-writer) for any
    # manual reader that still expects it.
    backend = payload.get("backend") or "claude"
    blob = json.dumps(payload, indent=2)
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"last-resolve-tick-{backend}.json").write_text(blob, encoding="utf-8")
        (runs_dir / "last-resolve-tick.json").write_text(blob, encoding="utf-8")
    except OSError:
        pass


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    lines = [
        f"issue-resolve-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  "
        f"backend={p.get('backend')}  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live)",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)",
        f"  target    : #{p.get('target_issue')}  {(p.get('issue_title') or '')[:54]}",
        f"  -> {p.get('reason')}",
    ]
    if p.get("spawned"):
        s = p["spawned"]
        lines.append(f"  spawned pid={s.get('pid')} issue=#{s.get('issue')} log={s.get('log')}")
    if not p.get("live") and p.get("action") == "would_spawn":
        lines.append("  DRY-RUN — re-run with --live to spawn the issue worker")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One guarded tick that spawns an issue-RESOLUTION worker (dry-run by default).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="hard cap on live workers, enforced by the preflight (default: 2)")
    ap.add_argument("--work-kind", default=None,
                    help="switcher work kind (engineering->t1, gardening->t2). Default "
                         "follows --backend: engineering for claude (t1 opus pool), "
                         "gardening for opencode/glm (t2 pool). The opencode pool has NO "
                         "t1 account, so an engineering route there REFUSE_NO_ACCOUNTs — "
                         "which silently stalled the docs lane until this default landed.")
    ap.add_argument("--lane", default=None,
                    help="explicit lane (default: the lane with the most open issues)")
    ap.add_argument("--backend", choices=BACKENDS, default="claude",
                    help="worker backend: claude (opus, t1) or opencode (glm-5.2, t2, "
                         "a separate quota pool). Default claude.")
    ap.add_argument("--exclude-lane", default="",
                    help="comma-separated lanes to drop from the busiest-pick (e.g. an "
                         "opus task excludes 'docs' so a glm task owns it). Ignored when "
                         "--lane is set.")
    ap.add_argument("--live", action="store_true",
                    help="actually spawn the worker (default: dry-run / plan only)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh")
    ap.add_argument("--cooldown-min", type=int, default=120,
                    help="skip an issue attempted within this many minutes (anti-churn; "
                         "0 disables). Default 120 — stops re-storming one un-landable issue.")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    exclude_lanes = {s.strip() for s in args.exclude_lane.split(",") if s.strip()}
    # The opencode/glm pool is tier 2 only; engineering routes to t1 and finds
    # nothing there. Derive the work-kind from the backend unless set explicitly.
    work_kind = args.work_kind or ("gardening" if args.backend == "opencode" else "engineering")
    payload = evaluate(root, max_workers=args.max_workers, work_kind=work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh,
                       cooldown_min=args.cooldown_min, backend=args.backend,
                       exclude_lanes=exclude_lanes)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
