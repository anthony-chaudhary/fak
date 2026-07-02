#!/usr/bin/env python3
r"""THE one-touch status card for the always-on issue dispatcher.

One command, one screen, the whole loop: is the always-on watchdog installed and
ticking, can it grow right now (the DoS-safe spawn gate), how many workers are
live, which account the switcher would pick, how big the open-issue backlog is,
and — the load-bearing honesty metric — what fraction of *closed* issues are
TRULY resolved (a DOS-witnessed resolving commit in git) versus merely
CLAIMED_CLOSED.

It is a pure FOLD over tools that already exist; it launches nothing and writes
nothing:

  * tools/dispatch_preflight.py   host-guard ∧ account-free ∧ under-cap  (fast)
  * tools/dos_supervisor_status.py  the [supervise] readiness card       (fast)
  * schtasks query                 is FleetDOSDispatchWatchdog installed? (fast)
  * tools/issue_lane_router.py     open backlog mapped to lanes          (gh; slow)
  * tools/issue_closure_audit.py   closure_rate = resolved/claimed       (gh; slow)

The two gh-backed folds are bounded and degrade to "n/a" on timeout, so the card
always returns promptly. ``--fast`` skips them entirely (pure-local, sub-5s).
Exit 0 when the dispatcher is healthy (safe to grow OR already at a healthy
target and host clean), 1 when something needs an operator's eye.

    python tools/dispatch_status.py            # the card
    python tools/dispatch_status.py --json     # machine-readable
    python tools/dispatch_status.py --fast      # skip gh-backed folds
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispatch_preflight  # noqa: E402  (pid-sidecar identity probe)

SCHEMA = "fleet-dispatch-status/1"
# The guarded always-on tick (tools/register_issue_dispatch.ps1). The older
# FleetDOSDispatchWatchdog keeps the un-gated kernel supervisor alive; this card
# tracks the DoS-safe issue dispatcher, so it reports the guarded task.
WATCHDOG_TASK = "FleetIssueDispatch"

RUNS_DIRNAME = ".dispatch-runs"
# Per-session fak-guard decision journals (one file per guarded worker), written by
# the dispatch worker's guard_wrap (dispatch_worker.py / cmd/dispatchworker). Each
# non-empty JSONL line is one kernel decision (internal/journal.Row).
GUARD_AUDIT_DIRNAME = "guard-audit"
# The decision-row kinds the journal records (internal/journal.rowFromEvent). DENY +
# RESULT_DENY are the refusals; QUARANTINE is a poisoned-result hold.
_GUARD_DENY_KINDS = ("DENY", "RESULT_DENY")
_GUARD_QUARANTINE_KIND = "QUARANTINE"
_GUARD_RECENT_LOOKBACK_MIN = 90
# resolve-<N>-<stamp>.log written by issue_resolve_dispatch.spawn_issue_worker.
_RESOLVE_LOG_RE = re.compile(r"resolve-(\d+)-(\d{8}-\d{6})\.log$")
_LEASEREF_PREFIX = "refs/fak/locks/"
_NOOP_BANNER_RE = re.compile(r"(?i)>\s*build\s*[·:]")
# The real-turn byte floor: a log at or below this carried no productive turn — it
# is a 0-byte spawn or a banner-only stub (e.g. a detached opencode worker that
# prints `> build · <model>` and exits). Mirrors the canonical
# ``issue_resolve_dispatch._STUB_LOG_MAX_BYTES`` (a drift test pins them equal) so a
# banner-only no-op counts as "produced nothing", not as output. See #1276.
_STUB_LOG_MAX_BYTES = 512
_BACKEND_STUB_LOOKBACK_MIN = 90
_RID_RE = re.compile(r"^RID-[A-Z0-9]+$")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def run_json(cmd: list[str], cwd: Path, timeout: int,
             ok_codes: set[int] | None = None) -> dict[str, Any]:
    ok_codes = ok_codes if ok_codes is not None else set(range(0, 16))
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, creationflags=_win_creationflags())
    except subprocess.TimeoutExpired:
        return {"_error": f"timed out after {timeout}s", "_cmd": cmd}
    except OSError as exc:
        return {"_error": str(exc), "_cmd": cmd}
    doc = _last_json(proc.stdout)
    doc.setdefault("_returncode", proc.returncode)
    if proc.returncode not in ok_codes and "_error" not in doc and not doc.get("schema"):
        doc["_error"] = (proc.stderr or proc.stdout or "").strip()[-300:]
    return doc


def _last_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def git_path(root: Path, name: str) -> Path:
    try:
        proc = subprocess.run(["git", "rev-parse", "--git-path", name], cwd=root,
                              capture_output=True, text=True, timeout=5,
                              creationflags=_win_creationflags())
    except (subprocess.TimeoutExpired, OSError):
        return root / ".git" / name
    if proc.returncode != 0:
        return root / ".git" / name
    path = (proc.stdout or "").strip()
    return (root / path).resolve() if path else root / ".git" / name


def merge_state(root: Path) -> dict[str, Any]:
    merge_head = git_path(root, "MERGE_HEAD")
    present = False
    try:
        present = merge_head.exists() and bool(merge_head.read_text(encoding="utf-8").strip())
    except OSError:
        present = False
    out: dict[str, Any] = {
        "merge_in_progress": present,
        "merge_head": str(merge_head),
    }
    if present:
        out["next_action"] = (
            "wait for MERGE_HEAD to clear before starting new worker edits; "
            "partial path commits are unsafe while a peer merge is in progress")
    return out


def _string_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v]
    return []


def _normalize_tree(t: str) -> str:
    t = str(t or "").strip().replace("\\", "/")
    if t.startswith("./"):
        t = t[2:]
    t = t.rstrip("/")
    for suffix in ("/**", "/*"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t.rstrip("/")


def _clean_tree(tree: Any) -> list[str]:
    out: list[str] = []
    for t in _string_list(tree):
        n = _normalize_tree(t)
        if n:
            out.append(n)
    return out


def _tree_overlap_one(a: str, b: str) -> bool:
    a, b = _normalize_tree(a), _normalize_tree(b)
    if not a or not b:
        return True
    if a in ("**", "**/*") or b in ("**", "**/*"):
        return True
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def trees_overlap(a: Any, b: Any) -> bool:
    ta, tb = _clean_tree(a), _clean_tree(b)
    if not ta or not tb:
        return True
    return any(_tree_overlap_one(x, y) for x in ta for y in tb)


def _lease_active_unix(rec: dict[str, Any]) -> int | None:
    acquired = _int(rec.get("acquired_unix"))
    renewed = _int(rec.get("renewed_unix"))
    if acquired is None and renewed is None:
        return None
    if acquired is None:
        return renewed
    if renewed is None:
        return acquired
    return max(acquired, renewed)


def _lease_expired(rec: dict[str, Any], now_ts: float) -> bool:
    ttl = _int(rec.get("ttl_seconds"), 0) or 0
    if ttl <= 0:
        return False
    active = _lease_active_unix(rec)
    if active is None:
        return False
    return now_ts >= active + ttl


def _lease_lane(lease_id: str) -> str:
    lease_id = str(lease_id or "").strip()
    if not lease_id.startswith("resolve-"):
        return lease_id
    lane = lease_id[len("resolve-"):]
    if re.search(r"-\d+$", lane):
        lane = re.sub(r"-\d+$", "", lane)
    return lane or lease_id


def _backlog_candidates(backlog: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = (backlog.get("lanes") or {}) if isinstance(backlog, dict) else {}
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for row in backlog.get("issues") or []:
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane") or "")
        issue = _int(row.get("number"))
        if not lane or issue is None:
            continue
        grp = lanes.get(lane) or {}
        cand = {
            "issue": issue,
            "lane": lane,
            "confidence": row.get("confidence"),
            "tree": _string_list(grp.get("tree")),
        }
        out.append(cand)
        seen.add((issue, lane))

    for lane, grp_any in lanes.items():
        if not isinstance(grp_any, dict):
            continue
        lane_s = str(lane)
        for issue_any in grp_any.get("issues") or []:
            issue = _int(issue_any)
            if issue is None or (issue, lane_s) in seen:
                continue
            out.append({
                "issue": issue,
                "lane": lane_s,
                "confidence": None,
                "tree": _string_list(grp_any.get("tree")),
            })
            seen.add((issue, lane_s))
    return out


def summarize_leases(records: list[dict[str, Any]], backlog: dict[str, Any],
                     *, now_ts: float | None = None) -> dict[str, Any]:
    """Classify refs/fak/locks records for the status card.

    Active leases block a candidate only when their tree overlaps a currently
    routed issue's lane tree. Expired records stay visible as reapable residue,
    but never block a candidate.
    """
    now_ts = time.time() if now_ts is None else now_ts
    backlog = backlog if isinstance(backlog, dict) else {}
    lanes = backlog.get("lanes") or {}
    candidate_source_available = "_skipped" not in backlog and not ("_error" in backlog and not lanes)
    candidates = _backlog_candidates(backlog)
    active: list[dict[str, Any]] = []
    expired: list[dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        lease_id = str(rec.get("id") or "").strip()
        if not lease_id:
            continue
        tree = _string_list(rec.get("tree_globs"))
        active_unix = _lease_active_unix(rec)
        age_seconds = max(0, int(now_ts - active_unix)) if active_unix is not None else None
        ttl = _int(rec.get("ttl_seconds"), 0) or 0
        expires_in = None
        if ttl > 0 and active_unix is not None:
            expires_in = int(active_unix + ttl - now_ts)
        row = {
            "id": lease_id,
            "lane": _lease_lane(lease_id),
            "holder": rec.get("holder"),
            "tree": tree,
            "age_seconds": age_seconds,
            "age_min": round(age_seconds / 60, 1) if age_seconds is not None else None,
            "ttl_seconds": ttl,
            "expires_in_seconds": expires_in,
            "generation": rec.get("generation"),
        }
        if _lease_expired(rec, now_ts):
            row["status"] = "EXPIRED"
            row["blocks_candidate"] = False
            row["blocking_candidates"] = []
            expired.append(row)
            continue
        row["status"] = "LIVE"
        if candidate_source_available:
            blockers = [
                c for c in candidates
                if trees_overlap(tree, c.get("tree"))
            ]
            row["blocks_candidate"] = bool(blockers)
            row["blocking_candidates"] = blockers[:8]
        else:
            row["blocks_candidate"] = None
            row["blocking_candidates"] = []
        active.append(row)

    active.sort(key=lambda r: (not bool(r.get("blocks_candidate")), str(r.get("lane") or ""), str(r.get("id") or "")))
    expired.sort(key=lambda r: str(r.get("id") or ""))
    return {
        "source": "refs/fak/locks",
        "candidate_source_available": candidate_source_available,
        "candidate_count": len(candidates),
        "active_count": len(active),
        "expired_count": len(expired),
        "blocking_count": sum(1 for r in active if r.get("blocks_candidate")),
        "active": active,
        "expired": expired[:8],
    }


def read_leaseref_records(root: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        proc = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", _LEASEREF_PREFIX],
            cwd=root, capture_output=True, text=True, timeout=10,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout or "git for-each-ref failed").strip()[-500:]

    records: list[dict[str, Any]] = []
    skipped = 0
    for ref in (proc.stdout or "").splitlines():
        ref = ref.strip()
        if not ref.startswith(_LEASEREF_PREFIX):
            continue
        if ref[len(_LEASEREF_PREFIX):].startswith("session-"):
            continue
        try:
            blob = subprocess.run(
                ["git", "cat-file", "blob", ref],
                cwd=root, capture_output=True, text=True, timeout=10,
                creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired):
            skipped += 1
            continue
        if blob.returncode != 0:
            skipped += 1
            continue
        try:
            rec = json.loads(blob.stdout or "{}")
        except ValueError:
            skipped += 1
            continue
        if isinstance(rec, dict):
            rec.setdefault("id", ref[len(_LEASEREF_PREFIX):])
            records.append(rec)
        else:
            skipped += 1
    if skipped:
        for rec in records:
            rec.setdefault("_skipped_records", skipped)
    return records, None


def read_lease_state(root: Path, backlog: dict[str, Any],
                     *, now_ts: float | None = None) -> dict[str, Any]:
    records, err = read_leaseref_records(root)
    if err:
        return {
            "source": "refs/fak/locks",
            "read_error": err,
            "candidate_source_available": False,
            "candidate_count": 0,
            "active_count": 0,
            "expired_count": 0,
            "blocking_count": 0,
            "active": [],
            "expired": [],
        }
    state = summarize_leases(records, backlog, now_ts=now_ts)
    skipped = max((_int(r.get("_skipped_records"), 0) or 0) for r in records) if records else 0
    if skipped:
        state["skipped_records"] = skipped
    return state


def _dispatch_lease_token(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "unknown"
    out: list[str] = []
    for ch in s:
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in "-_.":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-.") or "unknown"


def _default_lease_id_for_lane(lane: str) -> str:
    return "resolve-" + _dispatch_lease_token(lane)


def _spawn_lane(log: Path) -> str:
    try:
        first = log.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return ""
    for field in first.split():
        if field.startswith("lane="):
            return field[len("lane="):]
    return ""


def _banner_noop(log: Path) -> bool:
    try:
        if log.stat().st_size > _STUB_LOG_MAX_BYTES:
            return False
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(_NOOP_BANNER_RE.search(text))


def _read_worker_lease_id(stem: Path, lane: str) -> str:
    try:
        value = (stem.with_suffix(".lease-id")).read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    return _default_lease_id_for_lane(lane) if lane else ""


def _read_worker_tree(stem: Path) -> list[str]:
    try:
        obj = json.loads(stem.with_suffix(".lease-tree.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _string_list(obj)


def scan_live_dispatch_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> dict[str, Any]:
    if not runs_dir.is_dir():
        return {"available": True, "workers": []}
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore

            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            return {"available": False, "workers": [], "error": "psutil unavailable"}

    workers: list[dict[str, Any]] = []
    for log in runs_dir.glob("resolve-*.log"):
        m = _RESOLVE_LOG_RE.search(log.name)
        if not m:
            continue
        stem = log.with_suffix("")
        pid_file = stem.with_suffix(".pid")
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(
            pid_file, alive=alive, probe=probe):
            continue
        if _banner_noop(log):
            continue
        lane = _spawn_lane(log)
        lease_id = _read_worker_lease_id(stem, lane)
        workers.append({
            "worker": stem.name,
            "issue": int(m.group(1)),
            "stamp": m.group(2),
            "pid": pid,
            "lane": lane,
            "lease_id": lease_id,
            "tree": _read_worker_tree(stem),
            "log": log.name,
        })
    workers.sort(key=lambda r: str(r.get("worker") or ""))
    return {"available": True, "workers": workers}


def cross_check_worker_leases(worker_state: dict[str, Any],
                              leases: dict[str, Any]) -> dict[str, Any]:
    active_leases = {
        str(row.get("id") or ""): row
        for row in (leases.get("active") or [])
        if row.get("id")
    }
    if not worker_state.get("available", True):
        return {
            "available": False,
            "error": worker_state.get("error"),
            "clean_count": 0,
            "orphan_process_count": 0,
            "orphan_lease_count": 0,
            "clean": [],
            "orphan_process": [],
            "orphan_lease": [],
        }

    clean: list[dict[str, Any]] = []
    orphan_process: list[dict[str, Any]] = []
    matched: set[str] = set()
    for worker in worker_state.get("workers") or []:
        lease_id = str(worker.get("lease_id") or "")
        lease = active_leases.get(lease_id)
        if lease:
            matched.add(lease_id)
            clean.append({"worker": worker, "lease": lease})
        else:
            orphan_process.append({
                "worker": worker,
                "reason": "missing active dispatch lease" if lease_id else "worker has no lease id",
            })

    orphan_lease = [
        {"lease": lease, "reason": "active lease has no local live worker sidecar"}
        for lease_id, lease in sorted(active_leases.items())
        if lease_id not in matched
    ]
    return {
        "available": True,
        "clean_count": len(clean),
        "orphan_process_count": len(orphan_process),
        "orphan_lease_count": len(orphan_lease),
        "clean": clean,
        "orphan_process": orphan_process,
        "orphan_lease": orphan_lease,
    }


def worker_lease_crosscheck(
    runs_dir: Path,
    leases: dict[str, Any],
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> dict[str, Any]:
    workers = scan_live_dispatch_workers(runs_dir, alive=alive, probe=probe)
    return cross_check_worker_leases(workers, leases)


def has_key_named(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(has_key_named(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(has_key_named(v, key) for v in obj)
    return False


def run_ids_from_loop_ledger(ledger: Path, *, limit: int = 6) -> list[str]:
    if limit <= 0 or not ledger.exists():
        return []
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        loop_id = str(row.get("loop_id") or "")
        run_id = str(row.get("run_id") or "")
        if not loop_id.startswith("issue-resolve-") or not _RID_RE.fullmatch(run_id):
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        out.append(run_id)
        if len(out) >= limit:
            break
    return out


def dos_status_digest(root: Path, run_id: str) -> dict[str, Any]:
    if not _RID_RE.fullmatch(run_id):
        return {"run_id": run_id, "_error": "not a DOS RID"}
    try:
        proc = subprocess.run(
            ["dos", "status", "--workspace", str(root), "--json", run_id],
            cwd=root, capture_output=True, text=True, timeout=45,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"run_id": run_id, "_error": str(exc)}
    doc = _last_json(proc.stdout)
    if not doc:
        return {"run_id": run_id, "_error": (proc.stderr or proc.stdout or "no JSON").strip()[-500:]}
    if has_key_named(doc, "claimed"):
        return {"run_id": run_id, "_error": "dos status emitted forbidden claimed field",
                "reason": "RUN_STATUS_CLAIMED_FIELD"}
    doc.setdefault("run_id", run_id)
    doc["_returncode"] = proc.returncode
    return doc


def read_run_status_digests(
    root: Path,
    *,
    ledger: Path | None = None,
    limit: int = 6,
    status_reader: Any | None = None,
) -> list[dict[str, Any]]:
    ledger = ledger or root / ".fak" / "loops.jsonl"
    reader = status_reader or (lambda rid: dos_status_digest(root, rid))
    return [reader(rid) for rid in run_ids_from_loop_ledger(ledger, limit=limit)]


def silent_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Issue-resolution workers that exited having produced NOTHING — a
    ``resolve-<N>-<stamp>.log`` at or below the real-turn floor
    (``_STUB_LOG_MAX_BYTES``) whose ``.pid`` process is dead.

    "Produced nothing" is NOT only a 0-byte file: a detached worker that prints its
    spawn header + a TUI banner (e.g. opencode's ``> build · <model>``, ~122 bytes)
    and exits landed zero turns just the same, yet ``size != 0`` hid it from this
    card before #1276. The byte floor catches both; each row carries its ``size`` and
    a ``kind`` (``empty``/``stub``) so the render is honest about which it found.

    A ``claude -p`` worker writes nothing to stdout until its final message, so a
    sub-floor log with a *live* pid is still-running (not silent) and is excluded. A
    dead pid over a sub-floor log is the "spun, produced nothing" residual the cooldown
    self-corrects but leaves operator-invisible — this is the signal. Best effort:
    psutil is optional (its absence means we cannot prove a pid dead, so we report
    nothing rather than a false silent), exactly like
    ``issue_resolve_dispatch.live_resolution_issues``. Newest first.
    """
    if not runs_dir.is_dir():
        return []
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore

            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None  # cannot prove liveness -> report no silents (no false alarms)
    out: list[dict[str, Any]] = []
    for log in runs_dir.glob("resolve-*.log"):
        m = _RESOLVE_LOG_RE.search(log.name)
        if not m:
            continue
        try:
            size = log.stat().st_size
        except OSError:
            continue
        if size > _STUB_LOG_MAX_BYTES:
            continue  # over the real-turn floor -> produced output, not silent
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if alive is None and probe is None:
            continue  # no liveness oracle this run -> do not claim it is silent
        if dispatch_preflight.resolve_sidecar_pid_is_live(
            pid_file, alive=alive, probe=probe):
            continue  # still running -> not (yet) silent
        out.append({"issue": int(m.group(1)), "stamp": m.group(2), "log": log.name,
                    "pid": pid, "size": size, "kind": "empty" if size == 0 else "stub"})
    out.sort(key=lambda r: r["stamp"], reverse=True)
    return out


def watchdog_installed() -> dict[str, Any]:
    """Is the always-on watchdog scheduled task registered, and is it enabled?"""
    try:
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", WATCHDOG_TASK, "/FO", "LIST"],
            capture_output=True, text=True, timeout=15,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"installed": None, "error": str(exc)}
    if proc.returncode != 0:
        return {"installed": False, "status": None}
    status = None
    for line in proc.stdout.splitlines():
        if line.lower().strip().startswith("status:"):
            status = line.split(":", 1)[1].strip()
    return {"installed": True, "status": status}


def _int(v: Any, d: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def read_active_weekly_cap(runs_dir: Path, account_tag: str | None,
                           now_ts: float | None = None) -> dict[str, Any] | None:
    """The active weekly-cap hold for ``account_tag`` (if any), read from the
    dispatcher's persisted state (``account-cap-*.json``, written by the
    issue_resolve_dispatch weekly-cap gate). None when no unexpired hold matches.
    Read-only / best-effort, so the card can show WHY a logged-in account is held."""
    import datetime as _dt
    import time as _time
    try:
        now_ts = _time.time() if now_ts is None else now_ts
        now = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=now_ts)
    except Exception:
        return None
    if not runs_dir.is_dir():
        return None
    best: tuple[dict[str, Any], _dt.datetime] | None = None
    for path in runs_dir.glob("account-cap-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
            until = _dt.datetime.fromisoformat((st.get("until") or "").replace("Z", ""))
        except (OSError, ValueError):
            continue
        if now >= until or (account_tag and st.get("account") not in (None, account_tag)):
            continue
        if best is None or until < best[1]:
            best = (st, until)
    return best[0] if best else None


def read_backend_health(runs_dir: Path) -> list[dict[str, Any]]:
    """The backends currently held DEAD by the dispatcher's backend-health gate, read
    from ``backend-health-*.json`` (written by issue_resolve_dispatch.check_backend_health
    when a backend spins on a banner-only/0-byte streak). Each row carries the product,
    since-when, the lane reallocated to a healthy backend, and the evidence logs — so the
    card shows WHY a backend stopped spawning and where its work went. Read-only /
    best-effort; a corrupt or healthy sidecar is skipped. Newest-dead first."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for path in runs_dir.glob("backend-health-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("state") != "dead":
            continue
        out.append({k: st.get(k) for k in
                    ("product", "since", "abandoned_lane", "evidence_logs",
                     "reprobe_min", "last_reprobe")})
    out.sort(key=lambda r: str(r.get("since") or ""), reverse=True)
    return out


def _recent_backend_products(runs_dir: Path, *, lookback_min: int,
                             now_ts: float, backend_of_log: Any) -> list[str]:
    if not runs_dir.is_dir():
        return []
    horizon = now_ts - lookback_min * 60
    products: set[str] = set()
    for log in runs_dir.glob("resolve-*.log"):
        if not _RESOLVE_LOG_RE.search(log.name):
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        product = str(backend_of_log(log) or "claude")
        if product:
            products.add(product)
    return sorted(products)


def backend_stub_rates(
    runs_dir: Path,
    *,
    lookback_min: int = _BACKEND_STUB_LOOKBACK_MIN,
    now_ts: float | None = None,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Recent per-backend productive-vs-stub rollup from worker log content.

    This is intentionally independent of ``backend-health-*.json`` sidecars: a
    backend that stopped reaching its own dispatch tick may never persist a DEAD
    hold, but its recent ``resolve-*.log`` files can still prove that it is mostly
    spawning banner-only/0-byte no-ops. The per-log classification is delegated to
    ``issue_resolve_dispatch._classify_backend_logs`` so this status card shares
    the dispatcher's real-turn floor and quota-banner exception.
    """
    if not runs_dir.is_dir():
        return []
    import time
    try:
        import issue_resolve_dispatch as ird  # type: ignore
    except ImportError:
        return []
    now_ts = time.time() if now_ts is None else now_ts
    products = _recent_backend_products(
        runs_dir, lookback_min=lookback_min, now_ts=now_ts,
        backend_of_log=ird._backend_of_log)
    out: list[dict[str, Any]] = []
    for product in products:
        rows = ird._classify_backend_logs(
            runs_dir, product=product, lookback_min=lookback_min,
            now_ts=now_ts, alive=alive, probe=probe)
        if not rows:
            continue
        productive = sum(1 for r in rows if r.get("productive"))
        stub = len(rows) - productive
        evidence = [str(r.get("log")) for r in rows
                    if not r.get("productive") and r.get("log")][:5]
        out.append({
            "product": product,
            "lookback_min": lookback_min,
            "total": len(rows),
            "productive": productive,
            "stub": stub,
            "stub_rate": round(stub / len(rows), 3),
            "majority_stub": stub > productive,
            "evidence_logs": evidence,
        })
    out.sort(key=lambda r: (-float(r.get("stub_rate") or 0),
                            -int(r.get("stub") or 0),
                            str(r.get("product") or "")))
    return out


# A worker logs ``hook: <name> Failed`` when a lifecycle hook (the fak guard layer
# bound via the harness's hook config) fails to execute. ``claude`` binds the
# guard hooks natively; a non-claude backend (codex/opencode) runs its OWN native
# hook config, and when that config can't reach the dos hook CLI at runtime EVERY
# lifecycle hook fails — the worker stays productive while running UNHOOKED by the
# guard layer. Reuse the stub-rate lookback so both backend folds share one window.
_HOOK_FAIL_LOOKBACK_MIN = _BACKEND_STUB_LOOKBACK_MIN


def backend_hook_failures(
    runs_dir: Path,
    *,
    lookback_min: int = _HOOK_FAIL_LOOKBACK_MIN,
    now_ts: float | None = None,
    reader: Any | None = None,
) -> list[dict[str, Any]]:
    """Per-backend hook-failure rollup from recent worker-log content (#1277).

    The fak guard hooks (PreToolUse / PostToolUse / Stop) bind a worker through the
    harness's lifecycle-hook layer. A non-claude backend (codex/opencode) runs its
    OWN native hook config, and when that config can't reach the dos hook CLI at
    runtime every lifecycle hook logs ``hook: <name> Failed`` — the worker stays
    productive while running UNHOOKED by the guard layer. This fold counts those
    lines per backend over the recent ``resolve-*.log`` sessions and flags any
    backend whose EVERY recent session failed its hooks, so a fully-unhooked backend
    is no longer silent in the status card — the explicit "at minimum, surface the
    hook-failure rate" ask of #1277.

    Reuses ``dispatch_log_audit``'s hook detector + backend sidecar reader (one
    source of truth for the ``hook: … Failed`` signature). Best-effort: if that
    module can't import, the fold is empty (no false signal). Worst (unhooked, most
    failures) first.
    """
    if not runs_dir.is_dir():
        return []
    import time
    try:
        import dispatch_log_audit as dla  # type: ignore
    except ImportError:
        return []
    now_ts = time.time() if now_ts is None else now_ts
    horizon = now_ts - lookback_min * 60
    read = reader or dla._read_text
    by_backend: dict[str, dict[str, Any]] = {}
    for log in sorted(runs_dir.glob("resolve-*.log")):
        if not _RESOLVE_LOG_RE.search(log.name):
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        text = read(log)
        if text is None:
            continue
        backend = str(dla.backend_of_log(log) or "claude")
        row = by_backend.setdefault(backend, {
            "product": backend, "lookback_min": lookback_min, "sessions": 0,
            "sessions_with_hook_failures": 0, "hook_failures": 0, "evidence_logs": []})
        row["sessions"] += 1
        count = sum(int(f["count"]) for f in dla._match_hook_failures(text))
        if count > 0:
            row["sessions_with_hook_failures"] += 1
            row["hook_failures"] += count
            if len(row["evidence_logs"]) < 5:
                row["evidence_logs"].append(log.name)
    out: list[dict[str, Any]] = []
    for row in by_backend.values():
        s = int(row["sessions"])
        failed_sessions = int(row["sessions_with_hook_failures"])
        row["failure_session_rate"] = round(failed_sessions / s, 3) if s else 0.0
        # "fully unhooked" = every recent session of this backend failed its hooks,
        # i.e. the guard hook layer never bound on this backend over the window.
        row["all_sessions_unhooked"] = bool(
            s > 0 and failed_sessions == s and row["hook_failures"] > 0)
        out.append(row)
    out.sort(key=lambda r: (not r["all_sessions_unhooked"],
                            -int(r["hook_failures"]), str(r["product"])))
    return out


def guard_coverage(
    runs_dir: Path,
    *,
    lookback_min: int = _GUARD_RECENT_LOOKBACK_MIN,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Roll up the per-session ``fak guard`` decision journals on the dispatch path.

    The concurrent-dispatch fleet fronts every worker with ``fak guard`` by default
    (dispatch_worker.py and cmd/dispatchworker), and each guarded session owns a
    unique hash-chained journal under ``.dispatch-runs/guard-audit/*.jsonl`` whose
    every non-empty line is one kernel decision (``internal/journal.Row``). This fold
    is the WITNESS that the dispatch path actually ran THROUGH the kernel — and what
    the kernel decided (allow / deny / quarantine) — rather than a flag claiming it
    did. It does NOT invent a coverage percent it cannot ground: it reports the
    witnessed session + decision counts, which a self-report cannot fake.

    Read-only / best-effort. Returns a payload with:

      * ``sessions`` — guarded worker sessions on record (= journal files)
      * ``recent_sessions`` — journals touched within ``lookback_min``
      * ``empty_sessions`` — journals with 0 decision rows (booted under guard but
        proposed no adjudicated tool call — the silent empty-turn signature)
      * ``rows`` / ``recent_rows`` — total / recent kernel decisions
      * ``by_kind`` — the decision mix (DECIDE/DENY/RESULT_DENY/QUARANTINE/VDSO_HIT/…)
      * ``denied`` / ``quarantined`` — derived refusal counts
      * ``evidence`` — the most-recent journal filenames
    """
    import time

    audit_dir = runs_dir / GUARD_AUDIT_DIRNAME
    payload: dict[str, Any] = {
        "dir_present": audit_dir.is_dir(),
        "sessions": 0,
        "recent_sessions": 0,
        "empty_sessions": 0,
        "rows": 0,
        "recent_rows": 0,
        "by_kind": {},
        "denied": 0,
        "quarantined": 0,
        "lookback_min": lookback_min,
        "evidence": [],
    }
    if not audit_dir.is_dir():
        return payload

    now_ts = time.time() if now_ts is None else now_ts
    horizon = now_ts - lookback_min * 60
    by_kind: dict[str, int] = {}
    files: list[tuple[float, str, int]] = []  # (mtime, name, rows) for evidence/recency
    for jp in audit_dir.glob("*.jsonl"):
        try:
            mtime = jp.stat().st_mtime
            text = jp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                kind = str(json.loads(line).get("kind") or "UNKNOWN")
            except ValueError:
                kind = "MALFORMED"
            by_kind[kind] = by_kind.get(kind, 0) + 1
        payload["sessions"] += 1
        payload["rows"] += rows
        if rows == 0:
            payload["empty_sessions"] += 1
        if mtime >= horizon:
            payload["recent_sessions"] += 1
            payload["recent_rows"] += rows
        files.append((mtime, jp.name, rows))

    payload["by_kind"] = dict(sorted(by_kind.items()))
    payload["denied"] = sum(by_kind.get(k, 0) for k in _GUARD_DENY_KINDS)
    payload["quarantined"] = by_kind.get(_GUARD_QUARANTINE_KIND, 0)
    files.sort(key=lambda r: r[0], reverse=True)
    payload["evidence"] = [name for _, name, _ in files[:5]]
    return payload


def _total_commits(root: Path) -> int | None:
    """Commits reachable from HEAD, or None if git can't answer. Used to size the
    closure-audit window to the repo so it never silently scans a stale slice."""
    try:
        proc = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(root),
                              capture_output=True, text=True, timeout=15,
                              creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or "").strip()
    return int(out) if proc.returncode == 0 and out.isdigit() else None


def read_seat_inventory(root: Path, *, product: str | None = None) -> dict[str, Any]:
    """The operator-facing seat inventory (#1799): for every known account/seat, its
    dispatch_state (available/busy/cooling/unavailable) and, when not simply
    available, a specific hold_reason (e.g. ``auth_failed``, ``rate_limited``,
    ``cooldown_until=<ts>``, ``no_capacity``) -- never a bare "unavailable".

    Delegates entirely to ``fleet_accounts.seat_pool()`` (the existing explicit
    multi-seat pool) so this reuses the SAME hold-reason vocabulary the roster and
    the dispatch-preflight spawn gate already use, rather than inventing new state
    names. Pure-local (no gh, no subprocess) and best-effort: an import/read failure
    degrades to ``{"_error": ...}`` like the other fast, always-run folds on this
    card, never raises."""
    try:
        sys.path.insert(0, str(root / "tools"))
        import fleet_accounts  # noqa: PLC0415  (lazy: heavy module, only paid for here)
        rows = fleet_accounts.annotate_accounts(fleet_accounts.discover_accounts())
        leases = fleet_accounts.live_seat_leases(root / RUNS_DIRNAME)
        return fleet_accounts.seat_pool(rows, leases, product=product)
    except Exception as exc:  # best-effort card fold; never fail the whole card
        return {"_error": str(exc)}


def _seat_label(seat: dict[str, Any]) -> str:
    return str(seat.get("tag") or seat.get("account") or seat.get("seat") or "?")


def _limited_seat_labels(labels: list[str], *, limit: int = 4) -> str:
    kept = labels[:limit]
    if len(labels) > limit:
        kept.append(f"+{len(labels) - limit} more")
    return ", ".join(kept)


def _auth_failed_seat_action(seat_inventory: dict[str, Any]) -> str:
    tags = [
        _seat_label(s)
        for s in seat_inventory.get("seats", [])
        if str(s.get("hold_reason") or "") == "auth_failed"
    ]
    if not tags:
        return ""
    return (
        f"auth_failed={len(tags)} [{_limited_seat_labels(tags)}]; "
        "next action: run `fak accounts status` and re-login or remove the named seat(s)"
    )


def _seat_inventory_summary_line(seat_inventory: dict[str, Any]) -> str:
    if not seat_inventory.get("schema"):
        return ""
    by_state = seat_inventory.get("by_dispatch_state") or {}
    line = (
        f"seat inventory: {seat_inventory.get('total_seats', 0)} seat(s) - "
        f"available={by_state.get('available', 0)} busy={by_state.get('busy', 0)} "
        f"cooling={by_state.get('cooling', 0)} unavailable={by_state.get('unavailable', 0)}"
    )
    auth_action = _auth_failed_seat_action(seat_inventory)
    if auth_action:
        line += f"; {auth_action}"
    return line


def _github_rate_limit_error(*docs: dict[str, Any]) -> str:
    for doc in docs:
        err = str((doc or {}).get("_error") or "")
        if "rate limit" in err.lower() or "secondary rate" in err.lower():
            return err
    return ""


def _dispatch_limiter(pre: dict[str, Any], backlog: dict[str, Any],
                      closure: dict[str, Any], leases: dict[str, Any]) -> dict[str, Any]:
    base = dict(pre.get("capacity_limiter") or {})
    raw = dict(base.get("raw") or {})
    raw.setdefault("cap", pre.get("cap"))
    raw.setdefault("live", pre.get("live"))
    raw.setdefault("headroom", (pre.get("cap") - pre.get("live"))
                   if isinstance(pre.get("cap"), int) and isinstance(pre.get("live"), int)
                   else None)
    raw.setdefault("max_workers", pre.get("max_workers"))
    raw.setdefault("host_cap", pre.get("host_cap"))
    seat = pre.get("seat") or {}
    raw.setdefault("seat_total", seat.get("total"))
    raw.setdefault("seat_free", seat.get("free"))
    raw.setdefault("seat_leased", seat.get("leased"))
    raw["lane_leases_active"] = leases.get("active_count")
    raw["lane_leases_blocking"] = leases.get("blocking_count")

    gh_err = _github_rate_limit_error(backlog, closure)
    if gh_err:
        raw["github_error"] = gh_err
        return {"primary": "github_rate_limit", "term": "github_error", "raw": raw}

    blocking = _int(leases.get("blocking_count"), 0) or 0
    if blocking:
        return {"primary": "leases", "term": "lane_leases_blocking", "raw": raw}

    if base:
        base["raw"] = raw
        return base
    return {"primary": "unknown", "term": "unknown", "raw": raw}


def _dispatch_limiter_terms(limiter: dict[str, Any]) -> str:
    raw = limiter.get("raw") or {}
    parts = [
        f"cap={raw.get('cap')}",
        f"live={raw.get('live')}",
        f"headroom={raw.get('headroom')}",
        f"max={raw.get('max_workers')}",
        f"target={raw.get('dos_target')}",
        f"host_cap={raw.get('host_cap')}",
        f"host_binding={raw.get('host_binding')}",
        f"seats={raw.get('seat_total')}",
        f"free={raw.get('seat_free')}",
        f"leased={raw.get('seat_leased')}",
        f"lane_leases={raw.get('lane_leases_blocking')}/{raw.get('lane_leases_active')}",
    ]
    if raw.get("github_error"):
        parts.append("github_error=rate_limit")
    return " ".join(parts)


def collect(root: Path, *, max_workers: int, fast: bool,
            closure_commits: int) -> dict[str, Any]:
    pre = run_json([_py(), str(root / "tools" / "dispatch_preflight.py"),
                    "--json", "--max-workers", str(max_workers)], root, timeout=120)
    sup = run_json([_py(), str(root / "tools" / "dos_supervisor_status.py"),
                    "--json"], root, timeout=90)
    wd = watchdog_installed()

    backlog: dict[str, Any] = {"_skipped": "fast"} if fast else run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"], root,
        timeout=130)
    # Cover the WHOLE repo, not a slice. A closure audit whose --max-commits is
    # narrower than the repo's history can't bind a resolving commit older than the
    # window, so a long-since-shipped issue mis-buckets CLAIMED_CLOSED and the
    # closure_rate reads catastrophically low (the 0.20-vs-0.79 artifact). The
    # auditor caches every SHA verdict permanently, so a full-history window is
    # cheap on warm runs; we size it to the repo + headroom (never below the
    # operator's floor) and lift the issue limit above the real backlog so the
    # oldest -- disproportionately closed -- issues all load.
    total_commits = _total_commits(root)
    commit_window = max(closure_commits,
                        (total_commits + 200) if total_commits else closure_commits)
    closure: dict[str, Any] = {"_skipped": "fast"} if fast else run_json(
        [_py(), str(root / "tools" / "issue_closure_audit.py"), "--json",
         "--max-commits", str(commit_window), "--issue-limit", "4000"],
        root, timeout=300)
    # The RATE fold (closed/hour vs target) — the observable the loop's goal is
    # actually stated in. gh-backed, so it degrades to n/a under --fast/timeout
    # exactly like backlog/closure; it never flips the dispatcher-health verdict
    # (a below-target rate is information, not a broken dispatcher).
    throughput: dict[str, Any] = {"_skipped": "fast"} if fast else run_json(
        [_py(), str(root / "tools" / "dispatch_throughput.py"), "--json"],
        root, timeout=140, ok_codes=set(range(0, 16)))

    # Pure-local, always run (no gh/dos): which spawned workers exited producing
    # nothing. Cheap enough that --fast keeps it.
    silent = silent_workers(root / RUNS_DIRNAME)
    weekly_cap = read_active_weekly_cap(root / RUNS_DIRNAME,
                                        (pre.get("account") or {}).get("tag"))
    backend_health = read_backend_health(root / RUNS_DIRNAME)
    backend_stub_rate = backend_stub_rates(root / RUNS_DIRNAME)
    hook_failures = backend_hook_failures(root / RUNS_DIRNAME)
    guard = guard_coverage(root / RUNS_DIRNAME)
    run_status = read_run_status_digests(root)
    merge = merge_state(root)
    leases = read_lease_state(root, backlog)
    worker_leases = worker_lease_crosscheck(root / RUNS_DIRNAME, leases)
    seat_inventory = read_seat_inventory(root)

    return build_payload(root=root, pre=pre, sup=sup, wd=wd, backlog=backlog,
                         closure=closure, max_workers=max_workers, fast=fast,
                         silent=silent, weekly_cap=weekly_cap, throughput=throughput,
                         backend_health=backend_health,
                         backend_stub_rate=backend_stub_rate,
                         hook_failures=hook_failures, guard=guard,
                         run_status=run_status, merge=merge, leases=leases,
                         worker_leases=worker_leases, seat_inventory=seat_inventory)


def build_payload(*, root: Path, pre: dict, sup: dict, wd: dict, backlog: dict,
                  closure: dict, max_workers: int, fast: bool,
                  silent: list[dict[str, Any]] | None = None,
                  weekly_cap: dict[str, Any] | None = None,
                  throughput: dict[str, Any] | None = None,
                  backend_health: list[dict[str, Any]] | None = None,
                  backend_stub_rate: list[dict[str, Any]] | None = None,
                  hook_failures: list[dict[str, Any]] | None = None,
                  guard: dict[str, Any] | None = None,
                  run_status: list[dict[str, Any]] | None = None,
                  merge: dict[str, Any] | None = None,
                  leases: dict[str, Any] | None = None,
                  worker_leases: dict[str, Any] | None = None,
                  seat_inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    # --- dispatcher liveness / capacity ---
    cap = _int(pre.get("cap"))
    live = _int(pre.get("live"))
    host_safe = bool((pre.get("host") or {}).get("safe"))
    acct = pre.get("account") or {}
    pre_verdict = pre.get("verdict")

    # --- backlog --- (counts is the router's authoritative routed/unrouted fold)
    lanes = (backlog.get("lanes") or {}) if isinstance(backlog, dict) else {}
    bcounts = (backlog.get("counts") or {}) if isinstance(backlog, dict) else {}
    lane_counts: dict[str, int] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        lane_counts[ln] = len(iss) if hasattr(iss, "__len__") else _int(iss, 0) or 0
    open_issues = _int(bcounts.get("open"), sum(lane_counts.values()))
    routed = _int(bcounts.get("routed"))
    unrouted = _int(bcounts.get("unrouted"))
    backlog_na = "_skipped" in backlog or ("_error" in backlog and not lanes)

    # --- closure honesty ---
    counts = closure.get("counts") or {}
    closure_rate = closure.get("closure_rate")
    honest_close_rate = closure.get("honest_close_rate")
    closure_na = "_skipped" in closure or ("_error" in closure and closure_rate is None)
    open_witnessed = _int(counts.get("OPEN_WITNESSED"), 0)

    # --- throughput (closed/hour vs target) ---
    throughput = throughput or {}
    tp_na = "_skipped" in throughput or "_error" in throughput or not throughput.get("schema")

    # --- overall verdict ---
    # Healthy = host clean AND (can grow OR already at a healthy target). A flagged
    # host or an un-runnable safety check is the only thing that fails the card —
    # "no account free" / "at cap" are normal steady states, not breakage.
    reasons: list[str] = []
    if not host_safe:
        ok = False
        verdict = "HOST_FLAGGED"
        reasons.append("host resource guard flagged a process — reap/inspect before growing")
    elif pre_verdict == "REFUSE_INSPECT":
        ok = False
        verdict = "INSPECT"
        reasons.append(f"a safety preflight could not run: {pre.get('reason')}")
    elif pre_verdict == "REFUSE_NO_ACCOUNT":
        ok = True
        verdict = "BLOCKED_ON_ACCOUNT"
        reasons.append("no worker account free right now (switcher will resume when one frees)")
    elif pre_verdict == "REFUSE_AT_CAP":
        ok = True
        verdict = "AT_CAP"
        reasons.append(f"{live}/{cap} workers live — at the configured ceiling")
    else:
        ok = True
        verdict = "READY_TO_GROW"
        reasons.append(f"safe to spawn: {live}/{cap} live, account '{acct.get('tag')}' free")

    # A logged-in-but-quota-capped account makes the preflight read SPAWN_OK while
    # the dispatcher's weekly-cap gate is actually HOLDING. Surface that so the card
    # says WEEKLY_CAPPED, not the misleading READY_TO_GROW. Holding is a healthy
    # steady state (the t2 glm pool is unaffected), so ok stays True.
    if weekly_cap and verdict == "READY_TO_GROW":
        verdict = "WEEKLY_CAPPED"
        reasons = [f"account '{acct.get('tag')}' weekly-capped — resets "
                   f"{weekly_cap.get('reset_text') or '?'} (holding spawn until "
                   f"{weekly_cap.get('until')}); the t2 glm/docs pool is unaffected"]

    merge = merge or {}
    if merge.get("merge_in_progress"):
        ok = False
        verdict = "MERGE_IN_PROGRESS"
        reasons.insert(0, merge.get("next_action") or
                       "wait for MERGE_HEAD to clear before starting worker edits")

    if wd.get("installed") is False:
        reasons.append("always-on watchdog NOT installed (register_dos_dispatch_watchdog.ps1)")
    elif wd.get("installed"):
        reasons.append(f"always-on watchdog installed ({wd.get('status') or 'scheduled'})")

    silent = silent or []
    if silent:
        nums = ", ".join(f"#{w['issue']}" for w in silent[:6])
        reasons.append(f"{len(silent)} worker(s) exited producing nothing ({nums}) — inspect or re-scope")
    backend_health = backend_health or []
    if backend_health:
        names = ", ".join(f"{b.get('product')}->{b.get('abandoned_lane') or '?'}"
                          for b in backend_health[:4])
        reasons.append(f"{len(backend_health)} backend(s) held dead, lane reallocated "
                       f"({names}) — a healthy backend is covering; auto-restores on recovery")
    backend_stub_rate = backend_stub_rate or []
    majority_stub = [r for r in backend_stub_rate if r.get("majority_stub")]
    if majority_stub:
        names = ", ".join(f"{r.get('product')} {r.get('stub')}/{r.get('total')} stub"
                          for r in majority_stub[:4])
        reasons.append(f"backend stub-rate majority-stub over recent logs ({names}) — inspect backend output")

    # Hook-layer binding: a backend whose every recent session logs hook failures is
    # running UNHOOKED by the guard layer (productive but unguarded by the hook
    # backstop). Information, not breakage — like the stub-rate signal it adds a
    # reason but never flips ok (the commit-path / OFF_TRUNK guard is the backstop).
    hook_failures = hook_failures or []
    unhooked = [r for r in hook_failures if r.get("all_sessions_unhooked")]
    if unhooked:
        names = ", ".join(f"{r.get('product')} {r.get('hook_failures')} fail/"
                          f"{r.get('sessions')} sess "
                          f"({int(float(r.get('failure_session_rate') or 0) * 100)}%)"
                          for r in unhooked[:4])
        reasons.append(
            f"guard hook layer UNBOUND on {len(unhooked)} backend(s) ({names}) — "
            f"productive but running unhooked; the commit-path/OFF_TRUNK guard is the "
            f"backstop (#1277)")

    # Guard coverage: the witnessed proof the dispatch path ran THROUGH `fak guard`
    # (per-session decision journals), and the kernel's decision mix. Informational —
    # it adds a reason but never flips ok. A present-but-empty trail is its own signal
    # (workers booted under guard but proposed no adjudicated tool call).
    guard = guard or {}
    g_sessions = _int(guard.get("sessions"), 0) or 0
    g_rows = _int(guard.get("rows"), 0) or 0
    if g_sessions and g_rows:
        reasons.append(
            f"fak guard witnessed {g_rows} kernel decision(s) across {g_sessions} "
            f"dispatch session(s) ({guard.get('denied', 0)} denied, "
            f"{guard.get('quarantined', 0)} quarantined)")
    elif g_sessions:
        reasons.append(
            f"fak guard ran {g_sessions} dispatch session(s) but recorded 0 decisions "
            f"({guard.get('empty_sessions', 0)} empty) — workers booted under guard "
            f"but proposed no adjudicated tool call")

    if not tp_na:
        tp_verdict = throughput.get("verdict")
        tp_rate = throughput.get("completed_rate_per_hour")
        tp_target = throughput.get("target_per_hour")
        win = throughput.get("primary_window_hours")
        if tp_verdict in ("BELOW_TARGET", "AUDIT_ERROR"):
            reasons.append(f"throughput {tp_rate}/h completed over the {win}h analysis window — below the "
                           f"{tp_target}/h target")
        else:
            reasons.append(f"throughput {tp_verdict} ({tp_rate}/h completed over the {win}h analysis window, "
                           f"target {tp_target}/h)")

    run_status = run_status or []
    status_counts: dict[str, int] = {}
    status_errors = 0
    for digest in run_status:
        if digest.get("_error"):
            status_errors += 1
            continue
        verdict = str(((digest.get("liveness") or {}).get("verdict")) or "UNKNOWN")
        status_counts[verdict] = status_counts.get(verdict, 0) + 1
    if run_status:
        if status_errors:
            reasons.append(f"dos status digest read had {status_errors} error(s); inspect run_status")
        else:
            reasons.append(f"run truth from dos status digest for {len(run_status)} RID(s)")

    leases = leases or {}
    if leases.get("read_error"):
        reasons.append(f"lease read unavailable: {leases.get('read_error')}")
    elif leases.get("active_count"):
        blocking = _int(leases.get("blocking_count"), 0) or 0
        if leases.get("candidate_source_available") is False:
            reasons.append(
                f"{leases.get('active_count')} active lane lease(s); candidate blocking unknown "
                "(backlog fold unavailable)")
        elif blocking:
            blocked_nums: list[str] = []
            for row in leases.get("active") or []:
                if not row.get("blocks_candidate"):
                    continue
                for cand in row.get("blocking_candidates") or []:
                    issue = cand.get("issue")
                    if issue is not None:
                        blocked_nums.append(f"#{issue}")
            suffix = f" ({', '.join(blocked_nums[:6])})" if blocked_nums else ""
            reasons.append(
                f"{blocking}/{leases.get('active_count')} active lane lease(s) block "
                f"current candidate issue(s){suffix}")
        else:
            reasons.append(f"{leases.get('active_count')} active lane lease(s), none blocking current candidates")
    worker_leases = worker_leases or {}
    if worker_leases.get("available") is False:
        reasons.append(f"worker/lease cross-check unavailable: {worker_leases.get('error')}")
    elif worker_leases:
        op = _int(worker_leases.get("orphan_process_count"), 0) or 0
        ol = _int(worker_leases.get("orphan_lease_count"), 0) or 0
        clean = _int(worker_leases.get("clean_count"), 0) or 0
        if op or ol:
            reasons.append(
                f"worker/lease cross-check: clean={clean}, orphan-process={op}, orphan-lease={ol}")
        elif clean:
            reasons.append(f"worker/lease cross-check clean ({clean} matched worker/lease pair(s))")

    # Seat inventory (#1799): available/busy/cooling/unavailable counts across the
    # explicit account seat pool, so an operator sees WHY a seat is held without
    # digging through fleet_accounts directly. Informational only — never flips ok.
    seat_inventory = seat_inventory or {}
    if seat_inventory.get("_error"):
        reasons.append(f"seat inventory unavailable: {seat_inventory.get('_error')}")
    elif seat_inventory.get("schema"):
        reasons.append(_seat_inventory_summary_line(seat_inventory))

    limiter = _dispatch_limiter(pre, backlog, closure, leases)

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reasons": reasons,
        "workspace": str(root),
        "weekly_cap": weekly_cap,
        "dispatcher": {
            "cap": cap,
            "live": live,
            "headroom": (cap - live) if (cap is not None and live is not None) else None,
            "host_safe": host_safe,
            "preflight_verdict": pre_verdict,
            "limiter": limiter,
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "available")},
            "watchdog": wd,
        },
        "supervisor": {
            "verdict": sup.get("verdict"),
            "target": (sup.get("supervise") or {}).get("target"),
            "alive": (sup.get("supervise") or {}).get("alive"),
            "plans": sup.get("plans"),
        },
        "backlog": {
            "na": backlog_na,
            "open_issues": None if backlog_na else open_issues,
            "routed": None if backlog_na else routed,
            "by_lane": None if backlog_na else lane_counts,
            "unrouted": None if backlog_na else unrouted,
        },
        "closure": {
            "na": closure_na,
            "closure_rate": closure_rate,
            "honest_close_rate": honest_close_rate,
            "counts": counts or None,
            "open_witnessed_closable": None if closure_na else open_witnessed,
        },
        "throughput": {
            "na": tp_na,
            "verdict": None if tp_na else throughput.get("verdict"),
            "target_per_hour": None if tp_na else throughput.get("target_per_hour"),
            "primary_window_hours": None if tp_na else throughput.get("primary_window_hours"),
            "completed_rate_per_hour": None if tp_na else throughput.get("completed_rate_per_hour"),
            "raw_rate_per_hour": None if tp_na else throughput.get("raw_rate_per_hour"),
            "per_window": None if tp_na else (throughput.get("gh") or {}).get("per_window"),
            "loop_per_window": None if tp_na else (throughput.get("loop") or {}).get("per_window"),
            "last_loop_close_age_min": None if tp_na else (throughput.get("loop") or {}).get("last_loop_close_age_min"),
        },
        "workers": {
            "silent_count": len(silent),
            "silent": silent,
        },
        "backend_health": {
            "dead_count": len(backend_health),
            "dead": backend_health,
            "stub_rate": backend_stub_rate,
        },
        "hook_health": {
            "unhooked_count": len(unhooked),
            "by_backend": hook_failures,
        },
        "guard": guard,
        "run_status": {
            "source": "dos status",
            "count": len(run_status),
            "liveness": status_counts,
            "errors": status_errors,
            "digests": run_status,
        },
        "leases": leases,
        "worker_lease_check": worker_leases,
        "seat_inventory": seat_inventory or {},
        "git": {
            "merge_in_progress": bool(merge.get("merge_in_progress")),
            "merge_head": merge.get("merge_head"),
            "next_action": merge.get("next_action"),
        },
        "fast": fast,
    }


def _age_text(minutes: Any) -> str:
    if not isinstance(minutes, (int, float)):
        return "?"
    if minutes < 1:
        return "<1m"
    if minutes >= 60:
        hours = f"{minutes / 60:.1f}".rstrip("0").rstrip(".")
        return hours + "h"
    return f"{minutes:.1f}".rstrip("0").rstrip(".") + "m"


def _lease_block_text(row: dict[str, Any]) -> str:
    blocking = row.get("blocks_candidate")
    if blocking is None:
        return "candidate unknown"
    if not blocking:
        return "no candidate"
    nums = [f"#{c.get('issue')}" for c in (row.get("blocking_candidates") or [])
            if c.get("issue") is not None]
    return "blocks " + (",".join(nums[:4]) if nums else "candidate")


def _lease_summary_bits(leases: dict[str, Any], *, limit: int = 3) -> list[str]:
    rows = leases.get("active") or []
    bits: list[str] = []
    for row in rows[:limit]:
        bits.append(
            f"{row.get('id')} lane={row.get('lane') or '-'} "
            f"age={_age_text(row.get('age_min'))} {_lease_block_text(row)}")
    if len(rows) > limit:
        bits.append(f"+{len(rows) - limit} more")
    return bits


def _worker_lease_bucket_bits(rows: list[dict[str, Any]], *, key: str,
                              limit: int = 3) -> list[str]:
    bits: list[str] = []
    for row in rows[:limit]:
        obj = row.get(key) or {}
        bits.append(str(obj.get("worker") or obj.get("id") or obj.get("lease_id") or "?"))
    if len(rows) > limit:
        bits.append(f"+{len(rows) - limit} more")
    return bits


def render(p: dict[str, Any]) -> str:
    d = p.get("dispatcher") or {}
    s = p.get("supervisor") or {}
    b = p.get("backlog") or {}
    c = p.get("closure") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    lines = [
        f"╔═ DISPATCHER: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})",
        f"║ workers   : {d.get('live')}/{d.get('cap')} live (headroom {d.get('headroom')})  "
        f"host={'clean' if d.get('host_safe') else 'FLAGGED'}",
        f"║ limiter   : {(d.get('limiter') or {}).get('primary') or '-'} "
        f"({_dispatch_limiter_terms(d.get('limiter') or {})})",
        f"║ switcher  : account={a.get('tag') or '-'} (t{a.get('tier')}) "
        f"avail={a.get('available')}  preflight={d.get('preflight_verdict')}",
        "║ always-on : watchdog "
        + ("installed (" + str(wd.get('status') or 'scheduled') + ")"
           if wd.get("installed") else
           ("NOT installed" if wd.get("installed") is False else "unknown")),
        f"║ supervisor: {s.get('verdict')} alive={s.get('alive')}/{s.get('target')}  "
        f"plans={(s.get('plans') or {}).get('total_plans')} "
        f"units={(s.get('plans') or {}).get('total_units')}",
    ]
    seat_line = _seat_inventory_summary_line(p.get("seat_inventory") or {})
    if seat_line:
        lines.append("║ seats     : " + seat_line.removeprefix("seat inventory: "))
    if b.get("na"):
        lines.append("║ backlog   : n/a (--fast or gh timeout)")
    else:
        by = b.get("by_lane") or {}
        top = ", ".join(f"{k}={v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"║ backlog   : {b.get('open_issues')} open  [{top}]  unrouted={b.get('unrouted')}")
    if c.get("na"):
        lines.append("║ closure   : n/a (--fast or gh timeout)")
    else:
        cnt = c.get("counts") or {}
        lines.append(
            f"║ closure   : rate={c.get('closure_rate')} honest={c.get('honest_close_rate')}  "
            f"resolved={cnt.get('TRUE_RESOLVED')} data={cnt.get('DATA_RESOLVED')} "
            f"claimed={cnt.get('CLAIMED_CLOSED')} "
            f"closable-now={c.get('open_witnessed_closable')} (OPEN_WITNESSED)")
    tp = p.get("throughput") or {}
    if tp.get("na"):
        lines.append("║ rate      : n/a (--fast or gh timeout)")
    else:
        lines.append(
            f"║ rate      : {tp.get('verdict')}  {tp.get('completed_rate_per_hour')}/h completed "
            f"over the {tp.get('primary_window_hours')}h analysis window (target {tp.get('target_per_hour')}/h)")
    w = p.get("workers") or {}
    sc = w.get("silent_count") or 0
    if sc:
        nums = ", ".join(f"#{s['issue']}" for s in (w.get("silent") or [])[:6])
        lines.append(f"║ workers   : {sc} silent (<= {_STUB_LOG_MAX_BYTES} B log, exited) [{nums}]")
    bh = p.get("backend_health") or {}
    flagged_rates = [r for r in (bh.get("stub_rate") or []) if r.get("majority_stub")]
    if flagged_rates:
        bits = ", ".join(f"{r.get('product')}={r.get('stub')}/{r.get('total')} stub"
                         for r in flagged_rates[:4])
        lines.append(f"║ backend   : majority-stub recent logs [{bits}]")
    hh = p.get("hook_health") or {}
    unhooked_rows = [r for r in (hh.get("by_backend") or []) if r.get("all_sessions_unhooked")]
    if unhooked_rows:
        bits = ", ".join(f"{r.get('product')}={r.get('hook_failures')} fail/{r.get('sessions')} sess"
                         f" ({int(float(r.get('failure_session_rate') or 0) * 100)}%)"
                         for r in unhooked_rows[:4])
        lines.append(f"║ hooks     : guard layer UNBOUND [{bits}] (#1277)")
    gd = p.get("guard") or {}
    if gd.get("sessions"):
        lines.append(
            f"║ guard     : {gd.get('sessions')} session(s) ({gd.get('recent_sessions', 0)} recent), "
            f"{gd.get('rows', 0)} decision(s) [DENY={gd.get('denied', 0)} "
            f"QUAR={gd.get('quarantined', 0)}]  empty={gd.get('empty_sessions', 0)}")
    rs = p.get("run_status") or {}
    if rs.get("count"):
        bits = ", ".join(f"{k}={v}" for k, v in sorted((rs.get("liveness") or {}).items())) or "none"
        lines.append(f"║ run truth : dos status {rs.get('count')} RID(s), errors={rs.get('errors')} [{bits}]")
    leases = p.get("leases") or {}
    if leases.get("read_error"):
        lines.append(f"║ leases    : unavailable ({leases.get('read_error')})")
    elif leases.get("active_count"):
        bits = "; ".join(_lease_summary_bits(leases))
        lines.append(
            f"║ leases    : {leases.get('active_count')} active, "
            f"{leases.get('blocking_count', 0)} blocking [{bits}]")
    wl = p.get("worker_lease_check") or {}
    if wl.get("available") is False:
        lines.append(f"║ lease chk : unknown ({wl.get('error')})")
    elif wl:
        bits = []
        if wl.get("orphan_process_count"):
            bits.append("orphan-process "
                        + ",".join(_worker_lease_bucket_bits(wl.get("orphan_process") or [], key="worker")))
        if wl.get("orphan_lease_count"):
            bits.append("orphan-lease "
                        + ",".join(_worker_lease_bucket_bits(wl.get("orphan_lease") or [], key="lease")))
        detail = f" [{'; '.join(bits)}]" if bits else ""
        lines.append(
            f"║ lease chk : clean={wl.get('clean_count', 0)} "
            f"orphan-process={wl.get('orphan_process_count', 0)} "
            f"orphan-lease={wl.get('orphan_lease_count', 0)}{detail}")
    git = p.get("git") or {}
    if git.get("merge_in_progress"):
        lines.append(f"║ git       : MERGE_HEAD present — {git.get('next_action')}")
    lines.append("╚═ " + " | ".join(p.get("reasons") or []))
    return "\n".join(lines)


def render_md(payload: dict[str, Any], *, date: str) -> str:
    """The committed, human-readable status surface: which issues are synced to
    which lanes, how closure is progressing, and any worker that produced nothing.

    This is the plan-doc-equivalent for a plan-empty repo whose backlog is GitHub
    issues — an operator opens ONE file instead of grepping gitignored runtime.
    Date is git-derived by the caller (deterministic; the renderer takes no clock).
    """
    d = payload.get("dispatcher") or {}
    s = payload.get("supervisor") or {}
    b = payload.get("backlog") or {}
    c = payload.get("closure") or {}
    w = payload.get("workers") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}

    out = [
        # Jekyll front matter so the published status page keeps a stable <title> +
        # meta description (jekyll-seo-tag reads these). Without it, every --md regen
        # would strip the front matter the committed doc needs and the page would
        # read as discoverability debt to the SEO/AEO scorecard. Title/description are
        # deliberately date-independent so the page's identity is stable across regens.
        "---",
        'title: "fak issue dispatch status: fleet worker and backlog tracker"',
        'description: "Auto-generated fak issue dispatch status: dispatcher and worker '
        'state, open-issue backlog by lane, and closure-honesty rate across the fleet."',
        "---",
        "",
        f"# Issue dispatch status — {date}",
        "",
        "_Auto-generated by `tools/dispatch_status.py --md`. Do not hand-edit; "
        "re-run the tool (or the `FleetDispatchStatusDoc` task) to refresh._",
        "",
        f"- **dispatcher**: `{payload.get('verdict')}` "
        f"({'ok' if payload.get('ok') else 'ACTION'})",
        f"- **workers**: {d.get('live')}/{d.get('cap')} live "
        f"(headroom {d.get('headroom')}); host "
        f"{'clean' if d.get('host_safe') else '**FLAGGED**'}",
        f"- **primary limiter**: `{(d.get('limiter') or {}).get('primary') or '-'}` "
        f"({_dispatch_limiter_terms(d.get('limiter') or {})})",
        f"- **switcher account**: `{a.get('tag') or '-'}` (t{a.get('tier')}, "
        f"{a.get('model') or '?'}), available={a.get('available')}",
        "- **always-on watchdog**: "
        + ("installed (" + str(wd.get('status') or 'scheduled') + ")"
           if wd.get("installed") else
           ("**NOT installed**" if wd.get("installed") is False else "unknown")),
        f"- **supervisor**: `{s.get('verdict')}` "
        f"(alive {s.get('alive')}/{s.get('target')})",
    ]
    seat_line = _seat_inventory_summary_line(payload.get("seat_inventory") or {})
    if seat_line:
        out.append(f"- **seat inventory**: {seat_line.removeprefix('seat inventory: ')}")
    rs = payload.get("run_status") or {}
    if rs.get("count"):
        out.append(f"- **run status source**: `dos status` digests for {rs.get('count')} RID(s), "
                   f"errors={rs.get('errors')}")
    git = payload.get("git") or {}
    if git.get("merge_in_progress"):
        out.append(f"- **git wait state**: `MERGE_HEAD` present — {git.get('next_action')}")
    leases = payload.get("leases") or {}
    if leases.get("read_error"):
        out.append(f"- **lane leases**: unavailable (`{leases.get('read_error')}`)")
    elif leases.get("active_count"):
        out.append(f"- **lane leases**: {leases.get('active_count')} active; "
                   f"{leases.get('blocking_count', 0)} blocking current candidates")
    wl = payload.get("worker_lease_check") or {}
    if wl.get("available") is False:
        out.append(f"- **worker/lease cross-check**: unknown (`{wl.get('error')}`)")
    elif wl:
        out.append(f"- **worker/lease cross-check**: clean={wl.get('clean_count', 0)}, "
                   f"orphan-process={wl.get('orphan_process_count', 0)}, "
                   f"orphan-lease={wl.get('orphan_lease_count', 0)}")
    out += [
        "",
        "## Backlog by lane (issue → lane sync)",
        "",
    ]
    if b.get("na"):
        out.append("_Backlog n/a this run (gh fold skipped or timed out)._")
    else:
        out += [
            f"Open issues: **{b.get('open_issues')}** — routed {b.get('routed')}, "
            f"unrouted {b.get('unrouted')}.",
            "",
            "| lane | open issues |",
            "|---|---|",
        ]
        by_lane = b.get("by_lane") or {}
        for lane, n in sorted(by_lane.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {lane} | {n} |")

    out += ["", "## Closure honesty", ""]
    if c.get("na"):
        out.append("_Closure audit n/a this run (gh/dos fold skipped or timed out)._")
    else:
        cnt = c.get("counts") or {}
        honest = c.get("honest_close_rate")
        out += [
            f"`closure_rate` = **{c.get('closure_rate')}** "
            f"(TRUE_RESOLVED / (TRUE_RESOLVED + CLAIMED_CLOSED) — strict diff-witness)"
            + (f"; `honest_close_rate` = **{honest}** (also credits the DATA rung)."
               if honest is not None else "."),
            "",
            "| bucket | count |",
            "|---|---|",
            f"| TRUE_RESOLVED | {cnt.get('TRUE_RESOLVED', 0)} |",
            f"| DATA_RESOLVED | {cnt.get('DATA_RESOLVED', 0)} |",
            f"| CLAIMED_CLOSED | {cnt.get('CLAIMED_CLOSED', 0)} |",
            f"| OPEN_WITNESSED (closable now) | {c.get('open_witnessed_closable')} |",
        ]

    tp = payload.get("throughput") or {}
    out += ["", "## Throughput (closed issues per hour)", ""]
    if tp.get("na"):
        out.append("_Throughput n/a this run (gh fold skipped or timed out)._")
    else:
        out += [
            f"`verdict` = **{tp.get('verdict')}** — **{tp.get('completed_rate_per_hour')}/h** "
            f"completed over the trailing **{tp.get('primary_window_hours')}h** window "
            f"(target **{tp.get('target_per_hour')}/h**). Graded on the *completed* "
            "(resolved, not wontfix/dup) rate.",
            "",
            "| window | closed | completed | completed /h | loop-closed | loop /h |",
            "|---|---|---|---|---|---|",
        ]
        pw = tp.get("per_window") or {}
        lpw = tp.get("loop_per_window") or {}
        for key in ("1h", "3h", "6h", "12h", "24h"):
            g = pw.get(key)
            if not g:
                continue
            lp = lpw.get(key) or {}
            out.append(
                f"| {key} | {g.get('closed')} | {g.get('completed')} | "
                f"{g.get('completed_rate_per_hour')} | {lp.get('loop_closed', '-')} | "
                f"{lp.get('loop_rate_per_hour', '-')} |")
        last = tp.get("last_loop_close_age_min")
        out += ["",
                "Loop's last attributed close: "
                + (f"{last} min ago." if last is not None else "**none on record**.")
                + " A gh-rate far above the loop-rate means humans/peers are draining "
                "the backlog, not the dispatcher."]

    leases = payload.get("leases") or {}
    out += ["", "## Active lane leases", ""]
    if leases.get("read_error"):
        out.append(f"_Lease read unavailable: `{leases.get('read_error')}`._")
    elif not leases.get("active_count"):
        extra = ""
        if leases.get("expired_count"):
            extra = f" {leases.get('expired_count')} expired lease record(s) are reapable residue."
        out.append("No active lane leases under `refs/fak/locks/*`." + extra)
    else:
        if leases.get("candidate_source_available") is False:
            out.append(
                "_Backlog routing was unavailable, so candidate-blocking status is unknown._")
        elif leases.get("blocking_count"):
            out.append(
                f"**{leases.get('blocking_count')}** active lease(s) overlap current routed "
                "candidate issues; those candidates should wait or be repartitioned.")
        else:
            out.append("Active leases are present, but none overlap the current routed candidates.")
        if leases.get("expired_count"):
            out.append(
                f"{leases.get('expired_count')} expired lease record(s) are visible but non-blocking.")
        out += [
            "",
            "| lease id | lane | age | ttl | blocks candidate | holder | tree |",
            "|---|---|---:|---:|---|---|---|",
        ]
        for row in leases.get("active") or []:
            tree = ", ".join(f"`{t}`" for t in (row.get("tree") or [])) or "—"
            holder = str(row.get("holder") or "—")
            out.append(
                f"| `{row.get('id')}` | {row.get('lane') or '—'} | "
                f"{_age_text(row.get('age_min'))} | {row.get('ttl_seconds')}s | "
                f"{_lease_block_text(row)} | `{holder}` | {tree} |")

    wl = payload.get("worker_lease_check") or {}
    out += ["", "## Worker / lease cross-check", ""]
    if wl.get("available") is False:
        out.append(f"_Worker liveness unavailable: `{wl.get('error')}`._")
    elif not wl:
        out.append("_Worker/lease cross-check did not run._")
    else:
        out.append(
            f"clean={wl.get('clean_count', 0)}, "
            f"orphan-process={wl.get('orphan_process_count', 0)}, "
            f"orphan-lease={wl.get('orphan_lease_count', 0)}.")
        clean_rows = wl.get("clean") or []
        out += ["", "Clean matches:", ""]
        if not clean_rows:
            out.append("_No live worker has a matching active lease._")
        else:
            out += [
                "| worker | issue | pid | lease |",
                "|---|---:|---:|---|",
            ]
            for row in clean_rows:
                worker = row.get("worker") or {}
                lease = row.get("lease") or {}
                out.append(f"| `{worker.get('worker')}` | #{worker.get('issue')} | "
                           f"{worker.get('pid')} | `{lease.get('id')}` |")
        orphan_process = wl.get("orphan_process") or []
        out += ["", "Orphan processes:", ""]
        if not orphan_process:
            out.append("_No OS-visible dispatch worker is missing an active lease._")
        else:
            out += [
                "| worker | issue | pid | lease id | reason |",
                "|---|---:|---:|---|---|",
            ]
            for row in orphan_process:
                worker = row.get("worker") or {}
                out.append(f"| `{worker.get('worker')}` | #{worker.get('issue')} | "
                           f"{worker.get('pid')} | `{worker.get('lease_id') or '—'}` | "
                           f"{row.get('reason')} |")
        orphan_lease = wl.get("orphan_lease") or []
        out += ["", "Orphan leases:", ""]
        if not orphan_lease:
            out.append("_No active lease is missing a local live worker sidecar._")
        else:
            out += [
                "| lease | lane | holder | reason |",
                "|---|---|---|---|",
            ]
            for row in orphan_lease:
                lease = row.get("lease") or {}
                out.append(f"| `{lease.get('id')}` | {lease.get('lane') or '—'} | "
                           f"`{lease.get('holder') or '—'}` | {row.get('reason')} |")

    bh = payload.get("backend_health") or {}
    dead = bh.get("dead") or []
    stub_rates = bh.get("stub_rate") or []
    majority_stub = [r for r in stub_rates if r.get("majority_stub")]
    out += ["", "## Backend health / reallocation", ""]
    if not dead and not majority_stub:
        out.append("All backends healthy — none held dead, no lane reallocated.")
    elif not dead:
        out.append(
            "No `backend-health-*.json` sidecar is holding a backend dead, but the "
            "recent log sweep flags a majority-stub backend. That means the status "
            "card is no longer treating absence of a sidecar as proof of health.")
    else:
        out += [
            f"**{len(dead)}** backend(s) are spinning dead (a streak of banner-only / "
            "0-byte worker logs the weekly-cap gate doesn't catch — e.g. a credit-walled "
            "codex or a glm worker that prints only its startup banner). The dispatcher "
            "holds their spawns and a healthy backend claims the freed lane + budget; "
            "one re-probe worker is admitted per interval, so each auto-restores the "
            "moment it produces a real turn again.",
            "",
            "| backend | dead since | lane reallocated | re-probe (min) |",
            "|---|---|---|---|",
        ]
        for b in dead:
            out.append(f"| {b.get('product')} | {b.get('since')} | "
                       f"{b.get('abandoned_lane') or '—'} | {b.get('reprobe_min') or '—'} |")
    out += ["", "Backend stub-rate (recent resolve logs):", ""]
    if not stub_rates:
        out.append("_No recent resolve logs in the backend sweep window._")
    else:
        out += [
            "| backend | lookback | recent logs | productive | stub | stub rate | verdict | evidence |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in stub_rates:
            verdict = "**MAJORITY_STUB**" if row.get("majority_stub") else "ok"
            evidence = ", ".join(f"`{log}`" for log in (row.get("evidence_logs") or [])[:3]) or "—"
            rate = row.get("stub_rate")
            out.append(f"| {row.get('product')} | {row.get('lookback_min')}m | "
                       f"{row.get('total')} | {row.get('productive')} | {row.get('stub')} | "
                       f"{rate} | {verdict} | {evidence} |")

    hh = payload.get("hook_health") or {}
    hook_rows = hh.get("by_backend") or []
    out += ["", "## Hook health (guard-layer binding)", ""]
    if not hook_rows:
        out.append("_No recent resolve logs in the hook-failure sweep window._")
    else:
        unhooked = [r for r in hook_rows if r.get("all_sessions_unhooked")]
        if unhooked:
            out += [
                f"**{len(unhooked)}** backend(s) ran **UNHOOKED** by the guard layer over "
                "the recent window — every session logged `hook: <name> Failed`, so the fak "
                "guard hooks (PreToolUse / PostToolUse / Stop) never bound. `claude` binds "
                "the guard hooks natively; a non-claude backend (codex/opencode) runs its "
                "own native hook config, and when that config can't reach the dos hook CLI "
                "at runtime every lifecycle hook fails. Such a worker stays productive but "
                "runs unguarded by the hook backstop — the commit-path / `OFF_TRUNK` guard "
                "is what still holds the line (#1277).",
                "",
            ]
        else:
            out += ["All backends bound their guard hooks over the recent window — "
                    "no `hook: <name> Failed` storm.", ""]
        out += [
            "| backend | lookback | sessions | sessions w/ hook fail | fail-session rate | hook failures | verdict | evidence |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in hook_rows:
            verdict = "**UNHOOKED**" if row.get("all_sessions_unhooked") else "ok"
            evidence = ", ".join(f"`{log}`" for log in (row.get("evidence_logs") or [])[:3]) or "—"
            rate = row.get("failure_session_rate")
            out.append(f"| {row.get('product')} | {row.get('lookback_min')}m | "
                       f"{row.get('sessions')} | {row.get('sessions_with_hook_failures')} | "
                       f"{rate} | {row.get('hook_failures')} | {verdict} | {evidence} |")

    gd = payload.get("guard") or {}
    out += ["", "## Guard coverage (kernel decisions on the dispatch path)", ""]
    if not gd.get("dir_present"):
        out.append(
            "_No `.dispatch-runs/guard-audit/` journal yet — no guarded worker has run "
            "on this host. The dispatch worker fronts every session with `fak guard` by "
            "default (opt out `FLEET_DOGFOOD_GUARD=0`); the trail appears once one runs._")
    elif not gd.get("sessions"):
        out.append(
            "_The guard-audit directory exists but holds no journals — the guard wire is "
            "configured but never exercised by a launched worker._")
    else:
        by_kind = gd.get("by_kind") or {}
        out += [
            f"`fak guard` recorded **{gd.get('rows', 0)}** kernel decision(s) across "
            f"**{gd.get('sessions', 0)}** guarded dispatch session(s) "
            f"(**{gd.get('recent_sessions', 0)}** within {gd.get('lookback_min')}m) — the "
            "WITNESS that the concurrent-dispatch path ran THROUGH the kernel, not just "
            "got configured to. Each session owns a unique hash-chained journal "
            "(`fak audit verify <file>`); the decision mix is the kernel's verdict tally.",
            "",
            f"- **denied** (DENY + RESULT_DENY): {gd.get('denied', 0)}",
            f"- **quarantined**: {gd.get('quarantined', 0)}",
            f"- **empty sessions** (booted under guard, no adjudicated tool call): "
            f"{gd.get('empty_sessions', 0)}",
            "",
            "| decision kind | count |",
            "|---|---:|",
        ]
        for kind, n in sorted(by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {kind} | {n} |")
        evidence = gd.get("evidence") or []
        if evidence:
            out += ["", "Recent journals: "
                    + ", ".join(f"`{name}`" for name in evidence) + "."]

    sc = w.get("silent_count") or 0
    out += ["", "## Workers that produced nothing", ""]
    if not sc:
        out.append("None — every spawned worker either produced output or is still running.")
    else:
        out += [
            f"**{sc}** worker(s) exited at or below the real-turn floor "
            f"({_STUB_LOG_MAX_BYTES} B) — a 0-byte spawn or a banner-only stub "
            "(spawned, committed nothing). The anti-churn cooldown advances the picker "
            "past these, so the loop still progresses — but each is worth an operator's "
            "eye: an epic-shaped issue too large to land in one shot, or a dead backend "
            "spawning no-ops (a majority-stub backend is the tell — see #1276).",
            "",
            "| issue | spawned (utc stamp) | kind | bytes | log |",
            "|---|---|---|---|---|",
        ]
        for sw in (w.get("silent") or []):
            out.append(f"| #{sw.get('issue')} | {sw.get('stamp')} | {sw.get('kind') or '—'} | "
                       f"{sw.get('size') if sw.get('size') is not None else '—'} | `{sw.get('log')}` |")

    out += ["", "## Issue-contract repair flow", ""]
    out += [
        "When `fak issue contract` / `fak dispatch route --json` hold an issue "
        "below `DEFAULT_ISSUE_CONTRACT_MIN_SCORE`, that is a scaffold gap, not a "
        "reason to `--force` dispatch. Run the read-only repair-assist pass instead:",
        "",
        "1. `python tools/issue_contract_repair.py --lane <lane> --limit N --json` "
        "— classifies each held issue's contract reasons into a repair kind "
        "(`split`/`scope`/`route`/`noise`/`private`/`template`/`other`) and builds "
        "a manifest row per issue. Never edits, labels, comments on, or closes "
        "an issue.",
        "2. `template`-kind rows carry a dry-run-computed normalized-header fix "
        "(`ready: true`); every other kind lists exactly the missing contract "
        "fields as one-line human questions — content is never invented.",
        "3. An operator or follow-up agent answers the scaffolded questions (or "
        "reviews the `template` fix) and applies it via a manual `gh issue edit`.",
        "4. Re-run `fak issue contract --live` to confirm the score reaches the "
        "floor, then dispatch proceeds through the normal picker — no gate "
        "bypass needed.",
    ]

    out += ["", "---", "", "Reasons: " + "; ".join(payload.get("reasons") or [])]
    return "\n".join(out) + "\n"


def _rate_str(v: Any) -> str:
    """Trim a float rate to a compact form (0.80 -> 0.8, 2.0 -> 2) without trailing
    zero noise, leaving non-numbers untouched."""
    if isinstance(v, (int, float)):
        s = f"{float(v):.2f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(v)


def _join_limited(rows: list[str], *, limit: int = 3) -> str:
    kept = rows[:limit]
    if len(rows) > limit:
        kept.append(f"+{len(rows) - limit} more")
    return "; ".join(kept)


def _dispatch_capacity_line(payload: dict[str, Any]) -> str:
    d = payload.get("dispatcher") or {}
    a = d.get("account") or {}
    b = payload.get("backlog") or {}
    c = payload.get("closure") or {}
    l = payload.get("leases") or {}
    wl = payload.get("worker_lease_check") or {}

    parts: list[str] = []
    if d.get("live") is not None or d.get("cap") is not None:
        hr = d.get("headroom")
        parts.append(f"worker slots {d.get('live')}/{d.get('cap')} active"
                     + (f" ({hr} free)" if isinstance(hr, int) else ""))
    limiter = d.get("limiter") or {}
    if limiter.get("primary"):
        parts.append(f"limiter {limiter.get('primary')} ({_dispatch_limiter_terms(limiter)})")
    if a.get("tag"):
        parts.append(f"next account {a.get('tag')} t{a.get('tier')}"
                     + ("" if a.get("available") else " unavailable"))
    if not d.get("host_safe"):
        parts.append("host flagged")
    if not b.get("na") and b.get("open_issues") is not None:
        ur = b.get("unrouted")
        parts.append(f"backlog {b.get('open_issues')}"
                     + (f" ({ur} unrouted)" if ur else ""))
    if not c.get("na") and c.get("closure_rate") is not None:
        hk = c.get("honest_close_rate")
        parts.append(f"closure {_rate_str(c.get('closure_rate'))}"
                     + (f"/{_rate_str(hk)}" if hk is not None else ""))
    if l.get("active_count"):
        parts.append(f"leases {l.get('active_count')} active"
                     + (f" ({l.get('blocking_count')} blocking)" if l.get("blocking_count") else ""))
    if wl and wl.get("available") is not False:
        parts.append(f"lease-check clean {wl.get('clean_count', 0)}"
                     f"/orphan-proc {wl.get('orphan_process_count', 0)}"
                     f"/orphan-lease {wl.get('orphan_lease_count', 0)}")
    return "capacity: " + " · ".join(parts) if parts else ""


def _dispatch_trend_line(tp: dict[str, Any]) -> str:
    if tp.get("na"):
        return ""
    per_window = tp.get("per_window") or {}
    rates: list[str] = []
    for key in ("1h", "3h", "6h", "24h"):
        row = per_window.get(key) or {}
        rate = row.get("completed_rate_per_hour")
        if rate is not None:
            rates.append(f"{key} {_rate_str(rate)}/h")
    if not rates:
        rate = tp.get("completed_rate_per_hour")
        if rate is None:
            return ""
        rates.append(f"{tp.get('primary_window_hours')}h {_rate_str(rate)}/h")

    bits = ["completed " + " · ".join(rates)]
    target = tp.get("target_per_hour")
    if target is not None:
        bits.append(f"target {_rate_str(target)}/h")
    loop_windows = tp.get("loop_per_window") or {}
    primary = f"{tp.get('primary_window_hours')}h"
    loop_primary = loop_windows.get(primary) or {}
    if loop_primary.get("loop_rate_per_hour") is not None:
        bits.append(f"loop {primary} {_rate_str(loop_primary.get('loop_rate_per_hour'))}/h")
    last = tp.get("last_loop_close_age_min")
    if last is not None:
        bits.append(f"last loop close {_rate_str(last)}m ago")
    return "trend: " + "; ".join(bits)


def _dispatch_slack_buckets(payload: dict[str, Any]) -> dict[str, list[str]]:
    d = payload.get("dispatcher") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    sup = payload.get("supervisor") or {}
    tp = payload.get("throughput") or {}
    buckets = {"expected": [], "auto-solving": [], "action": []}

    verdict = str(payload.get("verdict") or "")
    preflight = str(d.get("preflight_verdict") or "")
    cap = payload.get("weekly_cap") or {}
    if cap:
        buckets["expected"].append(
            f"{a.get('tag') or 'account'} weekly-capped until "
            f"{cap.get('reset_text') or cap.get('until') or '?'}; scheduler waits")
    if verdict == "AT_CAP" or preflight == "REFUSE_AT_CAP":
        buckets["expected"].append("at configured worker-slot cap")
    if verdict == "BLOCKED_ON_ACCOUNT" or preflight == "REFUSE_NO_ACCOUNT":
        buckets["expected"].append("no free worker account; switcher resumes when one frees")
    if verdict == "STALLED" and payload.get("ok"):
        buckets["expected"].append("scheduler liveness says STALLED but gate marks it ok; see auto/action below")

    sup_verdict = str(sup.get("verdict") or "")
    if sup_verdict == "PLAN_SURFACE_EMPTY" and not sup.get("alive") and not sup.get("target"):
        buckets["expected"].append(
            "supervisor PLAN_SURFACE_EMPTY: expected for issue-driven dispatch; not session health")
    elif sup_verdict and sup_verdict not in ("READY", "OK", "READY_TO_CANARY"):
        buckets["action"].append(
            f"supervisor {sup_verdict} (alive {sup.get('alive')}/{sup.get('target')})")

    if wd.get("installed") is False:
        buckets["action"].append("always-on watchdog not installed; register FleetIssueDispatch")
    if not d.get("host_safe"):
        buckets["action"].append("host resource guard flagged a process; inspect before growing")
    if preflight == "REFUSE_INSPECT":
        buckets["action"].append("spawn preflight could not run; inspect the preflight error")
    auth_seat_action = _auth_failed_seat_action(payload.get("seat_inventory") or {})
    if auth_seat_action:
        buckets["action"].append(auth_seat_action)
    limiter = ((payload.get("dispatcher") or {}).get("limiter") or {})
    if limiter.get("primary") == "github_rate_limit":
        buckets["action"].append("GitHub rate limit is blocking the gh-backed status folds")

    workers = payload.get("workers") or {}
    if workers.get("silent_count"):
        nums = " ".join(f"#{s.get('issue')}" for s in (workers.get("silent") or [])[:5])
        buckets["auto-solving"].append(
            f"{workers.get('silent_count')} no-output worker(s) skipped by cooldown"
            + (f" ({nums})" if nums else "")
            + "; inspect only if the same issue repeats")

    bh = payload.get("backend_health") or {}
    stub_by_product = {
        str(r.get("product") or ""): r
        for r in (bh.get("stub_rate") or [])
        if r.get("majority_stub")
    }
    dead_products: set[str] = set()
    for r in (bh.get("dead") or [])[:4]:
        product = str(r.get("product") or "backend")
        dead_products.add(product)
        stub = stub_by_product.get(product)
        why = ""
        if stub:
            why = f"; evidence {stub.get('stub')}/{stub.get('total')} recent logs are stubs"
        reprobe = r.get("reprobe_min")
        buckets["auto-solving"].append(
            f"{product} held dead; lane {r.get('abandoned_lane') or '?'} reallocated"
            + (f"; re-probe every {reprobe}m" if reprobe else "")
            + why)
    for product, r in stub_by_product.items():
        if product in dead_products:
            continue
        buckets["action"].append(
            f"{product} majority-stub ({r.get('stub')}/{r.get('total')} recent logs); inspect backend output")

    hh = payload.get("hook_health") or {}
    for r in [x for x in (hh.get("by_backend") or []) if x.get("all_sessions_unhooked")][:4]:
        buckets["action"].append(
            f"{r.get('product')} guard hooks unbound "
            f"({r.get('sessions_with_hook_failures')}/{r.get('sessions')} sessions, "
            f"{r.get('hook_failures')} failures); workers ran unhooked")

    if not tp.get("na") and tp.get("verdict") in ("BELOW_TARGET", "AUDIT_ERROR"):
        buckets["action"].append(
            f"throughput {tp.get('verdict')}: {_rate_str(tp.get('completed_rate_per_hour'))}/h "
            f"vs target {_rate_str(tp.get('target_per_hour'))}/h")

    rs = payload.get("run_status") or {}
    if rs.get("errors"):
        buckets["action"].append(f"dos status had {rs.get('errors')} digest error(s)")
    git = payload.get("git") or {}
    if git.get("merge_in_progress"):
        buckets["action"].append(
            "peer merge in progress (MERGE_HEAD present); "
            f"{git.get('next_action') or 'wait before starting worker edits'}")
    leases = payload.get("leases") or {}
    if leases.get("read_error"):
        buckets["action"].append(f"lease read unavailable: {leases.get('read_error')}")
    elif leases.get("active_count"):
        bits = _lease_summary_bits(leases, limit=2)
        buckets["expected"].append(
            f"{leases.get('active_count')} active lane lease(s), "
            f"{leases.get('blocking_count', 0)} blocking current candidates"
            + (f" ({'; '.join(bits)})" if bits else ""))
    wl = payload.get("worker_lease_check") or {}
    if wl.get("available") is False:
        buckets["action"].append(f"worker/lease cross-check unavailable: {wl.get('error')}")
    elif wl:
        op = wl.get("orphan_process_count") or 0
        ol = wl.get("orphan_lease_count") or 0
        if op or ol:
            buckets["action"].append(
                f"worker/lease orphans: clean={wl.get('clean_count', 0)}, "
                f"orphan-process={op}, orphan-lease={ol}")
        elif wl.get("clean_count"):
            buckets["expected"].append(
                f"worker/lease cross-check clean ({wl.get('clean_count')} matched)")

    return buckets


def _dispatch_headline_state(payload: dict[str, Any]) -> str:
    buckets = _dispatch_slack_buckets(payload)
    if buckets["action"]:
        return "ACTION"
    if buckets["auto-solving"]:
        return "auto-solving"
    if buckets["expected"]:
        return "expected"
    if payload.get("ok"):
        return "healthy"
    return "ACTION"


def render_slack(payload: dict[str, Any]) -> str:
    r"""The COMPACT Slack body for the dispatch card — the signal-dense peer of
    ``render`` (which keeps the box-drawn rails for the terminal + committed doc).

    The boxed ``render`` is built for a monospace wall: ``╔═ ║ ╚═`` rails, a column
    label on every line (``workers   :``), and a ``╚═``-prefixed footer that restates
    every row again in prose. In a Slack channel — read on a phone, in mrkdwn, not a
    code fence — that chrome and that restated footer are pure noise: the reader
    re-reads the same fact twice and scans past box-drawing to reach a number.

    This renderer keeps the SIGNAL (every value an operator acts on) and drops the
    noise: ONE dense summary line carries capacity / account / backlog / closure /
    rate, then one targeted ``⚠``/``🔴`` line PER problem that needs an eye (a silent
    worker, a majority-stub or unhooked backend, a held backend, an uninstalled
    watchdog, a weekly cap). A healthy steady state collapses to the summary line plus
    a single ``✓`` — no restated footer, no rails, no fence. Pure given the payload."""
    tp = payload.get("throughput") or {}
    lines: list[str] = [
        "plane: scheduler/backlog, not session health",
    ]
    cap_line = _dispatch_capacity_line(payload)
    if cap_line:
        lines.append(cap_line)
    trend = _dispatch_trend_line(tp)
    if trend:
        lines.append(trend)

    buckets = _dispatch_slack_buckets(payload)
    for label in ("expected", "auto-solving", "action"):
        rows = buckets[label]
        if rows:
            lines.append(f"{label}: {_join_limited(rows)}")
    if not any(buckets.values()) and payload.get("ok"):
        lines.append("healthy: nothing needs an operator")
    return "\n".join(lines) if lines else "(no dispatcher signal)"


def slack_text(payload: dict[str, Any]) -> str:
    """The Slack message body for a status card: a one-line headline (so the channel
    preview and notification carry the verdict) above the COMPACT, signal-dense card
    (``render_slack``). The boxed ``render`` stays the terminal / committed-doc surface;
    Slack gets mrkdwn, not a monospace box, so the channel/phone reader scans state,
    not chrome (see the fleet-slack signal scorecard in tools/fleet_slack_status.py)."""

    verdict = payload.get("verdict")
    state = _dispatch_headline_state(payload)
    headline = f"*dispatch scheduler:* `{verdict}` ({state})"
    return headline + "\n" + render_slack(payload)


def post_to_slack(payload: dict[str, Any], *, channel: str = "",
                  dry_run: bool = False, transport: Any | None = None) -> dict[str, Any]:
    """Post the rendered status card to Slack via tools/slack_post. Never raises — a
    missing poster or a Slack failure becomes a typed verdict the caller logs, exactly
    like the rest of this read-only fold. Channel/token resolve through slack_post
    (``$FAK_DISPATCH_CHANNEL`` / the shared scoreboard token) unless ``channel`` is set."""
    try:
        import slack_post  # sibling module in tools/
    except Exception as exc:  # noqa: BLE001
        return {"posted": False, "error": f"slack_post unavailable: {exc}", "skipped": None}
    return slack_post.send(slack_text(payload), channel=channel, dry_run=dry_run,
                           transport=transport, include_signal_noise=False)


def git_date(root: Path) -> str:
    """The last-commit date (YYYY-MM-DD) — deterministic, no wall-clock in the tool."""
    try:
        proc = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=str(root),
                              capture_output=True, text=True, timeout=15,
                              creationflags=_win_creationflags())
        date = (proc.stdout or "").strip()
        return date or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _default_max_workers() -> int:
    """Mirror of dispatch_preflight.DEFAULT_MAX_WORKERS (built-in 8, FAK_MAX_WORKERS
    env knob applied) so the card's probe matches the gate's own ceiling instead of
    understating the fleet's headroom with a stale local default."""
    raw = os.environ.get("FAK_MAX_WORKERS", "").strip()
    try:
        if raw and int(raw) > 0:
            return int(raw)
    except ValueError:
        pass
    return 8


def main(argv: list[str] | None = None) -> int:
    default_workers = _default_max_workers()
    ap = argparse.ArgumentParser(description="One-touch always-on dispatcher status card.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=default_workers,
                    help="cap used by the spawn-gate preflight "
                         f"(default: {default_workers}; FAK_MAX_WORKERS retunes it)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the two gh-backed folds (backlog + closure); pure-local")
    ap.add_argument("--closure-commits", type=int, default=2500,
                    help="MINIMUM git-history budget for the closure audit; the actual "
                         "window auto-grows to the repo size + headroom so the audit "
                         "never scans a stale slice (default floor: 2500)")
    ap.add_argument("--md", default="",
                    help="write the committed markdown status doc to this path "
                         "(forces the full fold; --fast is ignored when --md is set)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--slack", nargs="?", const="__env__", default=None,
                    metavar="CHANNEL",
                    help="post the status card to Slack (optional channel id; default: "
                         "$FAK_DISPATCH_CHANNEL via tools/slack_post). Forces the full "
                         "fold so the posted card is never all-n/a.")
    ap.add_argument("--slack-dry-run", action="store_true",
                    help="with --slack: resolve the channel/token and report what WOULD "
                         "be posted without sending")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    # The committed doc must carry the real backlog/closure tables, so --md always
    # runs the full fold regardless of --fast. A LIVE Slack post is just as useless when
    # every row reads "n/a (fast)", so --slack forces the full fold too — but a
    # --slack-dry-run is a wiring check where speed matters, so it still honors --fast.
    live_slack = args.slack is not None and not args.slack_dry_run
    fast = args.fast and not args.md and not live_slack
    payload = collect(root, max_workers=args.max_workers, fast=fast,
                      closure_commits=args.closure_commits)

    if args.slack is not None:
        channel = "" if args.slack == "__env__" else args.slack
        slack_verdict = post_to_slack(payload, channel=channel,
                                      dry_run=args.slack_dry_run)
        payload["slack"] = slack_verdict
        if not args.json:
            if slack_verdict.get("posted"):
                print(f"slack: posted card to {slack_verdict.get('channel')} "
                      f"(ts={slack_verdict.get('ts')})")
            elif slack_verdict.get("dry_run"):
                print(f"slack (dry-run): would post to "
                      f"{slack_verdict.get('channel') or '(unset)'} "
                      f"[{slack_verdict.get('channel_source')}]")
            elif slack_verdict.get("skipped"):
                print(f"slack: skipped — {slack_verdict.get('skipped')}")
            else:
                print(f"slack: FAILED — {slack_verdict.get('error')}")

    if args.md:
        md_path = Path(args.md)
        if not md_path.is_absolute():
            md_path = root / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_md(payload, date=git_date(root)), encoding="utf-8")
        if not args.json:
            print(f"wrote {md_path} ({payload.get('verdict')}, "
                  f"open={ (payload.get('backlog') or {}).get('open_issues') }, "
                  f"silent={ (payload.get('workers') or {}).get('silent_count') })")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.md:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
