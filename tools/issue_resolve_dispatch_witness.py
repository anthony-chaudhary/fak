#!/usr/bin/env python3
"""No-commit classification, DOS commit witness sweeps, reblock streak holds, and loop ledger telemetry."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from issue_resolve_dispatch_core import (
    BASE_SHA_SIDECAR_SUFFIX,
    CLAIM_NO_COMMIT,
    CLAIM_UNWITNESSED,
    CLAIM_WITNESSED,
    DEFAULT_REBLOCK_STREAK_HOLD_N,
    DEFAULT_REBLOCK_STREAK_HOLD_TTL_H,
    LEASE_SIDECAR_SUFFIX,
    LOOP_ID_PREFIX,
    NO_COMMIT_AUTH_WALL,
    NO_COMMIT_BANNER_NOOP,
    NO_COMMIT_CLEAN_EXIT,
    NO_COMMIT_DIED_BEFORE_EPILOGUE,
    NO_COMMIT_GUARD_SPAWN_FAILED,
    NO_COMMIT_MISSING_LOG,
    NO_COMMIT_OFF_TRUNK,
    NO_COMMIT_POLICY_BLOCK,
    NO_COMMIT_PREVIEW_CONFIRM,
    NO_COMMIT_RESTART_EXHAUSTED,
    NO_COMMIT_SELF_MODIFY,
    OPEN_WITNESSED,
    SPAWN_CAUSE_CHILD_CRASH,
    SPAWN_CAUSE_EXEC_RACE,
    SPAWN_CAUSE_STALE_CRED,
    SPAWN_CAUSE_UNKNOWN,
    SPAWN_CAUSE_WEEKLY_LIMIT,
    WITNESS_SIDECAR_SUFFIX,
    _AUTH_GAP_RE,
    _GLM_WALL_RE,
    _GUARD_CHILD_SPAWN_FAILED_MARKER,
    _GUARD_EPILOGUE_MARKER,
    _HOLD_NO_COMMIT_REASONS,
    _ISSUE_TOKEN_RE,
    _LOG_ISSUE_RE,
    _MISSING_API_KEY_RE,
    _NOOP_BANNER_RE,
    _PREVIEW_CONFIRM_FEEDBACK_RE,
    _RESTART_EXHAUSTED_RE,
    _RESTART_EXHAUSTED_TAIL_BYTES,
    _RID_RE,
    _SPAWN_CRASH_RE,
    _STUB_LOG_MAX_BYTES,
    _WITNESS_OK,
    _WITNESS_STAMP_RE,
    _git_capture,
    _py,
    dispatch_preflight,
    fak_loop_cmd,
    fleet_trend,
    ird_attr,
    issue_closure_audit,
    issue_lane_router,
    lane_tree,
    no_window_creationflags,
    refusal_class,
)
from issue_resolve_dispatch_guards import (
    _cap_hit_from_text,
    _log_is_cap_banner,
    _log_tail_text,
    release_lane_lease,
)


def read_lease_sidecar(out_log: Path) -> dict[str, Any] | None:
    try:
        rec = json.loads(out_log.with_suffix(LEASE_SIDECAR_SUFFIX).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or not rec.get("id") or not rec.get("holder"):
        return None
    return rec


def classify_restart_exhaustion(log: Path) -> dict | None:
    """Extract the typed guard terminal from the resolver process log."""
    tail = _log_tail_text(log, nbytes=_RESTART_EXHAUSTED_TAIL_BYTES)
    matches = list(_RESTART_EXHAUSTED_RE.finditer(tail))
    if not matches:
        return None
    match = matches[-1]
    count = match.group("count") or match.group("limit")
    return {"reason": NO_COMMIT_RESTART_EXHAUSTED,
            "restart_count": int(count) if count else None,
            "dominant_cause": match.group("cause")}


def refine_unknown_no_commit(tail: str) -> str:
    """Name the residual no-commit from what the guard epilogue in ``tail`` shows (#5870)."""
    if _GUARD_CHILD_SPAWN_FAILED_MARKER in tail:
        return NO_COMMIT_GUARD_SPAWN_FAILED
    if _GUARD_EPILOGUE_MARKER in tail:
        return NO_COMMIT_CLEAN_EXIT
    return NO_COMMIT_DIED_BEFORE_EPILOGUE


def classify_no_commit_reason(log: Path) -> str:
    """Why did a finished worker land no resolving commit?"""
    try:
        if not log.exists():
            return NO_COMMIT_MISSING_LOG
    except OSError:
        return NO_COMMIT_MISSING_LOG
    restart = classify_restart_exhaustion(log)
    if restart is not None:
        return NO_COMMIT_RESTART_EXHAUSTED
    tail = _log_tail_text(log)
    if "SELF_MODIFY" in tail:
        return NO_COMMIT_SELF_MODIFY
    if "POLICY_BLOCK" in tail:
        return NO_COMMIT_POLICY_BLOCK
    if (_log_is_cap_banner(log) or _GLM_WALL_RE.search(tail)
            or _MISSING_API_KEY_RE.search(tail)):
        return NO_COMMIT_AUTH_WALL
    if "OFF_TRUNK" in tail:
        return NO_COMMIT_OFF_TRUNK
    if _PREVIEW_CONFIRM_FEEDBACK_RE.search(tail) and "exiting loop" in tail.lower():
        return NO_COMMIT_PREVIEW_CONFIRM
    try:
        small = log.stat().st_size <= _STUB_LOG_MAX_BYTES
    except OSError:
        small = False
    if small and _NOOP_BANNER_RE.search(tail):
        return NO_COMMIT_BANNER_NOOP
    return refine_unknown_no_commit(tail)


def _spawn_header_only(tail: str) -> bool:
    """True when ``tail`` holds nothing past the pre-exec ``# fak-spawn`` header line(s)."""
    for line in tail.splitlines():
        if line.strip() and not line.startswith("# fak-spawn"):
            return False
    return True


def classify_spawn_failed_cause(early: dict[str, Any]) -> str:
    """Attribute ONE early-exit SPAWN_FAILED event to a cause bucket (#2635)."""
    tail = str(early.get("tail") or "")
    if _cap_hit_from_text(tail) is not None or _GLM_WALL_RE.search(tail):
        return SPAWN_CAUSE_WEEKLY_LIMIT
    if _AUTH_GAP_RE.search(tail) or _MISSING_API_KEY_RE.search(tail):
        return SPAWN_CAUSE_STALE_CRED
    if _SPAWN_CRASH_RE.search(tail):
        return SPAWN_CAUSE_CHILD_CRASH
    if early.get("silent") or _spawn_header_only(tail):
        return SPAWN_CAUSE_EXEC_RACE
    return SPAWN_CAUSE_UNKNOWN


def held_no_commit_issues(witnessed: dict[str, Any] | None) -> set[int]:
    """Issue numbers to HOLD this tick because their last worker hit a structural hold."""
    held: set[int] = set()
    for rec in ((witnessed or {}).get("no_commit") or []):
        if isinstance(rec, dict) and rec.get("reason") in _HOLD_NO_COMMIT_REASONS:
            try:
                held.add(int(rec["issue"]))
            except (KeyError, TypeError, ValueError):
                continue
    return held


def issue_witness_history(runs_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """``{issue: [record, ...]}`` in LAUNCH order, from ``resolve-<issue>-<UTC>.witness``."""
    hist: dict[int, list[dict[str, Any]]] = {}
    if not runs_dir.is_dir():
        return hist
    for w in runs_dir.glob(f"resolve-*{WITNESS_SIDECAR_SUFFIX}"):
        m = _WITNESS_STAMP_RE.match(w.name)
        if not m:
            continue
        try:
            ts = dt.datetime.strptime(
                m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(
                    tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
        try:
            row = json.loads(w.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        hist.setdefault(int(m.group(1)), []).append({
            "ts": ts,
            "claim": str(row.get("claim") or ""),
            "reason": str(row.get("reason") or ""),
        })
    for runs in hist.values():
        runs.sort(key=lambda r: r["ts"])
    return hist


def reblock_streak_held_records(
    runs_dir: Path,
    *,
    streak_n: int = DEFAULT_REBLOCK_STREAK_HOLD_N,
    ttl_h: int = DEFAULT_REBLOCK_STREAK_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Issues whose last ``streak_n`` finished workers all hit the same re-blockable terminal (#5869)."""
    if streak_n <= 0 or ttl_h <= 0:
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    out: list[dict[str, Any]] = []
    for issue, all_runs in sorted(issue_witness_history(runs_dir).items()):
        runs = [r for r in all_runs if r["ts"] <= now]
        tail = runs[-streak_n:]
        if len(tail) < streak_n:
            continue
        reasons = {r["reason"] for r in tail}
        if len(reasons) != 1:
            continue
        reason = next(iter(reasons))
        if reason not in _HOLD_NO_COMMIT_REASONS:
            continue
        if any(r["claim"] != CLAIM_NO_COMMIT for r in tail):
            continue
        newest = tail[-1]["ts"]
        if newest < horizon:
            continue
        if updated_ts and (updated_ts.get(issue) or 0.0) > newest:
            continue
        out.append({
            "issue": issue,
            "number": issue,
            "reason": reason,
            "streak": streak_n,
            "last_ts": newest,
        })
    return out


def reblock_streak_held_issues(
    runs_dir: Path,
    *,
    streak_n: int = DEFAULT_REBLOCK_STREAK_HOLD_N,
    ttl_h: int = DEFAULT_REBLOCK_STREAK_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
) -> set[int]:
    """Issue numbers held after ``streak_n`` consecutive identical re-blockable terminals."""
    return {int(r["issue"]) for r in reblock_streak_held_records(
        runs_dir, streak_n=streak_n, ttl_h=ttl_h, now_ts=now_ts,
        updated_ts=updated_ts)}


def _subject_cites_issue(subject: str, issue: int) -> bool:
    """True when a commit ``subject`` names ``#<issue>`` at a word boundary."""
    return re.search(rf"(?<![\w-])#{int(issue)}\b", subject or "") is not None


def worker_resolving_sha(root: Path, issue: int, *, base_sha: str | None = None,
                         git: Any | None = None, scan_limit: int = 300) -> str | None:
    """The newest commit whose subject cites ``#<issue>``."""
    git = git or ird_attr("_git_capture", _git_capture)
    rev_args = ["log", "--no-color", "--pretty=format:%H\x1f%s"]
    if base_sha:
        rev_args.append(f"{base_sha}..HEAD")
    else:
        rev_args += ["-n", str(int(scan_limit))]
    rc, out = git(root, rev_args)
    if rc != 0 or not out:
        return None
    for line in out.splitlines():
        sha, _sep, subject = line.partition("\x1f")
        if sha and _subject_cites_issue(subject, issue):
            return sha.strip()
    return None


def _run_commit_audit(root: Path, sha: str, *, timeout: int = 60) -> dict[str, Any]:
    """Default ``dos commit-audit <sha> --json`` runner -> row for ``sha``."""
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os_mod.name == "nt":
        kwargs["creationflags"] = nowin()
    try:
        proc = sub_mod.run(["dos", "commit-audit", sha, "--workspace", str(root),
                            "--json"], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = (proc.stdout or "").strip()
    try:
        parsed = json.loads(out) if out else []
    except ValueError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for row in parsed:
            if isinstance(row, dict) and row.get("sha") and str(sha).startswith(str(row["sha"])):
                return row
        if parsed and isinstance(parsed[0], dict):
            return parsed[0]
    return {}


def audit_commit_witness(root: Path, sha: str, *, runner: Any | None = None) -> dict[str, Any]:
    """Grade ``sha`` through the DOS witness rung."""
    run = runner or ird_attr("_run_commit_audit", _run_commit_audit)
    doc = run(root, sha) or {}
    verdict = str(doc.get("verdict") or "")
    witness = str(doc.get("witness") or "")
    witnessed = verdict.upper() == "OK" and witness == _WITNESS_OK
    return {"witnessed": witnessed, "verdict": verdict or None, "witness": witness or None}


def _run_commit_audit_rows(root: Path, ref: str, *, timeout: int = 60) -> list[dict[str, Any]]:
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os_mod.name == "nt":
        kwargs["creationflags"] = nowin()
    try:
        proc = sub_mod.run(["dos", "commit-audit", ref, "--workspace", str(root),
                            "--json"], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        parsed = json.loads((proc.stdout or "").strip() or "[]")
    except ValueError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    return []


def locally_witnessed_issues(root: Path, *, base_ref: str = "origin/main",
                             git: Any | None = None,
                             audit_rows: Any | None = None) -> set[int]:
    """Open-loop duplicate guard for commits in ``base_ref..HEAD`` graded OK."""
    git = git or ird_attr("_git_capture", _git_capture)
    rc, out = git(root, ["log", "--no-color",
                         "--pretty=format:%H%x1f%B%x1e",
                         f"{base_ref}..HEAD"])
    if rc != 0 or not out:
        return set()
    cited: dict[str, set[int]] = {}
    for entry in out.split("\x1e"):
        entry = entry.strip()
        if not entry:
            continue
        sha, sep, text = entry.partition("\x1f")
        if not sep or not sha:
            continue
        nums = {int(m.group(1)) for m in _ISSUE_TOKEN_RE.finditer(text or "")}
        if nums:
            cited[sha.strip()] = nums
    if not cited:
        return set()
    rows_fn = audit_rows or ird_attr("_run_commit_audit_rows", _run_commit_audit_rows)
    rows = rows_fn(root, f"{base_ref}..HEAD")
    witnessed_shas = {
        str(r.get("sha") or "") for r in rows
        if str(r.get("verdict") or "").upper() == "OK"
        and str(r.get("witness") or "") == _WITNESS_OK
    }
    out_issues: set[int] = set()
    for sha, nums in cited.items():
        if any(sha.startswith(w) or w.startswith(sha) for w in witnessed_shas if w):
            out_issues.update(nums)
    return out_issues


def commit_audit_abstain_holds(root: Path, candidates: set[int], *,
                               git: Any | None = None,
                               audit_runner: Any | None = None,
                               scan_limit: int = 300) -> list[dict[str, Any]]:
    """Recent candidate-bound test commits whose witness ABSTAINs become holds."""
    wanted = {int(n) for n in candidates if n is not None}
    if not wanted:
        return []
    git = git or ird_attr("_git_capture", _git_capture)
    audit = audit_runner or ird_attr("_run_commit_audit", _run_commit_audit)
    rc, out = git(root, ["log", "--no-color", "-n", str(int(scan_limit)),
                         "--pretty=format:%H%x1f%B%x1e"])
    if rc != 0 or not out:
        return []
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in out.split("\x1e"):
        entry = entry.strip()
        if not entry:
            continue
        sha, sep, text = entry.partition("\x1f")
        if not sep or not sha:
            continue
        nums = {int(m.group(1)) for m in _ISSUE_TOKEN_RE.finditer(text or "")} & wanted
        nums -= seen
        if not nums:
            continue
        first_line = next((line.strip() for line in (text or "").splitlines()
                           if line.strip()), "")
        row = audit(root, sha.strip()) or {}
        verdict = str(row.get("verdict") or "").upper()
        if verdict != "ABSTAIN":
            continue
        test_files = row.get("test_files") or []
        test_subject = first_line.startswith("test(") or first_line.startswith("test:")
        if not test_files and not test_subject:
            continue
        for issue in sorted(nums):
            records.append({
                "issue": issue,
                "sha": sha.strip(),
                "code": "COMMIT_AUDIT_ABSTAIN",
                "verdict": verdict,
                "witness": row.get("witness"),
                "claim_kind": row.get("claim_kind"),
                "reason": row.get("reason") or "commit-audit abstained",
                "test_files": test_files,
            })
            seen.add(issue)
    return records


def open_witnessed_dispositions(root: Path, candidates: set[int], *,
                                git: Any | None = None,
                                audit_runner: Any | None = None,
                                scan_limit: int = 600) -> list[dict[str, Any]]:
    """Typed OPEN_WITNESSED dispositions for candidates already resolved in trunk."""
    wanted = {int(n) for n in candidates if n is not None}
    if not wanted:
        return []
    git = git or ird_attr("_git_capture", _git_capture)
    audit = audit_runner or ird_attr("_run_commit_audit", _run_commit_audit)
    ica = ird_attr("issue_closure_audit", issue_closure_audit)
    rc, out = git(root, ["log", "--no-color", "-n", str(int(scan_limit)),
                         "--pretty=format:%H%x1f%s%x1f%b%x1e"])
    if rc != 0 or not out:
        return []
    rows: list[dict[str, Any]] = []
    decided: set[int] = set()
    audited: dict[str, dict[str, Any]] = {}
    for entry in out.split("\x1e"):
        entry = entry.strip("\n")
        if not entry.strip():
            continue
        parts = entry.split("\x1f")
        if len(parts) < 2:
            continue
        sha = parts[0].strip()
        subject = parts[1].strip()
        body = parts[2] if len(parts) > 2 else ""
        if not sha:
            continue
        refs = ica.classify_refs(subject, body)
        nums = sorted(n for n, kind in refs.items()
                      if kind == ica.RESOLVING
                      and n in wanted and n not in decided)
        if not nums:
            continue
        row = audited.get(sha)
        if row is None:
            row = audit(root, sha) or {}
            audited[sha] = row
        verdict = str(row.get("verdict") or "").upper()
        witness = str(row.get("witness") or "")
        if verdict != "OK" or witness != _WITNESS_OK:
            continue
        if not ica.commit_binds_resolution(row, ""):
            continue
        for issue in nums:
            rows.append({
                "issue": issue, "sha": sha, "subject": subject,
                "code": OPEN_WITNESSED,
                "verdict": row.get("verdict"), "witness": row.get("witness"),
                "claim_kind": row.get("claim_kind"),
                "close_via": "tools/issue_resolve_witnessed.py",
            })
            decided.add(issue)
        if decided >= wanted:
            break
    return rows


def _run_close_arm(root: Path, cmd: list[str], *, timeout: int = 300) -> tuple[int, str]:
    """Default close-arm runner -> (returncode, stdout). Never raises."""
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os_mod.name == "nt":
        kwargs["creationflags"] = nowin()
    try:
        proc = sub_mod.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or "").strip()


def close_open_witnessed(root: Path, runs_dir: Path, rows: list[dict[str, Any]], *,
                         live: bool, runner: Any | None = None) -> dict[str, Any]:
    """Route OPEN_WITNESSED dispositions through the trusted close path."""
    if not rows:
        return {}
    run = runner or ird_attr("_run_close_arm", _run_close_arm)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    audit_path = runs_dir / f"open-witnessed-{stamp}.audit.json"
    synthetic = {
        "schema": "issue-closure-audit/synthetic/open-witnessed-dispatch/1",
        "issues": [{"number": int(r["issue"]), "bucket": OPEN_WITNESSED,
                    "title": str(r.get("title") or ""),
                    "witnessed_commits": [{"sha": str(r.get("sha") or ""),
                                           "subject": str(r.get("subject") or "")}]}
                   for r in rows if r.get("issue") is not None],
    }
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    except OSError as exc:
        return {"invoked": False, "error": f"could not write synthetic audit: {exc}"}
    cmd = [_py(), str(root / "tools" / "issue_resolve_witnessed.py"),
           "--workspace", str(root), "--audit-json", str(audit_path),
           "--limit", str(len(rows)), "--json"]
    if live:
        cmd.append("--live")
    rc, out = run(root, cmd)
    try:
        doc = json.loads(out or "{}")
    except ValueError:
        doc = {}
    if not isinstance(doc, dict) or not doc:
        return {"invoked": True, "ok": False, "returncode": rc, "live": live,
                "error": (out or "close arm produced no JSON")[-300:]}
    return {"invoked": True, "ok": bool(doc.get("ok")), "returncode": rc,
            "live": live, "verdict": doc.get("verdict"),
            "counts": doc.get("counts"),
            "closed_numbers": doc.get("closed_numbers"),
            "audit_json": str(audit_path)}


def witness_exited_workers(runs_dir: Path, root: Path, *, live: bool,
                           alive: set[int] | None = None, probe: Any | None = None,
                           git: Any | None = None,
                           audit_runner: Any | None = None,
                           lease_runner: Any | None = None) -> dict[str, Any]:
    """Bind each FINISHED worker's slot to a `dos commit-audit` witness (#1324)."""
    out: dict[str, Any] = {"live": live, "audited": [], "witnessed": [],
                           "unwitnessed": [], "no_commit": [],
                           "lease_released": [], "lease_release_failed": [],
                           "lease_release_retried": []}
    if not runs_dir.is_dir():
        return out
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    read_lease_fn = ird_attr("read_lease_sidecar", read_lease_sidecar)
    rel_lease_fn = ird_attr("release_lane_lease", release_lane_lease)
    resolving_sha_fn = ird_attr("worker_resolving_sha", worker_resolving_sha)
    audit_witness_fn = ird_attr("audit_commit_witness", audit_commit_witness)
    classify_nc_fn = ird_attr("classify_no_commit_reason", classify_no_commit_reason)
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore
            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None
    bucket_of = {CLAIM_WITNESSED: "witnessed", CLAIM_UNWITNESSED: "unwitnessed",
                 CLAIM_NO_COMMIT: "no_commit"}
    for log in sorted(runs_dir.glob("resolve-*.log")):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        witness_path = log.with_suffix(WITNESS_SIDECAR_SUFFIX)
        already_witnessed = witness_path.exists()
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if dp_mod.resolve_sidecar_pid_is_live(pid_file, alive=alive, probe=probe):
            continue
        if already_witnessed:
            if live:
                try:
                    witness_doc = json.loads(witness_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    witness_doc = {}
                prior_release = witness_doc.get("lease_release") if isinstance(witness_doc, dict) else {}
                if not (isinstance(prior_release, dict) and prior_release.get("released")):
                    lease = read_lease_fn(log)
                    if lease:
                        rel = rel_lease_fn(root, lease, runner=lease_runner)
                        out["lease_release_retried"].append({
                            "log": log.name,
                            "id": lease.get("id"),
                            "reason": "already_witnessed_dead_worker",
                            "released": bool(rel.get("released")),
                        })
                        if rel.get("released"):
                            out["lease_released"].append(rel)
                        else:
                            out["lease_release_failed"].append(rel)
                        if isinstance(witness_doc, dict):
                            witness_doc["lease_release"] = rel
                            try:
                                witness_path.write_text(
                                    json.dumps(witness_doc, sort_keys=True),
                                    encoding="utf-8")
                            except OSError:
                                pass
            continue
        issue = int(m.group(1))
        try:
            base = log.with_suffix(BASE_SHA_SIDECAR_SUFFIX).read_text(
                encoding="utf-8").strip() or None
        except OSError:
            base = None
        sha = resolving_sha_fn(root, issue, base_sha=base, git=git)
        if not sha:
            rec = {"issue": issue, "log": log.name, "sha": None,
                   "claim": CLAIM_NO_COMMIT, "verdict": None, "witness": None,
                   "reason": classify_nc_fn(log)}
            exhaustion = classify_restart_exhaustion(log)
            if exhaustion:
                rec.update(exhaustion)
        else:
            w = audit_witness_fn(root, sha, runner=audit_runner)
            rec = {"issue": issue, "log": log.name, "sha": sha,
                   "claim": CLAIM_WITNESSED if w["witnessed"] else CLAIM_UNWITNESSED,
                   "verdict": w["verdict"], "witness": w["witness"]}
        lease = read_lease_fn(log)
        if lease and lease.get("session_id"):
            rec["session_id"] = lease["session_id"]
        if live and lease:
            rel = rel_lease_fn(root, lease, runner=lease_runner)
            rec["lease_release"] = rel
            if rel.get("released"):
                out["lease_released"].append(rel)
            else:
                out["lease_release_failed"].append(rel)
        out["audited"].append(rec)
        out[bucket_of[rec["claim"]]].append(rec)
        if live:
            try:
                log.with_suffix(WITNESS_SIDECAR_SUFFIX).write_text(
                    json.dumps(rec, sort_keys=True), encoding="utf-8")
            except OSError:
                pass
    return out


def is_dos_run_id(run_id: object) -> bool:
    return bool(_RID_RE.fullmatch(str(run_id or "")))


def mint_dos_run_id(root: Path, process: str, parent: str | None = None) -> str | None:
    """Mint the DOS RID that `dos status` accepts, fail-open to caller fallback."""
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    cmd = ["dos", "run-id", "mint", process, "--root", str(root)]
    if parent:
        cmd += ["--parent", parent]
    kwargs: dict[str, Any] = {
        "cwd": root,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 30,
    }
    if os_mod.name == "nt":
        kwargs["creationflags"] = nowin()
    try:
        proc = sub_mod.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        doc = json.loads(proc.stdout)
    except ValueError:
        return None
    rid = str(doc.get("run_id") or "")
    return rid if proc.returncode == 0 and is_dos_run_id(rid) else None


def loop_id_for_payload(payload: dict[str, Any]) -> str:
    backend = str(payload.get("backend") or "claude").strip() or "claude"
    return f"{LOOP_ID_PREFIX}/{backend}"


def default_loop_ledger(root: Path) -> Path:
    configured = ird_attr("os", os).environ.get("FAK_LOOP_LEDGER")
    if configured:
        return Path(configured)
    return root / ".fak" / "loops.jsonl"


def _loop_metric_args(metrics: dict[str, int]) -> list[str]:
    out: list[str] = []
    for key in sorted(metrics):
        out += ["--metric", f"{key}={int(metrics[key])}"]
    return out


def _loop_evidence_args(evidence: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for kind, ref in evidence:
        if kind and ref:
            out += ["--evidence", f"{kind}={ref}"]
    return out


def _dispatch_collision_evidence(root: Path, payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Structured lane/lane_kind/mode/tree evidence for a refused dispatch tick (#4322)."""
    ilr = ird_attr("issue_lane_router", issue_lane_router)
    rc_fn = ird_attr("refusal_class", refusal_class)
    lt_fn = ird_attr("lane_tree", lane_tree)
    out: list[tuple[str, str]] = []
    contention = rc_fn(str(payload.get("verdict") or ""))
    if contention:
        out.append(("refusal_class", contention))
    lane = str(payload.get("lane") or "").strip()
    if not lane:
        return out
    out.append(("lane", lane))
    out.append(("lane_kind", "cluster"))
    out.append(("mode", "exclusive" if lane in ilr.EXCLUSIVE_LANES else "shared"))
    lease = payload.get("lease") if isinstance(payload.get("lease"), dict) else {}
    tree = lease.get("tree") if isinstance(lease.get("tree"), list) else None
    if not tree:
        try:
            tree = lt_fn(root, lane)
        except Exception:  # noqa: BLE001
            tree = None
    if tree:
        out.append(("tree", ",".join(str(t) for t in tree)))
    dirty = (payload.get("dirty_path_collision")
             if isinstance(payload.get("dirty_path_collision"), dict) else {})
    paths = dirty.get("dirty_paths") if isinstance(dirty.get("dirty_paths"), list) else None
    if paths:
        out.append(("paths", ",".join(str(p) for p in paths)))
    return out


def append_loop_event(root: Path, ledger: Path, event: dict[str, Any],
                      *, source: str = "issue_resolve_dispatch") -> dict[str, Any]:
    """Append one canonical loop-ledger row through `fak loop append`."""
    base_cmd = ird_attr("fak_loop_cmd", fak_loop_cmd)(root)
    if not base_cmd:
        return {"ok": False, "kind": event.get("kind"),
                "error": "fak binary unavailable; loop append skipped"}
    cmd = base_cmd + [
        "loop", "append",
        "--ledger", str(ledger),
        "--loop", str(event["loop_id"]),
        "--kind", str(event["kind"]),
        "--source", source,
        "--principal", str(event.get("principal") or event.get("backend") or "dispatcher"),
    ]
    for flag, key in (("--run", "run_id"), ("--status", "status"),
                      ("--reason", "reason"), ("--summary", "summary")):
        if event.get(key):
            cmd += [flag, str(event[key])]
    cmd += _loop_metric_args(event.get("metrics") or {})
    evidence = list(event.get("evidence") or [])
    if event.get("verified_state"):
        evidence.append(("verified_state", str(event["verified_state"])))
    cmd += _loop_evidence_args(evidence)
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    kwargs: dict[str, Any] = {
        "cwd": root, "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": 60,
    }
    if os_mod.name == "nt":
        kwargs["creationflags"] = nowin()
    try:
        proc = sub_mod.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "kind": event.get("kind"), "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "kind": event.get("kind"),
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip()[-500:],
        "stdout": (proc.stdout or "").strip()[-500:],
    }


def loop_run_id(payload: dict[str, Any], root: Path | None = None,
                mint: Any | None = None) -> str:
    existing = str(payload.get("run_id") or "")
    if is_dos_run_id(existing):
        return existing
    backend = str(payload.get("backend") or "claude").strip() or "claude"
    if root is not None:
        minted = (mint or mint_dos_run_id)(root, f"issue-resolve-dispatch-{backend}")
        if is_dos_run_id(minted):
            return str(minted)
    if existing:
        return existing
    spawned = payload.get("spawned") or {}
    if spawned.get("pid"):
        return f"resolve-{payload.get('target_issue') or 'none'}-{spawned.get('pid')}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"resolve-tick-{backend}-{stamp}"


def record_loop_tick(root: Path, payload: dict[str, Any],
                     *,
                     ledger: Path | None = None,
                     append: Any | None = None,
                     mint: Any | None = None) -> dict[str, Any]:
    """Lower one issue-resolve dispatcher tick into loop ledger events."""
    ledger = ledger or default_loop_ledger(root)
    append = append or ird_attr("append_loop_event", append_loop_event)
    run_id = str(loop_run_id(payload, root=root, mint=mint))
    payload["run_id"] = run_id
    loop_id = loop_id_for_payload(payload)
    pre = payload.get("preflight") or {}
    spawned = payload.get("spawned") or {}
    metrics: dict[str, int] = {
        "live": 1 if payload.get("live") else 0,
        "lane_issue_count": int(payload.get("lane_issue_count") or 0),
        "max_workers": int(payload.get("max_workers") or 0),
        "preflight_live": int(pre.get("live") or 0),
        "preflight_cap": int(pre.get("cap") or 0),
    }
    if payload.get("target_issue") is not None:
        metrics["target_issue"] = int(payload["target_issue"])
    if payload.get("prompt_chars") is not None:
        metrics["prompt_chars"] = int(payload["prompt_chars"])
    if spawned.get("pid"):
        metrics["pid"] = int(spawned["pid"])
    evidence: list[tuple[str, str]] = []
    if payload.get("target_issue") is not None:
        evidence.append(("issue", str(payload["target_issue"])))
    if spawned.get("log"):
        evidence.append(("log", str(spawned["log"])))
    if (payload.get("account") or {}).get("tag"):
        evidence.append(("account", str((payload.get("account") or {}).get("tag"))))
    admitted = bool(payload.get("ok")) and payload.get("action") in {"would_spawn", "spawned"}
    if not admitted:
        evidence.extend(_dispatch_collision_evidence(root, payload))

    events: list[dict[str, Any]] = [{
        "loop_id": loop_id, "run_id": run_id, "kind": "fire",
        "backend": payload.get("backend"),
        "summary": f"issue dispatch tick lane={payload.get('lane') or '-'}",
        "metrics": metrics, "evidence": evidence,
    }]
    events.append({
        "loop_id": loop_id, "run_id": run_id, "kind": "admit",
        "backend": payload.get("backend"),
        "status": "admitted" if admitted else "refused",
        "reason": str(payload.get("verdict") or payload.get("action") or ""),
        "summary": str(payload.get("reason") or "")[:200],
        "metrics": metrics, "evidence": evidence,
    })
    if payload.get("action") == "spawned":
        events.append({
            "loop_id": loop_id, "run_id": run_id, "kind": "start",
            "backend": payload.get("backend"),
            "status": "running",
            "reason": "SPAWNED",
            "summary": str(payload.get("reason") or "")[:200],
            "metrics": metrics, "evidence": evidence,
        })
    if payload.get("ok"):
        events.append({
            "loop_id": loop_id, "run_id": run_id, "kind": "end",
            "backend": payload.get("backend"),
            "status": "claimed_done",
            "reason": str(payload.get("verdict") or payload.get("action") or ""),
            "summary": str(payload.get("reason") or "")[:200],
            "metrics": metrics, "evidence": evidence,
        })
    rows = [append(root, ledger, ev) for ev in events]
    return {
        "ledger": str(ledger),
        "loop_id": loop_id,
        "run_id": run_id,
        "events": rows,
        "ok": all(r.get("ok") for r in rows) if rows else True,
    }


def append_fleet_trend_row(root: Path, payload: dict[str, Any], *,
                           append: Any | None = None,
                           now: str | None = None) -> dict[str, Any]:
    """#4594: feed the fleet-status trend ledger one partial row per live tick."""
    ft_mod = ird_attr("fleet_trend", fleet_trend)
    do_append = append or ft_mod.append
    pre = payload.get("preflight") or {}
    ledger = str(root / ft_mod.DEFAULT_LEDGER)
    ts = now or dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    try:
        row = do_append(ledger, {
            "scope": "backend",
            "backend": str(payload.get("backend") or "unknown"),
            "backend_live": float(pre.get("live") or 0),
        }, ts)
        return {"ok": True, "ledger": ledger, "row": row}
    except Exception as exc:
        return {"ok": False, "ledger": ledger, "error": str(exc)}
