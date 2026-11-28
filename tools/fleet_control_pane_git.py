#!/usr/bin/env python3
"""Git status, worktree doctor, trunk guard, commit, sync, and publish helpers for fleet_control_pane."""
from __future__ import annotations

import collections
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from fleet_control_pane_core import (
    COMMIT_SCHEMA,
    DEFAULT_WORKTREE_ALLOW_BRANCHES,
    DEFAULT_WORKTREE_MASTER_REF,
    DIRTY_GROUP_SUBJECTS,
    PANE_SOURCE_PATHS,
    PUBLISH_SCHEMA,
    SYNC_SCHEMA,
    TRUNK_GUARD_HOOKS_REL,
    TRUNK_GUARD_HOOK_REL,
    TRUNK_GUARD_INSTALLER_REL,
    display_command,
    git,
    git_stdout,
    iso_now,
    pane_attr,
)


def path_compare_key(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return os.path.normcase(os.path.normpath(str(resolved)))


def configured_hooks_path(root: Path, raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def trunk_guard_status(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    python = str(config.get("python") or sys.executable)
    installer = root / TRUNK_GUARD_INSTALLER_REL
    hook = root / TRUNK_GUARD_HOOK_REL
    hooks_dir = root / TRUNK_GUARD_HOOKS_REL
    command = [python, TRUNK_GUARD_INSTALLER_REL] if installer.exists() else []
    status: dict[str, Any] = {
        "supported": False,
        "installed": False,
        "ok": True,
        "setup_ok": True,
        "state": "skip",
        "detail": "not a git worktree",
        "installer": TRUNK_GUARD_INSTALLER_REL,
        "installer_exists": installer.exists(),
        "hook": TRUNK_GUARD_HOOK_REL,
        "hook_exists": hook.exists(),
        "expected": TRUNK_GUARD_HOOKS_REL,
        "actual": "",
        "command": command,
    }
    if not (root / ".git").exists():
        return status

    git_fn = pane_attr("git", git)
    git_probe = git_fn(["rev-parse", "--is-inside-work-tree"], root)
    if git_probe.returncode != 0:
        status.update({
            "ok": False,
            "setup_ok": False,
            "state": "blocked",
            "detail": git_probe.stderr.strip() or "git worktree probe failed",
        })
        return status
    status["supported"] = True

    if not installer.exists():
        status.update({
            "ok": False,
            "setup_ok": False,
            "state": "blocked",
            "detail": f"missing {TRUNK_GUARD_INSTALLER_REL}",
        })
        return status
    if not hook.exists():
        status.update({
            "ok": False,
            "setup_ok": False,
            "state": "blocked",
            "detail": f"missing {TRUNK_GUARD_HOOK_REL}",
        })
        return status

    configured = git_fn(["config", "--get", "core.hooksPath"], root)
    actual = configured.stdout.strip() if configured.returncode == 0 else ""
    actual_path = configured_hooks_path(root, actual)
    installed = bool(
        actual_path
        and path_compare_key(actual_path) == path_compare_key(hooks_dir)
    )
    status.update({
        "installed": installed,
        "ok": installed,
        "setup_ok": True,
        "state": "done" if installed else "todo",
        "actual": actual or "unset",
        "detail": (
            f"core.hooksPath={actual}"
            if installed
            else f"core.hooksPath={actual or 'unset'}; expected {TRUNK_GUARD_HOOKS_REL}"
        ),
    })
    return status


def parse_git_status_porcelain(raw: str) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    fields = [f for f in raw.split("\0") if f]
    i = 0
    while i < len(fields):
        item = fields[i]
        code = item[:2]
        path = item[3:] if len(item) > 3 else ""
        if code and code[0] in ("R", "C") and i + 1 < len(fields):
            old_path = fields[i + 1]
            entries.append({"code": code, "path": path, "old_path": old_path})
            i += 2
            continue
        entries.append({"code": code, "path": path})
        i += 1

    counts = collections.Counter()
    for entry in entries:
        code = entry["code"]
        if code == "??":
            counts["untracked"] += 1
        if "M" in code:
            counts["modified"] += 1
        if "A" in code:
            counts["added"] += 1
        if "D" in code:
            counts["deleted"] += 1
        if "R" in code:
            counts["renamed"] += 1
        if "C" in code:
            counts["copied"] += 1
        if "U" in code:
            counts["unmerged"] += 1
    return {
        "dirty": bool(entries),
        "dirty_total": len(entries),
        "counts": dict(counts),
        "entries": entries,
    }


def git_worktree_summary(root: Path) -> dict[str, Any]:
    git_fn = pane_attr("git", git)
    proc = git_fn(["status", "--porcelain=v1", "-z"], root, check=False)
    if proc.returncode != 0:
        return {
            "status_available": False,
            "dirty": False,
            "dirty_total": 0,
            "counts": {},
            "entries": [],
            "unmerged_paths": [],
            "merge_in_progress": False,
            "reason": proc.stderr.strip() or proc.stdout.strip() or "git status failed",
        }
    status = parse_git_status_porcelain(proc.stdout)
    entries = status.get("entries") or []
    status["status_available"] = True
    status["unmerged_paths"] = sorted(
        str(entry.get("path"))
        for entry in entries
        if entry.get("path") and "U" in str(entry.get("code") or "")
    )
    status["merge_in_progress"] = git_fn(["rev-parse", "--verify", "-q", "MERGE_HEAD"], root, check=False).returncode == 0
    return status


def worktree_allow_branches(config: dict[str, Any]) -> list[str]:
    raw = config.get("worktree_allow_branches", DEFAULT_WORKTREE_ALLOW_BRANCHES)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(branch) for branch in raw if str(branch).strip()]


def worktree_doctor_command(config: dict[str, Any], *, prune: bool = False, fetch: bool = True) -> list[str]:
    cmd = [str(config.get("python") or sys.executable), "tools/worktree_doctor.py"]
    if prune:
        cmd.append("--prune")
    if fetch:
        cmd.append("--fetch")
    master_ref = str(config.get("worktree_master_ref") or DEFAULT_WORKTREE_MASTER_REF)
    if master_ref != DEFAULT_WORKTREE_MASTER_REF:
        cmd.extend(["--master-ref", master_ref])
    for branch in worktree_allow_branches(config):
        cmd.extend(["--allow-branch", branch])
    return cmd


def collect_worktree_doctor(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    master_ref = str(config.get("worktree_master_ref") or DEFAULT_WORKTREE_MASTER_REF)
    try:
        import worktree_doctor  # type: ignore

        sigs = worktree_doctor.collect(str(root), master_ref, fetch=False)
        plan = worktree_doctor.make_plan(
            sigs,
            master_ref,
            allow_branches=worktree_allow_branches(config),
        )
        prune = list(plan.get("prune") or [])
        blocked = list(plan.get("blocked") or [])
        retained = list(plan.get("retained") or [])
        return {
            "available": True,
            "master_ref": master_ref,
            "total": len(sigs),
            "converged": bool(plan.get("converged")),
            "needs_human": bool(plan.get("needs_human")),
            "primary_offtrack": bool(plan.get("primary_offtrack")),
            "keeper": plan.get("keeper"),
            "prune_count": len(prune),
            "blocked_count": len(blocked),
            "retained_count": len(retained),
            "prune": prune[:10],
            "prune_omitted": max(0, len(prune) - 10),
            "blocked": blocked[:10],
            "blocked_omitted": max(0, len(blocked) - 10),
            "retained": retained[:10],
            "commands": {
                "inspect": display_command(worktree_doctor_command(config, prune=False, fetch=True)),
                "prune": display_command(worktree_doctor_command(config, prune=True, fetch=True)),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "commands": {
                "inspect": display_command(worktree_doctor_command(config, prune=False, fetch=True)),
            },
        }


def git_mutation_guard(root: Path) -> dict[str, Any]:
    summary_fn = pane_attr("git_worktree_summary", git_worktree_summary)
    worktree = summary_fn(root)
    unmerged_paths = [str(path) for path in worktree.get("unmerged_paths") or []]
    blockers: list[str] = []
    if worktree.get("merge_in_progress"):
        blockers.append("merge is in progress")
    if unmerged_paths:
        blockers.append(f"worktree has {len(unmerged_paths)} unmerged path(s)")
    return {
        "blocked": bool(blockers),
        "blockers": blockers,
        "reason": "; ".join(blockers),
        "worktree": {
            "status_available": worktree.get("status_available", False),
            "dirty": worktree.get("dirty", False),
            "dirty_total": worktree.get("dirty_total", 0),
            "counts": worktree.get("counts") or {},
            "unmerged_paths": unmerged_paths,
            "merge_in_progress": bool(worktree.get("merge_in_progress")),
        },
    }


def normalize_repo_path(root: Path, raw: str, *, allow_dir: bool = False) -> str:
    if not raw or not str(raw).strip():
        raise ValueError("empty path")
    root = root.resolve()
    candidate = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repo: {raw}") from exc
    if ".git" in rel.parts:
        raise ValueError(f"path enters .git: {raw}")
    if resolved.exists() and resolved.is_dir() and not allow_dir:
        raise ValueError(f"path is a directory; pass explicit files instead: {raw}")
    return rel.as_posix()


def split_z(raw: str) -> list[str]:
    return [p for p in raw.split("\0") if p]


def staged_paths(root: Path) -> list[str]:
    return split_z(git_stdout(["diff", "--cached", "--name-only", "-z"], root))


def path_status(root: Path, paths: list[str]) -> list[dict[str, str]]:
    if not paths:
        return []
    return parse_git_status_porcelain(
        git_stdout(["status", "--porcelain=v1", "-z", "--", *paths], root)
    )["entries"]


def commit_plan(
    config: dict[str, Any],
    *,
    paths: list[str],
    message: str,
    apply: bool = False,
    allow_dir: bool = False,
) -> dict[str, Any]:
    root = Path(config["root"]).resolve()
    selected = sorted(dict.fromkeys(
        normalize_repo_path(root, p, allow_dir=allow_dir) for p in paths
    ))
    if not selected:
        return {
            "schema": COMMIT_SCHEMA,
            "ok": False,
            "applied": False,
            "reason": "no paths selected",
        }
    if not message or not message.strip():
        return {
            "schema": COMMIT_SCHEMA,
            "ok": False,
            "applied": False,
            "paths": selected,
            "reason": "commit message is required",
        }

    git_fn = pane_attr("git", git)
    summary_fn = pane_attr("git_worktree_summary", git_worktree_summary)
    before_staged = staged_paths(root)
    foreign_before = [p for p in before_staged if p not in selected]
    status_before = path_status(root, selected)
    merge_in_progress = summary_fn(root).get("merge_in_progress", False)
    unmerged_selected = sorted(
        {str(entry.get("path")) for entry in status_before if "U" in str(entry.get("code") or "")}
    )
    changed_selected = sorted({e["path"] for e in status_before})
    commands = [
        ["git", "add", "--", *selected],
        ["git", "commit", "-s", "-m", message.strip()],
    ]
    base: dict[str, Any] = {
        "schema": COMMIT_SCHEMA,
        "generated_utc": iso_now(),
        "ok": False,
        "applied": False,
        "root": str(root),
        "paths": selected,
        "message": message.strip(),
        "status": status_before,
        "changed_selected": changed_selected,
        "merge_in_progress": merge_in_progress,
        "preexisting_staged": before_staged,
        "foreign_staged": foreign_before,
        "commands": [display_command(c) for c in commands],
    }
    if merge_in_progress:
        return {
            **base,
            "reason": "merge is in progress; finish or abort the merge before using the pane commit helper",
        }
    if unmerged_selected:
        return {
            **base,
            "unmerged_paths": unmerged_selected,
            "reason": "selected paths contain unresolved merge conflicts",
        }
    if foreign_before:
        return {
            **base,
            "reason": (
                "refusing because unrelated paths are already staged; "
                "commit or unstage them before using the pane commit helper"
            ),
        }
    if not changed_selected and not any(p in before_staged for p in selected):
        return {**base, "reason": "none of the selected paths have changes"}
    if not apply:
        return {**base, "ok": True, "reason": "dry run; pass --apply to stage and commit"}

    add_proc = git_fn(["add", "--", *selected], root)
    after_staged = staged_paths(root)
    foreign_after = [p for p in after_staged if p not in selected]
    if add_proc.returncode != 0:
        return {
            **base,
            "reason": add_proc.stderr.strip() or add_proc.stdout.strip() or "git add failed",
            "git_add_returncode": add_proc.returncode,
        }
    if foreign_after:
        return {
            **base,
            "preexisting_staged": before_staged,
            "post_add_staged": after_staged,
            "foreign_staged": foreign_after,
            "reason": "refusing because staging selected paths exposed foreign staged paths",
        }
    if not after_staged:
        return {**base, "reason": "selected paths produced no staged changes"}
    commit_proc = git_fn(["commit", "-s", "-m", message.strip()], root)
    if commit_proc.returncode != 0:
        return {
            **base,
            "post_add_staged": after_staged,
            "reason": commit_proc.stderr.strip() or commit_proc.stdout.strip() or "git commit failed",
            "git_commit_returncode": commit_proc.returncode,
        }
    sha = git_stdout(["rev-parse", "--short=12", "HEAD"], root).strip()
    return {
        **base,
        "ok": True,
        "applied": True,
        "commit": sha,
        "post_add_staged": after_staged,
        "stdout": commit_proc.stdout.strip()[-2000:],
        "stderr": commit_proc.stderr.strip()[-2000:],
    }


def dirty_group_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized in PANE_SOURCE_PATHS or normalized.startswith("tools/fleet_control_pane"):
        return "tools/fleet-control-pane"
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return "root"
    if len(parts) == 1:
        return "root"
    if parts[0] == "tools":
        if len(parts) == 2 and parts[1].startswith("_") and parts[1].endswith("_proof.png"):
            return "tools/proofs"
        return f"tools/{parts[1]}"
    if parts[0] == "fak":
        if len(parts) >= 3 and parts[1] in {"experiments", "internal"}:
            return f"fak/{parts[1]}/{parts[2]}"
        return "fak"
    return parts[0]


def dirty_group_commit_subject(group: str) -> str:
    return DIRTY_GROUP_SUBJECTS.get(group, f"TODO: describe {group} changes")


def dirty_commit_plan(
    entries: list[dict[str, str]],
    config: dict[str, Any],
    *,
    merge_in_progress: bool = False,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        path = entry.get("path")
        if not path:
            continue
        groups.setdefault(dirty_group_key(path), []).append(entry)
    out = []
    python = str(config.get("python") or sys.executable)
    for key in sorted(groups):
        group_entries = groups[key]
        paths = sorted(dict.fromkeys(str(entry.get("path")) for entry in group_entries if entry.get("path")))
        has_unmerged = any("U" in str(entry.get("code") or "") for entry in group_entries)
        blocked = merge_in_progress or has_unmerged
        command: list[str] = [
            python,
            "tools/fleet_control_pane.py",
            "commit",
            "--dirty-group",
            key,
        ]
        subject = dirty_group_commit_subject(key)
        command.extend(["-m", subject])
        item: dict[str, Any] = {
            "group": key,
            "count": len(paths),
            "paths": paths,
            "suggested_subject": subject,
        }
        if blocked:
            item["blocked"] = True
            item["reason"] = (
                "finish or abort the merge before using pane commit"
                if merge_in_progress
                else "resolve merge conflicts before using pane commit"
            )
        else:
            item["command"] = display_command(command)
        out.append(item)
    return {"groups": out, "count": len(out)}


def dirty_group_selection(config: dict[str, Any], groups: list[str]) -> dict[str, Any]:
    requested = [group for group in groups if group]
    collect_git_fn = pane_attr("collect_git", collect_git)
    git_status = collect_git_fn(config)
    plan = git_status.get("dirty_plan") or {}
    by_group = {
        str(group.get("group")): group
        for group in plan.get("groups") or []
        if group.get("group")
    }
    missing = [group for group in requested if group not in by_group]
    paths: list[str] = []
    for group in requested:
        paths.extend(str(path) for path in (by_group.get(group) or {}).get("paths") or [])
    return {
        "ok": not missing,
        "requested": requested,
        "missing": missing,
        "available": sorted(by_group),
        "paths": sorted(dict.fromkeys(paths)),
    }


def collect_git(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    if not (root / ".git").exists() or shutil.which("git") is None:
        return {"available": False, "reason": "not a git repo or git not on PATH"}
    git_fn = pane_attr("git", git)
    summary_fn = pane_attr("git_worktree_summary", git_worktree_summary)
    branch = git_fn(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
    head = git_fn(["rev-parse", "--short=12", "HEAD"], root).stdout.strip()
    status = summary_fn(root)
    entries = status.pop("entries", [])
    merge_in_progress = bool(status.get("merge_in_progress"))
    out: dict[str, Any] = {
        "available": True,
        "branch": branch,
        "head": head,
        **status,
        "dirty_sample": entries[:25],
        "dirty_omitted": max(0, len(entries) - 25),
        "dirty_plan": dirty_commit_plan(entries, config, merge_in_progress=merge_in_progress),
    }
    try:
        import safe_ff_sync  # type: ignore

        info = safe_ff_sync.assess(
            str(root),
            str(config.get("git_remote", "origin")),
            branch,
            do_fetch=False,
        )
        out["safe_ff"] = {
            k: info.get(k)
            for k in ("ok", "state", "reason", "target_ref", "write_count")
            if k in info
        }
        out["safe_ff"]["divergent_count"] = len(info.get("divergent", []) or [])
    except Exception as exc:  # git may have no remote, no upstream, etc.
        out["safe_ff"] = {"ok": None, "state": "unavailable", "reason": str(exc)}
    out["worktrees"] = collect_worktree_doctor(config)
    return out


def sync_plan(config: dict[str, Any], *, fetch: bool = False, apply: bool = False) -> dict[str, Any]:
    root = Path(config["root"])
    remote = str(config.get("git_remote", "origin"))
    guard_fn = pane_attr("git_mutation_guard", git_mutation_guard)
    mutation_guard = guard_fn(root) if apply else None
    cmd = [
        str(config.get("python") or sys.executable),
        "tools/safe_ff_sync.py",
        "apply" if apply else "check",
        "--repo",
        str(root),
        "--remote",
        remote,
    ]
    if fetch:
        cmd.append("--fetch")
    cmd.append("--json")
    base: dict[str, Any] = {
        "schema": SYNC_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "fetch": fetch,
        "remote": remote,
        "commands": [display_command(cmd)],
    }
    try:
        import safe_ff_sync  # type: ignore

        branch = safe_ff_sync.current_branch(str(root))
        info = safe_ff_sync.assess(str(root), remote, branch, fetch)
        info["branch"] = branch
        applied = False
        if apply:
            if mutation_guard and mutation_guard.get("blocked"):
                return {
                    **base,
                    "ok": False,
                    "state": info.get("state"),
                    "branch": branch,
                    "info": info,
                    "applied": False,
                    "sync_blocked": True,
                    "blockers": mutation_guard.get("blockers") or [],
                    "worktree": mutation_guard.get("worktree") or {},
                    "reason": mutation_guard.get("reason") or "worktree is not safe for sync",
                }
            if info.get("state") == "in-sync":
                applied = False
            elif info.get("state") == "behind" and info.get("ok"):
                info["new_head"] = safe_ff_sync.apply_ff(str(root), remote, branch, info)
                info["applied"] = True
                applied = True
            else:
                info["applied"] = False
        state = str(info.get("state") or "unknown")
        ok = state == "in-sync" or (state == "behind" and bool(info.get("ok")))
        if apply and state == "behind":
            ok = applied
        return {
            **base,
            "ok": ok,
            "state": state,
            "branch": branch,
            "info": info,
            "applied": applied,
            "reason": info.get("reason", ""),
        }
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "state": "unavailable",
            "applied": False,
            "reason": str(exc),
        }


def publish_plan(
    config: dict[str, Any],
    *,
    fetch: bool = True,
    apply: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    root = Path(config["root"])
    remote = str(config.get("git_remote", "origin"))
    git_fn = pane_attr("git", git)
    summary_fn = pane_attr("git_worktree_summary", git_worktree_summary)
    ahead_fn = pane_attr("ahead_commits", ahead_commits)
    base: dict[str, Any] = {
        "schema": PUBLISH_SCHEMA,
        "generated_utc": iso_now(),
        "apply": apply,
        "fetch": fetch,
        "allow_dirty": allow_dirty,
        "remote": remote,
        "applied": False,
    }
    try:
        import safe_ff_sync  # type: ignore

        branch = safe_ff_sync.current_branch(str(root))
        info = safe_ff_sync.assess(str(root), remote, branch, fetch)
        info["branch"] = branch
        push_cmd = ["git", "push", remote, f"HEAD:{branch}"]
        ahead = ahead_fn(root, remote, branch) if info.get("state") == "ahead" else {
            "target": f"{remote}/{branch}",
            "count": 0,
            "shown": 0,
            "limit": 20,
            "commits": [],
            "ok": True,
            "reason": "",
        }
        worktree = summary_fn(root)
        worktree_doc = {
            "status_available": worktree.get("status_available", False),
            "dirty": worktree.get("dirty", False),
            "dirty_total": worktree.get("dirty_total", 0),
            "counts": worktree.get("counts") or {},
            "unmerged_paths": worktree.get("unmerged_paths") or [],
            "merge_in_progress": worktree.get("merge_in_progress", False),
        }
        doc: dict[str, Any] = {
            **base,
            "branch": branch,
            "state": info.get("state"),
            "info": info,
            "ahead_commits": ahead,
            "worktree": worktree_doc,
            "commands": [display_command(push_cmd)],
            "reason": info.get("reason", ""),
        }
        if info.get("state") == "in-sync":
            return {**doc, "ok": True, "reason": "branch already published"}
        if info.get("state") != "ahead":
            return {
                **doc,
                "ok": False,
                "reason": info.get("reason") or f"publish requires ahead state, got {info.get('state')}",
            }
        blockers: list[str] = []
        if worktree_doc["merge_in_progress"]:
            blockers.append("merge is in progress")
        if worktree_doc["unmerged_paths"]:
            blockers.append(f"worktree has {len(worktree_doc['unmerged_paths'])} unmerged path(s)")
        if int(worktree_doc["dirty_total"] or 0) > 0 and not allow_dirty:
            blockers.append(f"worktree has {worktree_doc['dirty_total']} dirty path(s)")
        if blockers:
            return {
                **doc,
                "ok": False,
                "publish_blocked": True,
                "blockers": blockers,
                "reason": "; ".join(blockers),
            }
        if not apply:
            return {
                **doc,
                "ok": True,
                "reason": "dry run; pass --apply to push the ahead commits",
            }
        proc = git_fn(["push", remote, f"HEAD:{branch}"], root)
        return {
            **doc,
            "ok": proc.returncode == 0,
            "applied": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-2000:],
            "stderr": proc.stderr.strip()[-2000:],
            "reason": proc.stderr.strip() or proc.stdout.strip() or ("published" if proc.returncode == 0 else "git push failed"),
        }
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "state": "unavailable",
            "reason": str(exc),
        }


def ahead_commits(root: Path, remote: str, branch: str, limit: int = 20) -> dict[str, Any]:
    git_fn = pane_attr("git", git)
    target = f"{remote}/{branch}"
    proc = git_fn(["log", f"--max-count={limit}", "--format=%H%x00%h%x00%s", f"{target}..HEAD"], root, check=False)
    commits: list[dict[str, str]] = []
    if proc.returncode != 0:
        return {
            "target": target,
            "count": 0,
            "shown": 0,
            "limit": limit,
            "commits": commits,
            "ok": False,
            "reason": (proc.stderr or proc.stdout).strip()[-1000:],
        }
    for line in proc.stdout.splitlines():
        parts = line.split("\0", 2)
        if len(parts) != 3:
            continue
        sha, short, subject = parts
        commits.append({"sha": sha, "short": short, "subject": subject})
    count_proc = git_fn(["rev-list", "--count", f"{target}..HEAD"], root, check=False)
    try:
        count = int((count_proc.stdout or "").strip())
    except ValueError:
        count = len(commits)
    return {
        "target": target,
        "count": count,
        "shown": len(commits),
        "limit": limit,
        "commits": commits,
        "ok": count_proc.returncode == 0,
        "reason": "" if count_proc.returncode == 0 else (count_proc.stderr or count_proc.stdout).strip()[-1000:],
    }
