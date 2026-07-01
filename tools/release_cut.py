#!/usr/bin/env python3
"""Mechanical release-cut helper for fleet.

This automates the deterministic middle of `/release`: read the release
decision, bump VERSION, draft release notes, stage an explicit pathspec, guard
that staged set, and commit. It never pushes and never tags. Default mode is a
dry-run plan with no mutation.

Execute mode is intentionally conservative:

* takes the single-writer release lock, unless a parent release driver has
  already taken it and proves ownership with `--lock-already-held`;
* refuses dirty paths outside explicit `--include` paths;
* accepts `--from-manifest` as a structured source of explicit includes;
* stages only VERSION, the release note, and explicit includes;
* runs the staged-set guard before commit;
* leaves tag minting to `release_dry_run.py` + CI + the human/skill.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
VERSION_SUBJECT_RE = re.compile(r"^v\d+\.\d+\.\d+:")
CC_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?!?:\s*(?P<rest>.*)$")
GENERATION_ORDER = ("gen/now", "gen/next", "gen/second-next", "gen/future")


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.output or "") + f"\n(timed out after {timeout}s)"
    except OSError as exc:
        return 127, str(exc)


def repo_root() -> Path:
    code, out = run(["git", "rev-parse", "--show-toplevel"])
    return Path(out.strip()) if code == 0 and out.strip() else Path(__file__).resolve().parent.parent


def load_json(cmd: list[str], *, root: Path, timeout: int = 600) -> tuple[dict | None, str, int]:
    code, out = run(cmd, cwd=root, timeout=timeout)
    try:
        return json.loads(out), out, code
    except json.JSONDecodeError:
        return None, out, code


def shell_quote_path(path: str) -> str:
    return path.replace("\\", "/")


def paths_from_bump_report(report: dict) -> list[str]:
    out: list[str] = []
    for target in (report.get("targets") or {}).values():
        if not isinstance(target, dict) or target.get("skipped"):
            continue
        path = target.get("path")
        if isinstance(path, str) and path:
            out.append(shell_quote_path(path))
        # A target may nest a per-file list (e.g. release_bump's `install_docs`
        # target pin-bumps INSTALL.md under `files[].path`); pull those into the
        # commit pathspec too, or the bumped doc is left dirty-but-uncommitted
        # and trips release_cut's foreign-dirty guard.
        for f in target.get("files") or []:
            fp = f.get("path") if isinstance(f, dict) else None
            if isinstance(fp, str) and fp:
                out.append(shell_quote_path(fp))
    return dedupe(out)


def dedupe(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        p = shell_quote_path(path)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def default_headline(version: str, level: str, themes: list[str]) -> str:
    label = {"major": "Major", "minor": "Minor", "patch": "Patch"}.get(level, "Release")
    if themes:
        return f"{label} release for {', '.join(themes[:4])}"
    return f"{label} release v{version}"


def _yaml_str(value: str) -> str:
    return json.dumps(value)


def ascii_clean(value: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
    out = value
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.encode("ascii", errors="replace").decode("ascii")


def generation_from_text(text: str) -> str | None:
    for raw in (text or "").splitlines():
        key, sep, value = raw.strip().partition(":")
        if sep != ":" or key.strip().lower() != "generation":
            continue
        label = value.strip().lower()
        if label in GENERATION_ORDER:
            return label
        if label in {"now", "next", "second-next", "future"}:
            return f"gen/{label}"
        return None
    return None


def commit_generation(commit: dict) -> str | None:
    raw = str(commit.get("generation") or commit.get("gen") or "").strip().lower()
    if raw in GENERATION_ORDER:
        return raw
    if raw in {"now", "next", "second-next", "future"}:
        return f"gen/{raw}"
    for label in commit.get("labels") or []:
        text = str(label).strip().lower()
        if text in GENERATION_ORDER:
            return text
    return generation_from_text(str(commit.get("body") or ""))


def subject_with_generation(commit: dict) -> str:
    subject = ascii_clean(str(commit.get("subject") or "").strip())
    gen = commit_generation(commit)
    if subject and gen:
        return f"{subject} [{gen}]"
    return subject


def generation_counts(commits: list[dict]) -> dict[str, int]:
    counts = {gen: 0 for gen in GENERATION_ORDER}
    for commit in commits:
        subject = str(commit.get("subject") or "").strip()
        if not subject or VERSION_SUBJECT_RE.match(subject):
            continue
        gen = commit_generation(commit)
        if gen in counts:
            counts[gen] += 1
    return {gen: n for gen, n in counts.items() if n}


def grouped_subjects(commits: list[dict]) -> tuple[dict[str, list[str]], list[str]]:
    buckets: dict[str, list[str]] = {}
    order: list[str] = []
    for commit in commits:
        subject = subject_with_generation(commit)
        if not subject or VERSION_SUBJECT_RE.match(subject):
            continue
        m = CC_RE.match(subject)
        scope = (m.group("scope") or "").strip() if m else ""
        key = scope or "general"
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(subject)
    return buckets, order


def render_notes(version: str, *, date: str, level: str, themes: list[str],
                 headline: str, commits: list[dict]) -> str:
    headline = ascii_clean(headline)
    themes = [ascii_clean(t) for t in themes]
    buckets, order = grouped_subjects(commits)
    generations = generation_counts(commits)
    highlights: list[str] = []
    for theme in themes:
        if theme in buckets and buckets[theme]:
            highlights.append(buckets[theme][0])
    if not highlights:
        highlights = [s for scope in order for s in buckets[scope]][:6]

    lines = [
        "---",
        f"version: {version}",
        f"date: {date}",
        f"headline: {_yaml_str(headline)}",
        "themes:",
    ]
    for theme in themes[:8]:
        lines.append(f"  - {_yaml_str(theme)}")
    if generations:
        lines.append("generations:")
        for gen in GENERATION_ORDER:
            if generations.get(gen):
                lines.append(f"  {gen}: {generations[gen]}")
    lines.append("highlights:")
    for item in highlights[:8]:
        lines.append(f"  - {_yaml_str(item)}")
    lines.extend([
        "---",
        "",
        f"**TL;DR** - Automatic {level} release cut from {len(commits)} commit(s) since the previous tag.",
        "",
    ])
    if generations:
        lines.extend(["## Generation", ""])
        for gen in GENERATION_ORDER:
            if generations.get(gen):
                lines.append(f"- {gen}: {generations[gen]} commit(s)")
        lines.append("")
    for scope in order:
        lines.append(f"## {scope}")
        lines.append("")
        for subject in buckets[scope]:
            lines.append(f"- {subject}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def append_manifest_notes(notes: str, envelope: dict) -> str:
    producer = envelope.get("producer") or {}
    lines = [
        notes.rstrip(),
        "",
        "## Release manifest",
        "",
    ]
    if producer.get("skill") or producer.get("run_id") or producer.get("packet_path"):
        bits = [
            str(producer.get("skill") or "unknown"),
            str(producer.get("run_id") or "unknown"),
            str(producer.get("packet_path") or "unknown"),
        ]
        lines.append(f"- Producer: {' / '.join(bits)}")
    lines.append(f"- Staged manifest paths: {len(envelope.get('staged_paths') or [])}")
    if envelope.get("auto_deferred_paths"):
        lines.append("- Auto-deferred manifest paths: " + ", ".join(envelope["auto_deferred_paths"]))
    if envelope.get("commits_section_lines"):
        lines.extend(["", "### Manifest commits", ""])
        lines.extend(str(line) for line in envelope["commits_section_lines"])
    return "\n".join(lines).rstrip() + "\n"


def promotion_witness_from_env(env: dict[str, str] | None = None) -> dict | None:
    env = env or dict(os.environ)
    keys = {
        "source_branch": "FAK_RELEASE_SOURCE_BRANCH",
        "source_sha": "FAK_RELEASE_SOURCE_SHA",
        "target_branch": "FAK_RELEASE_TARGET_BRANCH",
        "target_sha": "FAK_RELEASE_TARGET_SHA",
        "source_range": "FAK_RELEASE_SOURCE_RANGE",
        "source_ci": "FAK_RELEASE_SOURCE_CI",
    }
    witness = {key: ascii_clean(str(env.get(var) or "").strip()) for key, var in keys.items()}
    if not any(witness.values()):
        return None
    return witness


def append_promotion_notes(notes: str, witness: dict) -> str:
    lines = [
        notes.rstrip(),
        "",
        "## Promotion source",
        "",
    ]
    if witness.get("source_branch") or witness.get("source_sha"):
        lines.append(f"- Source: {witness.get('source_branch') or '(unknown)'} {witness.get('source_sha') or '(unknown)'}")
    if witness.get("target_branch") or witness.get("target_sha"):
        lines.append(
            f"- Target before promotion: {witness.get('target_branch') or '(unknown)'} "
            f"{witness.get('target_sha') or '(unknown)'}"
        )
    if witness.get("source_range"):
        lines.append(f"- Promoted source range: {witness['source_range']}")
    if witness.get("source_ci"):
        lines.append(f"- Source CI: {witness['source_ci']}")
    return "\n".join(lines).rstrip() + "\n"


def dirty_paths(root: Path) -> list[str]:
    code, out = run(["git", "status", "--porcelain", "-z"], cwd=root)
    if code != 0:
        return []
    chunks = [c for c in out.split("\0") if c]
    paths: list[str] = []
    i = 0
    while i < len(chunks):
        item = chunks[i]
        path = item[3:] if len(item) > 3 else item
        paths.append(shell_quote_path(path))
        if item.startswith(("R ", "RM", "C ")):
            i += 1
        i += 1
    return paths


def upstream_state(root: Path) -> dict:
    code, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    branch = branch.strip()
    if code != 0 or branch == "HEAD":
        return {"state": "detached", "ok_for_release_cut": False, "reason": "not on a branch"}
    code, upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=root)
    if code != 0:
        return {
            "state": "no-upstream",
            "branch": branch,
            "upstream": None,
            "ahead": 0,
            "behind": 0,
            "ok_for_release_cut": True,
            "reason": "branch has no upstream",
        }
    upstream = upstream.strip()
    code, counts = run(["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"], cwd=root)
    if code != 0:
        return {"state": "unreadable", "branch": branch, "upstream": upstream,
                "ok_for_release_cut": False, "reason": counts.strip()[-300:]}
    parts = counts.split()
    ahead = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0
    behind = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    if ahead and behind:
        state = "diverged"
        ok = False
        reason = f"{branch} and {upstream} have diverged"
    elif behind:
        state = "behind"
        ok = False
        reason = f"{branch} is behind {upstream} by {behind} commit(s)"
    elif ahead:
        state = "ahead"
        ok = True
        reason = f"{branch} is ahead of {upstream} by {ahead} commit(s)"
    else:
        state = "in-sync"
        ok = True
        reason = f"{branch} is in sync with {upstream}"
    return {
        "state": state,
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "ok_for_release_cut": ok,
        "reason": reason,
    }


def unwind_failed_release_commit(root: Path, commit_sha: str | None) -> dict:
    """Soft-reset the release commit iff HEAD still points at that commit.

    The dry-run witness runs after the release commit exists because it must
    adjudicate committed bytes. If the witness fails, keeping that commit would
    leave VERSION half-cut. The HEAD check prevents this helper from rewinding a
    peer commit that landed while the witness was running in a hot worktree.
    """
    if not commit_sha:
        return {"ok": False, "skipped": True, "reason": "no release commit sha"}
    code, head = run(["git", "rev-parse", "HEAD"], cwd=root)
    current = head.strip()
    if code != 0 or not current:
        return {"ok": False, "skipped": True, "reason": "could not resolve HEAD", "detail": head.strip()[-300:]}
    if current != commit_sha:
        return {
            "ok": False,
            "skipped": True,
            "reason": "HEAD changed after release commit; refusing to reset",
            "head": current,
            "release_commit": commit_sha,
        }
    code, parent = run(["git", "rev-parse", f"{commit_sha}^"], cwd=root)
    parent_sha = parent.strip()
    if code != 0 or not parent_sha:
        return {"ok": False, "skipped": True, "reason": "could not resolve release commit parent", "detail": parent.strip()[-300:]}
    code, out = run(["git", "reset", "--soft", parent_sha], cwd=root)
    return {
        "ok": code == 0,
        "skipped": False,
        "mode": "soft",
        "from": commit_sha,
        "to": parent_sha,
        "detail": out.strip()[-300:] if code != 0 else None,
    }


def build_plan(root: Path, *, version: str | None, level: str | None,
               themes: list[str], headline: str | None, date: str,
               includes: list[str], force: bool, allow_hold: bool,
               limit_commits: int, require_ci_green: bool = False,
               from_manifest: str | None = None) -> dict:
    context, raw_context, context_code = load_json([
        sys.executable, str(root / "tools" / "release_context.py"),
        "--no-previews", "--limit-commits", str(limit_commits),
    ], root=root)
    if context is None or context_code != 0:
        return {"ok": False, "aborted": "release_context failed", "detail": raw_context[-500:]}

    decide_cmd = [
        sys.executable, str(root / "tools" / "release_decide.py"),
        "--json", "--limit-commits", str(limit_commits),
    ]
    if require_ci_green:
        decide_cmd.append("--require-ci-green")
    decision, raw_decision, decision_code = load_json(decide_cmd, root=root)
    if decision is None:
        return {"ok": False, "aborted": "release_decide emitted no JSON", "detail": raw_decision[-500:]}
    if decision_code not in (0, 2):
        return {"ok": False, "aborted": "release_decide failed", "detail": raw_decision[-500:]}
    if decision.get("decision") != "release" and allow_hold and not version:
        return {
            "ok": True,
            "held": True,
            "dry_run": True,
            "aborted": None,
            "decision": decision,
            "reason": decision.get("reason"),
        }
    if decision.get("decision") != "release" and not force and not version:
        return {
            "ok": False,
            "aborted": "release_decide held",
            "decision": decision,
        }

    target = (version or decision.get("next_version") or "").lstrip("v")
    if not SEMVER_RE.match(target):
        return {"ok": False, "aborted": f"invalid target version {target!r}", "decision": decision}
    cut_level = level or decision.get("level") or "patch"
    cut_themes = themes or [str(t) for t in (decision.get("themes") or [])]
    cut_headline = headline or default_headline(target, cut_level, cut_themes)

    bump, raw_bump, bump_code = load_json([
        sys.executable, str(root / "tools" / "release_bump.py"),
        target, "--dry-run",
    ], root=root)
    if bump is None or bump_code != 0:
        return {"ok": False, "aborted": "release_bump dry-run failed", "detail": raw_bump[-500:]}

    manifest_envelope: dict | None = None
    manifest_includes: list[str] = []
    if from_manifest:
        manifest_envelope, raw_manifest, manifest_code = load_json([
            sys.executable, str(root / "tools" / "release_manifest.py"),
            "consume", from_manifest, "--json",
        ], root=root)
        if manifest_envelope is None:
            return {"ok": False, "aborted": "release manifest unreadable", "detail": raw_manifest[-500:]}
        if manifest_code != 0 or not manifest_envelope.get("ok"):
            return {
                "ok": False,
                "aborted": "release manifest refused",
                "manifest": manifest_envelope,
                "detail": raw_manifest[-500:],
            }
        manifest_includes = [str(p) for p in manifest_envelope.get("staged_paths") or []]

    notes_rel = f"docs/releases/v{target}.md"
    include_paths = dedupe([shell_quote_path(p) for p in includes + manifest_includes])
    notes_preview = render_notes(
        target,
        date=date,
        level=cut_level,
        themes=cut_themes,
        headline=cut_headline,
        commits=context.get("commits_since_tag") or [],
    )
    if manifest_envelope:
        notes_preview = append_manifest_notes(notes_preview, manifest_envelope)
    promotion_witness = promotion_witness_from_env()
    if promotion_witness:
        notes_preview = append_promotion_notes(notes_preview, promotion_witness)
    paths = dedupe(paths_from_bump_report(bump) + [notes_rel] + include_paths)
    return {
        "ok": True,
        "version": target,
        "tag": f"v{target}",
        "level": cut_level,
        "themes": cut_themes,
        "headline": cut_headline,
        "date": date,
        "paths": paths,
        "include_paths": include_paths,
        "notes_file": notes_rel,
        "notes_exists": (root / notes_rel).exists(),
        "manifest": manifest_envelope,
        "promotion_witness": promotion_witness,
        "decision": decision,
        "context": {
            "last_tag": context.get("last_tag"),
            "latest_any_tag": context.get("latest_any_tag"),
            "n_commits": len(context.get("commits_since_tag") or []),
        },
        "upstream": upstream_state(root),
        "bump_report": bump,
        "notes_preview": notes_preview,
    }


def verify_release_lock(root: Path) -> tuple[int, str]:
    return run([sys.executable, str(root / "tools" / "release_lock.py"), "verify"], cwd=root)


def execute_plan(root: Path, plan: dict, *, includes: list[str], overwrite_notes: bool,
                 skip_dry_run: bool, ttl: int, allow_stale_upstream: bool,
                 lock_already_held: bool = False) -> dict:
    upstream = upstream_state(root)
    plan["upstream"] = upstream
    if not allow_stale_upstream and not upstream.get("ok_for_release_cut"):
        plan.update(ok=False, aborted="upstream is not fresh", detail=upstream.get("reason"))
        return plan

    paths = [shell_quote_path(p) for p in plan["paths"]]
    # Any path the cut plans to commit is allowed to be dirty going in — that's
    # VERSION, the release note, the explicit --includes, AND the bump targets
    # (e.g. INSTALL.md install-pin bumps) carried in plan["paths"].
    allowed_dirty = set(shell_quote_path(p) for p in includes)
    allowed_dirty.update(paths)
    foreign_dirty = [
        p for p in dirty_paths(root)
        if p not in allowed_dirty and p not in {"VERSION", plan["notes_file"]}
    ]
    if foreign_dirty:
        plan.update(ok=False, aborted="dirty paths outside release cut", foreign_dirty=foreign_dirty)
        return plan

    lock_acquired = False
    try:
        if lock_already_held:
            code, out = verify_release_lock(root)
            if code != 0:
                plan.update(ok=False, aborted="parent release lock not held", detail=out[-500:])
                return plan
            plan["release_lock"] = {"ok": True, "held_by_parent": True}
        else:
            lock_cmd = [sys.executable, str(root / "tools" / "release_lock.py"), "acquire", "--ttl", str(ttl)]
            for path in paths:
                lock_cmd.extend(["--snapshot", path])
            code, out = run(lock_cmd, cwd=root)
            if code != 0:
                plan.update(ok=False, aborted="could not acquire release lock", detail=out[-500:])
                return plan
            lock_acquired = True

        bump, raw_bump, bump_code = load_json([
            sys.executable, str(root / "tools" / "release_bump.py"), plan["version"],
        ], root=root)
        if bump is None or bump_code != 0:
            plan.update(ok=False, aborted="release_bump failed", detail=raw_bump[-500:])
            return plan

        notes_path = root / plan["notes_file"]
        if notes_path.exists() and not overwrite_notes:
            notes_written = False
        else:
            notes_path.parent.mkdir(parents=True, exist_ok=True)
            notes_path.write_text(plan["notes_preview"], encoding="utf-8", newline="")
            notes_written = True

        code, out = run(["git", "add", "--", *paths], cwd=root)
        if code != 0:
            plan.update(ok=False, aborted="git add failed", detail=out[-500:])
            return plan

        guard_cmd = [sys.executable, str(root / "tools" / "release_lock.py"), "guard"]
        for path in paths:
            guard_cmd.extend(["--allow", path])
        code, out = run(guard_cmd, cwd=root)
        if code != 0:
            plan.update(ok=False, aborted="release guard failed", detail=out[-500:])
            return plan

        subject = f"{plan['tag']}: {plan['headline']}"
        code, out = run(["git", "commit", "-m", subject, "--", *paths], cwd=root)
        if code != 0:
            plan.update(ok=False, aborted="git commit failed", detail=out[-500:])
            return plan
        code, sha = run(["git", "rev-parse", "HEAD"], cwd=root)
        plan["commit_sha"] = sha.strip() if code == 0 else None

        if not skip_dry_run:
            verdict, raw, dry_code = load_json([
                sys.executable, str(root / "tools" / "release_dry_run.py"),
                "--json", "HEAD",
            ], root=root, timeout=1200)
            plan["dry_run_verdict"] = verdict
            if verdict is None or dry_code != 0 or not verdict.get("ok"):
                unwind = unwind_failed_release_commit(root, plan.get("commit_sha"))
                if unwind.get("ok"):
                    plan["commit_sha"] = None
                plan.update(
                    ok=False,
                    aborted="release_dry_run failed on release commit",
                    detail=raw[-500:],
                    release_commit_unwind=unwind,
                )
                return plan

        plan.update(ok=True, executed=True, notes_written=notes_written, bump_report=bump)
        return plan
    finally:
        if lock_acquired:
            run([sys.executable, str(root / "tools" / "release_lock.py"), "release"], cwd=root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute a mechanical fleet release cut.")
    parser.add_argument("--version", default=None, help="target X.Y.Z; default from release_decide")
    parser.add_argument("--level", choices=["patch", "minor", "major"], default=None)
    parser.add_argument("--theme", action="append", default=[], help="release-note theme; repeatable")
    parser.add_argument("--headline", default=None)
    parser.add_argument("--date", default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--include", action="append", default=[], help="extra explicit path to stage; repeatable")
    parser.add_argument("--from-manifest", default=None,
                        help="consume release-manifest.json and stage only its dirty, shipped paths")
    parser.add_argument("--limit-commits", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="allow --version despite release_decide hold")
    parser.add_argument("--allow-hold", action="store_true",
                        help="dry-run exits 0 when release_decide says hold; for CI smoke only")
    parser.add_argument("--require-ci-green", action="store_true",
                        help="require release_decide to confirm a green main ci.yml base")
    parser.add_argument("--execute", action="store_true", help="mutate files and create the release commit")
    parser.add_argument("--overwrite-notes", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--allow-stale-upstream", action="store_true",
                        help="allow execute while behind/diverged/detached (not recommended)")
    parser.add_argument("--lock-already-held", action="store_true",
                        help="execute under a parent-held release lock; verifies ownership and does not release it")
    parser.add_argument("--ttl", type=int, default=1800)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    root = repo_root()
    plan = build_plan(
        root,
        version=args.version,
        level=args.level,
        themes=args.theme,
        headline=args.headline,
        date=args.date,
        includes=args.include,
        from_manifest=args.from_manifest,
        force=args.force,
        allow_hold=args.allow_hold,
        limit_commits=args.limit_commits,
        require_ci_green=args.require_ci_green,
    )
    if plan.get("held") and args.execute:
        plan.update(ok=False, aborted="release_decide held; execute refused")
    elif plan.get("ok") and args.execute:
        plan = execute_plan(
            root,
            plan,
            includes=plan.get("include_paths") or args.include,
            overwrite_notes=args.overwrite_notes,
            skip_dry_run=args.skip_dry_run,
            ttl=args.ttl,
            allow_stale_upstream=args.allow_stale_upstream,
            lock_already_held=args.lock_already_held,
        )
    else:
        plan["dry_run"] = True

    if args.as_json:
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        verdict = "OK" if plan.get("ok") else "REFUSED"
        print(f"[release-cut] {verdict} {plan.get('tag', '')}")
        if plan.get("aborted"):
            print(f"  aborted: {plan['aborted']}", file=sys.stderr)
        for path in plan.get("paths") or []:
            print(f"  path: {path}")
    return 0 if plan.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
