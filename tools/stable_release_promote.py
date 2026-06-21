#!/usr/bin/env python3
"""Replay-safe stable release promotion for fleet.

This is the executable spine behind `.claude/skills/stable-release/SKILL.md`:
it reads the stable-release gate context, writes the evidence file, and creates
the annotated `stable/<codename>` tag. It does not bump VERSION or create a new
rolling release.

Default mode is local and conservative: write the evidence file and local tag,
but do not push. Use `--push` only when performing the real promotion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

CODENAME_RE = re.compile(r"^\d{4}-\d{2}-[a-z][a-z0-9-]{2,20}$")


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


def git(root: Path, args: list[str], *, timeout: int = 600) -> tuple[int, str]:
    return run(["git", *args], cwd=root, timeout=timeout)


def stable_tag(codename: str) -> str:
    return f"stable/{codename}"


def tag_sha(root: Path, tag: str) -> str | None:
    code, out = git(root, ["rev-list", "-n1", tag])
    return out.strip() if code == 0 and out.strip() else None


def load_context(args: argparse.Namespace, root: Path) -> dict:
    cmd = [
        sys.executable,
        str(root / "tools" / "stable_release_context.py"),
        "--codename",
        args.codename,
        "--window-days",
        str(args.window_days),
        "--json",
    ]
    if args.from_tag:
        cmd.extend(["--from", args.from_tag])
    if args.skip_tests:
        cmd.append("--skip-tests")
    if args.skip_dos:
        cmd.append("--skip-dos")
    if args.skip_ci:
        cmd.append("--skip-ci")
    if args.force_promote_rationale:
        cmd.append("--force-promote")
    code, out = run(cmd, cwd=root, timeout=900)
    if code not in (0, 1):
        raise RuntimeError(out.strip()[-500:] or f"stable_release_context.py exited {code}")
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"stable_release_context.py did not emit JSON: {exc}") from exc


def _yaml_str(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def render_evidence_frontmatter(payload: dict, *, rationale: str | None = None,
                                generated_at_utc: str | None = None) -> str:
    generated_at_utc = generated_at_utc or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat()
    lines = [
        "codename: " + _yaml_str(payload.get("codename")),
        "underlying_version: " + _yaml_str(payload.get("candidate_tag")),
        "candidate_sha: " + _yaml_str(payload.get("candidate_sha")),
        "promoted_at_utc: " + _yaml_str(generated_at_utc),
        "window_days: " + _yaml_str(payload.get("window_days")),
        "forced: " + _yaml_str(bool((payload.get("summary") or {}).get("forced") or rationale)),
    ]
    previous = payload.get("previous_stable") or {}
    if previous:
        lines.append("previous_stable: " + _yaml_str(previous.get("tag")))
    if rationale:
        lines.append("force_promote_rationale: " + _yaml_str(rationale))
    lines.append("gate:")
    for name, row in sorted((payload.get("gate") or {}).items()):
        if not isinstance(row, dict):
            continue
        bits = [f"pass: {_yaml_str(bool(row.get('pass')))}"]
        if "advisory" in row:
            bits.append(f"advisory: {_yaml_str(bool(row.get('advisory')))}")
        if "verdict" in row:
            bits.append(f"verdict: {_yaml_str(row.get('verdict'))}")
        lines.append(f"  {name}: {{{', '.join(bits)}}}")
    return "\n".join(lines) + "\n"


def render_evidence_file(payload: dict, *, rationale: str | None = None,
                         generated_at_utc: str | None = None) -> str:
    candidate = payload.get("candidate_tag")
    sha = payload.get("candidate_sha")
    codename = payload.get("codename")
    gate_json = json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        "---",
        render_evidence_frontmatter(
            payload, rationale=rationale, generated_at_utc=generated_at_utc
        ).rstrip(),
        "---",
        "",
        f"# Stable promotion - {codename}",
        "",
        f"Promotes `{candidate}` at commit `{sha}`.",
        "",
        "## Known-good evidence",
        "",
    ]
    for name, row in sorted((payload.get("gate") or {}).items()):
        if not isinstance(row, dict):
            continue
        status = "PASS" if row.get("pass") else "FAIL"
        extra = ""
        if row.get("advisory"):
            extra = " (advisory)"
        lines.append(f"- `{name}`: {status}{extra}")
    if rationale:
        lines.extend(["", "## Force-promote rationale", "", rationale])
    lines.extend([
        "",
        "## Rollback target",
        "",
        f"`git checkout {stable_tag(str(codename))}`",
        "",
        "## Gate evidence",
        "",
        "```json",
        gate_json,
        "```",
        "",
    ])
    return "\n".join(lines)


def _frontmatter_text(text: str) -> str:
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---", 4)
    return text[4:end] if end != -1 else ""


def evidence_matches(path: Path, payload: dict) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return evidence_text_matches(text, payload)


def evidence_text_matches(text: str, payload: dict) -> bool:
    fm = _frontmatter_text(text)
    required = {
        "codename": str(payload.get("codename")),
        "underlying_version": str(payload.get("candidate_tag")),
        "candidate_sha": str(payload.get("candidate_sha")),
    }
    for key, value in required.items():
        pattern = re.compile(rf"^{re.escape(key)}:\s+['\"]?{re.escape(value)}['\"]?\s*$", re.M)
        if not pattern.search(fm):
            return False
    return True


def evidence_committed_at_head(root: Path, evidence_rel: Path, payload: dict) -> tuple[bool, str | None]:
    rel = str(evidence_rel).replace("\\", "/")
    code, text = git(root, ["show", f"HEAD:{rel}"])
    if code != 0:
        return False, f"{rel} is not committed at HEAD"
    if not evidence_text_matches(text, payload):
        return False, f"{rel} at HEAD does not match this stable promotion"
    return True, None


def promote_from_context(root: Path, payload: dict, *, rationale: str | None = None,
                         dry_run: bool = False, write_tag: bool = True,
                         push: bool = False) -> dict:
    codename = str(payload.get("codename") or "")
    candidate_tag = payload.get("candidate_tag")
    candidate_sha = payload.get("candidate_sha")
    idem = payload.get("idempotency") or {}
    summary = payload.get("summary") or {}
    errors: list[str] = []
    skips: list[str] = []
    tag_name = stable_tag(codename)

    if not CODENAME_RE.match(codename):
        errors.append("invalid_codename")
    if not candidate_tag or not candidate_sha:
        errors.append("no_candidate")
    if payload.get("tag_collision"):
        errors.append(f"tag_collision:{payload['tag_collision']}")
    if idem.get("tag_exists") and idem.get("tag_matches_candidate") is False:
        errors.append("stable_codename_points_at_different_commit")
    if not summary.get("all_green") and not rationale:
        blockers = summary.get("blockers") or []
        errors.append("gate_red:" + "; ".join(str(b) for b in blockers))

    evidence_rel = Path(str(idem.get("evidence_path") or f"docs/stable-releases/{codename}.md"))
    evidence_path = root / evidence_rel

    result = {
        "ok": False,
        "codename": codename,
        "candidate_tag": candidate_tag,
        "candidate_sha": candidate_sha,
        "stable_tag": tag_name,
        "evidence_path": str(evidence_rel).replace("\\", "/"),
        "dry_run": dry_run,
        "tag_created": False,
        "tag_pushed": False,
        "evidence_written": False,
        "idempotent_skips": skips,
        "errors": errors,
    }
    if errors:
        return result

    if dry_run:
        result["ok"] = True
        skips.extend(["dry_run_no_evidence_write", "dry_run_no_tag"])
        return result

    if evidence_path.exists() and evidence_matches(evidence_path, payload):
        skips.append("evidence_file_already_exists")
    else:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(render_evidence_file(payload, rationale=rationale),
                                 encoding="utf-8", newline="")
        result["evidence_written"] = True

    if push:
        committed, reason = evidence_committed_at_head(root, evidence_rel, payload)
        if not committed:
            result["errors"].append("evidence_not_committed_before_tag_push:" + str(reason))
            return result

    if write_tag:
        existing_sha = tag_sha(root, tag_name)
        if existing_sha:
            if existing_sha.lower() != str(candidate_sha).lower():
                result["errors"].append("stable_tag_exists_on_different_commit")
                return result
            skips.append("tag_already_exists_same_sha")
        else:
            code, out = git(root, [
                "tag", "-a", tag_name, str(candidate_sha),
                "-m", f"{tag_name} - promoted from {candidate_tag}",
            ])
            if code != 0:
                result["errors"].append("tag_create_failed:" + out.strip()[-300:])
                return result
            result["tag_created"] = True
    else:
        skips.append("tag_skipped_by_caller")

    if push:
        code, out = git(root, ["push", "origin", tag_name], timeout=300)
        if code != 0:
            result["errors"].append("tag_push_failed:" + out.strip()[-300:])
            return result
        result["tag_pushed"] = True

    result["ok"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a fleet rolling tag to stable/<codename>.")
    parser.add_argument("--codename", required=True)
    parser.add_argument("--from", dest="from_tag", default=None)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--force-promote-rationale", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-dos", action="store_true")
    parser.add_argument("--skip-ci", action="store_true")
    parser.add_argument("--skip-tag", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    root = repo_root()
    try:
        payload = load_context(args, root)
        result = promote_from_context(
            root,
            payload,
            rationale=args.force_promote_rationale,
            dry_run=args.dry_run,
            write_tag=not args.skip_tag,
            push=args.push,
        )
    except Exception as exc:
        result = {"ok": False, "errors": [str(exc)]}

    if args.as_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        verdict = "OK" if result.get("ok") else "REFUSED"
        print(f"[stable-release] {verdict} {result.get('stable_tag', '')}")
        if result.get("evidence_path"):
            print(f"  evidence: {result['evidence_path']}")
        for skip in result.get("idempotent_skips") or []:
            print(f"  skip: {skip}")
        for error in result.get("errors") or []:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
