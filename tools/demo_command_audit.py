#!/usr/bin/env python3
"""Audit documented demo commands for stale package/script references.

The hosted-link guard catches broken public URLs. This guard catches the local
equivalent: docs that tell a user to run `go run ./cmd/<demo>` or a helper script
that no longer exists. It is static and network-free by default; it does not execute
the demos. Documented `make <target>` entries are checked against the Makefile
target list too.

Run from the repo root:

    python tools/demo_command_audit.py
    python tools/demo_command_audit.py --json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import demo_registry as dr

SCHEMA = "fak-demo-command-audit/1"
DEFAULT_DOCS = ("docs/run-the-demos.md", "docs/demos.html")

_ENV_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_ENV_VALUE = r"(?:\"[^\"]*\"|'[^']*'|[^\s`<>]+)"
_ENV_PREFIX = rf"(?:(?:{_ENV_IDENT})={_ENV_VALUE}\s+)*"
_TAIL = r"[^\r\n`<]*"

GO_CMD_RE = re.compile(
    rf"(?P<command>{_ENV_PREFIX}go\s+(?P<verb>run|test|build)\s+(?:[^\r\n`<]*?\s)?"
    rf"\./cmd/(?P<name>[A-Za-z0-9_-]+)/?{_TAIL})"
)
GO_C_CMD_RE = re.compile(
    rf"(?P<command>{_ENV_PREFIX}go\s+-C\s+(?P<dir>[^\s`<>]+)\s+"
    rf"(?P<verb>run|test|build)\s+(?:[^\r\n`<]*?\s)?\./cmd/(?P<name>[A-Za-z0-9_-]+)/?{_TAIL})"
)
SCRIPT_CMD_RE = re.compile(
    rf"(?P<command>(?:bash|sh)\s+(?P<path>tools/[A-Za-z0-9_./-]+\.sh){_TAIL})"
)
PY_TOOL_CMD_RE = re.compile(
    rf"(?P<command>python(?:3)?\s+(?P<path>tools/[A-Za-z0-9_./-]+\.py){_TAIL})"
)
MAKE_CMD_RE = re.compile(
    rf"(?P<command>make\s+(?P<target>[A-Za-z0-9_.-]+){_TAIL})"
)
MAKE_TARGET_RE = re.compile(r"^(?P<target>[A-Za-z0-9_.-]+)\s*:(?!=)")
BARE_INLINE_CMD_RE = re.compile(r"(?:<code>|`)\s*(?P<target>\./cmd/[A-Za-z0-9_-]+)\s*(?:</code>|`)")


@dataclass(frozen=True)
class CommandRef:
    source: str
    line: int
    kind: str
    target: str
    command: str
    go_dir: str = ""

    def row(self) -> dict[str, Any]:
        row = {
            "source": self.source,
            "line": self.line,
            "kind": self.kind,
            "target": self.target,
            "command": self.command.strip(),
        }
        if self.go_dir:
            row["go_dir"] = self.go_dir
        return row

    @property
    def loc(self) -> str:
        return f"{self.source}:{self.line}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_sources(workspace: Path) -> list[str]:
    sources = list(DEFAULT_DOCS)
    sources.extend(
        p.relative_to(workspace).as_posix()
        for p in sorted((workspace / "cmd").glob("*/README.md"))
    )
    return sources


def extract_command_refs(source: str, text: str) -> list[CommandRef]:
    refs: list[CommandRef] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in GO_CMD_RE.finditer(line):
            kind = "go-" + m.group("verb")
            refs.append(CommandRef(source, line_no, kind, f"cmd/{m.group('name')}", m.group("command")))
        for m in GO_C_CMD_RE.finditer(line):
            kind = "go-" + m.group("verb")
            refs.append(
                CommandRef(
                    source,
                    line_no,
                    kind,
                    f"cmd/{m.group('name')}",
                    m.group("command"),
                    go_dir=m.group("dir").strip("\"'"),
                )
            )
        for m in SCRIPT_CMD_RE.finditer(line):
            refs.append(CommandRef(source, line_no, "shell-script", m.group("path"), m.group("command")))
        for m in PY_TOOL_CMD_RE.finditer(line):
            refs.append(CommandRef(source, line_no, "python-tool", m.group("path"), m.group("command")))
        for m in MAKE_CMD_RE.finditer(line):
            refs.append(CommandRef(source, line_no, "make-target", m.group("target"), m.group("command")))
    return refs


def make_targets(workspace: Path) -> set[str]:
    path = workspace / "Makefile"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    targets: set[str] = set()
    for line in text.splitlines():
        if line.startswith(("\t", " ", "#")):
            continue
        m = MAKE_TARGET_RE.match(line)
        if m:
            targets.add(m.group("target"))
    return targets


def _inside_workspace(workspace: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


def _path_defect(workspace: Path, ref: CommandRef) -> str | None:
    if ref.go_dir and ref.go_dir not in {".", workspace.name}:
        return f"{ref.loc} unsupported go -C directory in demo command: {ref.go_dir}"

    parts = Path(ref.target).parts
    if ".." in parts:
        return f"{ref.loc} {ref.kind} target escapes audited tree: {ref.target}"
    target = workspace / ref.target
    if not _inside_workspace(workspace, target):
        return f"{ref.loc} {ref.kind} target resolves outside workspace: {ref.target}"

    if ref.kind in {"go-build", "go-run"}:
        if not target.is_dir():
            return f"{ref.loc} {ref.kind.replace('-', ' ')} target missing: {ref.target}"
        if not (target / "main.go").is_file():
            return f"{ref.loc} {ref.kind.replace('-', ' ')} target has no main.go: {ref.target}"
        return None

    if ref.kind == "go-test":
        if not target.is_dir():
            return f"{ref.loc} go test target missing: {ref.target}"
        if not any(target.glob("*_test.go")):
            return f"{ref.loc} go test target has no *_test.go: {ref.target}"
        return None

    if ref.kind in {"shell-script", "python-tool"}:
        if not target.is_file():
            return f"{ref.loc} {ref.kind} target missing: {ref.target}"
        return None

    if ref.kind == "make-target":
        if ref.target not in make_targets(workspace):
            return f"{ref.loc} make target missing from Makefile: {ref.target}"
        return None

    return f"{ref.loc} unclassified command kind: {ref.kind}"


def documented_go_run_packages(refs: list[CommandRef]) -> set[str]:
    return {
        ref.target.removeprefix("cmd/")
        for ref in refs
        if ref.kind == "go-run" and ref.target.startswith("cmd/")
    }


def browser_demo_coverage_defects(refs: list[CommandRef], demos: tuple[dr.Demo, ...] = dr.DEMOS) -> list[str]:
    documented = documented_go_run_packages(refs)
    defects: list[str] = []
    for demo in demos:
        if demo.name not in documented:
            defects.append(f"browser demo registry entry is not documented with a go run command: cmd/{demo.name}")
    return defects


def collect(workspace: Path, *, sources: list[str] | None = None) -> dict[str, Any]:
    workspace = workspace.resolve()
    source_list = sources if sources is not None else default_sources(workspace)
    refs: list[CommandRef] = []
    defects: list[str] = []
    read_sources: list[str] = []

    for source in source_list:
        path = workspace / source
        if not _inside_workspace(workspace, path):
            defects.append(f"source resolves outside workspace: {source}")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            defects.append(f"read {source}: {exc}")
            continue
        read_sources.append(source)
        refs.extend(extract_command_refs(source, text))
        defects.extend(bare_cmd_defects(source, text))

    if not refs:
        defects.append("no documented demo commands found in audited sources")

    defects.extend(d for ref in refs if (d := _path_defect(workspace, ref)))
    coverage_defects = browser_demo_coverage_defects(refs) if sources is None else []
    defects.extend(coverage_defects)
    counts = Counter(ref.kind for ref in refs)
    documented_packages = documented_go_run_packages(refs)
    ok = not defects
    if ok:
        verdict, finding = "OK", "demo_command_refs_clean"
        reason = f"{len(refs)} documented command(s) audited across {len(read_sources)} source(s); all targets resolve"
        next_action = "rerun after adding or changing demo documentation"
    else:
        verdict, finding = "ACTION", "demo_command_ref_debt"
        reason = f"{len(defects)} documented-command defect(s) across {len(read_sources)} source(s)"
        next_action = "fix stale cmd package or tools/ script references in demo docs"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "sources": read_sources,
        "command_count": len(refs),
        "counts": dict(sorted(counts.items())),
        "documented_go_run_packages": sorted(documented_packages),
        "browser_demo_coverage": sorted(
            demo.name for demo in dr.DEMOS if demo.name in documented_packages
        ),
        "commands": [ref.row() for ref in refs],
        "defects": defects,
    }


def bare_cmd_defects(source: str, text: str) -> list[str]:
    defects: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in BARE_INLINE_CMD_RE.finditer(line):
            target = m.group("target")
            defects.append(
                f"{source}:{line_no} bare cmd path in inline code: {target}; "
                f"use `go run {target}` for runnable demo docs"
            )
    return defects


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"demo-command-audit: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"  next: {payload['next_action']}",
    ]
    if payload.get("counts"):
        counts = ", ".join(f"{k}={v}" for k, v in payload["counts"].items())
        lines.append(f"  commands: {counts}")
    if payload.get("defects"):
        lines.append("")
        lines.append("defects:")
        for defect in payload["defects"]:
            lines.append(f"  - {defect}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit documented demo command references.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument(
        "--source",
        action="append",
        default=None,
        help="relative doc path to audit; repeatable (default: demo docs + cmd/*/README.md)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON payload")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, sources=args.source)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
