#!/usr/bin/env python3
"""Run the deterministic headless demo witnesses.

The command/link audits prove docs point at real packages and scripts. The browser
HTTP smoke proves pages mount and APIs respond. This tool covers the other half of
docs/run-the-demos.md: the model-free headless commands that claim CI-usable,
deterministic invariants.

Run from the repo root:

    python tools/demo_headless_smoke.py
    python tools/demo_headless_smoke.py --only unseedemo-selfcheck --json
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fak-demo-headless-smoke/1"
DEFAULT_DOC = "docs/run-the-demos.md"
HEADLESS_START = "## 2. Headless"
HEADLESS_END = "## 3. "
GO_RUN_RE = re.compile(r"^\s*(go\s+run\s+\./cmd/[^\r\n#]+)")
README_DOCS_GLOB = "cmd/*/README.md"
EXITING_PACKAGES = {
    "a2ademo",
    "causalbench",
    "ctxplandemo",
    "cxlpooldemo",
    "deletioncert",
    "hwcachedemo",
    "memqdemo",
    "poisonedmcpdemo",
}
HEADLESS_FLAGS = {"-selfcheck", "-print", "-json", "-out", "-report", "-profiles"}
INTERACTIVE_OR_MODEL_PACKAGES = {"simpledemo"}


@dataclass(frozen=True)
class Witness:
    name: str
    argv: tuple[str, ...]
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...] = ()
    doc_command: str = ""
    json_outputs: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        if self.doc_command:
            return self.doc_command
        return " ".join(self.argv)


WITNESSES: tuple[Witness, ...] = (
    Witness("guarddemo-selfcheck", ("go", "run", "./cmd/guarddemo", "-selfcheck"), ("reproduced the documented safety-floor",)),
    Witness("turntaxdemo-selfcheck", ("go", "run", "./cmd/turntaxdemo", "-selfcheck"), ("reproduced the documented turn-tax",)),
    Witness("tokendemo-selfcheck", ("go", "run", "./cmd/tokendemo", "-selfcheck"), ("reproduced the documented ledger",)),
    Witness("unseedemo-selfcheck", ("go", "run", "./cmd/unseedemo", "-selfcheck"), ("7/7 invariants reproduced",)),
    Witness("dropindemo-selfcheck", ("go", "run", "./cmd/dropindemo", "-selfcheck"), ("universal-recipe invariants reproduced",)),
    Witness("ctxdemo-print", ("go", "run", "./cmd/ctxdemo", "-print"), ("fak · ctxdemo", "fak-win")),
    Witness("ctxdemo-print-json", ("go", "run", "./cmd/ctxdemo", "-print", "-json"), ('"scenario"', '"tokens"')),
    Witness("ctxdemo-bars-deep-research", ("go", "run", "./cmd/ctxdemo", "-bars", "-scenario", "deep-research"), ("deep-research", "fak makes the model re-read")),
    Witness("tokendemo-print-prefilter", ("go", "run", "./cmd/tokendemo", "-print", "-suite", "prefilter-bad-calls"), ("fak keeps 1,452 tokens out",)),
    Witness("tokendemo-print-reread", ("go", "run", "./cmd/tokendemo", "-print", "-suite", "reread-same-file"), ("3 re-reads served from cache",)),
    Witness("tokendemo-json", ("go", "run", "./cmd/tokendemo", "-json"), ('"suite": "prefilter-bad-calls"',)),
    Witness("unseedemo-print", ("go", "run", "./cmd/unseedemo", "-print"), ("ACT 1", "write-time evict")),
    Witness("unseedemo-json", ("go", "run", "./cmd/unseedemo", "-json"), ('"witness"', '"frames"')),
    Witness("guarddemo-print", ("go", "run", "./cmd/guarddemo", "-print"), ("WITHOUT fak", "WITH fak")),
    Witness("dropindemo-print", ("go", "run", "./cmd/dropindemo", "-print"), ("fak guard -- codex", "one static binary")),
    Witness("guarddemo-print-happy", ("go", "run", "./cmd/guarddemo", "-print", "-scenario", "turntax-happy"), ("0 breaches",)),
    Witness("turntaxdemo-print", ("go", "run", "./cmd/turntaxdemo", "-print"), ("tuned SOTA agent: 5 forced round-trips",)),
    Witness("turntaxdemo-print-happy", ("go", "run", "./cmd/turntaxdemo", "-print", "-suite", "turntax-happy"), ("0 forced round-trips",)),
    Witness("ctxdemo-bars", ("go", "run", "./cmd/ctxdemo", "-bars"), ("fak makes the model re-read",)),
    Witness("tokendemo-print", ("go", "run", "./cmd/tokendemo", "-print"), ("fak keeps 1,452 tokens out",)),
    Witness("a2ademo", ("go", "run", "./cmd/a2ademo"), ("a2ademo: OK",)),
    Witness("ctxplandemo-selfcheck", ("go", "run", "./cmd/ctxplandemo", "-selfcheck"), ("O(1) view planned under budget",)),
    Witness("hwcachedemo", ("go", "run", "./cmd/hwcachedemo"), ("prefill tokens saved by demoting instead of evicting",)),
    Witness("cxlpooldemo", ("go", "run", "./cmd/cxlpooldemo"), ("prefill tokens saved", "memory deduplicated")),
    Witness(
        "cxlpooldemo-profiles",
        ("go", "run", "./cmd/cxlpooldemo", "-profiles", "cmd/cxlpooldemo/calibration.example.json"),
        ("CXL switch-pool calibration", "memory deduplicated"),
    ),
    Witness("memqdemo", ("go", "run", "./cmd/memqdemo"), ("sealed span was refused",)),
    Witness(
        "memqdemo-report",
        ("go", "run", "./cmd/memqdemo", "-report", "{tmp}/memqdemo-report.json"),
        ("report written:",),
        doc_command="go run ./cmd/memqdemo -report memqdemo-report.json",
        json_outputs=("{tmp}/memqdemo-report.json",),
    ),
    Witness("poisonedmcpdemo", ("go", "run", "./cmd/poisonedmcpdemo"), ("QUARANTINED", "tool not allow-listed")),
    Witness("poisonedmcpdemo-json", ("go", "run", "./cmd/poisonedmcpdemo", "-json"), ('"results"', '"descriptions"')),
    Witness("causalbench-selfcheck", ("go", "run", "./cmd/causalbench", "-selfcheck"), ("causally evicted exactly the dependent",)),
    Witness(
        "causalbench-out",
        ("go", "run", "./cmd/causalbench", "-selfcheck", "-out", "{tmp}/causalbench-witness.json"),
        ("wrote",),
        doc_command="go run ./cmd/causalbench -selfcheck -out causalbench-witness.json",
        json_outputs=("{tmp}/causalbench-witness.json",),
    ),
    Witness("deletioncert-selfcheck", ("go", "run", "./cmd/deletioncert", "-selfcheck"), ("provable-deletion certificate minted",)),
    Witness(
        "deletioncert-out",
        ("go", "run", "./cmd/deletioncert", "-selfcheck", "-out", "{tmp}/deletioncert.json"),
        ("wrote",),
        doc_command="go run ./cmd/deletioncert -selfcheck -out deletioncert.json",
        json_outputs=("{tmp}/deletioncert.json",),
    ),
    Witness(
        "deletioncert-isolation-bench",
        ("go", "run", "./cmd/deletioncert", "-isolation-bench"),
        ("baseline leaked:", "benchmark passed"),
    ),
    Witness(
        "deletioncert-isolation-bench-out",
        ("go", "run", "./cmd/deletioncert", "-isolation-bench", "-out", "{tmp}/isolation-result.json"),
        ("wrote result to",),
        doc_command="go run ./cmd/deletioncert -isolation-bench -out isolation-result.json",
        json_outputs=("{tmp}/isolation-result.json",),
    ),
    Witness(
        "deletioncert-isolation-bench-seed",
        ("go", "run", "./cmd/deletioncert", "-isolation-bench", "-seed", "42"),
        ("seed: 42",),
        doc_command="go run ./cmd/deletioncert -isolation-bench -seed 42",
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def witness_map() -> dict[str, Witness]:
    return {w.name: w for w in WITNESSES}


def normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


def documented_headless_go_commands(workspace: Path, doc: str = DEFAULT_DOC) -> list[str]:
    path = workspace / doc
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    in_section = False
    commands: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if line.startswith(HEADLESS_START):
            in_section = True
            continue
        if in_section and line.startswith(HEADLESS_END):
            break
        if not in_section:
            continue
        m = GO_RUN_RE.match(line)
        if not m:
            continue
        command = normalize_command(m.group(1))
        if command not in seen:
            seen.add(command)
            commands.append(command)
    return commands


def _command_package(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for part in parts:
        if part.startswith("./cmd/"):
            return part[len("./cmd/"):]
    return ""


def _command_flags(command: str) -> set[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return {part for part in parts if part.startswith("-")}


def is_deterministic_readme_command(command: str) -> bool:
    pkg = _command_package(command)
    if not pkg or pkg in INTERACTIVE_OR_MODEL_PACKAGES:
        return False
    flags = _command_flags(command)
    if flags & HEADLESS_FLAGS:
        return True
    return pkg in EXITING_PACKAGES


def documented_readme_go_commands(workspace: Path) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()
    for path in sorted(workspace.glob(README_DOCS_GLOB)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = GO_RUN_RE.match(line)
            if not m:
                continue
            command = normalize_command(m.group(1))
            if command in seen or not is_deterministic_readme_command(command):
                continue
            seen.add(command)
            commands.append(command)
    return commands


def documented_witness_go_commands(workspace: Path) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()
    for command in documented_headless_go_commands(workspace) + documented_readme_go_commands(workspace):
        if command not in seen:
            seen.add(command)
            commands.append(command)
    return commands


def registry_command_defects(workspace: Path, witnesses: tuple[Witness, ...] = WITNESSES) -> list[str]:
    documented = set(documented_witness_go_commands(workspace))
    registered = {w.command for w in witnesses}
    defects: list[str] = []
    for command in sorted(documented - registered):
        defects.append(f"documented headless command has no witness: {command}")
    for command in sorted(registered - documented):
        defects.append(f"headless witness command is not documented in {DEFAULT_DOC}: {command}")
    return defects


def select_witnesses(names: list[str] | None) -> tuple[list[Witness], list[str]]:
    if not names:
        return list(WITNESSES), []
    by_name = witness_map()
    unknown = [name for name in names if name not in by_name]
    return [by_name[name] for name in names if name in by_name], unknown


def check_output(witness: Witness, rc: int, out: str) -> list[str]:
    defects: list[str] = []
    if rc != 0:
        defects.append(f"exit status {rc}")
    for needle in witness.must_contain:
        if needle not in out:
            defects.append(f"missing required output: {needle!r}")
    for needle in witness.must_not_contain:
        if needle in out:
            defects.append(f"forbidden output present: {needle!r}")
    return defects


def _format_arg(arg: str, tmp: Path | None) -> str:
    if tmp is None:
        return arg
    return arg.replace("{tmp}", str(tmp))


def _needs_tmp(witness: Witness) -> bool:
    return any("{tmp}" in arg for arg in witness.argv) or any("{tmp}" in p for p in witness.json_outputs)


def _check_json_outputs(witness: Witness, tmp: Path | None) -> list[str]:
    defects: list[str] = []
    for raw in witness.json_outputs:
        path = Path(_format_arg(raw, tmp))
        if not path.is_file():
            defects.append(f"missing JSON output: {path}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            defects.append(f"invalid JSON output {path}: {exc}")
    return defects


def run_witness(workspace: Path, witness: Witness, timeout_s: float) -> dict[str, Any]:
    tmp_ctx = tempfile.TemporaryDirectory(prefix="fak-demo-headless-") if _needs_tmp(witness) else None
    tmp = Path(tmp_ctx.name) if tmp_ctx is not None else None
    argv = [_format_arg(arg, tmp) for arg in witness.argv]
    try:
        r = subprocess.run(
            argv,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        defects = check_output(witness, r.returncode, out)
        defects.extend(_check_json_outputs(witness, tmp))
        tail = "\n".join(out.strip().splitlines()[-5:])
        return {
            "name": witness.name,
            "ok": not defects,
            "command": witness.command,
            "argv": " ".join(argv),
            "rc": r.returncode,
            "defects": defects,
            "tail": tail,
        }
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or "") + (exc.stderr or "")) if isinstance(exc.stdout, str) else ""
        return {
            "name": witness.name,
            "ok": False,
            "command": witness.command,
            "argv": " ".join(argv),
            "rc": 124,
            "defects": [f"timeout after {timeout_s:.1f}s"],
            "tail": "\n".join(out.strip().splitlines()[-5:]),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": witness.name,
            "ok": False,
            "command": witness.command,
            "argv": " ".join(argv),
            "rc": 1,
            "defects": [f"runner error: {exc}"],
            "tail": "",
        }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def collect(workspace: Path, *, names: list[str] | None = None, timeout_s: float = 120.0) -> dict[str, Any]:
    workspace = workspace.resolve()
    witnesses, unknown = select_witnesses(names)
    rows: list[dict[str, Any]] = []
    defects = [f"unknown witness: {name}" for name in unknown]
    registry = registry_command_defects(workspace) if names is None else []
    defects.extend(registry)

    if not registry:
        for witness in witnesses:
            row = run_witness(workspace, witness, timeout_s)
            rows.append(row)
            defects.extend(f"{witness.name}: {d}" for d in row.get("defects", []))

    ok = not defects
    if ok:
        verdict, finding = "OK", "headless_demo_witnesses_clean"
        reason = f"{len(rows)} deterministic headless demo witness(es) reproduced documented output"
        next_action = "rerun after changing headless demo commands, data, or invariants"
    else:
        verdict, finding = "ACTION", "headless_demo_witness_debt"
        reason = f"{len(defects)} headless-demo witness defect(s)"
        next_action = "fix the failing demo command or update the documented invariant with evidence"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "registry": {
            "documented": documented_witness_go_commands(workspace),
            "registered": sorted(w.command for w in WITNESSES),
            "defects": registry,
        },
        "witnesses": rows,
        "defects": defects,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"demo-headless-smoke: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"  next: {payload['next_action']}",
        "",
        "witnesses:",
    ]
    for row in payload.get("witnesses", []):
        status = "OK" if row.get("ok") else "FAIL"
        lines.append(f"  {status:4} {row.get('name')} :: {row.get('command')}")
        if row.get("tail") and not row.get("ok"):
            lines.append("       tail: " + str(row["tail"]).replace("\n", "\n             "))
    if payload.get("defects"):
        lines.append("")
        lines.append("defects:")
        for defect in payload["defects"]:
            lines.append(f"  - {defect}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run deterministic headless demo witnesses.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--only", action="append", default=None, help="witness name to run; repeatable")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-witness timeout in seconds")
    ap.add_argument("--list", action="store_true", help="list witness names and exit")
    ap.add_argument("--json", action="store_true", help="emit JSON payload")
    args = ap.parse_args(argv)

    if args.list:
        for witness in WITNESSES:
            print(f"{witness.name}\t{witness.command}")
        return 0

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, names=args.only, timeout_s=args.timeout)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
