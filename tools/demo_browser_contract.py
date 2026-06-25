#!/usr/bin/env python3
"""Audit browser-demo port and reverse-proxy metadata.

The dynamic browser smoke proves each browser demo can serve under a base path when
given an arbitrary loopback port. This static audit catches metadata drift around the
documented defaults: Go `defaultAddr`, local run docs, public-page commands, `PORT=`
examples, base-path env examples, and nginx `proxy_pass` snippets.

Run from the repo root:

    python tools/demo_browser_contract.py
    python tools/demo_browser_contract.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

import demo_registry as dr  # noqa: E402

SCHEMA = "fak-demo-browser-contract/1"
RUN_DOC = "docs/run-the-demos.md"
PUBLIC_DOC = "docs/demos.html"
DEFAULT_ADDR_RE = re.compile(r'const\s+defaultAddr\s*=\s*"(?P<addr>127\.0\.0\.1:(?P<port>\d+))"')


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def source_default_addr(workspace: Path, demo: dr.Demo) -> tuple[str, int]:
    text = _read(workspace / "cmd" / demo.name / "main.go")
    m = DEFAULT_ADDR_RE.search(text)
    if not m:
        return "", 0
    return m.group("addr"), int(m.group("port"))


def source_helper_defects(source_text: str, demo: dr.Demo) -> list[str]:
    defects: list[str] = []
    checks = {
        f'demoui.BasePathFlag(flag.CommandLine, "{demo.base_path}")': "shared base-path flag helper",
        "demoui.MountWithBasePath(": "shared base-path mount helper",
        "demoui.ListenAddr(": "shared PORT/listen helper",
        "demoui.LocalURL(": "shared startup URL helper",
    }
    for needle, label in checks.items():
        if needle not in source_text:
            defects.append(f"{demo.name}: main.go missing {label}: {needle}")
    if 'flag.String("base-path"' in source_text:
        defects.append(f"{demo.name}: main.go defines base-path flag directly; use demoui.BasePathFlag")
    if "func listenAddr(" in source_text:
        defects.append(f"{demo.name}: main.go defines local listenAddr; use demoui.ListenAddr")
    if "os.Getenv(demoui.DemoBasePathEnv)" in source_text:
        defects.append(f"{demo.name}: main.go reads {demo.name} base-path env directly; use demoui.BasePathFlag")
    return defects


def demo_contract_defects(workspace: Path, demo: dr.Demo, run_doc_text: str, public_doc_text: str) -> list[str]:
    defects: list[str] = []
    if demo.default_port <= 0:
        defects.append(f"{demo.name}: DEMOS registry missing default_port")
        return defects

    source_text = _read(workspace / "cmd" / demo.name / "main.go")
    addr, port = source_default_addr(workspace, demo)
    want_addr = f"127.0.0.1:{demo.default_port}"
    if addr != want_addr or port != demo.default_port:
        defects.append(f"{demo.name}: main.go defaultAddr {addr or '<missing>'}, want {want_addr}")
    defects.extend(source_helper_defects(source_text, demo))

    checks = {
        f"http://127.0.0.1:{demo.default_port}": "local loopback URL",
        f"PORT={demo.default_port} ./{demo.name}": "PORT launch example",
        f"FAK_DEMO_BASE_PATH={demo.base_path}": "base-path env example",
        f"location {demo.base_path}/": "nginx location",
        f"proxy_pass http://127.0.0.1:{demo.default_port};": "nginx proxy_pass",
    }
    for needle, label in checks.items():
        if needle not in run_doc_text:
            defects.append(f"{demo.name}: {RUN_DOC} missing {label}: {needle}")

    public_checks = {
        f"go run ./cmd/{demo.name}": "public local go-run command",
        f"http://127.0.0.1:{demo.default_port}": "public local loopback URL",
    }
    for needle, label in public_checks.items():
        if needle not in public_doc_text:
            defects.append(f"{demo.name}: {PUBLIC_DOC} missing {label}: {needle}")

    bare_public_cmd = f"<code>./cmd/{demo.name}</code>"
    if bare_public_cmd in public_doc_text:
        defects.append(f"{demo.name}: {PUBLIC_DOC} has bare command path, use go run form: {bare_public_cmd}")

    readme = _read(workspace / "cmd" / demo.name / "README.md")
    env_cmd = f"FAK_DEMO_BASE_PATH={demo.base_path} go run ./cmd/{demo.name}"
    if env_cmd not in readme:
        defects.append(f"{demo.name}: README missing base-path env command: {env_cmd}")
    return defects


def collect(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    run_doc_text = _read(workspace / RUN_DOC)
    public_doc_text = _read(workspace / PUBLIC_DOC)
    rows: list[dict[str, Any]] = []
    defects: list[str] = []

    if not run_doc_text:
        defects.append(f"read {RUN_DOC}: missing or empty")
    if not public_doc_text:
        defects.append(f"read {PUBLIC_DOC}: missing or empty")

    for demo in dr.DEMOS:
        addr, port = source_default_addr(workspace, demo)
        row_defects = demo_contract_defects(workspace, demo, run_doc_text, public_doc_text)
        rows.append({
            "demo": demo.name,
            "base_path": demo.base_path,
            "default_port": demo.default_port,
            "source_addr": addr,
            "source_port": port,
            "ok": not row_defects,
            "defects": row_defects,
        })
        defects.extend(row_defects)

    ok = not defects
    if ok:
        verdict, finding = "OK", "browser_demo_contract_clean"
        reason = f"{len(rows)} browser demo contract(s) match code and docs"
        next_action = "rerun after changing demo default ports, base paths, or deployment docs"
    else:
        verdict, finding = "ACTION", "browser_demo_contract_debt"
        reason = f"{len(defects)} browser-demo contract defect(s)"
        next_action = "sync code defaultAddr, demo docs, and per-demo README metadata"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "demos": rows,
        "defects": defects,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"demo-browser-contract: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"  next: {payload['next_action']}",
        "",
        "demos:",
    ]
    for row in payload.get("demos", []):
        status = "OK" if row.get("ok") else "FAIL"
        lines.append(
            f"  {status:4} {row['demo']} port={row['default_port']} "
            f"base={row['base_path']} source={row['source_addr'] or '<missing>'}"
        )
    if payload.get("defects"):
        lines.append("")
        lines.append("defects:")
        for defect in payload["defects"]:
            lines.append(f"  - {defect}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit browser demo default-port/base-path contracts.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit JSON payload")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
