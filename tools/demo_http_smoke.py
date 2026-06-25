#!/usr/bin/env python3
"""Smoke-test browser demos as real HTTP servers under base paths.

This is the dynamic companion to demo_command_audit.py and demo_live_links.py:
it builds the browser demo binaries, starts each on loopback with a reverse-proxy
base path, and fetches the mounted page plus one lightweight JSON API endpoint.
It does not touch the public network and does not require a model.

Run from the repo root:

    python tools/demo_http_smoke.py
    python tools/demo_http_smoke.py --demo guarddemo --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import demo_registry as dr

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fak-demo-http-smoke/1"
ABSOLUTE_BROWSER_REF_RE = re.compile(r"""(?P<quote>["'`])/(?P<prefix>api|assets|static|favicon|css|js)(?P<rest>[^"'`<\s]*)""")


Demo = dr.Demo
DEMOS = dr.DEMOS
demo_map = dr.demo_map
discover_browser_demo_names = dr.discover_browser_demo_names
registry_defects = dr.registry_defects
select_demos = dr.select_demos
normalize_base = dr.normalize_base
demo_url = dr.demo_url


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    status: int
    body: str
    location: str
    error: str = ""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def page_static_defects(workspace: Path, demo: Demo) -> list[str]:
    path = workspace / "cmd" / demo.name / "page.html"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"{demo.name}: read page.html: {exc}"]

    defects: list[str] = []
    if demo.api_path not in text:
        defects.append(f"{demo.name}: page.html does not reference registered API path {demo.api_path!r}")

    seen: set[str] = set()
    for m in ABSOLUTE_BROWSER_REF_RE.finditer(text):
        ref = "/" + m.group("prefix") + m.group("rest")
        if ref in seen:
            continue
        seen.add(ref)
        defects.append(f"{demo.name}: page.html has root-relative browser reference {ref!r}; use a relative path")
    return defects


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def fetch(url: str, *, timeout_s: float, follow_redirects: bool = True) -> FetchResult:
    headers = {"User-Agent": "fak-demo-http-smoke/1"}
    req = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener() if follow_redirects else urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            body = resp.read(512_000).decode("utf-8", errors="replace")
            return FetchResult(200 <= resp.status < 400, resp.status, body, resp.headers.get("Location", ""))
    except urllib.error.HTTPError as exc:
        body = exc.read(64_000).decode("utf-8", errors="replace")
        return FetchResult(False, exc.code, body, exc.headers.get("Location", ""), str(exc.reason))
    except (OSError, TimeoutError) as exc:
        return FetchResult(False, 0, "", "", str(exc))


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def validate_api_payload(demo: Demo, payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{demo.name} API returned {type(payload).__name__}, want JSON object"]

    defects: list[str] = []
    if demo.name in {"guarddemo", "ctxdemo"}:
        if not _nonempty_list(payload.get("scenarios")):
            defects.append(f"{demo.name} API missing non-empty scenarios list")
    elif demo.name == "turntaxdemo":
        if not _nonempty_list(payload.get("suites")):
            defects.append("turntaxdemo API missing non-empty suites list")
    elif demo.name == "demorace":
        if not _nonempty_list(payload.get("models")):
            defects.append("demorace API missing non-empty models list")
        ratio = payload.get("prefill_tok_ratio")
        if not isinstance(ratio, dict) or _float_value(ratio.get("b_over_c")) <= 1:
            defects.append("demorace API missing prefill_tok_ratio.b_over_c > 1")
    elif demo.name == "unseedemo":
        if not isinstance(payload.get("witness"), dict):
            defects.append("unseedemo API missing witness object")
        if not _nonempty_list(payload.get("frames")):
            defects.append("unseedemo API missing non-empty frames list")
        if not _nonempty_list(payload.get("fences")):
            defects.append("unseedemo API missing non-empty fences list")
    return defects


def wait_for_page(proc: subprocess.Popen[str], url: str, timeout_s: float) -> FetchResult:
    deadline = time.monotonic() + timeout_s
    last = FetchResult(False, 0, "", "", "not probed")
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return FetchResult(False, 0, "", "", f"process exited before serving (rc={proc.returncode})")
        last = fetch(url, timeout_s=1.0)
        if last.ok:
            return last
        time.sleep(0.2)
    return FetchResult(False, last.status, last.body, last.location, f"timed out waiting for {url}: {last.error}")


def executable_path(tmp: Path, demo: Demo) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return tmp / (demo.name + suffix)


def build_demo(workspace: Path, tmp: Path, demo: Demo, timeout_s: float) -> tuple[Path | None, str]:
    exe = executable_path(tmp, demo)
    cmd = ["go", "build", "-o", str(exe), "./cmd/" + demo.name]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"build failed to start for {demo.name}: {exc}"
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["unknown build error"]
        return None, f"go build ./cmd/{demo.name} failed: {detail[0]}"
    return exe, ""


def stop_process(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    try:
        stdout, stderr = proc.communicate(timeout=1)
    except Exception:  # noqa: BLE001
        return ""
    tail = ((stderr or "") + (stdout or "")).strip().splitlines()
    return tail[-1] if tail else ""


def smoke_server(workspace: Path, exe: Path, demo: Demo, timeout_s: float, base_source: str) -> dict[str, Any]:
    port = free_port()
    cmd = [str(exe), "-addr", f"127.0.0.1:{port}", *demo.extra_args]
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    if base_source == "flag":
        cmd.extend(("-base-path", demo.base_path))
    elif base_source == "env":
        env["FAK_DEMO_BASE_PATH"] = demo.base_path
    else:
        return {"base_source": base_source, "ok": False, "defects": [f"unknown base source: {base_source}"]}
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    server_tail = ""
    defects: list[str] = []
    try:
        page = wait_for_page(proc, demo_url(port, demo.base_path), timeout_s)
        if not page.ok:
            defects.append(f"page not reachable at {demo.base_path}/: {page.status} {page.error}")
        elif demo.page_marker not in page.body:
            defects.append(f"page missing marker {demo.page_marker!r}")

        root = fetch(f"http://127.0.0.1:{port}/", timeout_s=2.0, follow_redirects=False)
        if root.status not in {301, 302, 307, 308}:
            defects.append(f"root did not redirect to base path: status {root.status}")
        elif root.location != normalize_base(demo.base_path) + "/":
            defects.append(f"root redirect location {root.location!r}, want {normalize_base(demo.base_path) + '/'}")

        api = fetch(demo_url(port, demo.base_path, demo.api_path), timeout_s=timeout_s)
        if not api.ok:
            defects.append(f"API {demo.api_path} not reachable: {api.status} {api.error}")
        else:
            try:
                payload = json.loads(api.body)
            except json.JSONDecodeError as exc:
                defects.append(f"API {demo.api_path} did not return JSON: {exc}")
            else:
                defects.extend(validate_api_payload(demo, payload))
    finally:
        server_tail = stop_process(proc)

    return {
        "demo": demo.name,
        "ok": not defects,
        "port": port,
        "base_source": base_source,
        "base_path": demo.base_path,
        "api_path": demo.api_path,
        "defects": defects,
        "server_tail": server_tail,
    }


def smoke_demo(workspace: Path, tmp: Path, demo: Demo, timeout_s: float) -> dict[str, Any]:
    static_defects = page_static_defects(workspace, demo)
    if static_defects:
        return {"demo": demo.name, "ok": False, "defects": static_defects, "runs": []}

    exe, build_error = build_demo(workspace, tmp, demo, timeout_s)
    if build_error:
        return {"demo": demo.name, "ok": False, "defects": [build_error], "runs": []}

    assert exe is not None
    runs = [
        smoke_server(workspace, exe, demo, timeout_s, "flag"),
        smoke_server(workspace, exe, demo, timeout_s, "env"),
    ]
    defects = [f"{r['base_source']}: {d}" for r in runs for d in r.get("defects", [])]
    return {
        "demo": demo.name,
        "ok": not defects,
        "base_path": demo.base_path,
        "api_path": demo.api_path,
        "defects": defects,
        "runs": runs,
    }


def collect(workspace: Path, *, names: list[str] | None = None, timeout_s: float = 30.0) -> dict[str, Any]:
    workspace = workspace.resolve()
    demos, unknown = select_demos(names)
    rows: list[dict[str, Any]] = []
    defects = [f"unknown demo: {name}" for name in unknown]
    registry = registry_defects(workspace) if names is None else []
    defects.extend(registry)

    if not registry:
        with tempfile.TemporaryDirectory(prefix="fak-demo-http-smoke-") as td:
            tmp = Path(td)
            for demo in demos:
                row = smoke_demo(workspace, tmp, demo, timeout_s)
                rows.append(row)
                defects.extend(f"{demo.name}: {d}" for d in row.get("defects", []))

    ok = not defects
    if ok:
        verdict, finding = "OK", "browser_demo_http_clean"
        n_runs = sum(len(row.get("runs", [])) for row in rows)
        reason = f"{len(rows)} browser demo(s) built and served under base paths ({n_runs} server run(s))"
        next_action = "rerun after changing browser demo routing, handlers, or pages"
    else:
        verdict, finding = "ACTION", "browser_demo_http_debt"
        reason = f"{len(defects)} browser-demo HTTP defect(s)"
        next_action = "fix the failing demo server, base-path mount, page marker, or API endpoint"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "registry": {
            "discovered": discover_browser_demo_names(workspace),
            "registered": sorted(d.name for d in DEMOS),
            "defects": registry,
        },
        "demos": rows,
        "defects": defects,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"demo-http-smoke: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"  next: {payload['next_action']}",
        "",
        "demos:",
    ]
    for row in payload.get("demos", []):
        status = "OK" if row.get("ok") else "FAIL"
        lines.append(f"  {status:4} {row.get('demo')} {row.get('base_path')}/ api={row.get('api_path')}")
        for run in row.get("runs", []):
            rstatus = "OK" if run.get("ok") else "FAIL"
            lines.append(f"       {rstatus:4} base={run.get('base_source')} port={run.get('port')}")
            if run.get("server_tail") and not run.get("ok"):
                lines.append(f"            tail: {run['server_tail']}")
    if payload.get("defects"):
        lines.append("")
        lines.append("defects:")
        for defect in payload["defects"]:
            lines.append(f"  - {defect}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build and HTTP-smoke browser demos under base paths.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--demo", action="append", default=None, help="demo command to smoke; repeatable")
    ap.add_argument("--timeout", type=float, default=30.0, help="build/server/API timeout in seconds")
    ap.add_argument("--json", action="store_true", help="emit JSON payload")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, names=args.demo, timeout_s=args.timeout)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
