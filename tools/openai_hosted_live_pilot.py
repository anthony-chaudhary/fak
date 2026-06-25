#!/usr/bin/env python3
"""Hosted OpenAI live pilot for the fak guard proof packet.

This is intentionally optional: it makes a hosted OpenAI call through either the
saved Codex ChatGPT/OAuth login (`codex exec`) or, when explicitly selected, the
Platform API-key path. Without either auth source, it writes a structured
BLOCKED_ENV artifact that records the missing external state without leaking
secrets.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import openai_live_prereq_audit  # noqa: E402


SCHEMA = "fak-openai-hosted-live-pilot/1"
DEFAULT_MODEL = "gpt-5.5"
MARKER = "fak-openai-live-ok"
CODEX_LOGIN_MARKER = "fak-openai-login-ok"
REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("examples/dev-agent-policy.json")
BOOT_TIMEOUT = 30
CODEX_EXEC_TIMEOUT = 240


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact(text: str) -> str:
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted-openai-key]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(OPENAI_API_KEY=)[^\s]+", r"\1[redacted]", text)
    return text[:300]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_prereqs() -> dict[str, Any]:
    return openai_live_prereq_audit.collect()


def choose_auth_source(prereqs: dict[str, Any], auth_mode: str) -> tuple[str | None, list[str]]:
    sources = prereqs.get("auth_sources") if isinstance(prereqs.get("auth_sources"), dict) else {}
    platform_ready = bool(sources.get("platform_api_key") or prereqs.get("platform_api_ready"))
    codex_ready = bool(sources.get("codex_login") or prereqs.get("codex_login_ready"))
    blockers = [str(item) for item in prereqs.get("blockers") or []]

    if auth_mode == "codex-login":
        if codex_ready:
            return "codex_login", []
        codex_auth = prereqs.get("codex_auth") if isinstance(prereqs.get("codex_auth"), dict) else {}
        codex_blockers = [str(item) for item in codex_auth.get("blockers") or []]
        return None, codex_blockers or ["Codex ChatGPT login is not ready"]
    if auth_mode == "api-key":
        if platform_ready:
            return "platform_api_key", []
        api_blockers: list[str] = []
        env = prereqs.get("env") if isinstance(prereqs.get("env"), dict) else {}
        packages = prereqs.get("packages") if isinstance(prereqs.get("packages"), dict) else {}
        if not env.get("OPENAI_API_KEY_set"):
            api_blockers.append("OPENAI_API_KEY is not set")
        if not packages.get("openai"):
            api_blockers.append("openai package is not installed")
        return None, api_blockers or ["Platform API-key auth is not ready"]
    if auth_mode != "auto":
        return None, [f"unknown auth mode: {auth_mode}"]

    if codex_ready:
        return "codex_login", []
    if platform_ready:
        return "platform_api_key", []
    return None, blockers


def find_fak(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    exe = "fak.exe" if sys.platform == "win32" else "fak"
    local = REPO_ROOT / exe
    if local.is_file():
        return [str(local)]
    on_path = shutil.which("fak")
    if on_path:
        return [on_path]
    if shutil.which("go"):
        out = REPO_ROOT / exe
        result = subprocess.run(["go", "build", "-o", str(out), "./cmd/fak"], cwd=REPO_ROOT)
        if result.returncode == 0 and out.is_file():
            return [str(out)]
        return ["go", "run", "./cmd/fak"]
    raise RuntimeError("fak not found and no Go toolchain to build it")


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def wait_healthy(base_url: str) -> bool:
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/healthz", timeout=2) as resp:
                data = json.load(resp)
                if data.get("ok"):
                    return True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.2)
    return False


@contextlib.contextmanager
def kernel_context(kernel: str | None, fak: str | None) -> Iterator[str]:
    if kernel:
        if not wait_healthy(kernel.rstrip("/")):
            raise RuntimeError(f"could not reach fak at {kernel}")
        yield kernel.rstrip("/")
        return

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        find_fak(fak) + ["serve", "--addr", f"127.0.0.1:{port}", "--policy", str(POLICY)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not wait_healthy(base_url):
            out = ""
            try:
                proc.terminate()
                out = proc.communicate(timeout=5)[0] or ""
            except Exception:
                proc.kill()
            raise RuntimeError("fak server did not become healthy: " + redact(out[-800:]))
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except Exception:
            proc.kill()


class FakClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        if not isinstance(data, dict):
            raise RuntimeError(f"fak {path} returned non-object JSON")
        return data

    def adjudicate(self, tool: str, arguments: dict[str, Any], *, read_only: bool = False, trace_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": tool, "arguments": arguments, "read_only": read_only}
        if trace_id:
            payload["trace_id"] = trace_id
        return self.post("/v1/fak/adjudicate", payload)

    def admit(self, tool: str, result: Any, *, trace_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": tool, "result": result}
        if trace_id:
            payload["trace_id"] = trace_id
        return self.post("/v1/fak/admit", payload)


def verdict_kind(row: dict[str, Any]) -> str:
    verdict = row.get("verdict") if isinstance(row.get("verdict"), dict) else {}
    return str(verdict.get("kind") or "")


def verdict_reason(row: dict[str, Any]) -> str:
    verdict = row.get("verdict") if isinstance(row.get("verdict"), dict) else {}
    return str(verdict.get("reason") or "")


def run_guard_probe(base_url: str) -> dict[str, Any]:
    client = FakClient(base_url)
    denied = client.adjudicate("git_push", {}, trace_id="openai-hosted-deny")
    allowed = client.adjudicate("git_status", {}, read_only=True, trace_id="openai-hosted-allow")
    admitted = client.admit(
        "git_status",
        {"text": "On branch main\nnothing to commit, working tree clean"},
        trace_id="openai-hosted-allow",
    )
    ok = (
        verdict_kind(denied) == "DENY"
        and verdict_reason(denied) == "POLICY_BLOCK"
        and verdict_kind(allowed) == "ALLOW"
        and verdict_kind(admitted) in {"ALLOW", "DEFER"}
    )
    return {
        "status": "PASS" if ok else "FAIL",
        "dangerous_attempt": {
            "tool": "git_push",
            "expected": "DENY/POLICY_BLOCK",
            "verdict": denied.get("verdict") or {},
            "executed": False,
        },
        "useful_continuation": {
            "tool": "git_status",
            "expected": "ALLOW",
            "verdict": allowed.get("verdict") or {},
            "admit_verdict": admitted.get("verdict") or {},
        },
    }


def run_openai_probe(model: str) -> dict[str, Any]:
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=f"Return exactly this text and no other text: {MARKER}",
        )
        output_text = str(getattr(response, "output_text", "") or "")
        response_id = str(getattr(response, "id", "") or "")
        contains_marker = MARKER in output_text
        return {
            "status": "PASS" if contains_marker else "FAIL",
            "auth_source": "platform_api_key",
            "model": model,
            "response_id_present": bool(response_id),
            "output_text_len": len(output_text),
            "output_text_sha256": sha256_text(output_text),
            "contains_expected_marker": contains_marker,
        }
    except Exception as exc:  # noqa: BLE001 - live proof artifact must capture typed failures
        return {
            "status": "FAIL",
            "auth_source": "platform_api_key",
            "model": model,
            "error_type": type(exc).__name__,
            "error": redact(str(exc)),
        }


def summarize_codex_jsonl(text: str) -> dict[str, Any]:
    event_types: dict[str, int] = {}
    parsed = 0
    malformed = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        parsed += 1
        kind = str(event.get("type") or event.get("msg", {}).get("type") or "unknown")
        event_types[kind] = event_types.get(kind, 0) + 1
    return {"json_event_count": parsed, "json_malformed_lines": malformed, "json_event_types": event_types}


def run_codex_login_probe(*, model: str | None = None, codex_exe: str | None = None) -> dict[str, Any]:
    exe = codex_exe or shutil.which("codex") or "codex"
    with tempfile.TemporaryDirectory(prefix="fak-codex-openai-") as td:
        output_path = Path(td) / "last-message.txt"
        cmd = [
            exe,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--json",
            "--sandbox",
            "read-only",
            "-C",
            str(REPO_ROOT),
            "--output-last-message",
            str(output_path),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append(f"Return exactly this text and do not call tools: {CODEX_LOGIN_MARKER}")
        env = os.environ.copy()
        removed_api_key = "OPENAI_API_KEY" in env
        env.pop("OPENAI_API_KEY", None)
        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=CODEX_EXEC_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - live proof artifact captures operational failures
            return {
                "status": "FAIL",
                "auth_source": "codex_login",
                "model": model or "",
                "error_type": type(exc).__name__,
                "error": redact(str(exc)),
                "openai_api_key_env_removed": removed_api_key,
            }
        output_text = ""
        if output_path.is_file():
            output_text = output_path.read_text(encoding="utf-8", errors="replace")
        contains_marker = CODEX_LOGIN_MARKER in output_text
        stdout_summary = summarize_codex_jsonl(result.stdout or "")
        out: dict[str, Any] = {
            "status": "PASS" if result.returncode == 0 and contains_marker else "FAIL",
            "auth_source": "codex_login",
            "model": model or "",
            "codex_exec_exit_code": result.returncode,
            "contains_expected_marker": contains_marker,
            "output_text_len": len(output_text),
            "output_text_sha256": sha256_text(output_text),
            "stdout_len": len(result.stdout or ""),
            "stdout_sha256": sha256_text(result.stdout or ""),
            "stderr_len": len(result.stderr or ""),
            "stderr_sha256": sha256_text(result.stderr or ""),
            "openai_api_key_env_removed": removed_api_key,
        }
        out.update(stdout_summary)
        if result.returncode != 0:
            out["stderr_redacted"] = redact(result.stderr or "")
        return out


def collect(
    model: str,
    *,
    kernel: str | None = None,
    fak: str | None = None,
    auth_mode: str = "auto",
    codex_model: str | None = None,
    codex_exe: str | None = None,
) -> dict[str, Any]:
    prereqs = collect_prereqs()
    auth_source, auth_blockers = choose_auth_source(prereqs, auth_mode)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": now_utc(),
        "status": "BLOCKED_ENV",
        "model": model,
        "auth_mode": auth_mode,
        "auth_source": auth_source or "",
        "prereqs": prereqs,
        "privacy": {
            "copied_fields": [
                "status booleans",
                "package versions",
                "verdict metadata",
                "hosted output hashes",
                "Codex exec JSON event counts",
            ],
            "dropped": [
                "OPENAI_API_KEY value",
                "Codex access_token value",
                "Codex refresh_token value",
                "Codex id_token value",
                "raw hosted OpenAI response text",
                "request/response payloads",
            ],
        },
    }
    if not auth_source:
        payload["blockers"] = auth_blockers or prereqs.get("blockers") or []
        return payload

    try:
        with kernel_context(kernel, fak) as base_url:
            guard = run_guard_probe(base_url)
            if auth_source == "codex_login":
                hosted = run_codex_login_probe(model=codex_model, codex_exe=codex_exe)
            else:
                hosted = run_openai_probe(model)
    except Exception as exc:  # noqa: BLE001 - preserve live failure without leaking process output
        payload.update({"status": "FAIL", "error_type": type(exc).__name__, "error": redact(str(exc))})
        return payload

    payload["guard"] = guard
    payload["hosted_openai"] = hosted
    payload["status"] = "PASS" if guard.get("status") == "PASS" and hosted.get("status") == "PASS" else "FAIL"
    return payload


def render_md(payload: dict[str, Any]) -> str:
    prereqs = payload.get("prereqs") if isinstance(payload.get("prereqs"), dict) else {}
    guard = payload.get("guard") if isinstance(payload.get("guard"), dict) else {}
    hosted = payload.get("hosted_openai") if isinstance(payload.get("hosted_openai"), dict) else {}
    blockers = payload.get("blockers") or prereqs.get("blockers") or []
    lines = [
        "# OpenAI hosted live pilot",
        "",
        f"- generated: `{payload.get('created_at')}`",
        f"- status: **`{payload.get('status')}`**",
        f"- model: `{payload.get('model')}`",
        f"- auth_mode: `{payload.get('auth_mode')}`",
        f"- auth_source: `{payload.get('auth_source')}`",
        f"- hosted_openai_ready: `{prereqs.get('hosted_openai_ready')}`",
        f"- platform_api_ready: `{prereqs.get('platform_api_ready')}`",
        f"- codex_login_ready: `{prereqs.get('codex_login_ready')}`",
        f"- agents_sdk_ready: `{prereqs.get('agents_sdk_ready')}`",
        "",
        "## Guard",
        "",
        f"- status: `{guard.get('status')}`",
        f"- dangerous tool: `git_push` -> `{((guard.get('dangerous_attempt') or {}).get('verdict') or {}).get('kind')}` / `{((guard.get('dangerous_attempt') or {}).get('verdict') or {}).get('reason', '')}`",
        f"- useful tool: `git_status` -> `{((guard.get('useful_continuation') or {}).get('verdict') or {}).get('kind')}`",
        "",
        "## Hosted OpenAI",
        "",
        f"- status: `{hosted.get('status')}`",
        f"- auth_source: `{hosted.get('auth_source')}`",
        f"- response_id_present: `{hosted.get('response_id_present')}`",
        f"- codex_exec_exit_code: `{hosted.get('codex_exec_exit_code')}`",
        f"- contains_expected_marker: `{hosted.get('contains_expected_marker')}`",
        f"- output_text_sha256: `{hosted.get('output_text_sha256', '')}`",
        f"- json_event_count: `{hosted.get('json_event_count')}`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "This artifact records verdict metadata, hosted-output hashes, and Codex exec event counts only. It never writes API key values, Codex token values, raw hosted OpenAI response text, or request payloads.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("OPENAI_LIVE_MODEL", DEFAULT_MODEL))
    p.add_argument("--auth-mode", choices=["auto", "api-key", "codex-login"], default="auto")
    p.add_argument("--codex-model", default=os.environ.get("CODEX_LIVE_MODEL", ""), help="optional model override for codex exec")
    p.add_argument("--codex-exe", help="path to codex executable")
    p.add_argument("--kernel", help="use an existing fak base URL instead of starting one")
    p.add_argument("--fak", help="path to fak binary")
    p.add_argument("--out", type=Path, help="write JSON report")
    p.add_argument("--markdown", type=Path, help="write Markdown report")
    p.add_argument("--json", action="store_true", help="print JSON to stdout")
    p.add_argument("--fail-on-not-pass", action="store_true", help="exit nonzero unless status is PASS")
    args = p.parse_args(argv)

    payload = collect(
        args.model,
        kernel=args.kernel,
        fak=args.fak,
        auth_mode=args.auth_mode,
        codex_model=args.codex_model or None,
        codex_exe=args.codex_exe,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_md(payload), encoding="utf-8")
    if args.json or not (args.out or args.markdown):
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_not_pass and payload.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
