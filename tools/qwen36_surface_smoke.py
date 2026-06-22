#!/usr/bin/env python3
"""qwen36_surface_smoke.py - exercise Qwen3.6 through fak's three model surfaces.

The target server is any OpenAI-compatible endpoint serving Qwen/Qwen3.6-27B or a
compatible local quant. The runner can start fak's gateway, then probes:

  agent          fak agent live A/B loop, bounded by --agent-max-turns
  gateway-openai fak serve /v1/models plus optional /v1/chat/completions
  mcp-http       fak serve /mcp initialize + tools/list + fak_adjudicate

It is intentionally stdlib-only so the same file can run on the Mac and NVIDIA
laptop test beds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import os
import re
import shutil
import signal
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "fak.qwen36-surface-smoke.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_EXTRA_BODY = '{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'
DEFAULT_SURFACES = "agent,gateway-openai,mcp-http"
DEFAULT_REGISTRIES = (
    "tools/fleet_endpoints.local.json",
    "tools/fleet_endpoints.json",
    "tools/fleet_endpoints.example.json",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def split_command(command: str) -> list[str]:
    return shlex.split(command, posix=os.name != "nt")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def endpoint_registries(registry: str) -> list[Path]:
    if registry:
        path = Path(registry)
        return [path if path.is_absolute() else ROOT / path]
    return [ROOT / p for p in DEFAULT_REGISTRIES]


def load_endpoint_registry(registry: str) -> tuple[Path, dict[str, Any]]:
    checked = []
    for path in endpoint_registries(registry):
        checked.append(str(path))
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8-sig"))
    raise FileNotFoundError("no endpoint registry found; checked " + ", ".join(checked))


def tcp_open(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def tcp_probe(hosts: list[str], port: int, timeout_s: float) -> list[dict[str, Any]]:
    return [{"host": host, "port": port, "open": tcp_open(host, port, timeout_s)} for host in hosts]


def first_open_probe(probes: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((probe for probe in probes if probe.get("open")), None)


def find_tailscale() -> str:
    exe = shutil.which("tailscale")
    if exe:
        return exe
    if os.name == "nt":
        candidate = r"C:\Program Files\Tailscale\tailscale.exe"
        if os.path.exists(candidate):
            return candidate
    return ""


def tailscale_ping(host: str, timeout_s: float) -> dict[str, Any]:
    exe = find_tailscale()
    if not exe or not host:
        return {"host": host, "state": "UNAVAILABLE", "detail": "tailscale CLI not found"}
    timeout_arg = f"{max(1, int(timeout_s))}s"
    try:
        proc = subprocess.run(
            [exe, "ping", "--c", "1", "--timeout", timeout_arg, host],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(2.0, timeout_s + 1.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"host": host, "state": "ERROR", "detail": str(exc)}
    text = proc.stdout.strip()
    if proc.returncode == 0 and "pong" in text:
        match = re.search(r"\bin\s+(\d+)ms\b", text)
        return {
            "host": host,
            "state": "ONLINE",
            "rtt_ms": int(match.group(1)) if match else None,
            "detail": text[:300],
        }
    return {"host": host, "state": "OFFLINE", "detail": text[:300]}


def tailscale_status(timeout_s: float) -> dict[str, Any]:
    exe = find_tailscale()
    if not exe:
        return {"error": "tailscale CLI not found"}
    try:
        proc = subprocess.run(
            [exe, "status", "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(2.0, timeout_s + 1.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[:300]}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"error": f"tailscale status returned invalid JSON: {exc}"}
    return data if isinstance(data, dict) else {"error": "tailscale status JSON was not an object"}


def short_hostname(host: str) -> str:
    clean = host.strip().strip("[]").rstrip(".")
    if not clean:
        return ""
    try:
        ipaddress.ip_address(clean)
        return ""
    except ValueError:
        return clean.split(".", 1)[0] if "." in clean else ""


def unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def tailscale_ping_candidates(row: dict[str, Any], hosts: list[str]) -> list[str]:
    raw = [
        str(row.get("tailnet_host") or "").rstrip("."),
        str(row.get("magicdns") or "").rstrip("."),
        *hosts,
    ]
    candidates: list[str] = []
    for value in raw:
        candidates.append(value)
        short = short_hostname(value)
        if short:
            candidates.append(short)
    return unique_values(candidates)


def node_candidate_names(peer: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for value in [peer.get("HostName"), peer.get("DNSName"), *(peer.get("TailscaleIPs") or [])]:
        text = str(value or "").strip().rstrip(".")
        if not text:
            continue
        names.append(text)
        short = short_hostname(text)
        if short:
            names.append(short)
    return unique_values(names)


def find_tailscale_peer(node: str, status: dict[str, Any]) -> dict[str, Any] | None:
    target = node.strip().rstrip(".").lower()
    peers = []
    self_node = status.get("Self")
    if isinstance(self_node, dict):
        peers.append(self_node)
    peer_map = status.get("Peer")
    if isinstance(peer_map, dict):
        peers.extend(p for p in peer_map.values() if isinstance(p, dict))
    for peer in peers:
        if any(name.lower() == target for name in node_candidate_names(peer)):
            return peer
    return None


def endpoint_from_tailscale_node(node: str, serve_port: int, status: dict[str, Any]) -> dict[str, Any]:
    if status.get("error"):
        return {
            "name": node,
            "state": "TAILSCALE_UNAVAILABLE",
            "source": "tailscale-status",
            "detail": status["error"],
        }
    peer = find_tailscale_peer(node, status)
    if peer is None:
        return {
            "name": node,
            "state": "MISSING",
            "source": "tailscale-status",
            "detail": "node not found in tailscale status",
        }
    dns = str(peer.get("DNSName") or "").rstrip(".")
    ip = next((str(ip) for ip in (peer.get("TailscaleIPs") or []) if ":" not in str(ip)), "")
    aliases = [value for value in (ip, dns, short_hostname(dns) or str(peer.get("HostName") or node)) if value]
    return {
        "name": node,
        "tailnet_host": short_hostname(dns) or str(peer.get("HostName") or node),
        "magicdns": dns,
        "tailscale_ip": ip,
        "probe_hosts": [ip] if ip else aliases,
        "os": peer.get("OS"),
        "enabled": True,
        "serve_port": serve_port,
        "ssh": {
            "available": False,
            "probe": True,
            "port": 22,
            "method": "opportunistic TCP reachability probe; auth is not assumed",
            "auth_verified": False,
        },
        "source": "tailscale-status",
        "tailscale_online": bool(peer.get("Online")),
    }


def tailscale_ping_any(hosts: list[str], timeout_s: float) -> dict[str, Any]:
    attempts = []
    for host in hosts:
        result = tailscale_ping(host, timeout_s)
        attempts.append(result)
        if result.get("state") == "ONLINE":
            return {**result, "attempts": attempts}
    state = "UNAVAILABLE" if any(a.get("state") == "UNAVAILABLE" for a in attempts) else "OFFLINE"
    detail = attempts[-1].get("detail") if attempts else "no ping candidates"
    return {"host": hosts[0] if hosts else "", "state": state, "detail": detail, "attempts": attempts}


def resolve_endpoint_row(name: str, row: dict[str, Any], source: str, dry_run: bool, timeout_s: float) -> dict[str, Any]:
    hosts = [h for h in (row.get("tailscale_ip"), row.get("magicdns"), row.get("tailnet_host")) if h]
    probe_hosts_raw = row.get("probe_hosts") if isinstance(row.get("probe_hosts"), list) else []
    probe_hosts = [str(host) for host in probe_hosts_raw if str(host)]
    if not probe_hosts:
        probe_hosts = hosts
    port = int(row.get("serve_port") or 0)
    ssh_cfg = row.get("ssh") if isinstance(row.get("ssh"), dict) else {}
    ssh_port = int((ssh_cfg or {}).get("port") or 0)
    ssh_probe = bool((ssh_cfg or {}).get("available") or (ssh_cfg or {}).get("probe"))
    base_host = hosts[0] if hosts else ""
    ping_hosts = tailscale_ping_candidates(row, hosts)
    base_url = f"http://{base_host}:{port}/v1" if base_host and port else ""
    detail = {
        "name": name,
        "source": source,
        "enabled": bool(row.get("enabled")),
        "os": row.get("os"),
        "serve_port": port,
        "hosts": hosts,
        "probe_hosts": probe_hosts,
        "base_url": base_url,
        "tailscale_online": row.get("tailscale_online"),
        "ssh": {
            "available": bool((ssh_cfg or {}).get("available")),
            "probe": bool((ssh_cfg or {}).get("probe")),
            "port": ssh_port,
            "method": (ssh_cfg or {}).get("method"),
            "auth_verified": bool((ssh_cfg or {}).get("auth_verified")),
        },
    }
    if not row.get("enabled"):
        return {**detail, "state": "DISABLED", "detail": "endpoint has enabled=false; owner opt-in required"}
    if not hosts or not port:
        return {**detail, "state": "INVALID", "detail": "endpoint lacks host or serve_port"}
    if dry_run:
        return {**detail, "state": "PLANNED", "detail": "dry-run; serve port not probed"}
    serve_probes = tcp_probe(probe_hosts, port, timeout_s)
    serve_hit = first_open_probe(serve_probes)
    if serve_hit:
        return {
            **detail,
            "state": "READY",
            "detail": f"serve port open at {serve_hit['host']}:{port}",
            "base_url": f"http://{serve_hit['host']}:{port}/v1",
            "serve_tcp": serve_probes,
        }
    ssh_probes: list[dict[str, Any]] = []
    ssh_hit = None
    if ssh_port and ssh_probe:
        ssh_probes = tcp_probe(probe_hosts, ssh_port, min(timeout_s, 2.0))
        ssh_hit = first_open_probe(ssh_probes)
    ping = (
        {"host": ping_hosts[0] if ping_hosts else "", "state": "SKIPPED", "detail": "ssh TCP already proved node reachability"}
        if ssh_hit
        else tailscale_ping_any(ping_hosts, min(timeout_s, 5.0))
    )
    online_by_ping = ping.get("state") == "ONLINE"
    state = "ONLINE_NO_SERVE" if ssh_hit or online_by_ping else "OFFLINE"
    if ssh_hit:
        if detail["ssh"]["available"]:
            message = f"node accepts SSH at {ssh_hit['host']}:{ssh_port}, but serve port {port} is not live"
        else:
            message = (
                f"node has SSH TCP open at {ssh_hit['host']}:{ssh_port}, "
                f"but SSH auth is not verified and serve port {port} is not live"
            )
    elif online_by_ping:
        message = f"node responds to tailscale ping at {ping.get('host')}, but serve port {port} is not live"
    else:
        message = f"serve port {port} did not accept TCP on any registry host"
    return {
        **detail,
        "state": state,
        "detail": message,
        "serve_tcp": serve_probes,
        "tailscale_ping": ping,
        "ssh": {**detail["ssh"], "tcp": ssh_probes, "tcp_open": bool(ssh_hit)},
    }


def resolve_endpoint(name: str, registry: str, dry_run: bool, timeout_s: float) -> dict[str, Any]:
    reg_path, reg = load_endpoint_registry(registry)
    endpoints = reg.get("endpoints") or []
    row = next((e for e in endpoints if e.get("name") == name), None)
    if row is None:
        return {
            "name": name,
            "state": "MISSING",
            "registry": str(reg_path),
            "detail": "endpoint name not found in registry",
        }
    endpoint = resolve_endpoint_row(name, row, source=f"registry:{reg_path}", dry_run=dry_run, timeout_s=timeout_s)
    return {**endpoint, "registry": str(reg_path)}


def resolve_tailscale_node(node: str, serve_port: int, dry_run: bool, timeout_s: float) -> dict[str, Any]:
    row = endpoint_from_tailscale_node(node, serve_port, tailscale_status(timeout_s))
    if row.get("state"):
        return row
    return resolve_endpoint_row(node, row, source="tailscale-status", dry_run=dry_run, timeout_s=timeout_s)


def json_post(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any] | None, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), parse_json(raw), raw[:1000]
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), parse_json(raw), raw[:1000]
    except OSError as exc:
        return 0, None, str(exc)


def json_post_timed(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any] | None, str, float]:
    started = time.time()
    status, data, body = json_post(url, payload, timeout_s)
    return status, data, body, round(time.time() - started, 3)


def json_get(url: str, timeout_s: float) -> tuple[int, dict[str, Any] | None, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), parse_json(raw), raw[:1000]
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), parse_json(raw), raw[:1000]
    except OSError as exc:
        return 0, None, str(exc)


def parse_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def int_number(value: Any) -> int | None:
    numeric = number(value)
    return int(numeric) if numeric is not None else None


def chat_response_text(data: dict[str, Any] | None) -> str:
    choices = (data or {}).get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    text = first.get("text")
    return text if isinstance(text, str) else ""


def chat_perf_metrics(data: dict[str, Any] | None, duration_s: float, baseline_decode_tps: float = 0.0) -> dict[str, Any]:
    payload = data or {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    prompt_tokens = int_number(usage.get("prompt_tokens")) or int_number(timings.get("prompt_n"))
    completion_tokens = int_number(usage.get("completion_tokens")) or int_number(timings.get("predicted_n"))
    total_tokens = int_number(usage.get("total_tokens"))
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    prompt_tps = number(timings.get("prompt_per_second"))
    decode_tps = number(timings.get("predicted_per_second"))
    if decode_tps is None and completion_tokens and duration_s > 0:
        decode_tps = completion_tokens / duration_s

    metrics: dict[str, Any] = {
        "duration_s": duration_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tps": round(prompt_tps, 3) if prompt_tps is not None else None,
        "decode_tps": round(decode_tps, 3) if decode_tps is not None else None,
        "e2e_total_tps": round(total_tokens / duration_s, 3) if total_tokens and duration_s > 0 else None,
    }
    if timings:
        metrics["llama_timings"] = {
            key: timings.get(key)
            for key in (
                "prompt_n",
                "prompt_ms",
                "prompt_per_second",
                "predicted_n",
                "predicted_ms",
                "predicted_per_second",
            )
            if key in timings
        }
    if baseline_decode_tps > 0 and decode_tps is not None:
        metrics["baseline_decode_tps"] = baseline_decode_tps
        metrics["decode_vs_baseline"] = round(decode_tps / baseline_decode_tps, 3)
    return metrics


def env_for(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.provider_extra_body:
        env["FAK_PROVIDER_EXTRA_BODY_JSON"] = args.provider_extra_body
    env.setdefault("FAK_PLANNER_TIMEOUT_S", str(args.model_timeout_s))
    env.setdefault("FAK_HTTP_WRITE_TIMEOUT_S", str(args.model_timeout_s + 30))
    return env


def run_checked(cmd: list[str], env: dict[str, str], timeout_s: float, cwd: Path) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        return {
            "exit_code": proc.returncode,
            "duration_s": round(time.time() - started, 3),
            "stdout_excerpt": proc.stdout[-2000:],
            "stderr_excerpt": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "duration_s": round(time.time() - started, 3),
            "stdout_excerpt": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_excerpt": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "error": f"timeout after {timeout_s}s",
        }


def fak_cmd(args: argparse.Namespace) -> list[str]:
    return split_command(args.fak_command)


def start_gateway(args: argparse.Namespace, env: dict[str, str]) -> tuple[subprocess.Popen[str] | None, str, dict[str, Any]]:
    if args.gateway_url:
        return None, args.gateway_url.rstrip("/"), {"mode": "existing", "url": args.gateway_url.rstrip("/")}
    port = args.gateway_port or free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = fak_cmd(args) + [
        "serve",
        "--addr",
        f"127.0.0.1:{port}",
        "--provider",
        args.provider,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
    ]
    if args.dry_run:
        return None, url, {"mode": "planned", "command": cmd, "url": url}
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    deadline = time.time() + args.gateway_start_timeout_s
    last = ""
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            return proc, url, {"mode": "start-failed", "exit_code": code, "stderr_excerpt": stderr[-2000:]}
        status, _, body = json_get(url + "/healthz", timeout_s=1.0)
        last = body
        if status == 200:
            return proc, url, {"mode": "started", "url": url, "pid": proc.pid, "command": cmd}
        time.sleep(0.5)
    return proc, url, {"mode": "start-timeout", "url": url, "last_error": last, "command": cmd}


def stop_gateway(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def surface_agent(args: argparse.Namespace, env: dict[str, str], out_dir: Path) -> dict[str, Any]:
    out = out_dir / f"qwen36-agent-{args.node_name}.json"
    cmd = fak_cmd(args) + [
        "agent",
        "--provider",
        args.provider,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--max-turns",
        str(args.agent_max_turns),
        "--out",
        str(out),
    ]
    if args.dry_run:
        return {"surface": "agent", "status": "PLANNED", "command": cmd, "out": str(out)}
    res = run_checked(cmd, env=env, timeout_s=args.agent_timeout_s, cwd=ROOT)
    status = "PASS" if res.get("exit_code") == 0 and out.exists() else "FAIL"
    detail: dict[str, Any] = {"surface": "agent", "status": status, "command": cmd, "out": str(out), **res}
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            detail["live"] = data.get("live")
            detail["fak_turns"] = (data.get("fak") or {}).get("turns")
            detail["baseline_turns"] = (data.get("baseline") or {}).get("turns")
            detail["fak_hit_turn_cap"] = (data.get("fak") or {}).get("hit_turn_cap")
        except (OSError, json.JSONDecodeError) as exc:
            detail["artifact_error"] = str(exc)
    return detail


def surface_gateway_openai(args: argparse.Namespace, gateway_url: str) -> dict[str, Any]:
    status, data, body = json_get(gateway_url + "/v1/models", timeout_s=args.http_timeout_s)
    model_ids = [m.get("id") for m in (data or {}).get("data", []) if isinstance(m, dict)]
    ok = status == 200 and args.model in model_ids
    detail: dict[str, Any] = {
        "surface": "gateway-openai",
        "status": "PASS" if ok else "FAIL",
        "models_http_status": status,
        "models": model_ids,
        "body_excerpt": body,
    }
    if args.gateway_chat:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 16,
        }
        chat_status, chat_data, chat_body, chat_duration_s = json_post_timed(
            gateway_url + "/v1/chat/completions",
            payload,
            timeout_s=args.model_timeout_s,
        )
        detail["chat_http_status"] = chat_status
        detail["chat_body_excerpt"] = chat_body
        detail["chat_duration_s"] = chat_duration_s
        response_text = chat_response_text(chat_data)
        if response_text:
            detail["chat_response_excerpt"] = response_text[:500]
        detail["chat_perf"] = chat_perf_metrics(
            chat_data,
            chat_duration_s,
            baseline_decode_tps=args.perf_decode_baseline_tps,
        )
        if chat_status != 200 or not (chat_data or {}).get("choices"):
            detail["status"] = "FAIL"
        elif args.min_decode_tps > 0:
            decode_tps = detail["chat_perf"].get("decode_tps")
            if decode_tps is None:
                detail["status"] = "FAIL"
                detail["perf_gate"] = {
                    "status": "FAIL",
                    "reason": "decode_tps unavailable",
                    "min_decode_tps": args.min_decode_tps,
                }
            elif float(decode_tps) < args.min_decode_tps:
                detail["status"] = "FAIL"
                detail["perf_gate"] = {
                    "status": "FAIL",
                    "decode_tps": decode_tps,
                    "min_decode_tps": args.min_decode_tps,
                }
            else:
                detail["perf_gate"] = {
                    "status": "PASS",
                    "decode_tps": decode_tps,
                    "min_decode_tps": args.min_decode_tps,
                }
    return detail


def surface_mcp_http(args: argparse.Namespace, gateway_url: str) -> dict[str, Any]:
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "qwen36-smoke", "version": "dev"}},
    }
    list_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    adjudicate = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "fak_adjudicate",
            "arguments": {"tool": "search_kb", "arguments": {"query": "refund policy"}, "read_only": True},
        },
    }
    rows = []
    ok = True
    for payload in (init, list_tools, adjudicate):
        status, data, body = json_post(gateway_url + "/mcp", payload, timeout_s=args.http_timeout_s)
        has_result = bool(data and data.get("result") is not None)
        rows.append({"method": payload["method"], "http_status": status, "has_result": has_result, "body_excerpt": body})
        ok = ok and status == 200 and has_result
    return {"surface": "mcp-http", "status": "PASS" if ok else "FAIL", "calls": rows}


def write_report(path: str, report: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.6 Surface Smoke",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Node: `{report['node_name']}`",
        f"- Model: `{report['model']}`",
        f"- Base URL: `{report['base_url']}`",
        "",
        "| surface | status | note |",
        "|---|---|---|",
    ]
    for row in report["surfaces"]:
        note = row.get("detail") or row.get("out") or row.get("body_excerpt") or ""
        lines.append(f"| `{row['surface']}` | {row['status']} | {str(note)[:160]} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Smoke Qwen3.6 through fak agent, OpenAI gateway, and MCP gateway surfaces")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible upstream base URL, e.g. http://node:8000/v1")
    ap.add_argument("--endpoint", default="", help="fleet endpoint name from tools/fleet_endpoints*.json; resolves to base URL")
    ap.add_argument("--registry", default="", help="endpoint registry path; defaults to local/json/example search order")
    ap.add_argument("--tailscale-node", default="", help="Tailscale node name/DNS/IP; resolves from `tailscale status --json`")
    ap.add_argument("--serve-port", type=int, default=8131, help="model serve port used with --tailscale-node")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--api-key-env", default="NONE_LOCAL")
    ap.add_argument("--provider-extra-body", default=DEFAULT_EXTRA_BODY, help="JSON object merged into OpenAI-compatible upstream requests")
    ap.add_argument("--surfaces", default=DEFAULT_SURFACES, help="comma list: agent,gateway-openai,mcp-http")
    ap.add_argument("--fak-command", default="go -C fak run ./cmd/fak", help="command prefix used to invoke fak")
    ap.add_argument("--gateway-url", default="", help="existing fak serve URL; if omitted the runner starts one")
    ap.add_argument("--gateway-port", type=int, default=0)
    ap.add_argument("--gateway-chat", action="store_true", help="also run a live /v1/chat/completions through the gateway")
    ap.add_argument("--perf-decode-baseline-tps", type=float, default=0.0, help="annotate gateway chat decode tok/s as a ratio against this baseline")
    ap.add_argument("--min-decode-tps", type=float, default=0.0, help="fail gateway chat if measured decode tok/s is below this threshold")
    ap.add_argument("--gateway-start-timeout-s", type=float, default=30.0)
    ap.add_argument("--http-timeout-s", type=float, default=10.0)
    ap.add_argument("--model-timeout-s", type=int, default=300)
    ap.add_argument("--agent-max-turns", type=int, default=1)
    ap.add_argument("--agent-timeout-s", type=float, default=900.0)
    ap.add_argument("--node-name", default=socket.gethostname().lower())
    ap.add_argument("--out-dir", default="fak/experiments/qwen36")
    ap.add_argument("--out", default="", help="JSON report path")
    ap.add_argument("--markdown", default="", help="Markdown report path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    known = {"agent", "gateway-openai", "mcp-http"}
    unknown = sorted(set(surfaces) - known)
    if unknown:
        raise SystemExit(f"unknown surfaces: {', '.join(unknown)}")
    target_count = len([v for v in (args.base_url, args.endpoint, args.tailscale_node) if v])
    if target_count > 1:
        raise SystemExit("pass only one of --base-url, --endpoint, or --tailscale-node")
    endpoint: dict[str, Any] | None = None
    if args.endpoint:
        endpoint = resolve_endpoint(args.endpoint, args.registry, dry_run=args.dry_run, timeout_s=args.http_timeout_s)
        if endpoint.get("base_url"):
            args.base_url = str(endpoint["base_url"])
    if args.tailscale_node:
        endpoint = resolve_tailscale_node(
            args.tailscale_node,
            serve_port=args.serve_port,
            dry_run=args.dry_run,
            timeout_s=args.http_timeout_s,
        )
        if endpoint.get("base_url"):
            args.base_url = str(endpoint["base_url"])
    if not args.base_url and endpoint is None:
        raise SystemExit("pass --base-url, --endpoint, or --tailscale-node")

    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = env_for(args)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "node_name": args.node_name,
        "model": args.model,
        "base_url": args.base_url,
        "provider": args.provider,
        "provider_extra_body": args.provider_extra_body,
        "dry_run": args.dry_run,
        "endpoint": endpoint,
        "surfaces": [],
    }

    if endpoint is not None and endpoint.get("state") not in {"READY", "PLANNED"}:
        report["surfaces"].append({
            "surface": "endpoint",
            "status": "FAIL",
            "endpoint_state": endpoint.get("state"),
            "detail": endpoint.get("detail"),
        })
        report["summary"] = {"surfaces": 1, "passed": 0, "failed": 1}
        out = args.out or str(out_dir / f"qwen36-surface-smoke-{args.node_name}.json")
        write_report(out, report)
        if args.markdown:
            Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
            Path(args.markdown).write_text(markdown(report), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        return 3

    proc: subprocess.Popen[str] | None = None
    try:
        if "agent" in surfaces:
            report["surfaces"].append(surface_agent(args, env, out_dir))
        if "gateway-openai" in surfaces or "mcp-http" in surfaces:
            proc, gateway_url, gateway = start_gateway(args, env)
            report["gateway"] = gateway
            if gateway.get("mode") in {"start-failed", "start-timeout"}:
                for surface in [s for s in surfaces if s.startswith("gateway") or s.startswith("mcp")]:
                    report["surfaces"].append({"surface": surface, "status": "FAIL", "gateway": gateway})
            elif gateway.get("mode") == "planned":
                for surface in [s for s in surfaces if s.startswith("gateway") or s.startswith("mcp")]:
                    report["surfaces"].append({"surface": surface, "status": "PLANNED", "gateway": gateway})
            else:
                if "gateway-openai" in surfaces:
                    report["surfaces"].append(surface_gateway_openai(args, gateway_url))
                if "mcp-http" in surfaces:
                    report["surfaces"].append(surface_mcp_http(args, gateway_url))
    finally:
        stop_gateway(proc)

    report["summary"] = {
        "surfaces": len(report["surfaces"]),
        "passed": len([s for s in report["surfaces"] if s.get("status") in {"PASS", "PLANNED"}]),
        "failed": len([s for s in report["surfaces"] if s.get("status") == "FAIL"]),
    }

    out = args.out or str(out_dir / f"qwen36-surface-smoke-{args.node_name}.json")
    write_report(out, report)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
