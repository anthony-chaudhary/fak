#!/usr/bin/env python3
"""SWE-bench Verified resolve/completion: fak-gateway vs a raw serving engine.

The throughput ladder (``dgx_qwen36_27b_runner.py``) load-compares the fak gateway
against raw SGLang on tokens/sec. This is the missing *resolve-rate* arm: it runs a
SWE-bench Verified subset through a real coding agent (``mini-swe-agent``) twice —
once against the raw OpenAI endpoint, once against the same model fronted by
``fak serve`` — then grades both with the OFFICIAL harness and reports OVERALL
COMPLETION (did the agent finish + produce a patch) and RESOLVE-RATE side by side.

It is the head-to-head "comparable" metric the SWE-bench doc marks serving-node
gated: the question is whether routing every tool turn through fak's adjudication
gateway changes whether the instance gets solved. The historical default remains
Qwen3.6-27B on raw SGLang. The same driver can now be pointed at GLM-5.2 on raw
vLLM, including the 20-task SWE-bench Verified slice the DGX benchmark plan needs.

Design: self-contained (stdlib only), idempotent, and heavily logged so it can be
launched DETACHED via the Slack control bridge and polled from a fresh session —
everything lands in a host-shared run dir (default /tmp/swe-cmp-1) plus a DONE.rc
sentinel written by the launcher.

Serving path = external engine TP=8 + ``fak serve`` gateway, NOT fak's native CUDA
engine (which cannot serve a multi-GPU 27B/753B model today). Both agent arms hit
the SAME served weights; the only difference is the fak gateway in the request path.

Usage on the DGX (cwd /srv/fleet):
    python3 tools/dgx_swebench_compare.py --run-dir /tmp/swe-cmp-1 \
        --filter 'astropy__astropy-12907'        # one instance, many turns
    python3 tools/dgx_swebench_compare.py --run-dir /tmp/swe-cmp-1 \
        --slice 0:3 --skip-serve                  # reuse already-up servers
    python3 tools/dgx_swebench_compare.py --run-dir /tmp/swe-glm52-vllm-20 \
        --engine vllm --model zai-org/GLM-5.2-FP8 --served-model-name glm-5.2 \
        --raw-base-url http://127.0.0.1:8000/v1 --verified-count 20 \
        --skip-engine-serve --require-gpu-name H200      # or B200/GB200/etc.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- Historical DGX defaults (verified 2026-06-22; see qwen36 bring-up notes) ---
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_SERVED = "qwen36-27b"  # clean OpenAI model id (SGLang --served-model-name)
DEFAULT_ENGINE = "sglang"
DEFAULT_TP = 8
DEFAULT_MEM_FRACTION = "0.75"  # fits GPU0 alongside the resident GLM llama-server
DEFAULT_SGLANG_PY = "/root/sglang-stock/bin/python"
DEFAULT_ENGINE_PORT = 30000
DEFAULT_FAK_BIN = "/srv/fleet/tools/.bin/fak"
DEFAULT_FAK_PORT = 8080
VENV = Path("/root/venvs/mini-swe-agent")
MINI = str(VENV / "bin" / "mini-extra")
VPY = str(VENV / "bin" / "python")
DATASET = "princeton-nlp/SWE-bench_Verified"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status < 500
    except Exception:
        return False


def wait_health(url: str, label: str, deadline_s: int) -> bool:
    log(f"waiting for {label} at {url} (<= {deadline_s}s)")
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if http_ok(url):
            log(f"{label} READY after {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"!! {label} NOT ready after {deadline_s}s")
    return False


def popen_logged(cmd, logpath: Path, env=None):
    log(f"launch: {' '.join(shlex.quote(c) for c in cmd)}  (log -> {logpath})")
    fh = open(logpath, "w")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh


def write_done_rc(run_dir: Path, rc: int) -> None:
    """Write the detached-run completion sentinel documented by this runner."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "DONE.rc").write_text(f"{int(rc)}\n", encoding="utf-8")
    log(f"wrote {run_dir/'DONE.rc'} rc={int(rc)}")


def write_compare_preflight(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "COMPARE-PREFLIGHT.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    log(f"wrote {run_dir/'COMPARE-PREFLIGHT.json'}")


def root_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def raw_arm_name(engine: str) -> str:
    return f"raw-{engine}"


def raw_api_base(args) -> str:
    if args.raw_base_url:
        return args.raw_base_url.rstrip("/")
    return f"http://127.0.0.1:{args.engine_port}/v1"


def gateway_api_base(args) -> str:
    if args.gateway_base_url:
        base = args.gateway_base_url.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"
    return f"http://127.0.0.1:{args.fak_port}/v1"


def arm_specs(args) -> list[tuple[str, str]]:
    return [(raw_arm_name(args.engine), raw_api_base(args)),
            ("fak-gateway", gateway_api_base(args))]


def selected_arms(args) -> set[str]:
    return set(args.arms.split(",")) if args.arms else {raw_arm_name(args.engine), "fak-gateway"}


def command_exists(cmd: str) -> bool:
    if not cmd:
        return False
    exe = shlex.split(cmd, posix=(os.name != "nt"))[0].strip('"')
    p = Path(exe)
    if p.is_absolute() or p.parent != Path("."):
        return p.exists()
    return shutil.which(exe) is not None


def check_python_import(python: str, module: str, timeout_s: int = 30) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [python, "-c", f"import {module}; print('ok')"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return False, repr(exc)
    detail = (proc.stderr or proc.stdout or "").strip()
    return proc.returncode == 0, detail


def command_output(cmd: list[str], timeout_s: int = 10) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:
        return {"ok": False, "detail": repr(exc), "cmd": cmd}
    detail = (proc.stdout or proc.stderr or "").strip().splitlines()
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "detail": detail[0] if detail else "",
        "cmd": cmd,
    }


def python_expr(python: str, expr: str, timeout_s: int = 10) -> dict:
    return command_output([python, "-c", expr], timeout_s=timeout_s)


def endpoint_ready(api_base: str) -> tuple[bool, str]:
    models = api_base.rstrip("/") + "/models"
    health = root_url(api_base) + "/health"
    if http_ok(models):
        return True, models
    if http_ok(health):
        return True, health
    return False, f"{models} and {health} did not return <500"


def preflight_runtime(args) -> dict:
    runtime = {
        "mini_swe_agent": command_output([MINI, "--version"], timeout_s=10),
        "swebench_python": python_expr(VPY, "import sys; print(sys.version.split()[0])"),
        "swebench_version": python_expr(
            VPY,
            "import importlib.metadata as m; print(m.version('swebench'))",
        ),
    }
    if args.engine == "vllm":
        runtime["vllm_version"] = python_expr(
            VPY,
            "import importlib.metadata as m; print(m.version('vllm'))",
        )
        if not args.skip_engine_serve:
            runtime["vllm_command"] = command_output(
                shlex.split(args.vllm_command) + ["--version"], timeout_s=10
            )
    elif args.engine == "sglang":
        runtime["sglang_version"] = python_expr(
            args.sglang_python,
            "import importlib.metadata as m; print(m.version('sglang'))",
        )
    if not args.skip_gateway_serve:
        runtime["fak"] = command_output([args.fak_bin, "--version"], timeout_s=10)
    return runtime


def preflight_config(args) -> dict:
    return {
        "raw_base_url": raw_api_base(args),
        "gateway_base_url": gateway_api_base(args),
        "engine_port": args.engine_port,
        "fak_port": args.fak_port,
        "skip_engine_serve": bool(args.skip_engine_serve),
        "skip_gateway_serve": bool(args.skip_gateway_serve),
        "tp": args.tp,
        "mem_fraction": args.mem_fraction,
        "context_length": args.context_length,
        "engine_args": args.engine_args,
        "tool_call_parser": args.tool_call_parser,
        "grade_workers": args.grade_workers,
        "grade_timeout": args.grade_timeout,
        "agent_timeout": args.agent_timeout,
        "workers": args.workers,
        "max_iterations": args.max_iterations,
        "require_gpu_name": args.require_gpu_name,
    }


def compare_preflight(run_dir: Path, args, env_info: dict, want: set[str]) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("mini-swe-agent", command_exists(MINI), MINI)
    add("swebench-python", command_exists(VPY), VPY)
    ok, detail = check_python_import(VPY, "swebench.harness.run_evaluation")
    add("swebench-harness-import", ok, detail)

    if not args.skip_engine_serve:
        if args.engine == "sglang":
            add("sglang-python", command_exists(args.sglang_python), args.sglang_python)
        elif args.engine == "vllm":
            add("vllm-command", command_exists(args.vllm_command), args.vllm_command)
    elif raw_arm_name(args.engine) in want:
        ok, detail = endpoint_ready(raw_api_base(args))
        add(f"{raw_arm_name(args.engine)}-endpoint", ok, detail)

    if not args.skip_gateway_serve:
        add("fak-bin", command_exists(args.fak_bin), args.fak_bin)
    elif "fak-gateway" in want:
        ok, detail = endpoint_ready(gateway_api_base(args))
        add("fak-gateway-endpoint", ok, detail)

    if args.require_gpu_name:
        gpu0 = str(env_info.get("gpu0", ""))
        add("required-gpu-name", args.require_gpu_name in gpu0,
            f"gpu0={gpu0!r} require={args.require_gpu_name!r}")

    payload = {
        "schema": "fak.dgx-swebench-compare-preflight.v1",
        "ok": all(row["ok"] for row in checks),
        "model": args.model,
        "served_as": args.served_model_name,
        "engine": args.engine,
        "selection": args.filter or f"slice {args.slice}",
        "arms": sorted(want),
        "config": preflight_config(args),
        "runtime": preflight_runtime(args),
        "env": env_info,
        "checks": checks,
    }
    write_compare_preflight(run_dir, payload)
    return payload


def build_engine_command(args) -> list[str]:
    """Return the serving-engine command for the non-`--skip-serve` path.

    SGLang keeps the historical Qwen3.6 defaults. vLLM is intentionally generic:
    model-specific tool-call parser flags are passed via --engine-args so the
    report records exactly what the operator ran instead of baking a stale parser
    guess into this harness.
    """
    extra = shlex.split(args.engine_args) if args.engine_args else []
    if args.engine == "sglang":
        cmd = [args.sglang_python, "-m", "sglang.launch_server",
               "--model-path", args.model, "--served-model-name", args.served_model_name,
               "--port", str(args.engine_port), "--host", "127.0.0.1",
               "--tp", str(args.tp), "--trust-remote-code",
               "--mem-fraction-static", args.mem_fraction]
        if args.tool_call_parser:
            cmd += ["--tool-call-parser", args.tool_call_parser]
        return cmd + extra
    if args.engine == "vllm":
        cmd = shlex.split(args.vllm_command) + ["serve", args.model,
               "--served-model-name", args.served_model_name,
               "--tensor-parallel-size", str(args.tp),
               "--trust-remote-code",
               "--host", "127.0.0.1", "--port", str(args.engine_port)]
        if args.context_length:
            cmd += ["--max-model-len", str(args.context_length)]
        return cmd + extra
    raise ValueError(f"unknown engine {args.engine!r}")


def apply_selection_shortcuts(args) -> None:
    if args.verified_count > 0:
        args.filter = ""
        args.slice = f"0:{args.verified_count}"


def serve_engine(run_dir: Path, args):
    env = os.environ.copy()
    if args.engine == "sglang":
        env["PATH"] = str(Path(args.sglang_python).parent) + os.pathsep + env.get("PATH", "")
    # demote the TP memory-balance heuristic (GPU0 is GLM-occupied, so per-GPU free
    # VRAM is uneven) — --mem-fraction-static is the real OOM guard.
    env["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "0"
    return popen_logged(build_engine_command(args), run_dir / f"{args.engine}.log", env)


def tool_probe(model: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Use the bash tool to list files in /testbed."}],
        "tools": [{"type": "function", "function": {
            "name": "bash", "description": "run a bash command",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}],
        "max_tokens": 700,
    }


def probe_toolcall_result(api_base: str, model: str) -> dict:
    """Single tool-calling request; return a structured tool-call readiness probe.

    mini-swe-agent needs OpenAI tool_calls for its bash action. A model that answers
    in plain text can look "healthy" at /v1/models while every SWE-bench instance is
    doomed, so GLM/vLLM runs can promote this probe to a hard gate.
    """
    import urllib.error
    url = api_base.rstrip("/") + "/chat/completions"
    out = {"endpoint": api_base, "ok": False, "tool_calls": 0, "status": "FAIL"}
    try:
        req = urllib.request.Request(url, data=json.dumps(tool_probe(model)).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        ch = r["choices"][0]
        m = ch["message"]
        tc = m.get("tool_calls") or []
        out.update({
            "ok": bool(tc),
            "status": "PASS" if tc else "FAIL",
            "finish_reason": ch.get("finish_reason"),
            "tool_calls": len(tc),
            "content_len": len(m.get("content") or ""),
            "completion_tokens": r.get("usage", {}).get("completion_tokens"),
        })
        return out
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:160]
        except Exception:
            pass
        out.update({"status": "HTTP_ERROR", "http_status": e.code, "detail": body})
        return out
    except Exception as e:
        out.update({"status": "ERROR", "detail": repr(e)})
        return out


def format_toolcall_probe(probe: dict) -> str:
    if probe.get("ok"):
        return (f"{probe.get('endpoint')}: finish={probe.get('finish_reason')} "
                f"tool_calls=YES({probe.get('tool_calls')}) "
                f"content_len={probe.get('content_len')} "
                f"completion_tokens={probe.get('completion_tokens')}")
    if probe.get("status") == "HTTP_ERROR":
        return f"{probe.get('endpoint')}: HTTP {probe.get('http_status')} {probe.get('detail', '')}"
    if probe.get("detail"):
        return f"{probe.get('endpoint')}: {probe.get('status')} {probe.get('detail')}"
    return (f"{probe.get('endpoint')}: finish={probe.get('finish_reason')} "
            f"tool_calls=NONE content_len={probe.get('content_len')} "
            f"completion_tokens={probe.get('completion_tokens')}")


def probe_toolcall(api_base: str, model: str) -> str:
    """Backward-compatible text form for logs/tests."""
    return format_toolcall_probe(probe_toolcall_result(api_base, model))


def serve_fak(run_dir: Path, args):
    cmd = [args.fak_bin, "serve", "--addr", f"127.0.0.1:{args.fak_port}",
           "--provider", "openai",
           "--base-url", raw_api_base(args),
           "--model", args.served_model_name, "--api-key-env", "NONE_LOCAL"]
    return popen_logged(cmd, run_dir / "fak.log")


def served_models(api_base: str) -> str:
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as r:
            return r.read().decode()[:400]
    except Exception as e:
        return f"(error: {e})"


def run_agent(arm: str, api_base: str, run_dir: Path, args) -> dict:
    """Run mini-swe-agent on the subset against one endpoint. Returns a result dict."""
    preds_dir = run_dir / f"preds_{arm}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    cmd = [MINI, "swebench",
           "--subset", "verified", "--split", "test",
           "-w", str(args.workers), "-o", str(preds_dir),
           "-m", f"openai/{args.served_model_name}",
           "-c", "swebench.yaml",
           "-c", f"model.model_kwargs.api_base={api_base}",
           "-c", "model.model_kwargs.api_key=EMPTY"]
    if args.filter:
        cmd += ["--filter", args.filter]
    else:
        cmd += ["--slice", args.slice]
    if args.max_iterations:
        cmd += ["-c", f"agent.step_limit={args.max_iterations}"]
    if args.redo:
        cmd += ["--redo-existing"]

    log(f"=== ARM {arm}: agent against {api_base} ===")
    log(f"served models @ {api_base}: {served_models(api_base)}")
    alog = preds_dir / "agent.log"
    t0 = time.time()
    rc = None
    with open(alog, "w") as fh:
        try:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        except OSError as e:
            fh.write(f"failed to launch mini-swe-agent: {e}\n")
            log(f"!! {arm} agent LAUNCH FAILED: {e}")
            return {
                "arm": arm, "endpoint": api_base, "agent_rc": -127,
                "agent_sec": round(time.time() - t0, 1), "preds_path": None,
                "instances": 0, "completed": 0, "patch_bytes": 0,
                "instance_ids": [], "agent_error": f"launch failed: {e}",
            }
        try:
            rc = p.wait(timeout=args.agent_timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            log(f"!! {arm} agent TIMED OUT after {args.agent_timeout}s")
            rc = -9
    dt = time.time() - t0
    log(f"ARM {arm}: agent rc={rc} in {dt:.0f}s")

    preds_path, preds = find_predictions(preds_dir)
    completed = sum(1 for p in preds if (p.get("model_patch") or "").strip())
    patch_bytes = sum(len(p.get("model_patch") or "") for p in preds)
    return {
        "arm": arm, "endpoint": api_base, "agent_rc": rc,
        "agent_sec": round(dt, 1), "preds_path": str(preds_path) if preds_path else None,
        "instances": len(preds), "completed": completed, "patch_bytes": patch_bytes,
        "instance_ids": [p.get("instance_id") for p in preds],
    }


def find_predictions(preds_dir: Path):
    """Locate mini-swe-agent's predictions file and normalize to a list of dicts."""
    cands = [preds_dir / "preds.json"] + sorted(preds_dir.glob("*.json"))
    seen = set()
    for c in cands:
        if not c.exists() or c in seen:
            continue
        seen.add(c)
        try:
            data = json.loads(c.read_text())
        except Exception:
            continue
        recs = []
        if isinstance(data, dict):
            # dict keyed by instance_id -> pred, OR a single pred
            if "model_patch" in data or "model_name_or_path" in data:
                recs = [data]
            else:
                for k, v in data.items():
                    if isinstance(v, dict):
                        v.setdefault("instance_id", k)
                        recs.append(v)
        elif isinstance(data, list):
            recs = [r for r in data if isinstance(r, dict)]
        if recs and any("model_patch" in r for r in recs):
            return c, recs
    return None, []


def grade(arm: str, preds_path: str, run_dir: Path, args, submitted: int) -> dict:
    """Grade a predictions file with the official harness; parse resolve count.

    `submitted` is the number of instances we actually ran (the subset size) —
    used as the resolve-rate denominator. The harness report.json's
    `total_instances` is the WHOLE dataset (500 for Verified), so trusting it
    would report 1/500 instead of the honest 1/1 for a 1-instance subset."""
    if not preds_path:
        return {"available": False, "reason": "no predictions file produced"}
    run_id = f"swecmp-{arm}"
    abspreds = str(Path(preds_path).resolve())
    cmd = [VPY, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", DATASET, "--predictions_path", abspreds,
           "--run_id", run_id, "--max_workers", str(args.grade_workers)]
    log(f"=== GRADE {arm}: {' '.join(shlex.quote(c) for c in cmd)} ===")
    glog = run_dir / f"grade_{arm}.log"
    t0 = time.time()
    cwd = str(Path(preds_path).parent)
    resolved = total = -1
    grade_rc = None
    grade_error = ""
    try:
        with open(glog, "w") as fh:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 cwd=cwd, text=True)
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                m = re.search(r"Instances resolved:\s*(\d+)", line)
                if m:
                    resolved = int(m.group(1))
                m = re.search(r"Total instances:\s*(\d+)", line)
                if m:
                    total = int(m.group(1))
                m = re.search(r"Resolved\s+(\d+)\s*/\s*(\d+)", line)
                if m:
                    resolved, total = int(m.group(1)), int(m.group(2))
            grade_rc = p.wait(timeout=args.grade_timeout)
    except Exception as e:
        grade_error = repr(e)
        log(f"!! grade {arm} error: {e}")
    dt = time.time() - t0

    rep = find_report(cwd, run_id)
    res = {"available": True, "grade_sec": round(dt, 1), "run_id": run_id,
           "submitted": submitted, "grade_log": str(glog)}
    if grade_rc is not None:
        res["grade_rc"] = grade_rc
    if grade_error:
        res["available"] = False
        res["reason"] = f"grade harness failed: {grade_error}"
    if rep:
        res["report_path"] = str(rep)
        try:
            d = json.loads(rep.read_text())
            rids = d.get("resolved_ids") or d.get("resolved_instances") or []
            res["resolved_ids"] = rids
            res["resolved"] = len(rids) if isinstance(rids, list) else int(rids)
            res["dataset_total"] = d.get("total_instances")  # 500 for Verified
            res["available"] = True
            res.pop("reason", None)
        except Exception as e:
            log(f"!! parse report {rep}: {e}")
            res["available"] = False
            res["reason"] = f"could not parse grade report: {e}"
    if "resolved" not in res and resolved >= 0:
        res["resolved"] = resolved
    # denominator = instances WE ran (the subset), not the 500-set
    res["total"] = submitted
    if res.get("total"):
        res["resolve_pct"] = round(100 * res.get("resolved", 0) / res["total"], 1)
    if grade_rc not in (None, 0):
        res["available"] = False
        res["reason"] = f"grade harness exited {grade_rc}"
    log(f"GRADE {arm}: resolved={res.get('resolved')}/{res.get('total')} "
        f"(dataset_total={res.get('dataset_total')}) in {dt:.0f}s")
    return res


def find_report(cwd: str, run_id: str):
    base = Path(cwd)
    pats = [f"*{run_id}*.json",
            f"logs/run_evaluation/{run_id}/**/report.json",
            "logs/run_evaluation/**/report.json"]
    best = None
    for pat in pats:
        for f in base.glob(pat):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            if any(k in d for k in ("resolved_ids", "resolved_instances", "total_instances")):
                if best is None or f.stat().st_mtime > best.stat().st_mtime:
                    best = f
    return best


def write_compare(run_dir: Path, env_info: dict, results: list, args,
                  tool_call_probes: list[dict] | None = None) -> None:
    out = {"benchmark": "swe-bench-verified-resolve-compare",
           "schema": "fak.dgx-swebench-compare.v1",
           "model": args.model, "served_as": args.served_model_name,
           "engine": args.engine, "tp": args.tp,
           "dataset": DATASET, "selection": args.filter or f"slice {args.slice}",
           "env": env_info, "arms": results}
    arm_instance_ids = {
        r["agent"]["arm"]: r["agent"].get("instance_ids")
        for r in results
        if isinstance(r.get("agent"), dict)
        and isinstance(r["agent"].get("arm"), str)
        and isinstance(r["agent"].get("instance_ids"), list)
    }
    if arm_instance_ids:
        lists = list(arm_instance_ids.values())
        out["selection_instance_ids"] = lists[0]
        out["selection_instance_ids_match"] = all(ids == lists[0] for ids in lists)
    if tool_call_probes is not None:
        out["tool_call_probes"] = tool_call_probes
    (run_dir / "compare.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    def row(r):
        a, g = r["agent"], r["grade"]
        rr = f"{g.get('resolved','?')}/{g.get('total','?')}" if g.get("available") else "n/a"
        return (f"| {a['arm']} | `{a['endpoint']}` | {a['instances']} | "
                f"{a['completed']} | {a['patch_bytes']} | {a['agent_sec']}s | "
                f"{rr} | {g.get('resolve_pct','-')}% | {g.get('grade_sec','-')}s |")

    md = [f"# SWE-bench Verified — fak-gateway vs raw-{args.engine} (overall completion)",
          "",
          f"**Model:** `{args.model}` served as `{args.served_model_name}` "
          f"({args.engine}, TP={args.tp}) · "
          f"**Selection:** {args.filter or 'slice '+args.slice} · "
          f"**Host:** {env_info.get('host','?')}",
          "",
          "Both arms drive the identical `mini-swe-agent` against the SAME served "
          "weights; the only difference is whether each tool turn is routed through "
          f"the `fak serve` adjudication gateway or hits raw {args.engine} directly. "
          "*Overall completion* = the agent ran to the end and "
          "emitted a non-empty patch; *resolved* = the official harness "
          "(`swebench.harness.run_evaluation`) PASS_TO_PASS + FAIL_TO_PASS grade.",
          "",
          "| arm | endpoint | instances | completed | patch bytes | agent time | "
          "resolved | resolve% | grade time |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    md += [row(r) for r in results]
    if out.get("selection_instance_ids"):
        md += ["", "## SWE-bench Instances", ""]
        if out.get("selection_instance_ids_match"):
            md.append("Both arms ran the same instance IDs:")
        else:
            md.append("WARNING: arm instance IDs differed:")
        for instance_id in out["selection_instance_ids"]:
            md.append(f"- `{instance_id}`")
    if tool_call_probes is not None:
        md += ["", "## Tool-Call Self-Test", "",
               "| arm | endpoint | status | tool calls | detail |",
               "|---|---|---:|---:|---|"]
        for p in tool_call_probes:
            md.append(
                f"| {p.get('arm', '-')} | `{p.get('endpoint', '-')}` | "
                f"{p.get('status', '-')} | {p.get('tool_calls', 0)} | "
                f"{p.get('detail') or p.get('finish_reason') or ''} |"
            )
    # verdict
    g = {r["agent"]["arm"]: r["grade"] for r in results}
    a = {r["agent"]["arm"]: r["agent"] for r in results}
    raw_name = raw_arm_name(args.engine)
    if raw_name in g and "fak-gateway" in g:
        rs, fk = g[raw_name], g["fak-gateway"]
        same = rs.get("resolved") == fk.get("resolved")
        md += ["", "## Verdict",
               f"- **{raw_name}:** completed {a[raw_name]['completed']}/"
               f"{a[raw_name]['instances']}, resolved "
               f"{rs.get('resolved','?')}/{rs.get('total','?')}.",
               f"- **fak-gateway:** completed {a['fak-gateway']['completed']}/"
               f"{a['fak-gateway']['instances']}, resolved "
               f"{fk.get('resolved','?')}/{fk.get('total','?')}.",
               f"- Overall completion through the fak gateway "
               f"{'MATCHES' if same else 'DIFFERS from'} {raw_name} on this "
               f"selection — routing every tool turn through fak's adjudication "
               f"plane did {'not change' if same else 'change'} the resolve outcome."]
    (run_dir / "COMPARE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {run_dir/'compare.json'} and {run_dir/'COMPARE.md'}")


def grade_has_resolve(g: dict, submitted: int) -> bool:
    if g.get("available") is not True:
        return False
    if "grade_rc" in g and g.get("grade_rc") != 0:
        return False
    if not g.get("report_path"):
        return False
    if not g.get("grade_log"):
        return False
    if g.get("submitted") != submitted:
        return False
    if g.get("total") != submitted:
        return False
    if g.get("resolve_pct") is None:
        return False
    try:
        resolved = int(g.get("resolved"))
    except (TypeError, ValueError):
        return False
    resolved_ids = g.get("resolved_ids")
    if not isinstance(resolved_ids, list):
        return False
    if len(resolved_ids) != resolved:
        return False
    return 0 <= resolved <= submitted


def probe_env(args) -> dict:
    if hasattr(os, "uname"):
        host = os.uname().nodename
    else:
        host = os.environ.get("COMPUTERNAME", "?")
    info = {"host": host, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout or "nvidia-smi failed").strip())
        lines = out.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError("nvidia-smi returned no GPU rows")
        info["gpu0"] = lines[0] if lines else "?"
        info["ngpu"] = str(len(lines)) if lines else "?"
    except Exception as e:
        try:
            out = subprocess.run([args.sglang_python, "-c",
                                  "import torch;print(torch.cuda.get_device_name(0));print(torch.cuda.device_count())"],
                                 capture_output=True, text=True, timeout=60)
            lines = out.stdout.strip().splitlines()
            info["gpu0"] = lines[0] if lines else "?"
            info["ngpu"] = lines[1] if len(lines) > 1 else "?"
        except Exception:
            info["gpu0"] = f"probe error: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/tmp/swe-cmp-1")
    ap.add_argument("--engine", default=DEFAULT_ENGINE, choices=["sglang", "vllm"],
                    help="raw serving engine to compare against fak-gateway")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--served-model-name", default=DEFAULT_SERVED)
    ap.add_argument("--tp", type=int, default=DEFAULT_TP)
    ap.add_argument("--mem-fraction", default=DEFAULT_MEM_FRACTION)
    ap.add_argument("--context-length", type=int, default=0,
                    help="vLLM --max-model-len when starting the engine (0 = engine default)")
    ap.add_argument("--engine-port", type=int, default=DEFAULT_ENGINE_PORT)
    ap.add_argument("--fak-port", type=int, default=DEFAULT_FAK_PORT)
    ap.add_argument("--raw-base-url", default="",
                    help="existing raw engine /v1 base URL; overrides --engine-port")
    ap.add_argument("--gateway-base-url", default="",
                    help="existing fak gateway origin or /v1 base; overrides --fak-port")
    ap.add_argument("--sglang-python", default=DEFAULT_SGLANG_PY)
    ap.add_argument("--vllm-command", default="vllm")
    ap.add_argument("--fak-bin", default=DEFAULT_FAK_BIN)
    ap.add_argument("--tool-call-parser", default="qwen3_coder",
                    help="SGLang parser for OpenAI tool_calls; empty disables the flag")
    ap.add_argument("--engine-args", default="",
                    help="extra raw engine args, shell-split and appended to the serve command")
    ap.add_argument("--require-gpu-name", default="A100",
                    help="substring required in gpu0 name before claiming a result; empty disables")
    ap.add_argument("--filter", default="astropy__astropy-12907",
                    help="instance-id regex (1 instance, image cached). Empty => use --slice")
    ap.add_argument("--slice", default="0:1", help="used when --filter is empty")
    ap.add_argument("--verified-count", type=int, default=0,
                    help="shortcut for --filter '' --slice 0:N, e.g. 20 for a 20-task sample")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=0, help="agent step_limit (0=config default)")
    ap.add_argument("--agent-timeout", type=int, default=3600)
    ap.add_argument("--grade-workers", type=int, default=2)
    ap.add_argument("--grade-timeout", type=int, default=2400)
    ap.add_argument("--skip-serve", action="store_true", help="reuse already-running endpoints")
    ap.add_argument("--skip-engine-serve", action="store_true",
                    help="reuse the raw engine endpoint but start/probe fak-gateway")
    ap.add_argument("--skip-gateway-serve", action="store_true",
                    help="reuse the fak gateway endpoint but start/probe the raw engine")
    ap.add_argument("--preflight-only", action="store_true",
                    help="write COMPARE-PREFLIGHT.json and DONE.rc, then exit before serving/agents")
    ap.add_argument("--stop-serve", action="store_true", help="kill SGLang+gateway at the end")
    ap.add_argument("--redo", action="store_true", help="redo existing instances")
    ap.add_argument("--arms", default="",
                    help="comma-separated arms to run; default raw-<engine>,fak-gateway")
    ap.add_argument("--require-tool-calls", action="store_true",
                    help="fail before agents if any selected endpoint does not emit OpenAI tool_calls")
    ap.add_argument("--require-grade", action="store_true",
                    help="fail unless every selected arm has an official resolved/total grade")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    apply_selection_shortcuts(args)
    log(f"=== SWE-bench Verified resolve compare: fak-gateway vs raw-{args.engine} ===")
    log(f"run_dir={run_dir} model={args.model} served={args.served_model_name} selection={args.filter or args.slice}")
    env_info = probe_env(args)
    log(f"env: {json.dumps(env_info)}")
    if args.require_gpu_name and args.require_gpu_name not in str(env_info.get("gpu0", "")):
        log(f"!! gpu0 does not contain {args.require_gpu_name!r} — refusing to claim this result. Aborting.")
        write_done_rc(run_dir, 2)
        return 2

    procs = []
    if args.skip_serve:
        args.skip_engine_serve = True
        args.skip_gateway_serve = True

    want = selected_arms(args)
    preflight = compare_preflight(run_dir, args, env_info, want)
    if args.preflight_only:
        rc = 0 if preflight.get("ok") else 7
        write_done_rc(run_dir, rc)
        return rc
    if not preflight.get("ok"):
        bad = [row["name"] for row in preflight.get("checks", []) if not row.get("ok")]
        log("!! compare preflight failed: " + ", ".join(bad))
        write_done_rc(run_dir, 7)
        return 7

    if not args.skip_engine_serve:
        eng_p, _ = serve_engine(run_dir, args)
        procs.append((args.engine, eng_p))
        if not wait_health(root_url(raw_api_base(args)) + "/health", args.engine, 900):
            log(f"!! {args.engine} failed to come up; see {args.engine}.log")
            write_done_rc(run_dir, 3)
            return 3
    else:
        log(f"--skip-engine-serve: assuming raw {args.engine} endpoint already up at {raw_api_base(args)}")

    if not args.skip_gateway_serve:
        fak_p, _ = serve_fak(run_dir, args)
        procs.append(("fak", fak_p))
        # this fak build has no /health route (404), so probe /v1/models first.
        if not (wait_health(gateway_api_base(args).rstrip("/") + "/models", "fak-gateway", 120)
                or wait_health(root_url(gateway_api_base(args)) + "/health", "fak-gateway(health)", 30)):
            log("!! fak gateway failed to come up; see fak.log")
            write_done_rc(run_dir, 4)
            return 4
    else:
        log(f"--skip-gateway-serve: assuming fak-gateway endpoint already up at {gateway_api_base(args)}")

    # tool-call self-test BEFORE the agent runs — proves the serving config
    # emits OpenAI tool_calls (mini-swe-agent's bash action). A poll of run.log
    # sees this within ~90s, so a misconfig is caught before a long wasted run.
    log("--- tool-call self-test ---")
    tool_call_probes = []
    for arm, api_base in arm_specs(args):
        if arm not in want:
            continue
        probe = probe_toolcall_result(api_base, args.served_model_name)
        probe["arm"] = arm
        tool_call_probes.append(probe)
        log(f"  {arm}: " + format_toolcall_probe(probe))
    log("--- end self-test ---")
    exit_code = 0
    tool_failures = [p["arm"] for p in tool_call_probes if not p.get("ok")]
    if args.require_tool_calls and tool_failures:
        log("!! selected endpoint(s) did not emit tool_calls: " + ", ".join(tool_failures))
        write_compare(run_dir, env_info, [], args, tool_call_probes)
        exit_code = 5

    results = []
    if exit_code == 0:
        for arm, api_base in arm_specs(args):
            if arm not in want:
                continue
            a = run_agent(arm, api_base, run_dir, args)
            g = grade(arm, a.get("preds_path"), run_dir, args, a["instances"])
            if args.require_grade and not grade_has_resolve(g, a["instances"]):
                log(f"!! {arm} missing official resolved/total grade: {g}")
                exit_code = 6
            results.append({"agent": a, "grade": g})
            # checkpoint after each arm so a poll mid-run sees partial results
            write_compare(run_dir, env_info, results, args, tool_call_probes)

    write_compare(run_dir, env_info, results, args, tool_call_probes)
    log("=== DONE ===")

    if args.stop_serve:
        for name, p in procs:
            log(f"stopping {name} (pid {p.pid})")
            p.terminate()
    write_done_rc(run_dir, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
