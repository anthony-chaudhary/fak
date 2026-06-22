#!/usr/bin/env python3
"""SWE-bench Verified resolve/completion: fak-gateway vs raw-SGLang, on the DGX.

The throughput ladder (``dgx_qwen36_27b_runner.py``) load-compares the fak gateway
against raw SGLang on tokens/sec. This is the missing *resolve-rate* arm: it runs a
SWE-bench Verified subset through a real coding agent (``mini-swe-agent``) twice —
once against the raw SGLang OpenAI endpoint, once against the same model fronted by
``fak serve`` — then grades both with the OFFICIAL harness and reports OVERALL
COMPLETION (did the agent finish + produce a patch) and RESOLVE-RATE side by side.

It is the head-to-head "comparable" metric the SWE-bench doc marks DGX-gated:
the question is whether routing every tool turn through fak's adjudication gateway
changes whether the instance gets solved.

Design: self-contained (stdlib only), idempotent, and heavily logged so it can be
launched DETACHED via the Slack control bridge and polled from a fresh session —
everything lands in a host-shared run dir (default /tmp/swe-cmp-1) plus a DONE.rc
sentinel written by the launcher.

Serving path = SGLang TP=8 (bf16) + ``fak serve`` gateway, NOT fak's native CUDA
engine (which cannot serve a multi-GPU 27B). Both agent arms hit the SAME SGLang
weights; the only difference is the fak gateway in the request path.

Usage on the DGX (cwd /srv/fleet):
    python3 tools/dgx_swebench_compare.py --run-dir /tmp/swe-cmp-1 \
        --filter 'astropy__astropy-12907'        # one instance, many turns
    python3 tools/dgx_swebench_compare.py --run-dir /tmp/swe-cmp-1 \
        --slice 0:3 --skip-serve                  # reuse already-up servers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- DGX-fixed constants (verified 2026-06-22; see dgx-qwen36-27b-bringup memory) -
MODEL = "Qwen/Qwen3.6-27B"
SERVED = "qwen36-27b"  # clean OpenAI model id (SGLang --served-model-name)
TP = 8
MEM_FRACTION = "0.75"  # fits GPU0 alongside the resident GLM llama-server
SGLANG_PY = "/root/sglang-stock/bin/python"
SGLANG_PORT = 30000
FAK_BIN = "/srv/fleet/tools/.bin/fak"
FAK_PORT = 8080
VENV = Path("/root/venvs/mini-swe-agent")
MINI = str(VENV / "bin" / "mini-extra")
VPY = str(VENV / "bin" / "python")
DATASET = "princeton-nlp/SWE-bench_Verified"

ARMS = [("raw-sglang", SGLANG_PORT), ("fak-gateway", FAK_PORT)]


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


def serve_sglang(run_dir: Path):
    env = os.environ.copy()
    env["PATH"] = str(Path(SGLANG_PY).parent) + os.pathsep + env.get("PATH", "")
    # demote the TP memory-balance heuristic (GPU0 is GLM-occupied, so per-GPU free
    # VRAM is uneven) — --mem-fraction-static is the real OOM guard.
    env["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "0"
    cmd = [SGLANG_PY, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--served-model-name", SERVED,
           "--port", str(SGLANG_PORT), "--host", "127.0.0.1",
           "--tp", str(TP), "--trust-remote-code",
           "--mem-fraction-static", MEM_FRACTION,
           # mini-swe-agent drives the bash action via the OpenAI tool-calling API,
           # so SGLang MUST parse the model's tool calls into the `tool_calls`
           # field — else the agent sees "no tool calls", loops, and submits an
           # empty patch. Qwen3.6's chat template uses the XML
           # <function=name><parameter=arg> tool format (Qwen3-Coder style), so the
           # parser is `qwen3_coder` — NOT `qwen25` (Hermes JSON), whose grammar
           # mismatched and collapsed generation to 1 token. (--reasoning-parser is
           # omitted: mini reads tool_calls, not reasoning_content.)
           "--tool-call-parser", "qwen3_coder"]
    return popen_logged(cmd, run_dir / "sglang.log", env)


TOOL_PROBE = {
    "model": SERVED,
    "messages": [{"role": "user", "content": "Use the bash tool to list files in /testbed."}],
    "tools": [{"type": "function", "function": {
        "name": "bash", "description": "run a bash command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}],
    "max_tokens": 700,
}


def probe_toolcall(port: int) -> str:
    """Single tool-calling request; report whether the endpoint returns tool_calls.
    Logged before the agent runs so a poll validates the serving config early."""
    import urllib.error
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    try:
        req = urllib.request.Request(url, data=json.dumps(TOOL_PROBE).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        ch = r["choices"][0]
        m = ch["message"]
        tc = m.get("tool_calls")
        return (f"port {port}: finish={ch.get('finish_reason')} "
                f"tool_calls={'YES('+str(len(tc))+')' if tc else 'NONE'} "
                f"content_len={len(m.get('content') or '')} "
                f"completion_tokens={r.get('usage',{}).get('completion_tokens')}")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:160]
        except Exception:
            pass
        return f"port {port}: HTTP {e.code} {body}"
    except Exception as e:
        return f"port {port}: ERROR {e!r}"


def serve_fak(run_dir: Path):
    cmd = [FAK_BIN, "serve", "--addr", f"127.0.0.1:{FAK_PORT}",
           "--provider", "openai",
           "--base-url", f"http://127.0.0.1:{SGLANG_PORT}/v1",
           "--model", SERVED, "--api-key-env", "NONE_LOCAL"]
    return popen_logged(cmd, run_dir / "fak.log")


def served_models(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
            return r.read().decode()[:400]
    except Exception as e:
        return f"(error: {e})"


def run_agent(arm: str, port: int, run_dir: Path, args) -> dict:
    """Run mini-swe-agent on the subset against one endpoint. Returns a result dict."""
    preds_dir = run_dir / f"preds_{arm}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    api_base = f"http://127.0.0.1:{port}/v1"
    cmd = [MINI, "swebench",
           "--subset", "verified", "--split", "test",
           "-w", str(args.workers), "-o", str(preds_dir),
           "-m", f"openai/{SERVED}",
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
    log(f"served models @ {port}: {served_models(port)}")
    alog = preds_dir / "agent.log"
    t0 = time.time()
    rc = None
    with open(alog, "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
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
            p.wait(timeout=args.grade_timeout)
    except Exception as e:
        log(f"!! grade {arm} error: {e}")
    dt = time.time() - t0

    rep = find_report(cwd, run_id)
    res = {"available": True, "grade_sec": round(dt, 1), "run_id": run_id,
           "submitted": submitted}
    if rep:
        res["report_path"] = str(rep)
        try:
            d = json.loads(rep.read_text())
            rids = d.get("resolved_ids") or d.get("resolved_instances") or []
            res["resolved_ids"] = rids
            res["resolved"] = len(rids) if isinstance(rids, list) else int(rids)
            res["dataset_total"] = d.get("total_instances")  # 500 for Verified
        except Exception as e:
            log(f"!! parse report {rep}: {e}")
    if "resolved" not in res and resolved >= 0:
        res["resolved"] = resolved
    # denominator = instances WE ran (the subset), not the 500-set
    res["total"] = submitted
    if res.get("total"):
        res["resolve_pct"] = round(100 * res.get("resolved", 0) / res["total"], 1)
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


def write_compare(run_dir: Path, env_info: dict, results: list, args) -> None:
    out = {"benchmark": "swe-bench-verified-resolve-compare",
           "schema": "fak.dgx-swebench-compare.v1",
           "model": MODEL, "served_as": SERVED, "tp": TP,
           "dataset": DATASET, "selection": args.filter or f"slice {args.slice}",
           "env": env_info, "arms": results}
    (run_dir / "compare.json").write_text(json.dumps(out, indent=2))

    def row(r):
        a, g = r["agent"], r["grade"]
        rr = f"{g.get('resolved','?')}/{g.get('total','?')}" if g.get("available") else "n/a"
        return (f"| {a['arm']} | `{a['endpoint']}` | {a['instances']} | "
                f"{a['completed']} | {a['patch_bytes']} | {a['agent_sec']}s | "
                f"{rr} | {g.get('resolve_pct','-')}% | {g.get('grade_sec','-')}s |")

    md = [f"# SWE-bench Verified — fak-gateway vs raw-SGLang (overall completion)",
          "",
          f"**Model:** `{MODEL}` served as `{SERVED}` (SGLang TP={TP}, bf16) · "
          f"**Selection:** {args.filter or 'slice '+args.slice} · "
          f"**Host:** {env_info.get('host','?')}",
          "",
          "Both arms drive the identical `mini-swe-agent` against the SAME SGLang "
          "weights; the only difference is whether each tool turn is routed through "
          "the `fak serve` adjudication gateway (`:8080`) or hits raw SGLang "
          "(`:30000`) directly. *Overall completion* = the agent ran to the end and "
          "emitted a non-empty patch; *resolved* = the official harness "
          "(`swebench.harness.run_evaluation`) PASS_TO_PASS + FAIL_TO_PASS grade.",
          "",
          "| arm | endpoint | instances | completed | patch bytes | agent time | "
          "resolved | resolve% | grade time |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    md += [row(r) for r in results]
    # verdict
    g = {r["agent"]["arm"]: r["grade"] for r in results}
    a = {r["agent"]["arm"]: r["agent"] for r in results}
    if "raw-sglang" in g and "fak-gateway" in g:
        rs, fk = g["raw-sglang"], g["fak-gateway"]
        same = rs.get("resolved") == fk.get("resolved")
        md += ["", "## Verdict",
               f"- **raw-SGLang:** completed {a['raw-sglang']['completed']}/"
               f"{a['raw-sglang']['instances']}, resolved "
               f"{rs.get('resolved','?')}/{rs.get('total','?')}.",
               f"- **fak-gateway:** completed {a['fak-gateway']['completed']}/"
               f"{a['fak-gateway']['instances']}, resolved "
               f"{fk.get('resolved','?')}/{fk.get('total','?')}.",
               f"- Overall completion through the fak gateway "
               f"{'MATCHES' if same else 'DIFFERS from'} raw SGLang on this "
               f"selection — routing every tool turn through fak's adjudication "
               f"plane did {'not change' if same else 'change'} the resolve outcome."]
    (run_dir / "COMPARE.md").write_text("\n".join(md) + "\n")
    log(f"wrote {run_dir/'compare.json'} and {run_dir/'COMPARE.md'}")


def probe_env() -> dict:
    info = {"host": os.uname().nodename, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        out = subprocess.run([SGLANG_PY, "-c",
                              "import torch;print(torch.cuda.get_device_name(0));print(torch.cuda.device_count())"],
                             capture_output=True, text=True, timeout=60)
        lines = out.stdout.strip().splitlines()
        info["gpu0"] = lines[0] if lines else "?"
        info["ngpu"] = lines[1] if len(lines) > 1 else "?"
    except Exception as e:
        info["gpu0"] = f"probe error: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/tmp/swe-cmp-1")
    ap.add_argument("--filter", default="astropy__astropy-12907",
                    help="instance-id regex (1 instance, image cached). Empty => use --slice")
    ap.add_argument("--slice", default="0:1", help="used when --filter is empty")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=0, help="agent step_limit (0=config default)")
    ap.add_argument("--agent-timeout", type=int, default=3600)
    ap.add_argument("--grade-workers", type=int, default=2)
    ap.add_argument("--grade-timeout", type=int, default=2400)
    ap.add_argument("--skip-serve", action="store_true", help="reuse already-running endpoints")
    ap.add_argument("--stop-serve", action="store_true", help="kill SGLang+gateway at the end")
    ap.add_argument("--redo", action="store_true", help="redo existing instances")
    ap.add_argument("--arms", default="raw-sglang,fak-gateway")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== SWE-bench Verified resolve compare: fak-gateway vs raw-SGLang ===")
    log(f"run_dir={run_dir} model={MODEL} selection={args.filter or args.slice}")
    env_info = probe_env()
    log(f"env: {json.dumps(env_info)}")
    if "A100" not in str(env_info.get("gpu0", "")):
        log("!! gpu0 is not an A100 — refusing to claim a DGX result. Aborting.")
        return 2

    procs = []
    if not args.skip_serve:
        sgl_p, _ = serve_sglang(run_dir)
        procs.append(("sglang", sgl_p))
        if not wait_health(f"http://127.0.0.1:{SGLANG_PORT}/health", "SGLang", 900):
            log("!! SGLang failed to come up; see sglang.log")
            return 3
        fak_p, _ = serve_fak(run_dir)
        procs.append(("fak", fak_p))
        # this fak build has no /health route (404), so probe /v1/models first.
        if not (wait_health(f"http://127.0.0.1:{FAK_PORT}/v1/models", "fak-gateway", 120)
                or wait_health(f"http://127.0.0.1:{FAK_PORT}/health", "fak-gateway(health)", 30)):
            log("!! fak gateway failed to come up; see fak.log")
            return 4
    else:
        log("--skip-serve: assuming SGLang(:30000) + fak-gateway(:8080) already up")

    # tool-call self-test BEFORE the agent runs — proves the serving config
    # emits OpenAI tool_calls (mini-swe-agent's bash action). A poll of run.log
    # sees this within ~90s, so a misconfig is caught before a long wasted run.
    log("--- tool-call self-test ---")
    for arm, port in ARMS:
        log("  " + probe_toolcall(port))
    log("--- end self-test ---")

    want = set(args.arms.split(","))
    results = []
    for arm, port in ARMS:
        if arm not in want:
            continue
        a = run_agent(arm, port, run_dir, args)
        g = grade(arm, a.get("preds_path"), run_dir, args, a["instances"])
        results.append({"agent": a, "grade": g})
        # checkpoint after each arm so a poll mid-run sees partial results
        write_compare(run_dir, env_info, results, args)

    write_compare(run_dir, env_info, results, args)
    log("=== DONE ===")

    if args.stop_serve:
        for name, p in procs:
            log(f"stopping {name} (pid {p.pid})")
            p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
