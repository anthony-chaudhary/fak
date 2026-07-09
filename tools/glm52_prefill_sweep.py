#!/usr/bin/env python3
"""GLM-5.2 pure-fak PREFILL-latency sweep driver (lever L9; #3085 / #3086).

The prefill analogue of the pure-fak DECODE sweep (`tools/dgx_glm_throughput_run.sh`
+ `cmd/glmdsatput -out`). The decode drivers exist; the L9 lever table row in
`docs/notes/GLM52-PURE-FAK-PERF-FRONTIER-AND-LANDED-2026-07-08.md` reads
"real prefill path (chunked+FA) ... needs harness ... no prefill-sweep script yet".
This is that missing harness: it runs a prefill-dominant request at each prompt
length in {128, 512, 2048, 4096, 8192} against a GLM-5.2 `fak serve` endpoint,
records TTFT / prefill tok/s per length, and lands each length as a DISCOVERABLE
benchmark-ledger artifact under
`experiments/benchmark/runs/by-machine/<node>/<UTC>-glm52-prefill-sweep/p<len>/`
(the same manifest/result/RESULTS.md shape the decode sweep lands via benchcli).

HONESTY FENCE (load-bearing):
  * Writing this script produces NO measured prefill number. It *enables* the
    measurement — it is a driver the on-box sm_80 GPU peer runs. This box has no
    DGX/GPU access, so only the GPU-free `--dry-run` planner is exercised here.
  * The metric is single-stream prefill tok/s = prompt_tokens / TTFT (time to
    first token). It is served FULL-MLA on sm_80: GLM-5.2's native DSA sparse-
    attention kernel is sm_90-floored (see the note's hardware fence), so this is
    the full-MLA context curve, NOT the native-DSA path and NOT the 753B MoE
    aggregate serving rate.
  * Per the note (§ P0), the two largest lengths (>= 4096) may hit the sm_80 DSA
    illegal-memory-access. The live sweep therefore TOLERATES and RECORDS a
    per-length failure ({"status":"FAIL", ...}) and continues — one bad length
    never aborts the whole sweep or discards the smaller lengths that succeeded.

Stdlib-only (like `glm52_serve_preflight.py` / `glm52_serving_witness.py`) so it
runs on a bare DGX/handoff node with nothing installed.

Examples:
  # GPU-free plan (what the companion test exercises): print the full sweep plan,
  # request bodies, and land paths WITHOUT hitting any endpoint or GPU.
  python tools/glm52_prefill_sweep.py --dry-run --out plan.json

  # On the resident 8-GPU sm_80 host, once `fak serve` is up (see the note's
  # ready-to-dispatch box commands): run the real sweep and auto-land the ledger.
  python tools/glm52_prefill_sweep.py --endpoint http://127.0.0.1:8000/v1 \
      --model zai-org/GLM-5.2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:  # windowgate: CREATE_NO_WINDOW on git/helper subprocesses so background runs never flash a console
    from dispatch_worker import install_no_window_subprocess_defaults

    install_no_window_subprocess_defaults(subprocess)
except Exception:  # sibling module absent (imported off the tools/ path) — best-effort suppressor only
    pass

SCHEMA = "fak.glm52-prefill-sweep.v1"
# The per-length record schema. Deliberately distinct from the decode sweep's
# "glm-throughput/1" so a reader never mistakes a prefill row for a decode row.
RECORD_SCHEMA = "glm-prefill/1"
LINEAGE_SCHEMA = "fak-bench-lineage/1"
BENCHMARK_ARTIFACT_SCHEMA = "fak-benchmark-artifact/1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "zai-org/GLM-5.2"

# The #3085 sweep axis: establish the currently-unmeasured prefill baseline.
PREFILL_LENGTHS = [128, 512, 2048, 4096, 8192]
# Prefill-dominant: a large prompt with (near-)zero generation so the measured
# latency is the prompt's forward pass, not decode. 1 keeps TTFT well-defined for
# non-streaming endpoints (duration ~= prefill + a single decode step).
DEFAULT_MAX_TOKENS = 1
# Lengths at/above this are flagged fragile on sm_80 (the note's P0 DSA IMA).
FRAGILE_MIN_LEN = 4096

# The load-bearing scope caveat that travels WITH every landed number.
SCOPE = (
    "single-stream prefill tok/s = prompt_tokens / TTFT (time to first token); "
    "served full-MLA on sm_80 (GLM-5.2 DSA sparse-attn kernel is sm_90-floored, so "
    "this is the full-MLA context curve, not native DSA); NOT the 753B MoE "
    "aggregate serving rate"
)

FILLER_WORD = "word"


# --------------------------------------------------------------------------- #
# Time / lineage helpers (mirror internal/benchcli Stamp env overrides)
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def compact_stamp() -> str:
    """UTC stamp for the land dir name, matching the decode sweep's
    `date -u +%Y%m%dT%H%M%SZ`."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def resolve_node(override: str = "") -> str:
    if override.strip():
        return override.strip()
    node = _env("FAK_BENCH_NODE")
    if node:
        return node
    try:
        host = socket.gethostname().strip()
    except OSError:
        host = ""
    return host or "node"


def git_commit() -> str:
    """HEAD sha, honoring the FAK_BENCH_COMMIT override the decode sweep exports.
    Fail-soft to 'unknown' (never raises) so a lineage-free artifact can't ship."""
    override = _env("FAK_BENCH_COMMIT")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = (out.stdout or "").strip()
    return sha or "unknown"


def bench_lineage(node: str) -> dict[str, Any]:
    """The four-axis provenance benchcli.DecodeArtifact recognizes. git_commit or
    utc being non-empty is what makes the manifest discoverable by
    benchcli.BuildLineageIndex and bindable by `dos verify`."""
    return {
        "lineage_schema": LINEAGE_SCHEMA,
        "app_version": _env("FAK_APP_VERSION") or "unknown",
        "utc": _env("FAK_BENCH_UTC") or utc_now(),
        "git_commit": git_commit(),
        "go_version": "python-driver",  # this emitter is Python, not the Go harness
        "node": node,
    }


def _run_id(lineage: dict[str, Any], prompt_len: int) -> str:
    stamp = str(lineage.get("utc", "")).replace(":", "").replace("-", "")
    stamp = stamp.replace("T", "T").rstrip("Z") or "unknown-time"
    commit = str(lineage.get("git_commit", "unknown"))[:12] or "unknown"
    raw = f"{stamp}-glm52-prefill-p{prompt_len}-{commit}".lower()
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in raw).strip("-")


# --------------------------------------------------------------------------- #
# Pure planning (what --dry-run and the test exercise; no I/O)
# --------------------------------------------------------------------------- #

def synthetic_prompt(target_tokens: int) -> str:
    """A prompt that tokenizes to ~target_tokens. Most BPE tokenizers map a
    repeated short word to ~1 token each; the ACHIEVED prompt_tokens is read back
    from the endpoint's usage in live mode and is what the tok/s metric divides
    by, so this only needs to be prefill-DOMINANT at scale, not exact."""
    return " ".join([FILLER_WORD] * max(1, int(target_tokens)))


def prefill_payload(model: str, prompt_len: int, max_tokens: int, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": synthetic_prompt(prompt_len)}],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "stream": bool(stream),
    }
    if stream:
        # Ask for a final usage chunk so prompt_tokens is authoritative for tok/s.
        body["stream_options"] = {"include_usage": True}
    return body


def land_subdir(land_root: str, prompt_len: int) -> str:
    """One discoverable subdir per length, mirroring the decode sweep's
    per-config `L{L}-H{H}-tk{TK}` subdir convention."""
    root = (land_root or "").rstrip("/")
    return f"{root}/p{prompt_len}"


def build_plan(
    model: str,
    land_root: str,
    *,
    lengths: list[int] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = True,
    fragile_min_len: int = FRAGILE_MIN_LEN,
) -> list[dict[str, Any]]:
    """The full sweep plan: one step per prompt length with its request body and
    its land path. Pure — no network, no GPU, no filesystem."""
    lengths = list(lengths) if lengths is not None else list(PREFILL_LENGTHS)
    plan: list[dict[str, Any]] = []
    for prompt_len in lengths:
        body = prefill_payload(model, prompt_len, max_tokens, stream)
        content = body["messages"][0]["content"]
        plan.append({
            "prompt_len": prompt_len,
            "target_prompt_tokens": prompt_len,
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
            "prompt_chars": len(content),
            "request_body": body,
            "land_subdir": land_subdir(land_root, prompt_len) if land_root else "",
            "fragile_on_sm80": prompt_len >= fragile_min_len,
        })
    return plan


# --------------------------------------------------------------------------- #
# HTTP (live mode only — never reached in --dry-run)
# --------------------------------------------------------------------------- #

def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    got = _number(value)
    return int(got) if got is not None else None


def parse_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def json_get(url: str, timeout_s: float) -> tuple[int, dict[str, Any] | None, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), parse_json(raw), raw[:2000]
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), parse_json(raw), raw[:2000]
    except OSError as exc:
        return 0, None, str(exc)


def endpoint_reachable(base_url: str, timeout_s: float) -> dict[str, Any]:
    """Gate on the serve being up before the sweep (the driver self-runs step-0)."""
    url = base_url.rstrip("/") + "/models"
    status, data, body = json_get(url, timeout_s=timeout_s)
    ids = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        ids = [row.get("id") for row in data["data"] if isinstance(row, dict)]
    return {
        "url": url,
        "reachable": status == 200,
        "http_status": status,
        "model_ids": ids,
        "body_excerpt": body[:500],
    }


def measure_prefill_stream(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Streaming TTFT: time to the first content delta ~= the prefill forward pass.
    prompt_tokens is taken from the final usage chunk when present."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    ttft: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk_raw = line[len("data:"):].strip()
                if chunk_raw == "[DONE]":
                    break
                chunk = parse_json(chunk_raw)
                if chunk is None:
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    pt = _int(usage.get("prompt_tokens"))
                    if pt is not None:
                        prompt_tokens = pt
                    ct = _int(usage.get("completion_tokens"))
                    if ct is not None:
                        completion_tokens = ct
                choices = chunk.get("choices")
                if ttft is None and isinstance(choices, list) and choices:
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else {}
                    if isinstance(delta, dict) and delta.get("content"):
                        ttft = time.perf_counter() - started
            total = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "http_status": int(exc.code), "error": raw[:500]}
    except OSError as exc:
        return {"ok": False, "http_status": 0, "error": str(exc)}
    return {
        "ok": ttft is not None,
        "http_status": 200,
        "ttft_s": round(ttft, 6) if ttft is not None else None,
        "total_s": round(total, 6),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "source": "stream-ttft",
    }


def measure_prefill_blocking(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Non-streaming fallback: the whole request duration. With max_tokens=1 this
    is prefill-dominant. prompt_tokens comes from the response usage."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            duration = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "http_status": int(exc.code), "error": raw[:500]}
    except OSError as exc:
        return {"ok": False, "http_status": 0, "error": str(exc)}
    data_obj = parse_json(raw) or {}
    usage = data_obj.get("usage") if isinstance(data_obj.get("usage"), dict) else {}
    return {
        "ok": bool(data_obj.get("choices")),
        "http_status": 200,
        "ttft_s": round(duration, 6),
        "total_s": round(duration, 6),
        "prompt_tokens": _int(usage.get("prompt_tokens")),
        "completion_tokens": _int(usage.get("completion_tokens")),
        "source": "blocking-duration",
    }


def record_for_length(
    measurement: dict[str, Any],
    *,
    model: str,
    endpoint: str,
    prompt_len: int,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    """Fold one raw measurement into the discoverable per-length record. The
    `scope` field is load-bearing — it travels with the number."""
    prompt_tokens = measurement.get("prompt_tokens") or prompt_len
    ttft = measurement.get("ttft_s")
    prefill_tok_s = None
    if measurement.get("ok") and ttft and ttft > 0:
        prefill_tok_s = round(prompt_tokens / ttft, 3)
    rec: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "model": "glm_moe_dsa",
        "served_model": model,
        "endpoint": endpoint,
        "backend": f"fak serve @ {endpoint}",
        "prompt_len": prompt_len,
        "target_prompt_tokens": prompt_len,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "stream": stream,
        "ttft_s": ttft,
        "prefill_seconds": ttft,
        "prefill_tok_s": prefill_tok_s,
        "completion_tokens": measurement.get("completion_tokens"),
        "ttft_source": measurement.get("source"),
        "status": "OK" if measurement.get("ok") else "FAIL",
        "scope": SCOPE,
    }
    if not measurement.get("ok"):
        rec["error"] = measurement.get("error", "no first token observed")
        rec["http_status"] = measurement.get("http_status")
    return rec


# --------------------------------------------------------------------------- #
# Ledger land (mirror cmd/glmdsatput writeLedgerArtifact, in Python)
# --------------------------------------------------------------------------- #

def build_manifest(record: dict[str, Any], lineage: dict[str, Any], prompt_len: int) -> dict[str, Any]:
    """The discoverable manifest: the record body verbatim + a top-level `lineage`
    block (what benchcli.DecodeArtifact keys on) + a `benchmark_artifact` envelope
    carrying a run_id. Structurally identical intent to the Go harness's manifest,
    so benchcli.BuildLineageIndex folds it and `dos verify` can bind it."""
    manifest = dict(record)  # record fields survive at the top level (scope included)
    manifest["lineage"] = lineage
    manifest["benchmark_artifact"] = {
        "schema": BENCHMARK_ARTIFACT_SCHEMA,
        "run_id": _run_id(lineage, prompt_len),
        "timestamp": lineage.get("utc"),
        "fak_commit": lineage.get("git_commit"),
        "fak_version": lineage.get("app_version"),
        "harness": {"name": "glm52_prefill_sweep", "version": "1.0.0"},
        "model": {"name": "glm_moe_dsa", "precision": "served"},
        "results": {"metrics": {"prefill_tok_s": record.get("prefill_tok_s")}},
        "witness": {
            "test_path": "tools/glm52_prefill_sweep_test.py",
            "reproduction_command": (
                "python tools/glm52_prefill_sweep.py --endpoint <url>/v1 --model "
                + str(record.get("served_model", DEFAULT_MODEL))
            ),
        },
    }
    return manifest


def results_markdown(record: dict[str, Any]) -> str:
    lines = [
        "# fak NATIVE glm_moe_dsa PREFILL latency (pure-fak)",
        "",
        f"> **Scope (load-bearing):** `{record.get('scope', '')}`",
        "",
        "| field | value |",
        "|---|---|",
        f"| endpoint | {record.get('endpoint', '')} |",
        f"| served_model | {record.get('served_model', '')} |",
        f"| prompt_len (target) | {record.get('target_prompt_tokens', '')} |",
        f"| prompt_tokens (measured) | {record.get('prompt_tokens', '')} |",
        f"| TTFT (prefill s) | {record.get('ttft_s', '')} |",
        f"| **PREFILL** | **{record.get('prefill_tok_s', 'n/a')} tok/s** |",
        f"| status | {record.get('status', '')} |",
        "",
        "This artifact carries a benchcli lineage + benchmark_artifact envelope, so it "
        "is discoverable by fak's lineage index and bindable by `dos verify`.",
        "",
    ]
    return "\n".join(lines)


def write_ledger_artifact(land_subdir_path: str, record: dict[str, Any], lineage: dict[str, Any]) -> str:
    """Land one length as a discoverable artifact: manifest.json (lineage + envelope
    around the record), result.json (raw record, NO lineage so it is not
    double-counted by BuildLineageIndex), RESULTS.md. Returns the manifest path."""
    out = Path(land_subdir_path)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(record, lineage, int(record.get("prompt_len", 0)))
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "RESULTS.md").write_text(results_markdown(record), encoding="utf-8", newline="\n")
    return str(manifest_path)


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def build_dry_run_report(
    *,
    model: str,
    base_url: str,
    land_root: str,
    lengths: list[int],
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    plan = build_plan(model, land_root, lengths=lengths, max_tokens=max_tokens, stream=stream)
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": "plan",
        "dry_run": True,
        "model": model,
        "base_url": base_url,
        "land_root": land_root,
        "land_enabled": bool(land_root),
        "lengths": list(lengths),
        "max_tokens": max_tokens,
        "stream": stream,
        "scope": SCOPE,
        "plan": plan,
        "notes": [
            "PLAN ONLY: no endpoint or GPU was contacted; no prefill number is produced.",
            "The on-box sm_80 peer runs the live sweep (--endpoint) to produce the numbers.",
            f"Lengths >= {FRAGILE_MIN_LEN} are flagged fragile_on_sm80 (the note's P0 DSA "
            "illegal-memory-access); the live sweep records a per-length FAIL and continues.",
        ],
    }


def run_live_sweep(
    *,
    model: str,
    base_url: str,
    land_root: str,
    lengths: list[int],
    max_tokens: int,
    stream: bool,
    http_timeout_s: float,
    request_timeout_s: float,
    node: str,
) -> dict[str, Any]:
    reach = endpoint_reachable(base_url, timeout_s=http_timeout_s)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": "live",
        "dry_run": False,
        "model": model,
        "base_url": base_url,
        "node": node,
        "land_root": land_root,
        "land_enabled": bool(land_root),
        "lengths": list(lengths),
        "max_tokens": max_tokens,
        "stream": stream,
        "scope": SCOPE,
        "endpoint_gate": reach,
        "results": [],
    }
    if not reach["reachable"]:
        report["aborted"] = "endpoint not reachable"
        return report

    lineage = bench_lineage(node)
    chat_url = base_url.rstrip("/") + "/chat/completions"
    for prompt_len in lengths:
        payload = prefill_payload(model, prompt_len, max_tokens, stream)
        # Tolerate a per-length device fault (the note's sm_80 P0): a raised
        # exception here becomes a recorded FAIL row, never an aborted sweep.
        try:
            if stream:
                measurement = measure_prefill_stream(chat_url, payload, timeout_s=request_timeout_s)
            else:
                measurement = measure_prefill_blocking(chat_url, payload, timeout_s=request_timeout_s)
        except Exception as exc:  # noqa: BLE001 — one length must not sink the sweep
            measurement = {"ok": False, "http_status": 0, "error": f"{type(exc).__name__}: {exc}"}
        record = record_for_length(
            measurement,
            model=model,
            endpoint=base_url,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            stream=stream,
        )
        landed = ""
        if land_root:
            try:
                landed = write_ledger_artifact(land_subdir(land_root, prompt_len), record, lineage)
            except OSError as exc:
                landed = f"LAND_FAILED: {exc}"
        report["results"].append({"record": record, "manifest": landed})
    ok = sum(1 for r in report["results"] if r["record"]["status"] == "OK")
    report["summary"] = {
        "lengths": len(lengths),
        "ok": ok,
        "failed": len(lengths) - ok,
    }
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_lengths(raw: str) -> list[int]:
    if not raw.strip():
        return list(PREFILL_LENGTHS)
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or list(PREFILL_LENGTHS)


def default_land_root(node: str, stamp: str) -> str:
    return f"experiments/benchmark/runs/by-machine/{node}/{stamp}-glm52-prefill-sweep"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="GLM-5.2 pure-fak prefill-latency sweep driver (L9; #3085/#3086)",
        epilog=(
            "Scope: this driver ENABLES the measurement; running it produces no prefill "
            "number by itself. The metric is single-stream prefill tok/s = prompt_tokens/TTFT, "
            "served full-MLA on sm_80 (native DSA is sm_90-floored), NOT the 753B MoE serving "
            "rate. Lengths >= 4096 may hit the sm_80 DSA illegal-memory-access; the sweep "
            "records a per-length FAIL and continues."
        ),
    )
    ap.add_argument("--endpoint", "--base-url", dest="endpoint", default="",
                    help="OpenAI-compatible GLM-5.2 fak serve endpoint, e.g. http://127.0.0.1:8000/v1 "
                         "(omit with --dry-run to only print the plan)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lengths", default="", help="comma list of prompt lengths (default 128,512,2048,4096,8192)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="generation cap per request; small/zero keeps the request prefill-dominant (default 1)")
    ap.add_argument("--no-stream", action="store_true", help="use a blocking request (whole-duration) instead of streaming TTFT")
    ap.add_argument("--node", default="", help="override the land-dir node segment (default FAK_BENCH_NODE or hostname)")
    ap.add_argument("--stamp", default="", help="override the land-dir UTC stamp (default now, %%Y%%m%%dT%%H%%M%%SZ)")
    ap.add_argument("--out", default="experiments/glm52/prefill-sweep.json", help="write the sweep report/plan JSON here")
    ap.add_argument("--http-timeout-s", type=float, default=15.0)
    ap.add_argument("--request-timeout-s", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true", help="print the full sweep plan WITHOUT hitting any endpoint or GPU")
    args = ap.parse_args(argv)

    lengths = parse_lengths(args.lengths)
    stream = not args.no_stream
    node = resolve_node(args.node)
    stamp = args.stamp.strip() or compact_stamp()
    # Default-on land; opt out with GLM_LAND_DIR="" (mirrors the decode sweep's
    # `${GLM_LAND_DIR-<default>}` semantics: absent -> default, present -> value).
    land_root = os.environ.get("GLM_LAND_DIR", default_land_root(node, stamp))

    if args.dry_run or not args.endpoint:
        report = build_dry_run_report(
            model=args.model,
            base_url=args.endpoint or "(none: --dry-run)",
            land_root=land_root,
            lengths=lengths,
            max_tokens=args.max_tokens,
            stream=stream,
        )
        _write_out(args.out, report)
        _print_plan_summary(report)
        return 0

    report = run_live_sweep(
        model=args.model,
        base_url=args.endpoint,
        land_root=land_root,
        lengths=lengths,
        max_tokens=args.max_tokens,
        stream=stream,
        http_timeout_s=args.http_timeout_s,
        request_timeout_s=args.request_timeout_s,
        node=node,
    )
    _write_out(args.out, report)
    print(json.dumps(report.get("summary") or {"endpoint_gate": report.get("endpoint_gate")}, indent=2))
    # A sweep where the endpoint was unreachable or every length failed is a nonzero exit.
    if report.get("aborted"):
        return 1
    summary = report.get("summary") or {}
    return 0 if summary.get("ok", 0) > 0 else 1


def _write_out(path: str, report: dict[str, Any]) -> None:
    out = Path(path)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


def _print_plan_summary(report: dict[str, Any]) -> None:
    print(f"GLM-5.2 prefill sweep PLAN (dry-run) -- model={report['model']} land_root={report['land_root']}")
    print(f"scope: {report['scope']}")
    for step in report["plan"]:
        fragile = "  [fragile_on_sm80]" if step["fragile_on_sm80"] else ""
        print(f"  P={step['prompt_len']:<5} max_tokens={step['max_tokens']} "
              f"prompt_chars={step['prompt_chars']:<7} -> {step['land_subdir']}{fragile}")


if __name__ == "__main__":
    raise SystemExit(main())
