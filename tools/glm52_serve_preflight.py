#!/usr/bin/env python3
"""GLM-5.2 SGLang/vLLM serving-readiness preflight — a portable go/no-go gate.

GLM-5.2 (753B, `glm_moe_dsa`) uses DeepSeek-Sparse-Attention (DSA). The stock
SGLang/vLLM DSA kernels are gated to **Hopper (sm_90)** or **Blackwell (sm_100)**;
there is no Ampere (sm_80) support (vLLM #35021 reports at least three layers of
sm_80 incompatibility), and the only known lower-arch port targets Ada (sm_89,
RTX 4090) via a community `ada_dsa.py` — not stock, not Ampere. So whether a node
can serve GLM-5.2 in SGLang or vLLM is decided by two hard facts the operator
should never have to re-derive by hand: the GPU compute capability vs the DSA
kernel floor, and total VRAM vs the per-quant weight footprint.

This script turns that decision into one reproducible command. Run it ON a
candidate serving node (it reads `nvidia-smi` and probes for the engines), or
from any box as a PLANNER by passing the node's shape via `--gpu-name` /
`--gpu-count` / `--gpu-memory-total-gb`. It emits a structured JSON + Markdown
report with a per-engine verdict (READY / READY_PENDING_INSTALL / BLOCKED_ARCH /
BLOCKED_MEMORY) and the recommended quant + next action.

It is intentionally stdlib-only (like `glm52_serving_witness.py`) so it runs on
DGX/handoff nodes with nothing installed. It claims nothing about THIS checkout's
ability to serve 753B — it is a hardware/stack gate, and it fails closed.

Examples:
  # On a serving node (auto-detect GPUs + engines):
  python tools/glm52_serve_preflight.py --out preflight.json --markdown preflight.md

  # As a planner from the driver box (no GPU), for a hypothetical H200 node:
  python tools/glm52_serve_preflight.py --gpu-name "NVIDIA H200" --gpu-count 8 \
      --gpu-memory-total-gb 1128 --require-ready
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "fak.glm52-serve-preflight.v1"
ROOT = Path(__file__).resolve().parents[1]

# --- GLM-5.2 facts (from GLM-5.2-NATIVE-ENGINE-GAP / SGLang GLM-5.2 cookbook) ---
MODEL = "zai-org/GLM-5.2"
MODEL_FP8 = "zai-org/GLM-5.2-FP8"
TOTAL_PARAMS_B = 753.0

# Weight-resident VRAM per quantization, in GB (the published GLM-5.2 footprints).
# Real serving also needs KV-cache + activation headroom on top — see KV_OVERHEAD.
QUANT_WEIGHTS_GB: dict[str, float] = {
    "bf16": 1506.0,
    "fp8": 753.0,
    "nvfp4": 459.0,
    "w4afp8": 368.0,   # 4-bit INT experts + FP8 activations (Hopper-class hybrid)
    "int4": 376.0,
}
# Highest-fidelity first; the recommended quant is the first that fits AND whose
# arch floor the node clears. On Blackwell NVFP4 is hardware-native (preferred 4-bit);
# on Hopper NVFP4 is skipped and W4AFP8 (FP8 acts, SGLang-validated) outranks INT4.
QUANT_PREFERENCE = ["fp8", "nvfp4", "w4afp8", "int4"]

# Minimum compute capability per quant. DSA itself already needs sm_90, so every
# quant inherits that floor; NVFP4 additionally needs Blackwell FP4 tensor cores
# (sm_100) — it does NOT run on Hopper. On Hopper the 4-bit path is the SGLang-
# validated W4AFP8 hybrid or INT4 (AWQ/GPTQ).
QUANT_ARCH_FLOOR: dict[str, float] = {
    "bf16": 9.0,
    "fp8": 9.0,
    "nvfp4": 10.0,
    "w4afp8": 9.0,
    "int4": 9.0,
}

# DSA kernel floor for the STOCK engines.
DSA_STOCK_FLOOR_CC = 9.0  # Hopper sm_90; Blackwell sm_100 (>9.0) also supported.
ADA_PORT_CC = 8.9         # community-only Ada port (renning22/glm-5.2-4090, ada_dsa.py).

# sm_90+ data-center parts the DSA gate accepts. H100/H200 are Hopper (sm_90);
# B200/B300/GB200/GB300 are Blackwell (sm_100). The SGLang GLM-5.2 cookbook
# validated H200/B200/B300/GB300; H100 and GB200 clear the same arch floor.
SUPPORTED_GPUS = ["H100", "H200", "B200", "B300", "GB200", "GB300"]

# GPU model name -> CUDA compute capability, when nvidia-smi can't report
# compute_cap directly. Checked in order; first substring hit wins, so more
# specific names (a100 before a10) come first.
NAME_TO_CC: list[tuple[str, float]] = [
    ("gb300", 10.0), ("gb200", 10.0), ("b300", 10.0), ("b200", 10.0), ("b100", 10.0),
    ("h200", 9.0), ("h100", 9.0), ("h800", 9.0),
    ("a100", 8.0), ("a800", 8.0), ("a40", 8.6), ("a30", 8.0), ("a10", 8.6),
    ("rtx 4090", 8.9), ("rtx 4080", 8.9), ("l40s", 8.9), ("l40", 8.9), ("l4", 8.9),
    ("4090", 8.9), ("ada", 8.9),
    ("rtx 3090", 8.6), ("rtx 3080", 8.6), ("3090", 8.6),
    ("v100", 7.0), ("t4", 7.5),
]

Runner = Callable[[Sequence[str]], "CmdResult"]


class CmdResult:
    """Minimal injectable command result (stdlib-only, test-friendly)."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def real_runner(cmd: Sequence[str]) -> CmdResult:
    try:
        proc = subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CmdResult(127, "", str(exc))
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #

def capability_from_name(name: str) -> float | None:
    """Map a GPU model name to its CUDA compute capability, or None if unknown."""
    text = (name or "").lower()
    for needle, cc in NAME_TO_CC:
        if needle in text:
            return cc
    return None


def arch_label(cc: float | None) -> str:
    if cc is None:
        return "unknown"
    if cc >= 10.0:
        return "Blackwell (sm_100)"
    if cc >= 9.0:
        return "Hopper (sm_90)"
    if cc == ADA_PORT_CC:
        return "Ada (sm_89)"
    if cc >= 8.0:
        return "Ampere (sm_80/86)"
    if cc >= 7.0:
        return "Volta/Turing (sm_70/75)"
    return f"sm_{int(cc * 10)}"


def required_vram_gb(quant: str, kv_overhead: float) -> float:
    weights = QUANT_WEIGHTS_GB[quant]
    return round(weights * (1.0 + kv_overhead), 1)


# Per-rank fixed reserve (NCCL buffers + CUDA context + CUDA-graph/activation
# scratch) that does NOT shard with tensor-parallel. Bounded so the validated
# even-8-GPU H100/H200 nodes stay READY while a node that clears AGGREGATE VRAM
# but can't hold its per-card TP shard is still caught.
PER_RANK_RESERVE_GB = 7.0


def required_per_gpu_vram_gb(quant: str, kv_overhead: float, gpu_count: int) -> float:
    """Per-card VRAM a tensor-parallel shard needs: weights/TP scaled by KV
    overhead, plus the non-sharding per-rank reserve. gpu_count<=1 degrades to the
    single-rank footprint."""
    if gpu_count <= 1:
        return required_vram_gb(quant, kv_overhead)
    per_rank_weights = QUANT_WEIGHTS_GB[quant] / gpu_count
    return round(per_rank_weights * (1.0 + kv_overhead) + PER_RANK_RESERVE_GB, 1)


def quant_arch_ok(quant: str, cc: float | None) -> bool:
    """Does this GPU's compute capability support this quant's kernels?"""
    floor = QUANT_ARCH_FLOOR.get(quant, DSA_STOCK_FLOOR_CC)
    return cc is not None and cc >= floor


def recommended_quant(total_vram_gb: float, kv_overhead: float, cc: float | None = None) -> str | None:
    """Highest-fidelity quant whose weights+overhead fit AND whose arch floor the
    node clears. When cc is None (unknown arch) the arch gate is skipped."""
    for quant in QUANT_PREFERENCE:
        if cc is not None and not quant_arch_ok(quant, cc):
            continue
        if total_vram_gb >= required_vram_gb(quant, kv_overhead):
            return quant
    return None


def evaluate_engine(
    engine: str,
    *,
    cc: float | None,
    total_vram_gb: float,
    gpu_count: int,
    engine_present: bool,
    engine_version: str,
    quant: str,
    kv_overhead: float,
    per_gpu_vram_gb: float | None = None,
) -> dict[str, Any]:
    """Per-engine go/no-go for serving GLM-5.2. Fails closed on unknown arch."""
    need = required_vram_gb(quant, kv_overhead)
    fit_quant = recommended_quant(total_vram_gb, kv_overhead, cc)
    aggregate_ok = total_vram_gb >= need
    # A node can clear AGGREGATE VRAM yet be unable to hold its per-card TP shard
    # (weights/TP + KV + the non-sharding per-rank reserve). When per-card data is
    # known, gate on BOTH; when it is not, the aggregate check stands alone.
    per_gpu_need = required_per_gpu_vram_gb(quant, kv_overhead, gpu_count)
    per_gpu_ok = per_gpu_vram_gb is None or per_gpu_vram_gb >= per_gpu_need
    memory_ok = aggregate_ok and per_gpu_ok
    arch_ok = cc is not None and cc >= DSA_STOCK_FLOOR_CC
    quant_ok = quant_arch_ok(quant, cc)
    notes: list[str] = []

    if cc is None:
        verdict = "BLOCKED_ARCH"
        notes.append("GPU compute capability is unknown; pass --gpu-name or run on the node.")
    elif not arch_ok:
        verdict = "BLOCKED_ARCH"
        if cc == ADA_PORT_CC:
            notes.append(
                "Ada (sm_89): stock SGLang/vLLM still gate DSA to sm_90+. Only the "
                "community port renning22/glm-5.2-4090 (ada_dsa.py) runs here — not stock."
            )
        else:
            notes.append(
                f"{arch_label(cc)} is below the DSA kernel floor (sm_90). Stock "
                "SGLang/vLLM cannot serve GLM-5.2 here (vLLM #35021). Use llama.cpp "
                "(MLA path, runs on Ampere/CPU) or target any sm_90+ part: Hopper "
                "(H100/H200) or Blackwell (B200/B300/GB200/GB300)."
            )
    elif not quant_ok:
        verdict = "BLOCKED_QUANT_ARCH"
        hint = f" — re-run with --quant {fit_quant}" if fit_quant else ""
        notes.append(
            f"{quant.upper()} needs sm_{int(QUANT_ARCH_FLOOR.get(quant, DSA_STOCK_FLOOR_CC) * 10)} "
            f"(Blackwell FP4 tensor cores); {arch_label(cc)} can't run it. On Hopper use "
            f"INT4 (AWQ/GPTQ) or the SGLang-validated W4AFP8 hybrid{hint}."
        )
    elif not memory_ok:
        verdict = "BLOCKED_MEMORY"
        if aggregate_ok and not per_gpu_ok:
            # Aggregate clears, per-card shard does not — the OOM-at-load case the
            # aggregate-only check used to miss.
            notes.append(
                f"aggregate VRAM clears but the per-GPU TP shard (~{per_gpu_need} GB/card "
                f"incl. ~{PER_RANK_RESERVE_GB:.0f} GB reserve) exceeds the {per_gpu_vram_gb:.0f} GB "
                f"smallest card — would OOM at load. Use more/larger GPUs or a smaller quant."
            )
        elif fit_quant:
            notes.append(
                f"{quant} needs ~{need} GB but only {total_vram_gb:.0f} GB present; "
                f"{fit_quant} would fit — re-run with --quant {fit_quant}."
            )
        else:
            notes.append(
                f"{quant} needs ~{need} GB; no supported quant fits in {total_vram_gb:.0f} GB. "
                "Add GPUs or use a larger node."
            )
    elif not engine_present:
        verdict = "READY_PENDING_INSTALL"
        notes.append(f"Arch + memory OK; {engine} is not installed — pip install it, then serve.")
    else:
        verdict = "READY"
        notes.append(f"{engine} {engine_version} present; arch + memory OK for {quant}.")

    if quant == "int4" and verdict in {"READY", "READY_PENDING_INSTALL"}:
        notes.append(
            "int4 = self-quant only: there is no official GLM-5.2 INT4 (AWQ/GPTQ) repo, "
            "so the serve script has no default checkpoint — set MODEL=<your-int4-hf-repo>."
        )

    return {
        "engine": engine,
        "verdict": verdict,
        "ready": verdict in {"READY", "READY_PENDING_INSTALL"},
        "arch_ok": arch_ok,
        "memory_ok": memory_ok,
        "engine_present": engine_present,
        "engine_version": engine_version,
        "quant": quant,
        "required_vram_gb": need,
        "total_vram_gb": round(total_vram_gb, 1),
        "required_per_gpu_vram_gb": per_gpu_need,
        "per_gpu_vram_gb": per_gpu_vram_gb,
        "gpu_count": gpu_count,
        "recommended_quant": fit_quant,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Detection (injectable runner)
# --------------------------------------------------------------------------- #

def _to_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def detect_gpus(runner: Runner, smi_path: str | None) -> dict[str, Any]:
    """Read GPUs via nvidia-smi; returns names, per-GPU MiB, compute_cap if available."""
    if not smi_path:
        return {"source": "nvidia-smi", "status": "UNAVAILABLE", "detail": "nvidia-smi not found", "gpus": []}
    query = "index,name,memory.total,compute_cap"
    res = runner([smi_path, f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    have_cc = True
    if res.returncode != 0:
        have_cc = False
        res = runner([smi_path, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    if res.returncode != 0:
        return {"source": "nvidia-smi", "status": "ERROR", "detail": res.stderr.strip()[:500], "gpus": []}
    gpus: list[dict[str, Any]] = []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if have_cc and len(parts) >= 4:
            idx, name, mem, cc = parts[0], parts[1], parts[2], parts[3]
        elif len(parts) >= 3:
            idx, name, mem, cc = parts[0], parts[1], parts[2], ""
        else:
            continue
        gpus.append({
            "index": idx,
            "name": name,
            "memory_total_mib": _to_float(mem),
            "compute_cap": cc if cc and cc.upper() != "N/A" else "",
        })
    return {"source": "nvidia-smi", "status": "OK" if gpus else "EMPTY", "gpus": gpus}


def resolve_node(
    detected: dict[str, Any],
    *,
    override_name: str,
    override_count: int,
    override_total_gb: float,
) -> dict[str, Any]:
    """Fold detected GPUs + manual overrides into a single node shape."""
    gpus = detected.get("gpus") if isinstance(detected.get("gpus"), list) else []
    name = override_name or (gpus[0]["name"] if gpus else "")
    count = override_count or len(gpus)

    cc: float | None = None
    cc_source = ""
    if override_name:
        # Explicit --gpu-name means "evaluate THIS node" (planner): the override
        # wins over whatever card happens to be in this box.
        cc = capability_from_name(override_name)
        cc_source = "name-map" if cc is not None else ""
    elif gpus and gpus[0].get("compute_cap"):
        cc = _to_float(gpus[0]["compute_cap"])
        cc_source = "nvidia-smi"
    elif name:
        cc = capability_from_name(name)
        cc_source = "name-map" if cc is not None else ""

    if override_total_gb > 0:
        total_gb = override_total_gb
        mem_source = "override"
    elif gpus:
        total_mib = sum((g.get("memory_total_mib") or 0) for g in gpus)
        total_gb = round(total_mib / 1024, 1)
        mem_source = "nvidia-smi"
    else:
        total_gb = 0.0
        mem_source = ""

    # Smallest card decides the per-GPU/TP-shard fit. Prefer the real per-card data
    # (detected nvidia-smi rows); on the override path (no per-card data) fall back
    # to an even split, which assumes a homogeneous node.
    per_gpu_gb: float | None = None
    per_gpu_source = ""
    if gpus and any(g.get("memory_total_mib") for g in gpus):
        per_gpu_gb = round(min((g.get("memory_total_mib") or 0) for g in gpus) / 1024, 1)
        per_gpu_source = "min-card"
    elif total_gb > 0 and count > 0:
        per_gpu_gb = round(total_gb / count, 1)
        per_gpu_source = "even-split"

    return {
        "gpu_name": name,
        "gpu_count": count,
        "compute_cap": cc,
        "compute_cap_source": cc_source,
        "arch": arch_label(cc),
        "total_vram_gb": total_gb,
        "total_vram_source": mem_source,
        "per_gpu_vram_gb": per_gpu_gb,
        "per_gpu_vram_source": per_gpu_source,
    }


def detect_engine(name: str, runner: Runner, python_exe: str) -> dict[str, Any]:
    """Probe whether `import <name>` works and report its __version__."""
    code = f"import {name},sys; print(getattr({name}, '__version__', '?'))"
    res = runner([python_exe, "-c", code])
    present = res.returncode == 0
    version = res.stdout.strip().splitlines()[0] if (present and res.stdout.strip()) else ""
    return {"present": present, "version": version, "detail": "" if present else res.stderr.strip()[:200]}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(
    *,
    runner: Runner | None = None,
    python_exe: str = sys.executable,
    smi_path: str | None = None,
    smi_autodetect: bool = True,
    engines: Sequence[str] = ("sglang", "vllm"),
    override_name: str = "",
    override_count: int = 0,
    override_total_gb: float = 0.0,
    quant: str = "fp8",
    kv_overhead: float = 0.15,
    probe_engines: bool = True,
) -> dict[str, Any]:
    runner = runner or real_runner
    if smi_path is None and smi_autodetect:
        smi_path = shutil.which("nvidia-smi")

    detected = detect_gpus(runner, smi_path)
    node = resolve_node(
        detected,
        override_name=override_name,
        override_count=override_count,
        override_total_gb=override_total_gb,
    )

    engine_probes: dict[str, dict[str, Any]] = {}
    if probe_engines:
        for name in engines:
            engine_probes[name] = detect_engine(name, runner, python_exe)
    else:
        engine_probes = {name: {"present": False, "version": "", "detail": "not probed"} for name in engines}

    evaluations = []
    for name in engines:
        probe = engine_probes[name]
        evaluations.append(evaluate_engine(
            name,
            cc=node["compute_cap"],
            total_vram_gb=node["total_vram_gb"],
            gpu_count=node["gpu_count"],
            engine_present=bool(probe.get("present")),
            engine_version=str(probe.get("version") or ""),
            quant=quant,
            kv_overhead=kv_overhead,
            per_gpu_vram_gb=node.get("per_gpu_vram_gb"),
        ))

    any_ready = any(e["ready"] for e in evaluations)
    fit_quant = recommended_quant(node["total_vram_gb"], kv_overhead, node["compute_cap"])
    if any_ready:
        node_verdict = "READY" if any(e["verdict"] == "READY" for e in evaluations) else "READY_PENDING_INSTALL"
    elif all(e["verdict"] == "BLOCKED_ARCH" for e in evaluations):
        node_verdict = "BLOCKED_ARCH"
    elif any(e["verdict"] == "BLOCKED_QUANT_ARCH" for e in evaluations):
        node_verdict = "BLOCKED_QUANT_ARCH"
    elif any(e["verdict"] == "BLOCKED_MEMORY" for e in evaluations):
        node_verdict = "BLOCKED_MEMORY"
    else:
        node_verdict = "BLOCKED"

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "model": MODEL,
        "model_total_params_b": TOTAL_PARAMS_B,
        "quant": quant,
        "kv_overhead": kv_overhead,
        "dsa_stock_floor_cc": DSA_STOCK_FLOOR_CC,
        "supported_gpus": SUPPORTED_GPUS,
        "nvidia_smi": detected,
        "node": node,
        "engine_probes": engine_probes,
        "engines": evaluations,
        "summary": {
            "node_verdict": node_verdict,
            "any_engine_ready": any_ready,
            "ready_engines": [e["engine"] for e in evaluations if e["ready"]],
            "blocked_engines": [e["engine"] for e in evaluations if not e["ready"]],
            "recommended_quant": fit_quant,
            "arch": node["arch"],
            "compute_cap": node["compute_cap"],
            "total_vram_gb": node["total_vram_gb"],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    node = report.get("node", {})
    lines = [
        "# GLM-5.2 SGLang/vLLM serving-readiness preflight",
        "",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Model: `{report.get('model', '')}` ({report.get('model_total_params_b')}B)",
        f"- Node verdict: **`{s.get('node_verdict')}`**",
        f"- GPU: `{node.get('gpu_name', '') or 'unknown'}` × `{node.get('gpu_count')}` "
        f"— `{s.get('arch')}` (cc `{s.get('compute_cap')}`)",
        f"- Total VRAM: `{s.get('total_vram_gb')}` GB  |  recommended quant: `{s.get('recommended_quant')}`",
        f"- DSA stock kernel floor: sm_90 (Hopper) / sm_100 (Blackwell); supported GPUs: "
        f"`{', '.join(report.get('supported_gpus', []))}`",
        "",
        "| Engine | Verdict | Arch OK | Mem OK | Installed | Quant | Need (GB) | Have (GB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in report.get("engines", []):
        lines.append(
            f"| `{e['engine']}` | `{e['verdict']}` | {e['arch_ok']} | {e['memory_ok']} | "
            f"{e['engine_present']}{(' ' + e['engine_version']) if e['engine_version'] else ''} | "
            f"`{e['quant']}` | {e['required_vram_gb']} | {e['total_vram_gb']} |"
        )
    lines.append("")
    for e in report.get("engines", []):
        for note in e.get("notes", []):
            lines.append(f"- `{e['engine']}`: {note}")
    lines += [
        "",
        f"> KV headroom is modeled as a flat {report.get('kv_overhead')} fraction of weights, "
        "independent of served context length. At long context (the serve script defaults to "
        "131072, up to 1M) the real KV/activation working set is larger and not modeled here — "
        "verify on-node before committing to a high `--context-length`.",
        "",
        "After a READY/READY_PENDING_INSTALL node serves the endpoint, capture the issue-#130 "
        "evidence with `tools/glm52_serving_witness.py --base-url <url>/v1 --engine-cache-engine <engine>`.",
        "",
        "> **Scope of this witness.** It proves the *fak-fronts-an-external-engine* (SGLang/vLLM) "
        "form of #130 only — fak governs and fronts the weights an outside engine serves. It does "
        "**not** prove native in-kernel GLM-5.2 serving, which is the separate multi-month native "
        "track (`docs/notes/native-753b-track-staged-plan.md`; the external-vs-native evidence "
        "boundary is drawn in `docs/serving/glm52-full-size-serving-witness.md` §6).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 SGLang/vLLM serving-readiness preflight")
    ap.add_argument("--engine", action="append", choices=["sglang", "vllm"], default=[],
                    help="engine(s) to evaluate; default both")
    ap.add_argument("--quant", default="fp8", choices=list(QUANT_WEIGHTS_GB.keys()))
    ap.add_argument("--kv-overhead", type=float, default=0.15,
                    help="fractional VRAM headroom over weights for KV-cache/activations (default 0.15)")
    ap.add_argument("--gpu-name", default="", help="override/planner: GPU model name, e.g. 'NVIDIA H200'")
    ap.add_argument("--gpu-count", type=int, default=0, help="override/planner: number of GPUs")
    ap.add_argument("--gpu-memory-total-gb", type=float, default=0.0, help="override/planner: total VRAM in GB")
    ap.add_argument("--no-probe-engines", action="store_true", help="skip importing sglang/vllm (planner mode)")
    ap.add_argument("--out", type=Path, help="write JSON report")
    ap.add_argument("--markdown", type=Path, help="write Markdown report")
    ap.add_argument("--require-ready", action="store_true",
                    help="exit nonzero unless at least one engine is READY/READY_PENDING_INSTALL")
    args = ap.parse_args(argv)

    engines = tuple(dict.fromkeys(args.engine)) or ("sglang", "vllm")
    report = build_report(
        engines=engines,
        override_name=args.gpu_name,
        override_count=args.gpu_count,
        override_total_gb=args.gpu_memory_total_gb,
        quant=args.quant,
        kv_overhead=max(0.0, args.kv_overhead),
        probe_engines=not args.no_probe_engines,
    )
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    if args.require_ready and not report["summary"]["any_engine_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
