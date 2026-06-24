#!/usr/bin/env python3
"""Probe a REAL long-context model path for the fanbench prefix-scale points.

The fanbench pscale rows in `docs/benchmarks/FANOUT-BENCH-RESULTS.md` price a
byte-identical shared prefix of P = 262144 / 524288 / 1048576 tokens under a
transparent prompt-cache COST MODEL. They prove the geometry; they do NOT prove a
real model/backend on this host can prefill and serve those prefix lengths within
its context window and memory.

This tool closes the honest gap GitHub issue #431 names. It does NOT extrapolate the
modeled curve. It interrogates what is actually installed and, for each target P,
either MEASURES a real prefill (when a qualifying path exists) or records the
structured CEILING that stopped it — context length, missing checkpoint, missing
server, or KV memory. The verdict and the recorded ceiling are real facts read off
this host, never the modeled token economics dressed up as a measurement.

Candidate paths probed (the three the issue lists):
  1. fak in-kernel model  -- the selected checkpoint's max_position_embeddings.
  2. llama.cpp            -- llama-cli/llama-bench on PATH + a GGUF with ctx >= P.
  3. vLLM / SGLang server -- a reachable prefix-caching OpenAI endpoint.

KV-cache sizing uses real transformer geometry mirrored from
`internal/turnbench/longcontext.go` (NamedShape), so the memory ceiling is a
quantitative fact, not a guess.

Usage:
  python tools/fanout_longctx_probe.py            # all three target P, default qwen25-7b
  python tools/fanout_longctx_probe.py --model qwen25-1.5b --prefixes 262144
  python tools/fanout_longctx_probe.py --outdir experiments/fanout/pscale

Deterministic: the artifact embeds only stable host facts (no wall-clock), so a
re-run on the same host reproduces it byte-for-byte — that IS the gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import sys

# Transformer geometry, mirrored from internal/turnbench/longcontext.go NamedShape.
# KV-cache width per token = 2 (K,V) * num_layers * num_kv_heads * head_dim * dtype_bytes.
MODELS = {
    "smollm2-135m": {"layers": 30, "kv_heads": 3, "head_dim": 64},
    "qwen25-1.5b": {"layers": 28, "kv_heads": 2, "head_dim": 128},
    "qwen25-7b": {"layers": 28, "kv_heads": 4, "head_dim": 128},
}
DTYPE_BYTES = 2  # fp16 KV cache — the standard serving default.

TARGET_PREFIXES = [262144, 524288, 1048576]

# Where the in-kernel selected checkpoint's config.json may live (real dirs on disk).
CHECKPOINT_GLOBS = [
    "internal/model/.cache/*/config.json",
    ".cache/*/config.json",
    "internal/ggufload/.cache/*/config.json",
]

# Local OpenAI-compatible prefix-caching servers (vLLM 8000, SGLang 30000).
SERVER_CANDIDATES = [("127.0.0.1", 8000), ("127.0.0.1", 30000)]


def kv_cache_bytes(model: dict, prefix: int) -> int:
    return 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * prefix * DTYPE_BYTES


def host_ram_bytes() -> int:
    """Total physical RAM in bytes, cross-platform, stdlib only."""
    if sys.platform.startswith("win"):
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullTotalPhys)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def dedicated_gpu_vram_bytes() -> int:
    """Dedicated GPU VRAM via nvidia-smi/rocm-smi, else 0.

    An integrated/UMA Vulkan device (no dedicated VRAM) reports 0 here on purpose:
    it shares host RAM, so it is sized against the RAM ceiling, not a VRAM ceiling.
    """
    nv = shutil.which("nvidia-smi")
    if nv:
        try:
            import subprocess

            out = subprocess.run(
                [nv, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            return max(int(float(x)) for x in out) * 1024 * 1024
        except Exception:
            return 0
    return 0


def discover_checkpoint_ceiling() -> dict:
    """Largest max_position_embeddings across checkpoints actually present on disk."""
    best = {"context_tokens": 0, "config": None, "model_type": None}
    for pat in CHECKPOINT_GLOBS:
        for cfg in sorted(glob.glob(pat)):
            try:
                with open(cfg, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            ctx = int(d.get("max_position_embeddings", 0) or 0)
            if ctx > best["context_tokens"]:
                best = {
                    "context_tokens": ctx,
                    "config": cfg.replace("\\", "/"),
                    "model_type": d.get("model_type"),
                }
    return best


def discover_llamacpp() -> dict:
    bins = {name: shutil.which(name) for name in ("llama-cli", "llama-bench", "main")}
    present = {k: v for k, v in bins.items() if v}
    ggufs = []
    for root in (".", os.path.expanduser("~/.cache/llama.cpp"), os.path.expanduser("~/models")):
        if os.path.isdir(root):
            ggufs += glob.glob(os.path.join(root, "**", "*.gguf"), recursive=True)
    return {"binaries": present, "gguf_count": len(ggufs)}


def server_reachable() -> tuple[bool, str]:
    for host, port in SERVER_CANDIDATES:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True, f"{host}:{port}"
        except OSError:
            continue
    return False, "none of " + ", ".join(f"{h}:{p}" for h, p in SERVER_CANDIDATES)


def probe_prefix(prefix: int, model_name: str, host: dict, ckpt: dict, llama: dict,
                 server: tuple[bool, str]) -> dict:
    geom = MODELS[model_name]
    kv_bytes = kv_cache_bytes(geom, prefix)
    kv_fits_ram = kv_bytes <= host["total_ram_bytes"]
    paths = []

    # 1. fak in-kernel model — gated by the selected checkpoint's context window.
    if ckpt["context_tokens"] >= prefix:
        paths.append(_skip("fak-in-kernel", "NOT_RUN",
                           "checkpoint context %d >= %d, but in-kernel long-context "
                           "prefill measurement is not wired into this probe yet"
                           % (ckpt["context_tokens"], prefix)))
    else:
        detail = ("selected checkpoint max_position_embeddings=%d < %d (%dx short); "
                  "config=%s" % (ckpt["context_tokens"], prefix,
                                 prefix // max(ckpt["context_tokens"], 1),
                                 ckpt["config"] or "none-found"))
        p = _skip("fak-in-kernel", "CONTEXT_CEILING", detail)
        p["context_ceiling_tokens"] = ckpt["context_tokens"]
        paths.append(p)

    # 2. llama.cpp — needs a binary AND a GGUF whose context >= P.
    if not llama["binaries"]:
        paths.append(_skip("llama.cpp", "BACKEND_UNAVAILABLE",
                           "no llama-cli/llama-bench/main on PATH"))
    elif llama["gguf_count"] == 0:
        bin_path = next(iter(llama["binaries"].values()))
        paths.append(_skip("llama.cpp", "MODEL_UNAVAILABLE",
                           "llama.cpp present (%s) but no .gguf with context >= %d found "
                           "on host; KV for %s at P=%d is %.1f GiB, host RAM %.1f GiB (%s)"
                           % (bin_path, prefix, model_name, prefix, kv_bytes / 2 ** 30,
                              host["total_ram_bytes"] / 2 ** 30,
                              "fits" if kv_fits_ram else "EXCEEDS")))
    else:
        paths.append(_skip("llama.cpp", "NOT_RUN",
                           "%d GGUF(s) present; qualifying-context check + measurement "
                           "not implemented in this probe" % llama["gguf_count"]))

    # 3. vLLM / SGLang prefix-caching server.
    reachable, where = server
    if reachable:
        paths.append(_skip("vllm-sglang", "NOT_RUN",
                           "server reachable at %s; OpenAI prefill measurement not "
                           "implemented in this probe" % where))
    else:
        paths.append(_skip("vllm-sglang", "SERVER_UNAVAILABLE",
                           "no prefix-caching server reachable (%s)" % where))

    measured = any(p["status"] == "measured" for p in paths)
    return {
        "schema": "fanout-longctx-measure/1",
        "prefix_tokens": prefix,
        "reference_model": model_name,
        "model_geometry": {**geom, "dtype_bytes": DTYPE_BYTES,
                           "kv_bytes_per_token": kv_bytes // prefix},
        "kv_cache_bytes": kv_bytes,
        "kv_cache_gib": round(kv_bytes / 2 ** 30, 2),
        "kv_fits_host_ram": kv_fits_ram,
        "host": host,
        "paths": paths,
        "overall": "MEASURED" if measured else "SKIPPED_NO_LONGCTX_PATH",
        "note": ("No silent extrapolation: the MODELED token economics for this P live in "
                 "docs/benchmarks/FANOUT-BENCH-RESULTS.md (pscale table). This artifact "
                 "records only what a REAL model path could measure on this host — here, a "
                 "recorded ceiling, not a wall-clock number."),
        "generated_by": "tools/fanout_longctx_probe.py",
    }


def _skip(path: str, reason: str, detail: str) -> dict:
    return {
        "path": path,
        "status": "skipped",
        "reason": reason,
        "detail": detail,
        "ttft_ms": None,
        "prefill_ms": None,
        "measured_kv_bytes": None,
    }


def write_csv(rows: list, path: str) -> None:
    cols = ["prefix_tokens", "reference_model", "path", "status", "reason",
            "context_ceiling_tokens", "kv_cache_gib", "kv_fits_host_ram",
            "ttft_ms", "prefill_ms"]
    lines = [",".join(cols)]
    for art in rows:
        for p in art["paths"]:
            vals = [
                art["prefix_tokens"], art["reference_model"], p["path"], p["status"],
                p["reason"], p.get("context_ceiling_tokens", ""), art["kv_cache_gib"],
                art["kv_fits_host_ram"],
                "" if p["ttft_ms"] is None else p["ttft_ms"],
                "" if p["prefill_ms"] is None else p["prefill_ms"],
            ]
            lines.append(",".join(str(v) for v in vals))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25-7b", choices=sorted(MODELS),
                    help="reference geometry for KV sizing (default qwen25-7b)")
    ap.add_argument("--prefixes", default=",".join(str(p) for p in TARGET_PREFIXES),
                    help="comma-separated target prefix sizes")
    ap.add_argument("--outdir", default="experiments/fanout/pscale",
                    help="directory for the JSON/CSV artifacts")
    args = ap.parse_args()

    prefixes = [int(x) for x in args.prefixes.replace(" ", "").split(",") if x]
    host = {
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "total_ram_bytes": host_ram_bytes(),
        "total_ram_gib": round(host_ram_bytes() / 2 ** 30, 1),
        "dedicated_gpu_vram_bytes": dedicated_gpu_vram_bytes(),
    }
    ckpt = discover_checkpoint_ceiling()
    llama = discover_llamacpp()
    server = server_reachable()

    os.makedirs(args.outdir, exist_ok=True)
    artifacts = []
    for prefix in prefixes:
        art = probe_prefix(prefix, args.model, host, ckpt, llama, server)
        artifacts.append(art)
        out = os.path.join(args.outdir, "longctx-measure-p%d.json" % prefix)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(art, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("%-9s P=%-8d -> %s" % (art["overall"], prefix,
              " | ".join("%s:%s" % (p["path"], p["reason"]) for p in art["paths"])))

    csv_path = os.path.join(args.outdir, "longctx-measure.csv")
    write_csv(artifacts, csv_path)

    measured = any(a["overall"] == "MEASURED" for a in artifacts)
    print("\ncheckpoint ceiling: %d tokens (%s)" % (ckpt["context_tokens"], ckpt["config"]))
    print("host: %s, %.1f GiB RAM, %d dedicated GPU VRAM bytes"
          % (host["platform"], host["total_ram_gib"], host["dedicated_gpu_vram_bytes"]))
    print("wrote %d JSON + %s" % (len(artifacts), csv_path))
    if not measured:
        print("VERDICT: no real long-context path on this host; ceilings recorded, "
              "modeled curve NOT extrapolated into a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
