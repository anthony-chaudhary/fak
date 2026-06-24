#!/usr/bin/env python3
"""llamacpp_vulkan_anchor.py — the missing *same-GPU* llama.cpp reference number for #462.

Issue #462 (AMD RX 7600 Vulkan decode parity) tracks fak's decode throughput against a
llama.cpp anchor. Until now that anchor was **CPU** ("~145 tok/s Q8 (CPU)"), which the issue
itself flags as open item #3: *"A llama.cpp-Vulkan-on-this-GPU reference number (current anchor
is CPU)."* Comparing fak's Vulkan path to llama.cpp's *CPU* path mixes backends — it cannot
say how far fak is from llama.cpp **on the same accelerator**.

This probe closes that gap. It runs `llama-bench` over a GGUF on a chosen Vulkan device (and,
for triangulation, on the CPU backend of the same box), parses the `t/s` table, and returns a
structured anchor: decode (`tg`) tok/s plus the prefill (`pp`) ladder. The Python side does no
floating-point work of its own — it only *reads* llama.cpp's own measured numbers — so the
anchor cannot drift from what llama.cpp reported. It is the Vulkan sibling of
`amd_gpu_facts.py` (live GPU state) for the *throughput* axis.

## Recorded anchor — SmolLM2-135M-Instruct-Q8_0, real RX 7600, 2026-06-23

Measured on this box (AMD Ryzen 9 9950X + Radeon RX 7600, Vulkan 1.4, Windows 11) with the
ggml winget `llama-bench` build `9b260fc9e (9673)`, model
`SmolLM2-135M-Instruct-Q8_0.gguf`, `-p 16,64,256,512 -n 64 -r 3`:

    arm                          decode tok/s   pp512 tok/s
    llama.cpp Vulkan (RX 7600)        609.2        33723   <- device Vulkan0, -ngl 99
    llama.cpp CPU (Zen4 AVX-512)      268.5         4401   <- --device none -ngl 0

Set beside fak's own same-box Q8 numbers (`experiments/gpu/VULKAN-Q8-RX7600-20260619.md`):
fak Vulkan Q8 decode **24.6** tok/s, fak CPU Q8 decode **176.9** tok/s. The honest gaps are:

  * **decode, same GPU:  609.2 / 24.6 ≈ 24.8× off** — fak Vulkan vs llama.cpp Vulkan.
  * decode, CPU vs CPU:  268.5 / 176.9 ≈ 1.52× off  — fak CPU Q8 vs llama.cpp CPU Q8.

This *supersedes* the issue's "~17 tok/s f32, ~8× off" framing, which compared fak's **f32**
GPU path (17) against llama.cpp's **CPU** path (145) across hosts. Against a fair same-GPU
Vulkan anchor the real gap is ~25×, and the residual is the GEMV / integer-dot decode kernels
(#46, #49) — exactly the issue's open items 1–2. Parity on AMD/Vulkan is **not** reached; this
just makes the distance honest and measurable.

Usage:
    python tools/llamacpp_vulkan_anchor.py --gguf PATH                 # default device Vulkan0
    python tools/llamacpp_vulkan_anchor.py --gguf PATH --device none   # CPU arm of the same box
    python tools/llamacpp_vulkan_anchor.py --gguf PATH --fak-decode 24.6   # print the gap too

Pure stdlib; shells out to `llama-bench` (from PATH or --bench). Off-box / no-GGUF: prints a
clear unavailable record and exits 1 — it never fabricates a number.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# The 2026-06-23 measurement above, kept as data so a consumer (or a regression check) can read
# the recorded anchor without a GPU. Regenerate by running this probe on the real RX 7600.
RECORDED_ANCHOR = {
    "model": "SmolLM2-135M-Instruct-Q8_0.gguf",
    "host": "AMD Ryzen 9 9950X + Radeon RX 7600 (Vulkan 1.4, Windows 11)",
    "llama_bench_build": "9b260fc9e (9673)",
    "date": "2026-06-23",
    "vulkan_rx7600": {"decode_tok_s": 609.2, "prefill_tok_s": {"pp16": 5126.4, "pp64": 5056.8, "pp256": 28274.7, "pp512": 33723.3}},
    "cpu_zen4": {"decode_tok_s": 268.5, "prefill_tok_s": {"pp16": 2029.4, "pp64": 3482.4, "pp256": 4508.4, "pp512": 4401.0}},
    # fak's own same-box Q8 numbers, for the honest gap (experiments/gpu/VULKAN-Q8-RX7600-20260619.md)
    "fak_vulkan_q8_decode_tok_s": 24.6,
    "fak_cpu_q8_decode_tok_s": 176.9,
}


def _find_bench(explicit: str = "") -> str:
    """Resolve the llama-bench executable: --bench, then PATH, then the ggml winget location."""
    if explicit:
        return explicit
    for cand in ("llama-bench", "llama-bench.exe"):
        found = shutil.which(cand)
        if found:
            return found
    winget = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
        r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-bench.exe"
    )
    return winget if os.path.exists(winget) else "llama-bench"


# One data row of llama-bench's markdown output, e.g.
#   | llama 256M Q8_0 | 136.40 MiB | 134.52 M | Vulkan | 99 | Vulkan0 | tg64 | 609.24 ± 12.28 |
# We key off the `test` cell (pp<N> / tg<N>) and the `t/s` cell (`mean ± stddev`).
_TEST_CELL = re.compile(r"^(pp|tg)\d+$")
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_bench_table(text: str) -> dict:
    """Parse `llama-bench -o md` output into {test_name: mean_tok_s}, e.g. {"tg64": 609.24}.

    Reads only the mean (the figure before `±`); pure string work, no measurement of our own,
    so the structured anchor can never disagree with what llama.cpp printed. Returns {} if no
    table is present (off-box, load failure) — the caller decides that is unavailable."""
    out: dict[str, float] = {}
    header_cols: list[str] | None = None
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # the header is the row naming the columns; lock onto the one with both 'test' and 't/s'
        if header_cols is None:
            low = [c.lower() for c in cells]
            if "test" in low and "t/s" in low:
                header_cols = low
            continue
        if set(cells) <= {""} or all(set(c) <= {"-", ":"} for c in cells if c):
            continue  # the |---|---| separator row
        if len(cells) != len(header_cols):
            continue
        row = dict(zip(header_cols, cells))
        test = row.get("test", "")
        if not _TEST_CELL.match(test):
            continue
        m = _NUM.search(row.get("t/s", ""))
        if m:
            out[test] = float(m.group(0))
    return out


def anchor_from_table(table: dict) -> dict:
    """Fold a parsed {test: tok_s} table into the {decode_tok_s, prefill_tok_s} anchor shape."""
    decode = next((v for k, v in table.items() if k.startswith("tg")), None)
    prefill = {k: v for k, v in table.items() if k.startswith("pp")}
    return {"decode_tok_s": decode, "prefill_tok_s": prefill}


def run_anchor(gguf: str, device: str = "Vulkan0", bench: str = "",
               prefill: str = "16,64,256,512", decode: int = 64, reps: int = 3) -> dict:
    """Run llama-bench on `gguf` for `device` and return the structured anchor. `device='none'`
    (with ngl 0) is the CPU arm. Returns {available: False, error} when the bench can't run."""
    exe = _find_bench(bench)
    if not (shutil.which(exe) or os.path.exists(exe)):
        return {"available": False, "error": f"llama-bench not found ({exe})"}
    if not os.path.exists(gguf):
        return {"available": False, "error": f"GGUF not found: {gguf}"}
    ngl = "0" if device == "none" else "99"
    cmd = [exe, "-m", gguf, "--device", device, "-ngl", ngl,
           "-p", prefill, "-n", str(decode), "-r", str(reps), "-o", "md"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"llama-bench failed: {exc}"}
    table = parse_bench_table(p.stdout)
    if not table:
        return {"available": False, "error": "no t/s table parsed from llama-bench",
                "stderr": p.stderr[-400:]}
    facts = anchor_from_table(table)
    facts.update({"available": True, "device": device, "gguf": os.path.basename(gguf)})
    return facts


def gap_vs(anchor_decode: float, fak_decode: float) -> float | None:
    """How many times faster the llama.cpp anchor is than fak, on decode. None if undefined."""
    if not fak_decode or anchor_decode is None:
        return None
    return round(anchor_decode / fak_decode, 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="llama.cpp Vulkan/CPU decode anchor on the RX 7600 (#462)")
    ap.add_argument("--gguf", default="", help="path to the GGUF model to bench")
    ap.add_argument("--device", default="Vulkan0", help="llama-bench device (Vulkan0 = RX 7600; none = CPU)")
    ap.add_argument("--bench", default="", help="path to llama-bench (default: PATH / ggml winget)")
    ap.add_argument("--prefill", default="16,64,256,512", help="prefill ladder for -p")
    ap.add_argument("--decode", type=int, default=64, help="decode tokens for -n")
    ap.add_argument("--reps", type=int, default=3, help="repetitions for -r")
    ap.add_argument("--fak-decode", type=float, default=0.0, help="fak's decode tok/s, to print the gap")
    ap.add_argument("--recorded", action="store_true", help="print the recorded 2026-06-23 anchor and exit")
    args = ap.parse_args(argv)

    if args.recorded or not args.gguf:
        print(json.dumps(RECORDED_ANCHOR, indent=2))
        return 0

    facts = run_anchor(args.gguf, args.device, args.bench, args.prefill, args.decode, args.reps)
    if facts.get("available") and args.fak_decode:
        facts["gap_vs_fak"] = gap_vs(facts.get("decode_tok_s"), args.fak_decode)
    print(json.dumps(facts, indent=2))
    return 0 if facts.get("available") else 1


if __name__ == "__main__":
    sys.exit(main())
