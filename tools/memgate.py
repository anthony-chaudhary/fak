#!/usr/bin/env python3
"""memgate.py — pre-flight memory-pressure gate for heavy model loads.

The fleet runs multi-GB model loads on a shared, fixed-RAM box (Apple M3 Pro, 36 GB
unified; Linux DGX). Loading a 15 GB Qwen3.6-27B GGUF on top of a `llama-server`
already holding the same model in Metal unified memory is a silent OOM: the kernel
compresses/swaps, decode collapses to a crawl (0.1-0.7 tok/s), and either the new
load or a peer's process dies. This gate makes that failure mode LOUD before launch:
it reads the real available memory, names the big holders competing for it, and
either admits the load, waits for space to free, or refuses with a structured reason.

The macOS catch (why `memory_pressure`'s "Pages free" lies under a GPU load): Apple
Silicon unified memory means a Metal `-ngl 99` model is resident as WIRED pages,
counted neither free nor purgeable, but consuming the same physical pool a new CPU
load needs. So this gate reports `wired` separately and folds a conservative
available estimate = free + purgeable - safety_margin, surfacing a HIGH-WIRED flag
when a GPU model is clearly resident (wired > 40% of RAM is the heuristic).

Usage:
    python tools/memgate.py --require-gb 20                 # one-shot; exit 0 if ok, 1 if not
    python tools/memgate.py --require-gb 20 --wait 300      # poll up to 300s for space
    python tools/memgate.py --require-gb 20 --json          # structured snapshot only
    python tools/memgate.py                                  # snapshot, no requirement

Cross-platform: macOS (`vm_stat` + `sysctl hw.memsize`) and Linux (`/proc/meminfo`).
Pure stdlib. Exits: 0 = admitted/sufficient, 1 = refused/insufficient, 2 = read error.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time

# Conservative floor reserved for the OS / compositor / shell — never hand this to a model.
SAFETY_MARGIN_GB = 2.0
# Wired pages above this fraction of RAM strongly implies a resident GPU/Metal model on Apple Silicon.
HIGH_WIRED_FRACTION = 0.40
# Big-holder RSS threshold for the "who is competing" list.
HOLDER_GB = 1.0


def _run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=10
    ).stdout


def read_darwin() -> dict:
    out = _run(["vm_stat"])
    page_size = int(_run(["sysctl", "-n", "hw.pagesize"]).strip() or "16384")
    total = int(_run(["sysctl", "-n", "hw.memsize"]).strip())
    vals = {}
    for line in out.splitlines():
        line = line.strip().rstrip(".")
        if ":" not in line:
            continue
        key, _, num = line.partition(":")
        key = key.strip().lower()
        num = num.strip().split()[0]
        try:
            vals[key] = int(num)
        except ValueError:
            pass
    # vm_stat reports page counts; the keys we care about differ in casing/phrasing across macOS.
    free = vals.get("pages free", vals.get("free pages", 0)) * page_size
    purgeable = vals.get("pages purgeable", 0) * page_size
    wired = vals.get("pages wired down", vals.get("wired pages", 0)) * page_size
    compressed = vals.get("pages occupied by compressor", 0) * page_size
    return {
        "total_bytes": total,
        "free_bytes": free,
        "purgeable_bytes": purgeable,
        "wired_bytes": wired,
        "compressed_bytes": compressed,
        # Conservative: reclaim the purgeable, keep the safety margin, never touch wired.
        "available_bytes": max(free + purgeable - int(SAFETY_MARGIN_GB * 1e9), 0),
    }


def read_linux() -> dict:
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            num = rest.strip().split()[0]
            try:
                vals[key] = int(num) * 1024  # /proc/meminfo is in kB
            except ValueError:
                pass
    total = vals.get("MemTotal", 0)
    free = vals.get("MemFree", 0)
    avail = vals.get("MemAvailable", free + vals.get("Cached", 0))
    return {
        "total_bytes": total,
        "free_bytes": free,
        "available_bytes": max(avail - int(SAFETY_MARGIN_GB * 1e9), 0),
        "purgeable_bytes": vals.get("Cached", 0),
        "wired_bytes": 0,  # not a Linux concept; slab/unreclaimable approximated elsewhere
        "compressed_bytes": 0,
    }


def read_memory() -> dict:
    sysname = platform.system().lower()
    if sysname == "darwin":
        return read_darwin()
    if sysname == "linux":
        return read_linux()
    raise RuntimeError(f"unsupported platform for memgate: {sysname}")


def big_holders() -> list[dict]:
    """Processes with RSS >= HOLDER_GB — the competing tenants the operator can free."""
    out = _run(["ps", "-axo", "pid,rss,comm"])
    holders = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except ValueError:
            continue
        rss_gb = rss_kb / 1e6
        if rss_gb < HOLDER_GB:
            continue
        holders.append({"pid": pid, "rss_gb": round(rss_gb, 2), "comm": parts[2][:60]})
    holders.sort(key=lambda h: h["rss_gb"], reverse=True)
    return holders


def snapshot() -> dict:
    mem = read_memory()
    total_gb = mem["total_bytes"] / 1e9
    wired_gb = mem.get("wired_bytes", 0) / 1e9
    snap = {
        "platform": platform.system().lower(),
        "total_gb": round(total_gb, 2),
        "free_gb": round(mem["free_bytes"] / 1e9, 2),
        "available_gb": round(mem["available_bytes"] / 1e9, 2),
        "purgeable_gb": round(mem.get("purgeable_bytes", 0) / 1e9, 2),
        "wired_gb": round(wired_gb, 2),
        "compressed_gb": round(mem.get("compressed_bytes", 0) / 1e9, 2),
        "safety_margin_gb": SAFETY_MARGIN_GB,
        "high_wired": wired_gb / total_gb > HIGH_WIRED_FRACTION if total_gb else False,
        "holders": big_holders(),
    }
    snap["note"] = (
        "wired > 40% of RAM — a Metal/GPU model is likely resident in unified memory; "
        "'available' does NOT include wired, so a new CPU model load may still OOM "
        "against the GPUresident. Stop the GPU holder first."
        if snap["high_wired"]
        else "ok"
    )
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description="memory-pressure gate for heavy loads")
    ap.add_argument("--require-gb", type=float, help="GB the upcoming load needs (weights + KV + compute)")
    ap.add_argument("--wait", type=int, default=0, help="poll up to N seconds for memory to free (default: no wait)")
    ap.add_argument("--json", action="store_true", help="emit only the structured snapshot")
    ap.add_argument("--interval", type=float, default=5.0, help="poll interval seconds (with --wait)")
    args = ap.parse_args()

    if args.json and not args.require_gb:
        print(json.dumps(snapshot(), indent=2))
        return 0

    if not args.require_gb:
        snap = snapshot()
        print(json.dumps(snap, indent=2))
        return 0

    deadline = time.time() + args.wait if args.wait else 0
    while True:
        snap = snapshot()
        have = snap["available_gb"]
        ok = have >= args.require_gb and not snap["high_wired"]
        snap["require_gb"] = args.require_gb
        snap["admit"] = ok
        snap["shortfall_gb"] = round(max(args.require_gb - have, 0.0), 2)
        if ok or not args.wait or time.time() >= deadline:
            if args.json or not sys.stdout.isatty():
                print(json.dumps(snap, indent=2))
            else:
                verdict = "ADMIT" if ok else "REFUSE"
                print(f"[memgate] {verdict}: need {args.require_gb}GB, available {have}GB "
                      f"(shortfall {snap['shortfall_gb']}GB, wired {snap['wired_gb']}GB "
                      f"{'HIGH' if snap['high_wired'] else 'ok'})")
                if not ok and snap["holders"]:
                    print("[memgate] big holders to consider stopping:")
                    for h in snap["holders"][:5]:
                        print(f"           pid {h['pid']:>6}  {h['rss_gb']:>5}GB  {h['comm']}")
            return 0 if ok else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
