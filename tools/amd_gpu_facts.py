#!/usr/bin/env python3
"""amd_gpu_facts.py — the `nvidia-smi`-equivalent GPU-state probe for AMD cards on Windows.

The NVIDIA benchmark path probes NVIDIA GPUs with `nvidia-smi` (the nvidia probe
is private-side tooling). This box's accelerator is an AMD Radeon RX 7600,
and AMD's CLI equivalents (`rocm-smi` / `amd-smi`) require the ROCm SDK, which is NOT installed
here. So this probe reads the same facts — device identity, total/used VRAM, live compute
utilization, driver version — from sources that DO work on a stock Windows + AMD-driver box:

  * Win32_VideoController (CIM)        -> name, adapter RAM, driver version
  * `\\GPU Adapter Memory(*)\\Dedicated Usage` perf counter -> live VRAM bytes in use
  * `\\GPU Engine(*engtype_Compute*)\\Utilization Percentage` -> live compute-engine load

It returns the SAME shape of structured dict the nvidia probe does — `{"available": bool, ...}`
with a clear `error` when a source is missing — so a benchmark harness can call whichever
matches the host and fold the result uniformly. It is the missing observability that would
have caught the Vulkan-backend VRAM leak (a 1080-byte vkAllocateMemory failing because the
8 GB card was full) before it surfaced as a mid-benchmark crash.

Usage:
    python tools/amd_gpu_facts.py                 # one-shot JSON snapshot
    python tools/amd_gpu_facts.py --watch 0.5     # sample every 0.5s until Ctrl-C (smi-style)
    python tools/amd_gpu_facts.py --name 7600     # filter to a device whose name matches

Pure stdlib + PowerShell (already required by this repo's tooling); no pip deps.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time


def _pwsh(script: str, timeout: float = 20.0) -> tuple[bool, str, str]:
    """Run a PowerShell snippet, return (ok, stdout, stderr). PowerShell is the only way to
    reach Win32_VideoController + the GPU perf counters from stdlib without pywin32."""
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        return False, "", "no PowerShell (pwsh/powershell) on PATH"
    try:
        p = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode == 0, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", f"timeout after {timeout}s"
    except OSError as exc:
        return False, "", str(exc)


# One PowerShell pass gathers every fact as a JSON object, so the Python side does no parsing
# of human-formatted counter text. engtype_Compute is the Vulkan/CUDA-style compute queue;
# we sum all compute-engine instances (the driver splits work across several).
_PROBE = r"""
$ErrorActionPreference = 'SilentlyContinue'
$vc = Get-CimInstance Win32_VideoController |
      Where-Object { $_.Name -match 'Radeon|AMD' } | Select-Object -First 1
$dedicated = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples |
             Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
$compute = (Get-Counter '\GPU Engine(*engtype_Compute)\Utilization Percentage').CounterSamples |
           Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
$total = (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples |
         Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
# Per-engine-type breakdown: AMD's Vulkan compute often lands under engtype_Graphics or _Other,
# NOT _Compute, so reporting the busiest engine type is what actually tracks GPU work here.
$byType = @{}
foreach ($s in (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
  if ($s.InstanceName -match 'engtype_(\w+)') {
    $t = $Matches[1]
    if (-not $byType.ContainsKey($t)) { $byType[$t] = 0.0 }
    $byType[$t] += [double]$s.CookedValue
  }
}
$engines = ($byType.GetEnumerator() | ForEach-Object {
  [pscustomobject]@{ type = $_.Key; util_pct = [math]::Round($_.Value, 1) }
})
[pscustomobject]@{
  name            = $vc.Name
  driver_version  = $vc.DriverVersion
  adapter_ram     = [int64]$vc.AdapterRAM   # NOTE: caps at ~4 GB for >4 GB cards (WMI WORD wrap)
  vram_used_bytes = [int64]$dedicated
  compute_util_pct = [math]::Round([double]$compute, 1)
  total_util_pct   = [math]::Round([double]$total, 1)
  engines          = @($engines)
} | ConvertTo-Json -Compress
"""


def amd_facts(name_filter: str = "") -> dict:
    """Probe the AMD GPU's current state. Returns a dict with `available` and, on success,
    device identity + live VRAM + compute utilization. Mirrors the nvidia-side probe."""
    ok, out, err = _pwsh(_PROBE)
    if not ok or not out:
        return {"available": False, "error": err or "GPU perf counters returned nothing"}
    try:
        facts = json.loads(out)
    except json.JSONDecodeError as exc:
        return {"available": False, "error": f"probe JSON parse failed: {exc}", "raw": out}
    if name_filter and name_filter.lower() not in (facts.get("name") or "").lower():
        return {"available": False, "error": f"no GPU matching {name_filter!r}", "saw": facts.get("name")}
    # vram_used_bytes is the authoritative live figure; adapter_ram is WMI-capped, so report
    # used in MiB and flag the total as approximate.
    used = facts.get("vram_used_bytes") or 0
    facts["vram_used_mib"] = round(used / (1024 * 1024), 1)
    facts["available"] = True
    facts["note"] = (
        "adapter_ram is WMI-WORD-capped (~4GB) for >4GB cards; vram_used_bytes is exact. "
        "On AMD, Vulkan compute runs under the '3d' engine, so busiest_engine / total_util_pct "
        "track GPU work — compute_util_pct (engtype_Compute) reads ~0 even mid-decode."
    )
    # Surface the busiest engine on the one-shot snapshot too (watch mode already shows it),
    # so the honest "is the GPU working" signal travels with the JSON. On AMD Vulkan the headline
    # compute_util_pct is misleading (~0), so a consumer that only reads it would think the card
    # is idle while it decodes; busiest_engine is the field that actually tracks the work.
    engs = sorted(facts.get("engines") or [], key=lambda e: e.get("util_pct", 0) or 0, reverse=True)
    if engs:
        facts["busiest_engine"] = engs[0].get("type")
        facts["busiest_util_pct"] = engs[0].get("util_pct")
    return facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AMD GPU state probe (nvidia-smi-equivalent on Windows)")
    ap.add_argument("--name", default="", help="only report a device whose name matches this substring")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SEC",
                    help="sample every SEC seconds until interrupted (smi-style live view)")
    args = ap.parse_args(argv)

    if args.watch <= 0:
        # Probe ONCE: the printed snapshot and the exit code must describe the SAME sample.
        # The old code re-probed for the return value, so a transient counter blip could print
        # a healthy snapshot yet exit 1 (a TOCTOU), and every one-shot call paid the PowerShell
        # launch + Get-Counter sampling cost twice.
        facts = amd_facts(args.name)
        print(json.dumps(facts, indent=2))
        return 0 if facts.get("available") else 1

    # live-watch mode: one compact line per sample (the `nvidia-smi -l` analogue).
    try:
        while True:
            f = amd_facts(args.name)
            if f.get("available"):
                # On AMD, Vulkan compute is usually NOT under engtype_Compute, so the busiest
                # engine (derived in amd_facts) is the honest "is it working" signal.
                busiest = (f"{f['busiest_engine']}={f['busiest_util_pct']}%"
                           if f.get("busiest_engine") else "n/a")
                print(f"{time.strftime('%H:%M:%S')}  "
                      f"util(total)={f.get('total_util_pct'):>6}%  "
                      f"busiest={busiest:<18}  "
                      f"vram={f.get('vram_used_mib'):>8} MiB  "
                      f"[{f.get('name')}]", flush=True)
            else:
                print(f"{time.strftime('%H:%M:%S')}  unavailable: {f.get('error')}", flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
