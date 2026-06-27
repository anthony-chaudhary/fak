#!/usr/bin/env python3
"""crash_audit.py — one-command "why did this box crash?" forensic probe for Windows fleet hosts.

This repo's fleet runs many long-lived agent sessions on a single Windows workstation, and when
that host hard-crashes every session dies with it. Reconstructing *why* by hand means a dozen
`Get-WinEvent` incantations against the System log (Kernel-Power 41, WER 1001, EventLog 6008),
decoding the bugcheck code from hex, and cross-checking RAM topology / dump config — exactly the
sweep done by hand for the 2026-06-18 `0x4E PFN_LIST_CORRUPT` crash (see
`CRASH-AUDIT-2026-06-18-pfn-list-corrupt.md`). This tool automates that sweep so the next crash
is a single command instead of a forensic afternoon.

What it reads (all from sources that work WITHOUT admin — the event log is readable unelevated;
only the dump *files* under C:\\Windows\\Minidump need elevation, which the tool detects and
reports rather than failing on):

  * Kernel-Power 41        -> reboot-side record + raw BugcheckCode (the canonical "did the
                              kernel bugcheck, or was it a power-loss/hard-hang?" signal: code 0
                              with no dump == power loss / lockup, code != 0 == a real bugcheck)
  * WER 1001               -> the saved minidump path per crash
  * EventLog 6008          -> actual "previous shutdown was unexpected" wall-clock of each dirty
                              stop; this is the best crash-time signal after an auto-reboot
  * WHEA-Logger            -> hardware machine-checks (their ABSENCE on non-ECC RAM is itself a
                              clue: bit-flips corrupt silently instead of being logged)
  * Resource-Exhaustion 2004 -> commit/memory exhaustion (rules OOM in or out)
  * disk/Ntfs/volmgr errors  -> storage faults; benign volmgr 162 dump-write records are filtered
  * RAM topology + ECC + pagefile + CrashControl registry -> the config context a bugcheck
                              code is only meaningful against (e.g. memory-corruption codes on a
                              non-ECC 4-DIMM config point hard at the memory subsystem)

It decodes each bugcheck to its name + a fault CLASS (memory / driver / cpu_hw / storage /
software / power-loss) and folds the recent cluster into a single VERDICT with concrete next
actions. Output is JSON by default (pipe it anywhere) or `--report` for a human summary.

Pure stdlib + PowerShell, same as `tools/amd_gpu_facts.py`; no pip deps. The analysis half
(`decode_bugcheck`, `audit`) is a pure function of the probe dict, so it unit-tests without a
Windows box.

Blind-spot closure: a memory/driver verdict cannot tell bad RAM from a buggy driver without the
minidump stack, so `audit` always reports a `blind_spot` saying whether the leading bugcheck is
dump-attributable, and `main` best-effort runs `cdb`/WinDbg `!analyze -v` on each in-window dump
when a debugger is installed (`--no-analyze` skips it). An attributed dump names the faulting
module and clears the blind spot.

Usage:
    python tools/crash_audit.py                 # JSON snapshot of the crash forensics
    python tools/crash_audit.py --report        # human-readable report + verdict
    python tools/crash_audit.py --days 30       # only crashes within the last N days (default 90)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

# --- Bugcheck dictionary -------------------------------------------------------------------
# code -> (symbolic name, fault class, one-line hint). Classes drive the verdict; "memory"
# and "memory_or_driver" both count toward the memory-subsystem heuristic. This is the subset
# that actually shows up on fleet/consumer hardware — not the full ~300-entry MS table.
_FAULT_HINTS = {
    "memory": "corrupted kernel memory structure - bad RAM/IMC or a driver scribbling out of bounds",
    "memory_or_driver": "memory access fault - bad RAM, unstable memory config, or a faulty driver",
    "memory_or_hw": "kernel trap - bad RAM, CPU/IMC instability, or aggressive overclock",
    "driver": "a kernel-mode driver faulted - identify it with WinDbg `!analyze -v <dump>`",
    "cpu_hw": "hardware machine-check - CPU, PSU, motherboard, or a severe memory error",
    "storage": "the storage stack stalled - disk/controller/driver timeout",
    "software": "a critical user-mode process died - usually a user-mode software fault",
    "power_loss": "no bugcheck recorded - power loss, hard lockup, or thermal cutoff (no dump)",
    "unknown": "uncommon/unmapped bugcheck - analyze the dump directly",
}

# Keyed by integer bugcheck code.
_BUGCHECKS: dict[int, tuple[str, str]] = {
    0x00: ("(no bugcheck / power-loss)", "power_loss"),
    0x0A: ("IRQL_NOT_LESS_OR_EQUAL", "memory_or_driver"),
    0x18: ("REFERENCE_BY_POINTER", "driver"),
    0x1A: ("MEMORY_MANAGEMENT", "memory_or_driver"),
    0x1E: ("KMODE_EXCEPTION_NOT_HANDLED", "driver"),
    0x3B: ("SYSTEM_SERVICE_EXCEPTION", "driver"),
    0x4E: ("PFN_LIST_CORRUPT", "memory"),
    0x50: ("PAGE_FAULT_IN_NONPAGED_AREA", "memory_or_driver"),
    0x7A: ("KERNEL_DATA_INPAGE_ERROR", "storage"),
    0x7E: ("SYSTEM_THREAD_EXCEPTION_NOT_HANDLED", "driver"),
    0x7F: ("UNEXPECTED_KERNEL_MODE_TRAP", "memory_or_hw"),
    0x9C: ("MACHINE_CHECK_EXCEPTION", "cpu_hw"),
    0xA0: ("INTERNAL_POWER_ERROR", "driver"),
    0xC2: ("BAD_POOL_CALLER", "driver"),
    0xC4: ("DRIVER_VERIFIER_DETECTED_VIOLATION", "driver"),
    0xC5: ("DRIVER_CORRUPTED_EXPOOL", "memory_or_driver"),
    0xCA: ("PNP_DETECTED_FATAL_ERROR", "driver"),
    0xD1: ("DRIVER_IRQL_NOT_LESS_OR_EQUAL", "driver"),
    0xEF: ("CRITICAL_PROCESS_DIED", "software"),
    0xF4: ("CRITICAL_OBJECT_TERMINATION", "software"),
    0x101: ("CLOCK_WATCHDOG_TIMEOUT", "cpu_hw"),
    0x109: ("CRITICAL_STRUCTURE_CORRUPTION", "memory_or_driver"),
    0x124: ("WHEA_UNCORRECTABLE_ERROR", "cpu_hw"),
    0x133: ("DPC_WATCHDOG_VIOLATION", "driver"),
    0x139: ("KERNEL_SECURITY_CHECK_FAILURE", "memory_or_driver"),
    0x1000007E: ("SYSTEM_THREAD_EXCEPTION_NOT_HANDLED_M", "driver"),
    0x1000008E: ("KERNEL_MODE_EXCEPTION_NOT_HANDLED_M", "memory_or_driver"),
}

# Classes that implicate the memory subsystem (RAM / IMC / memory config).
_MEMORY_CLASSES = {"memory", "memory_or_driver", "memory_or_hw"}

# Recorded dispositions live next to this script so a forensic conclusion ships as data, not just
# prose. See load_dispositions / match_disposition below and tools/crash_dispositions.json.
_DISPOSITIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_dispositions.json")


def decode_bugcheck(code: int) -> dict:
    """Map an integer bugcheck code to {code_hex, name, fault_class, hint}. Unknown codes get a
    stable 'unknown' shape rather than raising, so a never-before-seen crash still reports."""
    name, klass = _BUGCHECKS.get(code, ("UNKNOWN_BUGCHECK", "unknown"))
    return {
        "code": code,
        "code_hex": f"0x{code:08x}",
        "name": name,
        "fault_class": klass,
        "hint": _FAULT_HINTS.get(klass, _FAULT_HINTS["unknown"]),
    }


def load_dispositions(path: str | None = None) -> list[dict]:
    """Load the recorded crash dispositions (tools/crash_dispositions.json). Returns the list of
    disposition dicts, or [] if the file is missing/empty/malformed — a missing record file must
    never break the audit, it just means 'no disposition recorded yet'. Pure I/O, no Windows."""
    p = path or _DISPOSITIONS_FILE
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    disps = data.get("dispositions") if isinstance(data, dict) else None
    return disps if isinstance(disps, list) else []


def match_disposition(crash: dict | None, probe: dict, dispositions: list[dict]) -> dict | None:
    """Return the recorded disposition whose `match` fits this crash on this host, else None.

    Match is intentionally coarse so a RECURRING host-level fault (new params/timestamp each time,
    same fault class) keeps matching the one recorded disposition instead of re-opening as a fresh
    `cause: unclear`. A `match` block may constrain on `fault_class` (against the crash) and/or
    substrings of `board` / `cpu` (against the probe's hardware). All present keys must match; an
    empty `match` matches nothing (guards against an accidental catch-all). First match wins."""
    if not crash:
        return None
    board = str(probe.get("board") or "")
    cpu = str(probe.get("cpu") or "")
    for d in dispositions:
        m = d.get("match") or {}
        if not m:
            continue
        if "fault_class" in m and m["fault_class"] != crash.get("fault_class"):
            continue
        if "board" in m and str(m["board"]) not in board:
            continue
        if "cpu" in m and str(m["cpu"]) not in cpu:
            continue
        return d
    return None


def _pwsh(script: str, timeout: float = 40.0) -> tuple[bool, str, str]:
    """Run a PowerShell snippet, return (ok, stdout, stderr). PowerShell is the only stdlib-free
    way to reach Get-WinEvent + the CIM hardware classes. Mirrors amd_gpu_facts._pwsh."""
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


# One PowerShell pass emits the whole forensic snapshot as a JSON object so the Python side does
# zero text parsing. Every Get-WinEvent is wrapped so a missing log/provider yields [] not a throw.
_PROBE = r"""
$ErrorActionPreference = 'SilentlyContinue'
function Evt($p){ try { Get-WinEvent -FilterHashtable $p -ErrorAction Stop } catch { @() } }

# Kernel-Power 41 is the canonical reboot-side crash record: it carries BugcheckCode even when WER
# didn't run. EventLog 6008 below supplies the actual dirty-shutdown wall-clock when available.
$crashes = @(Evt @{LogName='System'; Id=41} | Select-Object -First 40 | ForEach-Object {
  $x=[xml]$_.ToXml(); $h=@{}
  foreach($d in $x.Event.EventData.Data){ $h[$d.Name]=$d.'#text' }
  [pscustomobject]@{
    time          = $_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss')
    bugcheck_code = [int]$h['BugcheckCode']
    p1 = $h['BugcheckParameter1']; p2 = $h['BugcheckParameter2']
    p3 = $h['BugcheckParameter3']; p4 = $h['BugcheckParameter4']
    power_button  = $h['PowerButtonTimestamp']
  }
})

# WER 1001 gives the dump path per crash (joined to crashes by time on the Python side).
$dumps = @(Evt @{LogName='System'; Id=1001; ProviderName='Microsoft-Windows-WER-SystemErrorReporting'} |
  Select-Object -First 40 | ForEach-Object {
    $m=$_.Message
    $d = if($m -match 'saved in:\s*([A-Za-z]:\\[^.\r\n]+\.dmp)'){ $matches[1].Trim() } else { '' }
    [pscustomobject]@{ time=$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); dump=$d }
  })

$dirty = @(Evt @{LogName='System'; Id=6008} | Select-Object -First 20 | ForEach-Object {
  $x=[xml]$_.ToXml()
  $raw_time = [string]$x.Event.EventData.Data[0]
  $raw_date = [string]$x.Event.EventData.Data[1]
  $shutdown = ''
  $clean = ("$raw_date $raw_time" -replace '[\u200e\u200f]', '').Trim()
  try {
    $dt = [datetime]::Parse($clean, [System.Globalization.CultureInfo]::GetCultureInfo('en-US'))
    $shutdown = $dt.ToString('yyyy-MM-dd HH:mm:ss')
  } catch {}
  [pscustomobject]@{
    logged=$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss')
    shutdown_time=$shutdown
    detail=(($_.Message -split "`n")[0]).Trim()
  }
})

$whea = @(Evt @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} | Select-Object -First 20 | ForEach-Object {
  [pscustomobject]@{ time=$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); id=$_.Id; level=$_.LevelDisplayName }
})

$rex = @(Evt @{LogName='System'; ProviderName='Microsoft-Windows-Resource-Exhaustion-Detector'; Id=2004} |
  Select-Object -First 10 | ForEach-Object { $_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') })

$disk = @(Evt @{LogName='System'; ProviderName='disk','Disk','Ntfs','volmgr'} |
  Where-Object { $_.Level -le 2 -and !($_.ProviderName -eq 'volmgr' -and $_.Id -eq 162) } |
  Select-Object -First 15 | ForEach-Object {
    [pscustomobject]@{ time=$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); id=$_.Id; provider=$_.ProviderName }
  })

$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$bios = Get-CimInstance Win32_BIOS
$bb = Get-CimInstance Win32_BaseBoard
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1

$dimms = @(Get-CimInstance Win32_PhysicalMemory | ForEach-Object {
  [pscustomobject]@{
    gb=[int][math]::Round($_.Capacity/1GB,0); speed=[int]$_.Speed
    configured=[int]$_.ConfiguredClockSpeed; part=($_.PartNumber).Trim(); loc=$_.DeviceLocator
  }
})
# MemoryErrorCorrection: 3=None 4=Parity 5=Single-bit ECC 6=Multi-bit ECC
$arr = Get-CimInstance Win32_PhysicalMemoryArray | Select-Object -First 1

$cc = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl'
$dumpdir = 'C:\Windows\Minidump'
$dump_access = $true
try { $null = Get-ChildItem $dumpdir -ErrorAction Stop } catch { $dump_access = $false }

[pscustomobject]@{
  now           = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  last_boot     = $os.LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')
  crashes       = $crashes
  dumps         = $dumps
  dirty_shutdowns = $dirty
  whea          = $whea
  resource_exhaustion = $rex
  disk_errors   = $disk
  ram_total_gb  = [int][math]::Round($cs.TotalPhysicalMemory/1GB,0)
  dimms         = $dimms
  memory_error_correction = [int]$arr.MemoryErrorCorrection
  pagefile_auto = [bool]$cs.AutomaticManagedPagefile
  cpu           = $cpu.Name
  board         = "$($bb.Manufacturer) $($bb.Product)"
  bios_version  = $bios.SMBIOSBIOSVersion
  bios_date     = $bios.ReleaseDate.ToString('yyyy-MM-dd')
  crash_dump_enabled = [int]$cc.CrashDumpEnabled
  minidump_accessible = $dump_access
} | ConvertTo-Json -Depth 6 -Compress
"""

_DUMP_MODE = {0: "none", 1: "complete", 2: "kernel", 3: "small(minidump)", 7: "automatic"}
_ECC_MODE = {3: "None (non-ECC)", 4: "Parity", 5: "Single-bit ECC", 6: "Multi-bit ECC"}


def collect(timeout: float = 40.0) -> dict:
    """Gather the raw forensic snapshot via PowerShell. Returns {available: False, error} on any
    failure so callers never crash on a non-Windows box or a denied query."""
    ok, out, err = _pwsh(_PROBE, timeout=timeout)
    if not ok or not out:
        return {"available": False, "error": err or "probe returned nothing"}
    try:
        probe = json.loads(out)
    except json.JSONDecodeError as exc:
        return {"available": False, "error": f"probe JSON parse failed: {exc}", "raw": out[:500]}
    probe["available"] = True
    return probe


def _find_debugger() -> str:
    """Return the first available Microsoft kernel debugger on PATH (cdb preferred — it is
    scriptable and exits on its own). '' if none, so attribution degrades to a reported blind
    spot instead of failing."""
    for name in ("cdb", "cdb.exe", "kd", "kd.exe", "windbg.exe"):
        path = shutil.which(name)
        if path:
            return path
    return ""


def _extract_faulting_module(text: str) -> str:
    """Parse cdb/kd `!analyze -v` output for the faulting image. cdb emits lines like
    `MODULE_NAME: nt` and `IMAGE_NAME:  ntoskrnl.exe`; IMAGE_NAME names the actual driver
    file and is the most actionable attribution. Returns '' if nothing parsed."""
    image = module = ""
    for line in text.splitlines():
        s = line.strip()
        if not image and s.startswith("IMAGE_NAME:"):
            image = s.split(":", 1)[1].strip()
        elif not module and s.startswith("MODULE_NAME:"):
            module = s.split(":", 1)[1].strip()
    return image or module


def analyze_dump(path: str, timeout: float = 30.0) -> dict:
    """Read-only dump attribution — the closure of this audit's named blind spot. If a
    Microsoft debugger is on PATH and the dump is readable, run `cdb -z <dump> -c
    "!analyze -v; q"` and extract the faulting module. NEVER raises: on any failure
    (no debugger, unreadable dump, timeout, parse miss) it returns ``attributed: False``
    with an ``error`` so the caller can surface a blind spot rather than crash.
    """
    result = {"attributed": False, "faulting_module": "", "tool": "", "error": "", "raw_excerpt": ""}
    if not path:
        result["error"] = "no dump path"
        return result
    exe = _find_debugger()
    if not exe:
        result["error"] = "no Microsoft debugger (cdb/kd/windbg) on PATH"
        return result
    result["tool"] = exe
    if not os.path.isfile(path):
        result["error"] = f"dump not readable: {path}"
        return result
    try:
        proc = subprocess.run(
            [exe, "-z", path, "-c", "!analyze -v; kb; q"],
            capture_output=True, text=True, timeout=timeout,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        result["error"] = f"debugger timed out after {timeout}s"
        return result
    except OSError as exc:
        result["error"] = f"debugger failed: {exc}"
        return result
    result["raw_excerpt"] = text[:2000]
    module = _extract_faulting_module(text)
    if module:
        result["attributed"] = True
        result["faulting_module"] = module
    else:
        result["error"] = "debugger ran but no MODULE_NAME/IMAGE_NAME parsed"
    return result


def collect_attributions(crashes: list[dict], timeout: float = 30.0) -> dict[str, dict]:
    """Best-effort: attribute each in-window crash's dump (deduped by path). Returns a
    {dump_path: analyze_dump_result} map. Cheap when no debugger is installed (just PATH
    probes) and bounded by ``timeout`` per dump otherwise. Safe to call with crashes that
    have no dump path."""
    out: dict[str, dict] = {}
    for c in crashes:
        path = c.get("dump") or ""
        if path and path not in out:
            out[path] = analyze_dump(path, timeout=timeout)
    return out


def _parse(ts: str):
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _nearest_dump(dumps: list[dict], event_time: str, max_delta: timedelta = timedelta(minutes=10)) -> str:
    event_dt = _parse(event_time)
    if event_dt is None:
        return ""
    best: tuple[timedelta, str] | None = None
    for d in dumps:
        dump_dt = _parse(d.get("time", ""))
        dump_path = d.get("dump") or ""
        if dump_dt is None or not dump_path:
            continue
        delta = abs(dump_dt - event_dt)
        if delta > max_delta:
            continue
        if best is None or delta < best[0]:
            best = (delta, dump_path)
    return best[1] if best else ""


def _recent_shutdown_time(dirty_shutdowns: list[dict], event_time: str) -> str:
    """Return EventLog 6008's previous-shutdown time for the same reboot, when it is credible.

    Kernel-Power 41 is logged during the next boot, not at the instant the machine died. EventLog
    6008 carries the actual previous shutdown wall-clock, but Windows occasionally logs stale
    6008 details months later after sleep/boot oddities. Accept it only when the 6008 log record
    is adjacent to this Kernel-Power record and the shutdown time is recent enough.
    """
    event_dt = _parse(event_time)
    if event_dt is None:
        return ""
    best: tuple[timedelta, str] | None = None
    for row in dirty_shutdowns:
        logged_dt = _parse(row.get("logged", ""))
        shutdown_dt = _parse(row.get("shutdown_time", ""))
        if logged_dt is None or shutdown_dt is None:
            continue
        if abs(logged_dt - event_dt) > timedelta(minutes=10):
            continue
        if shutdown_dt > event_dt or event_dt - shutdown_dt > timedelta(hours=12):
            continue
        delta = abs(logged_dt - event_dt)
        if best is None or delta < best[0]:
            best = (delta, row["shutdown_time"])
    return best[1] if best else ""


def _is_dump_write_event(event: dict) -> bool:
    try:
        event_id = int(event.get("id") or 0)
    except (TypeError, ValueError):
        event_id = 0
    return str(event.get("provider") or "").lower() == "volmgr" and event_id == 162


def audit(probe: dict, window_days: int = 90, dump_attributions: dict | None = None,
          dispositions: list[dict] | None = None) -> dict:
    """Pure analysis: turn a probe dict into decoded crashes + a verdict. No I/O, so it unit-tests
    without Windows. `now` comes from the probe (not the wall clock) to keep it deterministic.

    ``dump_attributions`` (optional, {dump_path: analyze_dump result}) closes the audit's named
    blind spot: without reading the minidump a memory/driver verdict cannot exclude the other.
    When supplied, each crash carries its attribution and ``blind_spot`` reports whether the
    leading bugcheck is now dump-verifiable.

    ``dispositions`` (optional, list of recorded-disposition dicts; defaults to
    ``load_dispositions()``) is the durable answer to 'we already diagnosed this'. When the leading
    bugcheck matches a recorded disposition for this host, the result carries a ``disposition``
    block so callers report the KNOWN cause + remediation status instead of re-deriving an
    unexplained one. Pass ``[]`` to ignore the record file."""
    if not probe.get("available"):
        return {"available": False, "error": probe.get("error", "probe unavailable")}

    now = _parse(probe.get("now", "")) or datetime(1970, 1, 1)
    horizon = now - timedelta(days=window_days)
    dumps = probe.get("dumps", [])
    dirty_shutdowns = probe.get("dirty_shutdowns", [])
    disk_errors = [d for d in probe.get("disk_errors", []) if not _is_dump_write_event(d)]

    crashes = []
    for c in probe.get("crashes", []):
        logged_time = c.get("time", "")
        shutdown_time = c.get("shutdown_time") or _recent_shutdown_time(dirty_shutdowns, logged_time)
        display_time = shutdown_time or logged_time
        t = _parse(display_time)
        if t is None or t < horizon:
            continue
        dec = decode_bugcheck(int(c.get("bugcheck_code", 0)))
        rec = {
            "time": display_time,
            "logged_time": logged_time,
            "shutdown_time": shutdown_time,
            **dec,
            "params": [c.get("p1"), c.get("p2"), c.get("p3"), c.get("p4")],
            "dump": _nearest_dump(dumps, logged_time),
        }
        if dump_attributions is not None and rec["dump"]:
            rec["attribution"] = dump_attributions.get(rec["dump"])
        crashes.append(rec)

    ecc = int(probe.get("memory_error_correction", 0) or 0)
    is_ecc = ecc in (5, 6)
    dimms = probe.get("dimms", [])
    dimm_count = len(dimms)
    configured = [d.get("configured", 0) for d in dimms if d.get("configured")]
    min_configured = min(configured) if configured else 0
    # DDR5 JEDEC base is 4800 MT/s; running below it means the board down-clocked (the classic
    # 4-DIMM dual-rank fallback), which is correct behavior but flags a stressed memory topology.
    downclocked = bool(min_configured and min_configured < 4800)

    bugchecks = [c for c in crashes if c["code"] != 0]
    powerloss = [c for c in crashes if c["code"] == 0]
    classes = {c["fault_class"] for c in bugchecks}
    mem_hits = [c for c in bugchecks if c["fault_class"] in _MEMORY_CLASSES]

    verdict, confidence, actions = _verdict(
        bugchecks, powerloss, classes, mem_hits, is_ecc, dimm_count, downclocked,
        whea=probe.get("whea", []),
        rex=probe.get("resource_exhaustion", []),
        disk=disk_errors,
    )

    blind_spot = _blind_spot(bugchecks, dump_attributions)

    if dispositions is None:
        dispositions = load_dispositions()
    lead_bugcheck = bugchecks[0] if bugchecks else None
    disposition = match_disposition(lead_bugcheck, probe, dispositions)

    return {
        "available": True,
        "window_days": window_days,
        "now": probe.get("now"),
        "last_boot": probe.get("last_boot"),
        "crash_count": len(crashes),
        "bugcheck_count": len(bugchecks),
        "powerloss_count": len(powerloss),
        "latest_crash": crashes[0] if crashes else None,
        "crashes": crashes,
        "memory": {
            "total_gb": probe.get("ram_total_gb"),
            "dimm_count": dimm_count,
            "ecc": _ECC_MODE.get(ecc, f"code {ecc}"),
            "is_ecc": is_ecc,
            "min_configured_mts": min_configured,
            "downclocked_below_jedec": downclocked,
        },
        "platform": {
            "cpu": probe.get("cpu"),
            "board": probe.get("board"),
            "bios": f"{probe.get('bios_version')} ({probe.get('bios_date')})",
        },
        "dump_config": {
            "mode": _DUMP_MODE.get(probe.get("crash_dump_enabled"), str(probe.get("crash_dump_enabled"))),
            "minidump_accessible": probe.get("minidump_accessible"),
        },
        "signals": {
            "whea_events": len(probe.get("whea", [])),
            "resource_exhaustion_events": len(probe.get("resource_exhaustion", [])),
            "disk_errors": len(disk_errors),
        },
        "verdict": verdict,
        "confidence": confidence,
        "actions": actions,
        "blind_spot": blind_spot,
        "disposition": disposition,
    }


def _blind_spot(bugchecks: list[dict], dump_attributions: dict | None) -> dict:
    """The named blind spot of this audit (CRASH-AUDIT §4): a memory-/driver-/cpu-class verdict
    rests on dump evidence the audit can't always read, and a buggy driver produces bugchecks
    identical to bad RAM. This reports whether the LEADING bugcheck is dump-attributable, so the
    verdict is never mistaken for a certainty it isn't.

    Returns {present, reason, command}. ``present`` is False only when the leading bugcheck's
    dump was successfully attributed (a named faulting module) or there are no bugchecks.
    """
    if not bugchecks:
        return {"present": False, "reason": "no bugchecks in window", "command": ""}
    lead = bugchecks[0]
    dump = lead.get("dump") or ""
    klass = lead.get("fault_class", "unknown")
    if not dump:
        return {
            "present": True,
            "reason": ("no minidump captured for the leading bugcheck; a driver fault cannot be "
                       "excluded, and the next crash stays unverifiable until dumps are enabled"),
            "command": "Set CrashControl=2 (kernel dump) so the next bugcheck leaves attributable evidence.",
        }
    if dump_attributions is None:
        return {
            "present": True,
            "reason": (f"dump exists ({dump}) but stack attribution was not attempted; a {klass}-class "
                       "verdict cannot exclude a driver fault without reading the minidump"),
            "command": f'cdb -z "{dump}" -c "!analyze -v; q"   (or WinDbg !analyze -v, elevated)',
        }
    attr = dump_attributions.get(dump)
    if not attr or not attr.get("attributed"):
        err = (attr or {}).get("error") or "attribution unavailable"
        return {
            "present": True,
            "reason": f"dump exists ({dump}) but could not be attributed: {err}",
            "command": f'cdb -z "{dump}" -c "!analyze -v; q"   (elevated; needs Debugging Tools for Windows)',
        }
    module = attr.get("faulting_module", "?")
    return {
        "present": False,
        "reason": f"dump attributed to faulting image {module} — driver-vs-RAM question is settled by the stack",
        "command": "",
    }


def _verdict(bugchecks, powerloss, classes, mem_hits, is_ecc, dimm_count, downclocked,
             whea, rex, disk) -> tuple[str, str, list[str]]:
    """Fold the crash cluster + config into a single verdict string, a confidence, and an ordered
    action list. Ordered by severity: CPU machine-check > memory-subsystem > driver > power-loss."""
    actions: list[str] = []

    if not bugchecks and not powerloss:
        return ("No crashes in the window - system is stable.", "n/a", [])

    # OOM / storage side-signals get appended regardless of the primary verdict.
    side = []
    if rex:
        side.append(f"{len(rex)} memory/commit-exhaustion event(s) - investigate OOM pressure.")
    if disk:
        side.append(f"{len(disk)} disk/filesystem error(s) - check drive health (SMART).")

    # 1. Hardware machine-check is the most specific signal.
    if any(c["fault_class"] == "cpu_hw" for c in bugchecks) or any(
        w.get("level") in ("Error", "Critical") for w in whea
    ):
        actions = [
            "Run WinDbg `!analyze -v` on the dump - a 0x124/0x9C dump names the failing component.",
            "Test CPU/PSU/VRM: stress-test, check power delivery, reseat CPU.",
            "Update BIOS/AGESA and clear CMOS to defaults (remove any overclock/PBO offset).",
        ]
        return ("HARDWARE machine-check (CPU/PSU/board or severe memory error).",
                "high", actions + side)

    # 2. Memory-subsystem heuristic: memory-class bugcheck on a non-ECC, high-DIMM-count, or
    #    down-clocked config is the strongest pattern for a marginal RAM/IMC fault.
    if mem_hits:
        stressed = (dimm_count >= 4) or downclocked
        if not is_ecc and stressed:
            conf = "high"
            why = ("memory-corruption bugcheck on a non-ECC, high-capacity config "
                   f"({dimm_count} DIMMs"
                   + (", down-clocked below JEDEC base" if downclocked else "") + ")")
        else:
            conf = "medium"
            why = "memory-corruption-class bugcheck"
        actions = [
            "Run MemTest86 (bootable, several passes) OR Windows Memory Diagnostic during a "
            "maintenance window - do NOT reboot while the fleet is live.",
            "Update BIOS to the latest AGESA (newer AGESA materially improves 4-DIMM / "
            "high-capacity DDR5 stability on AM5).",
            "If it persists: raise SoC / VDDG / VDDP voltage a notch within spec, or drop to one "
            "DIMM pair to isolate a weak module; reseat, then RMA the failing stick.",
            "Switch dump mode to kernel (CrashControl=2) so the next dump attributes the fault.",
        ]
        return (f"MEMORY SUBSYSTEM suspect - {why}.", conf, actions + side)

    # 3. Driver fault.
    if classes & {"driver", "storage"}:
        actions = [
            "WinDbg `!analyze -v <dump>` - the stack names the faulting driver.",
            "Update or roll back the implicated driver (GPU/storage/network are usual suspects).",
            "Optionally enable Driver Verifier on suspect drivers to force-attribute the next fault.",
        ]
        return ("DRIVER fault - a kernel-mode driver is the likely culprit.", "medium", actions + side)

    # 4. Dumpless power-loss / hard hang.
    if powerloss and not bugchecks:
        actions = [
            "No bugcheck/dump - suspect power delivery (PSU, wall power, loose ATX/EPS), a hard "
            "lockup, or thermal cutoff.",
            "Check PSU + cabling; review temperatures under load; ensure dumps are enabled so the "
            "next event leaves evidence.",
        ]
        return ("POWER-LOSS / hard hang (no bugcheck recorded).", "medium", actions + side)

    # 5. Software.
    if classes & {"software"}:
        return ("SOFTWARE - a critical process terminated; usually a user-mode software fault.",
                "low", ["Inspect the dump and recent app/update history."] + side)

    return ("Inconclusive - analyze the dump directly with WinDbg `!analyze -v`.", "low",
            ["Open the latest dump in WinDbg and run `!analyze -v`."] + side)


def _report(a: dict) -> str:
    """Render the audit dict as a human-readable report."""
    if not a.get("available"):
        return f"crash_audit: unavailable — {a.get('error')}"
    L = []
    L.append("=" * 78)
    L.append(f"  CRASH AUDIT - last {a['window_days']} days   (as of {a['now']})")
    L.append("=" * 78)
    p = a["platform"]
    L.append(f"  {p['cpu']}")
    L.append(f"  {p['board']}   BIOS {p['bios']}")
    m = a["memory"]
    L.append(f"  RAM: {m['total_gb']} GB across {m['dimm_count']} DIMM(s) @ "
             f"{m['min_configured_mts']} MT/s  |  ECC: {m['ecc']}"
             + ("  [down-clocked below JEDEC base]" if m["downclocked_below_jedec"] else ""))
    d = a["dump_config"]
    L.append(f"  Dump mode: {d['mode']}  |  minidumps readable: {d['minidump_accessible']}")
    s = a["signals"]
    L.append(f"  Signals: WHEA={s['whea_events']}  OOM={s['resource_exhaustion_events']}  "
             f"disk-errors={s['disk_errors']}")
    L.append("-" * 78)
    L.append(f"  {a['crash_count']} crash(es): {a['bugcheck_count']} bugcheck(s), "
             f"{a['powerloss_count']} power-loss/hard-hang")
    for c in a["crashes"][:12]:
        tag = c["code_hex"] if c["code"] else "power-loss"
        L.append(f"    {c['time']}  {tag:<12} {c['name']}")
        if c.get("logged_time") and c.get("logged_time") != c.get("time"):
            L.append(f"        logged on reboot: {c['logged_time']}")
        L.append(f"        -> {c['hint']}")
        if c.get("dump"):
            L.append(f"        dump: {c['dump']}")
        attr = c.get("attribution")
        if attr and attr.get("attributed"):
            L.append(f"        attributed: {attr.get('faulting_module','?')}  (via {attr.get('tool','?')})")
    L.append("-" * 78)
    blind = a.get("blind_spot") or {}
    if blind.get("present"):
        L.append(f"  BLIND SPOT: {blind.get('reason','')}")
        if blind.get("command"):
            L.append(f"    -> {blind['command']}")
        L.append("-" * 78)
    L.append(f"  VERDICT ({a['confidence']} confidence): {a['verdict']}")
    if a["actions"]:
        L.append("  ACTIONS:")
        for i, act in enumerate(a["actions"], 1):
            L.append(f"    {i}. {act}")
    disp = a.get("disposition")
    if disp:
        L.append("-" * 78)
        recurring = " (accepted recurring)" if disp.get("recurring") else ""
        fak_bug = "NO — host fault, not a fak/fleet code bug" if disp.get("is_fak_bug") is False \
            else ("yes — code bug" if disp.get("is_fak_bug") else "unknown")
        L.append(f"  RECORDED DISPOSITION [{disp.get('id','?')}]{recurring}")
        L.append(f"    cause: {disp.get('cause','?')}   fak/fleet code bug: {fak_bug}")
        if disp.get("summary"):
            L.append(f"    {disp['summary']}")
        if disp.get("remediation_status"):
            L.append(f"    remediation status: {disp['remediation_status']}")
        if disp.get("evidence_doc"):
            L.append(f"    evidence: {disp['evidence_doc']}")
        if disp.get("tracking_issues"):
            L.append("    tracking: #" + ", #".join(str(i) for i in disp["tracking_issues"]))
        L.append("    -> this fault has a recorded disposition; it is NOT an unexplained cause.")
    L.append("=" * 78)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Windows crash forensics + root-cause verdict.")
    ap.add_argument("--days", type=int, default=90, help="only consider crashes within N days (default 90)")
    ap.add_argument("--report", action="store_true", help="human-readable report instead of JSON")
    ap.add_argument("--raw", action="store_true", help="dump the raw PowerShell probe JSON and exit")
    ap.add_argument("--no-analyze", action="store_true",
                    help="skip automatic cdb/WinDbg dump attribution (the blind-spot closure)")
    args = ap.parse_args()

    probe = collect()
    if args.raw:
        print(json.dumps(probe, indent=2))
        return 0 if probe.get("available") else 1

    result = audit(probe, window_days=args.days)
    # Best-effort dump attribution: close the audit's blind spot by running the Microsoft
    # debugger on each in-window crash's dump when one is installed. Cheap (PATH probes) and
    # bounded per dump; --no-analyze skips it for a fast JSON snapshot.
    if not args.no_analyze and result.get("available") and result.get("crashes"):
        attributions = collect_attributions(result["crashes"])
        if attributions:
            result = audit(probe, window_days=args.days, dump_attributions=attributions)
    if args.report:
        print(_report(result))
    else:
        print(json.dumps(result, indent=2))
    return 0 if result.get("available") else 1


if __name__ == "__main__":
    sys.exit(main())
