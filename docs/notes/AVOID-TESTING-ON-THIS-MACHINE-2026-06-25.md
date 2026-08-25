---
title: "Avoid running tests / local serves directly on this machine (for now)"
description: "Avoid the native Windows test suite or a local fak serve on this shared box; Defender has repeatedly quarantined transient Go test executables and stray serves add memory pressure."
---

# Avoid running tests / local serves directly on this machine (for now)

**Date:** 2026-06-25
**Updated:** 2026-08-25 (issue #8919)
**Scope:** the Windows dev box this repo is checked out on (`C:\work\fak`), not CI, WSL, or the GPU nodes.

> **Superseded diagnosis:** The August 25, 2026 audit for issue #8919 supersedes
> this note's earlier attribution to Windows Application Control. Current Windows
> event evidence identifies Microsoft Defender Antivirus real-time ML detections
> and quarantines of transient Go test executables instead.

## The rule

Do **not** run the test suite or stand up a local `fak serve` directly on this Windows
machine right now. Two reasons, one hard and one operational:

1. **Native `go test` is repeatedly disrupted here by Defender Antivirus.** The
   Defender Operational event log records **78 detections covering 57 unique binaries**
   from **July 26 through August 8, 2026**. Recorded threat-family labels include
   **Wacatac, Wacapew, Sabsik, and Bearfoos**. Go creates fresh transient test
   executables, so a new binary identity is scanned and can trigger the same detection
   and quarantine behavior again; retrying with a fresh build does not remove the cause.
   These labels and actions establish what Defender reported and did. They do **not**,
   by themselves, establish that the detections are false positives or that any binary
   is safe.

2. **This box is a shared, busy multi-session tree.** It also hosts the live dispatch
   fleet and unrelated workloads. Local `fak serve` / Vulkan-GGUF test processes left
   running here pile up; a single stray `fak-vulkan-pressuretrim-latest serve` was
   holding **4.3 GB** with zero established connections when this note was written.
   They are leftover memory pressure.

## What to do instead

- **Tests:** preserve the supported WSL path: run `./test.ps1` from the repo root
  under WSL, or let CI run them. For the fast gate, use the repository's isolated
  build/validation verbs where applicable; do not treat a quarantined native Windows
  test executable as a test result.
- **Do not weaken endpoint protection:** do not disable Microsoft Defender Antivirus,
  turn off real-time protection, or add broad process, folder, repository, Go-cache,
  or temporary-directory exclusions. Escalate any narrowly scoped remediation through
  the machine's security owner with the event evidence; this note does not declare the
  binaries safe.
- **Local serve for a real model:** use the GPU nodes / cloud serve (the in-kernel L4
  `fak-realmodel` path), not a long-lived `fak serve` on this Windows box.
- **If you must spin up a local serve to reproduce something:** kill it when done. Verify
  with `Get-NetTCPConnection -State Listen -LocalPort <port>` and stop the owning PID.

## Read-only Defender event diagnostic

This PowerShell command only reads the Defender Operational log. It does not change
Defender settings, restore quarantined files, or create exclusions:

```powershell
$start = [datetime]'2026-07-26T00:00:00'
$end = [datetime]'2026-08-09T00:00:00'
Get-WinEvent -FilterHashtable @{
  LogName = 'Microsoft-Windows-Windows Defender/Operational'
  Id = 1116, 1117
  StartTime = $start
  EndTime = $end
} | Select-Object TimeCreated, Id, ProviderName, Message
```

Event IDs 1116 and 1117 expose Defender detection and action records. Preserve the
raw event output for diagnosis, but do not infer false-positive status or binary safety
from family labels, filenames, counts, or quarantine actions alone.

## Cleanup recipe (what was done on 2026-06-25)

Find stray local fak serves and idle big processes:

```powershell
# Top memory holders
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 20 Name, Id, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,0)}}

# Local serves that nobody is connected to (safe to kill)
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'fak.*serve' } |
  Select-Object ProcessId, CommandLine
Get-NetTCPConnection -State Established | Where-Object { $_.OwningProcess -eq <PID> }   # empty => idle => kill
```

Killed this pass: `fak-vulkan-pressuretrim-latest serve` (idle local Vulkan serve, ~4.3 GB),
`fak.exe serve --session-id human-try` (idle dogfood serve, port 8080), and a hung 11-hour
`gcloud config list` (~1 GB). Left the live dispatch fleet and the `C:\work\job` batch alone.
