# Avoid running tests / local serves directly on this machine (for now)

**Date:** 2026-06-25
**Scope:** the Windows dev box this repo is checked out on (`C:\work\fak`), not CI, not the GPU nodes.

## The rule

Do **not** run the test suite or stand up a local `fak serve` directly on this Windows
machine right now. Two reasons, one hard and one operational:

1. **Native `go test` is already blocked here.** An OS Application-Control policy refuses
   the freshly-compiled test binaries (see [`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md) "Windows" note).
   The supported path is WSL: `./test.ps1` from the repo root. Trying to run `go test`
   natively just produces a confusing OS-level failure, not a real test result.

2. **This box is a shared, busy multi-session tree.** It also hosts the live dispatch
   fleet (opencode GLM-5.2 workers, the dos-mcp swarm, the 90s dispatch fill-loop) and an
   unrelated `C:\work\job` job-search batch. Local `fak serve` / Vulkan-GGUF test processes
   left running here pile up — a single stray `fak-vulkan-pressuretrim-latest serve` was
   holding **4.3 GB** with zero established connections when this note was written. They are
   pure leftover memory pressure.

## What to do instead

- **Tests:** run under WSL (`./test.ps1`) or let CI run them. For the fast gate, `make test-fast`
  builds + vets but the OS-control caveat still applies to the test binaries — prefer WSL.
- **Local serve for a real model:** use the GPU nodes / cloud serve (the in-kernel L4
  `fak-realmodel` path), not a long-lived `fak serve` on this Windows box.
- **If you must spin up a local serve to reproduce something:** kill it when done. Verify
  with `Get-NetTCPConnection -State Listen -LocalPort <port>` and stop the owning PID.

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
