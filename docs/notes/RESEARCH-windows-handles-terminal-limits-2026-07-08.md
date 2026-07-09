---
title: "Windows handles, consoles, and spawn storms: which ceilings are real (2026-07-08)"
description: "Deep research into the Windows resource limits blamed for fleet instability on the agent host. Finding: the handle ceiling is not a ceiling we hit, the pwsh 0xE9 crash is a fragility bug not a limit, and the real stall is CreateProcess cost. Four levers exist; three of the knobs people reach for are red herrings."
---

# Windows handles, consoles, and spawn storms: which ceilings are real

Date: 2026-07-08. Question asked: *"why is this Windows handles/terminal thing such an
issue — I find it hard to believe it's not adjustable or tunable."*

**Answer in one line:** it is not tunable because, for the most part, **it is not a limit**.
Three unrelated failure modes were conflated under one name. Exactly one of them has a
registry knob, and that knob is irrelevant to us. The other two have real fixes — they are
architectural, not configuration.

## The conflation

`AGENTS.md` and the issue tree carry three distinct Windows symptoms that get discussed as
one "handles/terminal" problem:

| # | Symptom | Witnessed | Actually is |
|---|---|---|---|
| A | Two orphaned `find` procs held **98.8% of system handles** | 2026-06-21, AGENTS.md | a **leak**, not a ceiling |
| B | `pwsh.exe` FailFast, Win32 `0xE9` *"No process is on the other end of the pipe"* | #2170 | a **fragility bug** in the ConPTY chain, not a ceiling |
| C | Whole-machine stall at low CPU/RAM/disk | #3153, `fak stallscan` | **`CreateProcess` cost** + soft-fault lock contention, not a ceiling |

None of the three is "we ran out of handles."

## Witnessed on this host (2026-07-08)

Live probe, not theory:

```
System-wide handles: 243,735          # per-process cap is 16,711,680
Top handle holders                     Handles   Threads
  WindowsTerminal  pid 17836            17,582       902
  chrome           pid 48772            11,162       566
  System           pid 4                 8,948       639
  svchost          pid 8492              7,261       325
```

Three facts fall straight out:

1. **The whole box uses 243,735 handles — 1.5% of what a *single* process is allowed.**
   Handle count is not now, and has never been, our ceiling.
2. **`WindowsTerminal` is the #1 handle holder *and* the #1 thread holder on the machine**,
   by a wide margin — 17,582 handles and 902 threads, versus ~7k/300 for the next real
   contenders. It is comfortably past Russinovich's *"more than ten thousand handles… likely
   either poorly designed or has a handle leak"* line. This is the literal "Windows
   handles/terminal thing": **the terminal emulator is the leak, not the fleet.** It is also
   the same process that held 2,767 threads and froze dispatch on 2026-07-01. Nothing in
   Windows is refusing us; one GUI app is accreting.
3. `netsh int ipv4 show excludedportrange protocol=tcp` shows **live `hns` reservations
   inside the dynamic range** — 54606–55493 in four consecutive 100-port blocks, plus
   50000–50059 administered. Any fixed-port bind landing there fails `WSAEACCES` while
   looking, to a caller, exactly like exhaustion. The dynamic range itself is stock
   (49152, 16384 ports).

## A. Kernel handles — a hard limit, never reached, not tunable, and not our problem

- Per-process ceiling is **16,777,216 (2^24)**, usable **16,711,680** on 64-bit. Russinovich:
  *"In one of the rare cases where Windows sets a hard-coded upper limit on a resource…"*
  It is a compile-time Executive constant. **There is no registry knob and no API.** Anyone
  who tells you otherwise is confusing it with the tunable paged-pool size, or with the
  C-runtime `_setmaxstdio` `FILE*` limit — neither touches the kernel handle cap.
- **System-wide there is no separate count limit at all**: *"The total number of open handles
  in the system is limited only by the amount of memory available."* The binding resource is
  **paged pool** (handle tables) and **nonpaged pool** (the objects themselves).
- Therefore the real failure is **pool exhaustion**, which surfaces as
  **`ERROR_NO_SYSTEM_RESOURCES` (1450)** in *unrelated* processes — that is why a runaway
  `find` takes the whole box down rather than just killing itself.
- Russinovich's own triage rule: *"Any process that has more than ten thousand handles open
  at any given point is likely either poorly designed or has a handle leak."*

**Correction to a folklore item:** `PspCidTable` bounds process/thread **IDs**, not general
object handles. It is not what we exhaust.

> **Verdict:** not tunable, and correctly so — the 16M cap exists precisely as a *leak
> backstop*. Our exposure is a leak (`find`), and the mitigation is detection + reaping,
> which `tools/runaway_process_scan.ps1` and `runaway_process_reaper.ps1` already do.

## B. GDI objects, USER objects, desktop heap — the knobs everyone reaches for, all red herrings here

These are the famous "10,000 handle" ceilings, and they *are* genuinely tunable:

| Resource | Default | Registry tunable | Valid range |
|---|---|---|---|
| GDI objects/proc | 10,000 | `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows\GDIProcessHandleQuota` | 256 – 65,536 |
| USER objects/proc | 10,000 | `…\USERProcessHandleQuota` | 200 – **18,000** |
| Desktop heap | `SharedSection=1024,20480,768` | `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\SubSystems\Windows` | interactive capped at 20480 |

`SharedSection` triple = *shared heap, KB* / *per interactive desktop, KB* / *per
non-interactive (Session-0) desktop, KB*.

**But they do not apply to us.** The controlling rule, from the win32k engineers'
NTDebugging series:

> *"If an application does not depend on user32.dll, it does not consume desktop heap."*

Desktop heap and GDI/USER handles are charged **only on object creation** — windows, menus,
hooks, bitmaps, DCs. A headless `pwsh.exe` / `node.exe` / `fak.exe` that never draws charges
essentially nothing, regardless of the 10,000 quota. The historical disasters were GUI
Terminal-Server farms and *many-distinct-account Session-0 services* squeezed into the tiny
768 KB non-interactive heap. On 64-bit Windows 11 session view space became a *dynamic*
kernel range, removing the dominant 32-bit cause outright.

The only real charge for a console fleet is that **each console gets a `conhost.exe`**, and
conhost does create window objects. It is small — you would need thousands of simultaneous
consoles on one desktop before it registered.

> **Verdict: do not tune `GDIProcessHandleQuota` / `USERProcessHandleQuota` /
> `SharedSection`.** They are tunable, they look relevant, and they will change nothing.
> Cheap canary if paranoid: **Event Log → System → Win32k Event 243** ("desktop heap
> allocation failed"), logged once per session.

## C. The console / ConPTY chain — fragility, not a ceiling. This is symptom B.

**There is no documented limit on the number of consoles or pseudoconsoles**, per-session or
system-wide, and no registry knob for one. Scaling is bounded only by resources. So the
`0xE9` crash is not "too many terminals."

What it actually is:

1. A process may have **at most one console** (`AllocConsole` fails if one exists). Windows
   consoles are not byte streams — clients talk to a **console host** (`conhost.exe`, or
   `OpenConsole.exe` in Terminal's bundled path) via IOCTLs brokered by the **`ConDrv.sys`**
   kernel driver.
2. `CreatePseudoConsole()` **spins up its own host process** per ConPTY, wired to the caller
   through an input pipe and an output pipe. There is no threads-only pseudoconsole: it is
   **process-per-PTY**.
3. A child's liveness is therefore coupled to a *chain*: `child ↔ conhost/OpenConsole ↔ terminal`.
4. When any link dies out of order, the next Console API call from the child fails with
   **`ERROR_NO_DATA` (233 / 0xE9)** — *"No process is on the other end of the pipe."*
5. **PowerShell treats console-buffer queries as infallible.**
   `ConsoleHostRawUserInterface.get_CursorPosition()` → `ConsoleControl.GetConsoleScreenBufferInfo()`
   returns 0xE9 → PowerShell throws `HostException` → unrecoverable inside
   `ConsoleHost.InputLoop.Run` → it calls **`Environment.FailFast()`**. That is an
   **uncatchable process abort** ("Unknown Hard Error"). This is exactly the stack in #2170.

This is a known, documented class with upstream history:

- `microsoft/terminal#16212` — pwsh FailFast via `GetConsoleScreenBufferInfo` when the
  ConPTY link between `OpenConsole.exe` and `pwsh.exe` destabilises.
- `PowerShell/PowerShell#12640` — the same 0xE9 "while getting console output buffer information."
- `warpdotdev/warp#11398` and `wezterm#7774` — **root cause was a bundled ConPTY/OpenConsole
  pair older than the pwsh/.NET it hosted**; fixed by bumping ConPTY.
- `microsoft/terminal` PR #10415 — a `.reset()`/`.release()` mixup **leaked handles** so
  "threads weren't aware of the other side collapsing," leaving **orphaned** conhost/OpenConsole.
- `terminal#1810` (`ClosePseudoConsole` hang), PR #14160 (shutdown deadlock on bulk output).

So ConPTY **has** a documented history of leaking handles/processes and deadlocking on
teardown, and there **is** a real race where the host dies and clients take 0xE9.

> **Verdict:** no tunable exists. Two real fixes:
> 1. **Do not inherit a fragile console.** Spawn with `CREATE_NO_WINDOW` (hidden console) or
>    `DETACHED_PROCESS` (no console at all) and redirect stdio to **pipes we own**, so child
>    I/O lifetime binds to our handles, not to a shared conhost that can die underneath it.
>    Never call console screen-buffer APIs on those children — those are what raise 0xE9.
>    (Note: `CREATE_NO_WINDOW` is **ignored** if combined with `CREATE_NEW_CONSOLE` or
>    `DETACHED_PROCESS`.) `internal/windowgate` already does `CREATE_NO_WINDOW`; the gap is
>    treating 0xE9 as a **structured, survivable event** rather than letting a host FailFast.
> 2. **Keep the bundled ConPTY current** where we drive interactive PTYs — a stale
>    ConPTY vs newer pwsh/.NET is a *known* cause of this exact crash.

## D. Pipes, ports, threads — bounded by memory, mostly non-issues

- **Pipes:** no count limit. `nMaxInstances` accepts up to `PIPE_UNLIMITED_INSTANCES` (255)
  per *named* pipe; overall bound is **nonpaged pool**. There is **no `ulimit -n`** on Windows.
  The EMFILE analog is `ERROR_TOO_MANY_OPEN_FILES` (4), but the real bound is the 16M table.
- **libuv/Node:** `uv_pipe_t` on Windows is a **Win32 named pipe** under `\\.\pipe\…` driven
  by IOCP. The child's end is created **non-overlapped** by default — the root of several
  Windows-only defects (`libuv#95` full-duplex spawn deadlock, `joyent/libuv#1023` IOCP +
  duplicated handles). Node's `fork` is **not** OS `fork` — it is a full `CreateProcess`.
- **Ephemeral TCP ports:** default range is **49152–65535 (16,384 ports)**. Inspect with
  `netsh int ipv4 show dynamicport tcp`; change with
  `netsh int ipv4 set dynamicport tcp start=<n> num=<r>`.
  **`MaxUserPort` is STALE** — it governed the pre-Vista 1025–5000 model and is effectively
  ignored on Windows 10/11. Any advice to set it is outdated.
  **A `listen()` on a fixed loopback port consumes no ephemeral port** — exhaustion is an
  *outbound* phenomenon. A few dozen MCP servers cannot exhaust it. Rough math: at default
  16,384 ports and ~120 s TIME_WAIT you need ~136 new outbound conns/sec sustained.
  - **The real trap:** with **WSL2 / Hyper-V / Docker** present, `hns` reserves large blocks
    of 49152–65535 that *shift on reboot*, so a fixed-port bind fails with `WSAEACCES`
    ("forbidden") though nothing is listening. Audit with
    `netsh int ipv4 show excludedportrange protocol=tcp`. Fix: bind below 49152, pre-reserve
    with `netsh int ipv4 add excludedportrange`, or bind port 0 and advertise the assigned port.
- **Threads:** no fixed cap. Each thread locks a **24 KB kernel stack in nonpaged RAM**
  (64-bit); the bound is resident-available memory, then the **commit limit**. Our
  `FAK_HOST_THREADS_PER_CORE=1000` (32,000-thread budget) is already a *raised* setting.
  The 2026-07-01 freeze — **`WindowsTerminal` pid 85884 with 2,767 threads** — was not an OS
  ceiling; `proc_resource_guard`'s `2,000/proc` is **our policy number**, and the 2,767 was
  WindowsTerminal accreting threads. Nothing in Windows was refusing us.

## E. The actual stall: `CreateProcess` cost + soft-fault lock contention (symptom C)

This is where the real money is.

- Windows has no `fork`. `CreateProcess` builds a process from scratch every time: new
  address space, map the image section, **resolve and map the whole DLL import graph**, run
  each `DllMain(PROCESS_ATTACH)` **serialized under the loader lock**, apply relocations/ASLR,
  build PEB/TEB, stacks, handle table, CSRSS registration.
- Measured: **Linux process creation is >20× faster than Windows** on identical hardware
  (bitsnbites, i7-6820HQ / Ryzen 1800X) — *"even a Raspberry Pi 3 is faster than a stock
  Windows 10 Pro install on an octa-core Ryzen 1800X."* The same source warns results are
  *"very sensitive to background services such as Windows Defender."*
- Each spawn faults in its EXE + DLL pages (**transition** faults — resident, shared) and
  zero-fills stacks/heap/PEB/TEB (**demand-zero** faults). Both are **soft**: no disk I/O.
  Thousands of spawns ⇒ millions of soft faults, each serializing on the **per-process
  working-set lock** and the system-wide **PFN database lock**.
- **This is why every meter reads low.** CPU% is low because threads are *blocked on MM
  locks*, not burning user cycles. RAM is fine because soft faults re-map already-resident
  pages. Disk queue is flat because soft faults do **zero disk I/O**. This is precisely the
  fingerprint `internal/stallscan` was built to catch, and its axes are the right ones.

**The Defender finding — and a live bug in our own script.** Defender hooks process creation
and file I/O via a minifilter, scanning the EXE and every DLL on each spawn, directly on the
critical path. Microsoft Learn states plainly:

> *"When you add a process to the process exclusion list, Microsoft Defender Antivirus
> doesn't scan files opened by that process… **The process itself, however, is scanned
> unless it's added to the file exclusion list.**"*

`tools/host_stall_mitigations.ps1` adds `-ExclusionProcess fak.exe, claude.exe, python.exe,
node.exe` — which excludes *files those processes open*, but **does not stop Defender
scanning the `fak.exe` / `node.exe` images themselves on every single `CreateProcess`.** For a
spawn-storm workload that is the exact cost we were trying to remove. It needs a matching
**path/file exclusion of the executables**, not just `-ExclusionProcess`. (Accept the security
trade-off knowingly: a process exclusion also disables Network Protection and ASR inspection
for that process, and an image-name exclusion like `node.exe` matches that name from *any*
path — prefer full paths.)

## F. Job Objects — the control surface we are not using

This is the strongest available lever and it is a Windows-native API, not a tunable.

A **job object** groups processes; limits apply to all members and are **inherited by children**
spawned via `CreateProcess`. Jobs nest (Windows 8+).

- **Admission control against spawn storms:** `JOBOBJECT_BASIC_LIMIT_INFORMATION` with
  `JOB_OBJECT_LIMIT_ACTIVE_PROCESS` + `ActiveProcessLimit = N`. Beyond N active members,
  further `CreateProcess` calls **fail** until members exit. A hard, kernel-enforced cap on
  parallelism — set N ≈ logical CPUs.
- **Orphan elimination, by construction:** `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` terminates
  every member when the last job handle closes. This is the canonical fix for
  *"parent crashed, children linger."* It directly addresses our **Session-0 S4U zombie
  `opencode.exe`** problem (68 procs ≈ 2,067 threads, un-killable from a non-elevated shell)
  — those orphans could not exist under a job.
- **Priority containment:** `JOB_OBJECT_LIMIT_PRIORITY_CLASS` caps what members may request;
  a member's `SetPriorityClass` above the job limit **"succeeds but is silently ignored."**
  Pin fleet workers to Below-Normal so a storm cannot starve the operator's terminal.
- **CPU rate control** (`JOBOBJECT_CPU_RATE_CONTROL_INFORMATION`, Win8+): hard cap
  (`CpuRate` = percent × 100), weight-based share, or min/max rate.
- Escape hatches exist (`JOB_OBJECT_LIMIT_BREAKAWAY_OK` + `CREATE_BREAKAWAY_FROM_JOB`).

`internal/procguard` does descendant reaping *after the fact*. A job object makes reaping
**structural** and adds admission control we currently implement in Go policy
(`host_cap`, `--max-workers`) that the kernel would enforce for free.

## Ranked action list

**Tune (cheap, do now)**
1. **Fix the Defender exclusion gap.** Add path/file exclusions for the fleet executables
   themselves (`fak.exe`, `node.exe`, `claude.exe`, `python.exe`), not only
   `-ExclusionProcess`. Today's script does not stop image scanning on spawn. *(highest
   value-per-minute item in this note)*
2. **Audit `netsh int ipv4 show excludedportrange protocol=tcp`** on this host — WSL2/Docker
   `hns` reservations are the most common cause of a "port in use / access denied" bind
   failure that looks like exhaustion but is not.
3. **Add a handle axis to `fak stallscan`.** `runaway_process_scan.ps1` already reads
   system-wide handle totals and warns past 1M; the live classifier in
   `internal/stallscan` does **not**. A monotonically climbing per-process handle count is
   the only early warning for the paged-pool-exhaustion class (symptom A). Threshold on
   *growth*, not absolute count. The 1M system-wide warning line is also mis-scaled for
   *per-process* leak detection — at 243k system-wide today, a single process passing ~10k
   and still climbing is the signal worth alerting on.
4. **Stop hosting the fleet under `WindowsTerminal`.** It is the largest handle *and* thread
   consumer on the box (17,582 / 902), it is what tripped `proc_resource_guard`'s
   2,000-thread policy and froze dispatch on 2026-07-01, and it is un-reapable by design
   (PROTECTED). Restarting it is the operator workaround; the durable fix is to run
   long-lived fleet processes detached from the interactive terminal (see item 5/6) so an
   operator GUI app is never in the fleet's dependency path. Its thread/handle growth is an
   app-level accretion — **no OS knob addresses it.**

**Redesign (real fixes)**
5. **Put every fleet-spawned process in a Job Object** with `ActiveProcessLimit`,
   `KILL_ON_JOB_CLOSE`, and a Below-Normal `PriorityClass`. Makes orphans structurally
   impossible and moves admission control into the kernel. Supersedes a chunk of
   `procguard` + preflight policy.
6. **Sever the console chain.** Ensure fleet children are spawned `CREATE_NO_WINDOW` /
   `DETACHED_PROCESS` with stdio on pipes *we own*, and treat `0xE9`/`ERROR_NO_DATA` as a
   structured, survivable child-failure event. This is exactly acceptance witness #1 of
   #2170, and it is now clear the mechanism must **tolerate** 0xE9 rather than assume a
   live console.
7. **Spawn less.** `CreateProcess` is >20× Linux's cost and cannot be tuned away. Long-lived
   worker processes over per-command spawns; never `pwsh -c` per item (a full extra process +
   profile + DLL load + AV scan each). `CREATE_NO_WINDOW`/`DETACHED_PROCESS` are
   *correctness* flags, **not** performance flags — they do not reduce the dominant
   address-space/DLL/AV cost.

**Do not do**
8. Do **not** raise `GDIProcessHandleQuota`, `USERProcessHandleQuota`, or `SharedSection`.
   Headless processes don't consume these. Tuning them changes nothing and weakens a
   per-process leak circuit-breaker.
9. Do **not** set `MaxUserPort`. It is deprecated on Windows 10/11.
10. Do **not** look for a knob to raise the per-process handle cap. It does not exist; the
    16M cap is a deliberate leak backstop and we are ~3 orders of magnitude below it
    system-wide (243,735 total vs 16,711,680 for one process).

## Honest fences

- Nothing in this note is a shipped code change; it is research + a ranked plan. Items 1–3
  are small and independently shippable; 4–6 are epics.
- The `>20×` spawn-cost ratio is from bitsnbites' graphical results; a canonical "X ms per
  `CreateProcess`" figure is workload- and AV-dependent. **The checkable next step is to
  measure it on this box** — time N spawns with and without the item-1 image exclusion. That
  measurement would also witness item 1.
- The claim that item 1 is a real gap rests on Microsoft Learn's `-ExclusionProcess`
  semantics, not on a local A/B. Measuring it (above) is what would convert it from
  *documented-should* to *witnessed*.
- Whether `procguard` + `windowgate` already tolerate a 0xE9 child death, or whether the
  parent dies with it, is the **invalidating assumption** flagged in
  `WINDOWS-SHELL-TUI-FAULT-BOUNDARY-TRIAGE-2026-07-04.md` and is still unresolved. This note
  does not resolve it.

## Sources

- Russinovich, *Pushing the Limits of Windows: Handles* — https://learn.microsoft.com/en-us/archive/blogs/markrussinovich/pushing-the-limits-of-windows-handles
- Russinovich, *Pushing the Limits of Windows: Processes and Threads* — https://learn.microsoft.com/en-us/archive/blogs/markrussinovich/pushing-the-limits-of-windows-processes-and-threads
- Microsoft Learn, *Handle Limitations* — https://learn.microsoft.com/en-us/windows/win32/sysinfo/handle-limitations
- Microsoft Learn, *GDI Objects* / *USER Objects* — https://learn.microsoft.com/en-us/windows/win32/sysinfo/gdi-objects · https://learn.microsoft.com/en-us/windows/win32/sysinfo/user-objects
- Microsoft Learn (KB 947246), *Desktop heap limitation* — https://learn.microsoft.com/en-us/troubleshoot/windows-server/performance/desktop-heap-limitation-out-of-memory
- NTDebugging, *Desktop Heap Overview* / *part 2* — https://learn.microsoft.com/en-us/archive/blogs/ntdebugging/desktop-heap-overview · https://learn.microsoft.com/en-us/archive/blogs/ntdebugging/desktop-heap-part-2
- Turner, *Introducing the Windows Pseudo Console (ConPTY)* — https://devblogs.microsoft.com/commandline/windows-command-line-introducing-the-windows-pseudo-console-conpty/
- *Inside the Windows Console* — https://devblogs.microsoft.com/commandline/windows-command-line-inside-the-windows-console/
- Microsoft Learn, *Pseudoconsoles* / *Process Creation Flags* — https://learn.microsoft.com/en-us/windows/console/pseudoconsoles · https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
- `microsoft/terminal` #16212, #1810, #18634, PR #10415, PR #14160, discussion #12115 — https://github.com/microsoft/terminal
- `PowerShell/PowerShell` #12640 — https://github.com/PowerShell/PowerShell/issues/12640
- `warpdotdev/warp` #11398 · `wezterm` #7774 (stale-ConPTY root cause) — https://github.com/warpdotdev/warp/issues/11398 · https://github.com/wezterm/wezterm/issues/7774
- Microsoft Learn, *Job Objects* / *JOBOBJECT_CPU_RATE_CONTROL_INFORMATION* — https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects · https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information
- Microsoft Learn, *Configure process-opened file exclusions (Defender)* — https://learn.microsoft.com/en-us/defender-endpoint/configure-process-opened-file-exclusions-microsoft-defender-antivirus
- Microsoft Learn, *TCP/IP port exhaustion troubleshooting* — https://learn.microsoft.com/en-us/troubleshoot/windows-client/networking/tcp-ip-port-exhaustion-troubleshooting
- libuv `uv_pipe_t` docs; libuv #95, joyent/libuv #1023; nodejs/node #21632 — https://docs.libuv.org/en/v1.x/pipe.html · https://github.com/libuv/libuv/issues/95 · https://github.com/nodejs/node/issues/21632
- bitsnbites, *Benchmarking OS primitives* — https://www.bitsnbites.eu/benchmarking-os-primitives/
- Microsoft, *The Basics of Page Faults* — https://techcommunity.microsoft.com/blog/askperf/the-basics-of-page-faults/373120
