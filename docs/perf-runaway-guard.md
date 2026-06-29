---
title: "fak runaway-process guard and resource leak audit"
description: "A standing guard that flags and optionally reaps any process whose thread, handle, or memory level runs away, plus verified leak-audit findings for this repo."
---

# Runaway-process guard & leak audit

A standing guard against the one host failure the fleet watchdogs did **not**
cover — a single process whose OS-resource *level* (threads / handles / memory)
runs away and pins the machine — plus the verified findings from a memory-leak /
CPU-over-pin audit of this repo.

## The incident class

A process can leak OS threads without bound. The witnessed case: an external
`llama.cpp` `llama-cli` invoked **CPU-only with no `-t`/`--threads` bound** climbed
to **~129,427 threads on one process** and pinned the host — ~74% average CPU,
processor-queue length 26–41 (a sustained queue >1–2 per core means CPU-bound),
and ~73,000 context-switches/sec (thrashing). One process, one bad invocation,
whole machine unusable.

The existing fleet watchdogs (`fleet_supervisor_watchdog.py`,
`fleet_resume_watchdog.py`, `fleet_dos_dispatch_watchdog.py`, …) keep the
supervisor and sessions **alive** — they answer *"did it stop?"*. None of them
answers *"is a live process consuming a pathological amount of resource right
now?"*. That is the gap this guard fills.

## The guard: `tools/proc_resource_guard.py`

A control-pane loop first, an opt-in reaper second.

- **Read-only status (default).** Scans every live process via the platform's own
  tools (PowerShell `Get-Process` on Windows, `ps -eo pid,nlwp,rss,comm` on
  Linux — no third-party deps) and flags any process over a threshold.
  `ok:false` ⇒ ACTION (a runaway is live).
- **Single-shot for the *level* dimensions.** Thread count is the load-bearing
  signal — 129k threads is unambiguous and needs no second sample — so the guard
  never has to poll a counter twice. The one exception is the opt-in **CPU-pin**
  dimension (see below), which is a *rate* and so needs two samples.
- **Thresholds.** Default thread ceiling **2000** — far above the busiest
  legitimate process observed (the NT `System` kernel at ~600) yet far below the
  pathological 129k. Handle and working-set ceilings are opt-in
  (`--max-handles`, `--max-ws-mb`; `0` disables a dimension). Exempt a known
  heavy app with `--allow NAME`.
- **Opt-in reaper.** `--enact` kills flagged **non-protected** runaways
  (`taskkill /T /F` on Windows, `SIGKILL` on POSIX). It **never** kills an
  OS-critical process (`System`, `csrss`, `lsass`, `systemd`, `launchd`, …) or
  the guard's own process tree, even when flagged. Default is report-only.

```sh
# status (read-only; what the control pane runs)
python tools/proc_resource_guard.py --json

# human view, lower threshold to see the current top consumers
python tools/proc_resource_guard.py --max-threads 250

# DESTRUCTIVE: reap flagged non-protected runaways (operator opt-in)
python tools/proc_resource_guard.py --enact
```

Wired into `tools/control_pane.loops.json` as the **`proc-resource-guard`** loop
(read-only `--json`, no `--enact`). Logs each scan to `tools/_watchdog/proc_guard.log`.

## The CPU-pin dimension (the *single-threaded* runaway)

The thread/handle/working-set ceilings catch a process that runs away in *level*.
They are blind to the other loud failure: a process that runs away in *rate* — a
**single thread pinning one core to 100% forever**. A stuck spin loop, a `while
true` left running in a terminal, an inference binary wedged busy-waiting on the
CPU: each burns a whole core indefinitely while showing a completely normal thread
count (often **1**), handle count, and memory. On a many-core host this is easy to
miss — one pinned core is only `100/ncores`% of total CPU (≈6% on a 16-core box),
so the machine "feels a little slow" and no level gauge ever trips.

`--max-cpu-pct` is the witness for that class. Because CPU% is a *rate*, it is the
one dimension that takes more than one sample: the guard reads cumulative
CPU-seconds two (or more) times `--cpu-window` apart and computes **per-core**
percent, top-style — **100 = one full core saturated**, 400 = four cores.

The honest hard part is telling a *pin* from a *legitimate burst* (a compile, a
test run, a scorecard pass all saturate a core for a few seconds — and should
**not** be reaped). The guard separates them with `--cpu-samples`: with N samples it
takes N−1 consecutive windows and scores each process by the **minimum** window —
so a process is only flagged if it stayed over the threshold in *every* window. A
burst that ends mid-measurement scores its quiet window and drops out. This turns
the single-shot guard into a "sustained over (N−1)×`--cpu-window` seconds" test
with **no external state to persist**.

```sh
# report any process pinning >90% of one core, sustained over two 2s windows (4s)
python tools/proc_resource_guard.py --max-cpu-pct 90 --cpu-window 2 --cpu-samples 3

# the safe enacting shape: must hold >90%/core across three 2s windows (6s) within a
# run AND be flagged in 2 consecutive runs before it is killed
python tools/proc_resource_guard.py --max-cpu-pct 90 --cpu-window 2 --cpu-samples 4 \
    --cpu-reap-confirm 2 --enact
```

**Two-level confirmation — burst vs. pin, then seconds vs. minutes.** The within-run
`--cpu-samples` min-over-windows tells a *burst* from a *pin* over a few **seconds**.
It cannot tell a legitimate minutes-long CPU job (a big single-threaded link, a CPU
inference, a heavy scorecard) from a wedged loop — duration is the only honest
separator, and the only honest way to measure **minutes** is across consecutive
scheduled runs. So auto-reaping a **CPU-only** pin (its only breach is CPU) is gated a
second time on **`--cpu-reap-confirm N`**: the guard keeps a tiny start-time-keyed
streak ledger (`tools/_watchdog/cpu_pin_streak.json`) and only kills a pin flagged in N
consecutive runs. Keying on `(pid, start-time)` makes it reuse-safe: a recycled pid
carries a different start time, so it gets a fresh streak instead of inheriting a dead
process's count. Thread/handle runaways and orphaned helpers are unambiguous and
**always reap immediately**, ungated. A pin that has not yet reached the count is
surfaced as `cpu-unconfirmed` (still ACTION) but not killed.

**Threshold guidance.** A single-threaded runaway pins ≈100%/core, so a reaping
threshold of **90** (a near-full core) sits safely above almost all legitimate
bursty work while still catching the wedge. A standing reaper should pair it with
`--cpu-reap-confirm 2` (so a kill needs the pin to survive across scheduled ticks —
minutes, not one 6 s window) and never reaps a protected OS process or the guard's
own tree. On POSIX `cputimes` is integer-second, so use a window ≥ a few seconds.
Witnessed on this host: a single-threaded `code_slop_scorecard.py` worker sustaining
66%/core for 4 s was *reported* at `--max-cpu-pct 40` and, being report-only by
default, correctly **not killed** — exactly the legitimate transient that both the
90%/6 s bar and the cross-run confirmation exclude.

### Root-cause hygiene for inference launchers

The trigger class is *"an inference binary spawned with no thread bound"*. Our
committed launchers should always emit a bound:

- `fak/experiments/qwen36/llama_token3_selfconsistency.sh` already passes `-t`. ✅
- `tools/qwen36_node_server.py` now emits an **explicit, bounded `--threads`** for
  the `cpu` profile (leaves two cores free so a resident CPU server cannot
  silently run on all cores and over-pin a shared host). GPU profiles keep
  llama's own default (decode is offloaded); an explicit `--threads` always wins.
  This is headroom **hygiene**, not the leak fix — see the audit note below.

## Orphan-sprawl reaping (the *quiet* slowdown)

The runaway above is the loud failure. The quiet one — a host that "feels slow"
while every resource gauge reads idle — is **sprawl**: a long-uptime fleet node
accumulates ephemeral children that outlive the sessions that spawned them. The
dominant case here is the **per-session DOS MCP server**: every `claude` /
`opencode` session launches `python -m dos_mcp.server` over stdio; when the
client exits cleanly the server should follow, but a crash / detach can leave it
resident, serving no one. A few hundred of those plus stray launcher shells is
death-by-a-thousand-cuts, not a spike — so the level-based thresholds above never
catch it.

The same guard reaps it, **opt-in and evidence-based** (it refuses to guess):

- **`--reap-orphans`** flags an ephemeral helper matching a pattern (default
  `dos_mcp.server`, extend with `--orphan-pattern SUBSTR`) **whose owner PID is
  no longer alive** (or was reparented to init). The liveness test is
  direction-safe under PID reuse: a reused parent PID reads as *alive*, so the
  helper is spared — a missed reap, never a wrong one.
- **`--reap-idle-shells`** flags a launcher shell (`pwsh`/`powershell`/`bash`)
  with **zero live children** aged past `--idle-shell-age-min` (default 30 min).
  A shell that still wraps a live session has a child and is never touched — so
  the guard running *inside* such a shell can never reap its own ancestor.

Both reuse the protected-names guard, the `--enact` gate, and the ledger. A
flagged orphan makes `ok:false` (ACTION) exactly like a runaway.

```sh
# report orphaned MCP servers + idle launcher shells (read-only)
python tools/proc_resource_guard.py --reap-orphans --reap-idle-shells

# DESTRUCTIVE: reap them (operator opt-in; protected names + own tree spared)
python tools/proc_resource_guard.py --reap-orphans --enact
```

The standing **`proc-resource-guard`** control-pane loop now runs `--reap-orphans`
in its read-only fold, so an orphaned MCP server surfaces as ACTION the same way
a runaway does (idle-shell detection stays manual to avoid flagging interactive
shells). Reaping still requires a deliberate `--enact` run.

## Standing scheduled home

The control-pane fold above surfaces a runaway only when the pane is run. For a
host that should *always* be watched without an operator in the loop,
`tools/register_proc_resource_guard.ps1` installs the guard as a restart-durable
Scheduled Task (`FleetProcResourceGuard`), the Python-guard peer of the find/grep
reaper's `register_runaway_reaper.ps1`:

```powershell
# install REPORT-ONLY (safe default): logs every scan, kills nothing
.\tools\register_proc_resource_guard.ps1

# flip to ENACTING: reaps runaways + orphans immediately, and a CPU pin only after it
# holds >90%/core across 4x2s = 6s sustained AND persists across 2 consecutive runs
# (--cpu-reap-confirm 2) -- a legit compile/test burst or minutes-long job is never killed
.\tools\register_proc_resource_guard.ps1 -Enact

.\tools\register_proc_resource_guard.ps1 -Action status   # mode + last run
.\tools\register_proc_resource_guard.ps1 -Action remove
```

It runs windowless (S4U, session 0) as this user, so it can enumerate and — when
enacting — terminate this user's runaways without elevation, and `-StartWhenAvailable`
resumes it after a missed reboot tick. The reaping safety is the guard's, not the
scheduler's: protected OS processes and the guard's own tree are never killed.

> Why this and not "kill anything with a dead parent": a dead parent PID is **not**
> proof of abandonment — a detached `-p` worker is *born* with a dead parent and is
> doing real work. The guard only reaps a process that (a) matches a known
> ephemeral-helper pattern **and** (b) has a dead owner, or (a) is a childless
> aged launcher shell. Both are positive evidence of "serving no live session,"
> which is the whole point — the same trust-from-evidence stance DOS itself takes.

## Audit findings (adversarially verified)

A five-slice audit (Go goroutines/timers, Go memory, Python servers, Python
loops, invocation hygiene) produced 8 candidates; each was re-checked by an
independent skeptic asking *"does this actually bite in normal long-running
operation?"*. **2 confirmed real; 6 rejected.**

### Confirmed (both fixed)

1. **LOW — unbounded quarantine bookkeeping → FIXED.**
   `fak/internal/ctxmmu/mmu.go` (`MMU.quarantineResult` / `MMU.Clear`, `var Default`).
   The `held map[string]abi.Ref` and `cleared map[string]bool` maps were append-only
   (one entry per quarantine event, monotonic `q<N>` ids, no `delete`). `Default`
   is a rank-10 `ResultAdmitter` hit on every admitted result, so under sustained
   screened-UNSAFE traffic the maps grew for the process lifetime.
   **Fix landed:** the maps are now FIFO-bounded to `DefaultHeldLimit` (8192,
   mirroring the vDSO's bounded ledgers). On overflow the **oldest** quarantine ids
   are evicted — **fail-closed**: a page-in of an evicted id is refused exactly like
   an unknown id, so sealed bytes stay absent (the safe direction), and the durable
   record across a process boundary is the recall leaf's persisted `Held()`. A
   configurable `NewWithHeldLimit(limit)` is exported; tests in `ctxmmu_test.go`
   (`TestHeldMapBoundedFIFO`) cover eviction + fail-closed page-in and pass.

2. **HIGH — unbounded content-addressed store → FIXED (pin-aware bounded LRU).**
   `fak/internal/blob/store.go` (`var Default`, the single production `RegionBackend`
   + `"blob"` `PageOutBackend`). The `blobs` map was **append-only**: every >256 B
   served arg/result body deposited a CAS entry that was never freed, so a
   long-running `fak serve` grew RSS monotonically with distinct traffic.
   **Fix landed:** a byte-bounded LRU (`DefaultMaxBytes` 1 GiB, override with
   `FAK_BLOB_MAX_BYTES`, `0` = unbounded legacy). Eviction is **pin-aware** — it only
   ever drops digests no live holder still references, so it cannot break soundness:
   - A new `abi.CASPinner` capability (`Pin`/`Unpin`, **refcounted by digest** for
     content-dedup) is the modular seam; `abi.PinResolved` / `abi.UnpinResolved` are
     the one-line helpers both holders call (no duplicated assertion logic).
   - **vDSO** pins a tier-2 entry's digest on fill and unpins on *every* removal path
     (LRU-evict, lookup-time revoke-evict, and the bulk `Revoke()`), all **under
     `v.mu`** so a concurrent eviction can never win the race before the entry is
     reachable to a hit.
   - **ctxmmu** pins a held quarantine handle on hold and unpins on FIFO-eviction.
   - Audited to need **no** pin: gateway/modelengine (resolve within the producing
     call), `recall`/`cdb` (own private disk-backed CAS, never the global store),
     agent transcript (prompt-resolve), and the oversize `_paged` pointers
     (write-only, no retrieval site).
   Direct proofs: `blob` unit tests (`TestByteBoundEvictsUnpinnedNotPinned`,
   `TestPinIsRefcounted`, `TestUnboundedStoreNeverEvicts`) **plus** end-to-end
   soundness tests that a vDSO hit (`vdso/caspin_test.go`) and a gated quarantine
   page-in (`ctxmmu/caspin_test.go`) both survive heavy CAS eviction while an
   unpinned control blob is correctly evicted.

3. **NEW (surfaced by the pin audit) — `normgate` held quarantine map unbounded →
   FIXED.** `fak/internal/normgate/normgate.go` mirrors ctxmmu's quarantine machinery
   (`g.held[id]` via `abi.PageOut("blob")`) but its `held` map had **no bound** — a
   second instance of the same leak class that the first finder pass missed.
   **Fix landed:** FIFO-bounded to `DefaultHeldLimit` (8192). No CAS pin is needed:
   `held` is **write-only** today (normgate has no page-in / `Clear`), so the
   paged-out bytes are never resolved later and are safe for the bound to reclaim
   (documented in-code; if a gated page-in is ever added, pin like ctxmmu). The
   missing retrieval path is a latent design gap tracked separately.

> Validation: Go test execution runs on this host (an earlier "blocked" note was
> stale). `go build ./...`, `go vet ./...`, `gofmt`, and the **entire** `go test ./...`
> suite pass with these changes — zero failures. A pre-existing Windows-only flake in
> `internal/journal` (`TestPerWriteDurableFlush_StatsMatchDurableRows`) — a leaked
> `audit.jsonl` handle that blocked `t.TempDir` cleanup — was **also fixed** here (a
> `defer j.Close()` after Open; the durability-without-Close assertions still run
> first). That handle leak is itself a member of the audited class.

### Rejected (verified *not* defects in normal operation)

- `tools/qwen36_node_server.py` "unbounded threads" — **rejected.** An unset
  `--threads` resolves to `std::thread::hardware_concurrency()` (a bounded 8–64),
  not an unbounded/growing pool; the 129k was an external `llama-cli` pathology,
  not normal unset-default behavior. (The headroom change above is hygiene, not a
  leak fix.)
- `tools/bench_endpoint_server.py`, `tools/fleet_bottleneck.py` (×2) —
  `ThreadingHTTPServer` spawns one **daemon** thread per connection that exits when
  the request ends; thread count tracks concurrent connections (bounded by a
  localhost/tailnet scrape + bounded auto-refresh), not self-amplifying growth.
  Minor slow-client hardening at most, not the incident class.
- `tools/qwen36_watch_nodes.py` busy-wait — **rejected.** A one-shot CLI, hard-
  bounded by `max_wait_s` and gated on per-node network I/O; at most a momentary
  micro-spin at the very end, never a sustained pin.
