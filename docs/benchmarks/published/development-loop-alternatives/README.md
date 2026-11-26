---
title: "Campaign record: Development-loop alternatives: frozen tool reuse pilot"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak development-loop-alternatives (2026-11-02): Development-loop alternatives: frozen tool reuse pilot."
---

<!-- fak-docs-key: development-loop-alternatives-20261102 -->
# Development-loop alternatives: frozen tool reuse pilot

Native ticket: `development-loop-alternatives-20261102`. Private factory evidence,
not GPU inference performance. Run on 2026-11-02, 04:54:17–04:54:59 Pacific
standard time (12:54 UTC). Treat 100× as a hypothesis, not an achieved result.

## Result and decision

Reuse the existing immutable tool artifact mechanism before changing module
boundaries or introducing another cache. One first-use invocation took **33.434 s**;
four subsequent invocations took **2.742, 2.015, 1.627, 2.156 s**, median **2.086 s**.
The first-use/median-hit ratio is **16.03×**, a **93.76%** reduction for this query.
This is a cache-state comparison, not a change to Fak or a measured improvement in
accepted software changes per hour. The ordinary native launcher already caches.
Do not count each repeated read as another shipped capability.

All five invocations exited zero and returned identical bytes. The new artifact
also hashes identically to the previously cached executable. A changed closure
requested a build; conflicting source pins refused. No check was skipped to get
the faster result. See [measurements](matched-pilot.json), [identity](artifact-identity.json),
[actual output](accepted-ticket-output.json), and [controls](cache-controls.json).

The first-use build spent 28.699 s in recorded phases: seed discovery 12.805 s,
source export 7.852 s, dependency discovery 1.656 s, Go build 6.352 s, residual
0.034 s. Compilation is **19.00% of the 33.434 s invocation**, so eliminating it
entirely caps that invocation's speedup at **1.235×**. It is 22.13% of recorded
build time, giving a build-only ceiling of **1.284×**. The warm path still takes
seconds and its child execution varies. See the [raw manifest](pilot-build-manifest.json)
and [native operation records](pilot-operations.jsonl).

For total development time, use `S = 1 / ((1-f) + f/r)`, where `f` is the measured
fraction affected and `r` its speedup. With this 16.03× substep ratio, a hypothetical
50% affected fraction gives only 1.883× overall; even a wholly affected workflow
cannot gain 100× from this lever. A 100× total improvement with an infinitely fast
lever requires at least 99% removable total time. We have no matched total-workflow
denominator, so no total-workflow speedup is claimed.

## Frozen task, quality floor, and envelope

Task: read the experiment's already-claimed native ticket as JSON. Acceptance:
correct ticket ID, EXECUTING state, original worker/epoch, exact output equality,
exit zero, committed execution from the same full private/public source pair,
and cache refusal for conflicting pins. This is a bounded real native read,
not a synthetic compiler sleep or a model benchmark.

Command in `C:/work/fak-private`, unchanged for all five runs:

```text
fak-run --committed --strict --module-ref github.com/anthony-chaudhary/fak-private=41ffe0fdbec35ae38941cf4d6a9586cf3babb11d --module-ref github.com/anthony-chaudhary/fak=81f5f5464277cc176011db0fe19a789f7b003cd1 fak-ticket show development-loop-alternatives-20261102 --json
```

Use a new task-owned `FAK_RUN_CACHE` for the first run; preserve it for four hits.
Use the existing warm `%USERPROFILE%/AppData/Local/go-build` compiler cache for all
runs. Keep `GOMAXPROCS=2`, `GOMEMLIMIT=1GiB`, `GOFLAGS=-p=2`, Windows/amd64 and
Go 1.26.10 fixed. Clear inherited module-pin maps; pass both explicit pins. The
launcher SHA256 was unchanged before/after. Both source pair and executable
SHA256 are retained in the artifacts. Ordinary freshness semantics are not
replaced globally: pin a deliberately selected experiment/tool bundle only.

Run sequential hidden child processes; measure elapsed time with Stopwatch,
capture both output streams, exit, launcher CPU and 250 ms working-set samples.
Native operation records independently report child CPU, partial resource samples,
and cache resolution. Native host admission acquired slot 0 after **8 ms** and
held it **29.137 s** for the sole build; no pool setting was changed. The existing
host exposed three slots, not an experiment-selected capacity. Join the lease by
ticket `01793624057477606600-38992.ticket`, not PID alone: the PID had an older,
unrelated use. See [lease rows](pilot-build-lease.jsonl).

The input query is fixed, but the desktop remains shared and loaded. One cold
artifact observation followed by four hits does not estimate cold-tail latency or
prove a randomized causal effect independent of host load. The compiler cache was
not cold. Native cache records and the absence of additional builds establish the
mechanism; the timing spread constrains the claim. No cold compiler stress run,
GPU run, broad suite, release qualification, or test deletion occurred.

## Resource and work accounting

- Timed five-command block: **41.976 s** summed command wall time. First build
  CPU is not total CPU: launcher CPU was 3.141 s cold versus 0.484–0.641 s on hits;
  compiler descendants were not aggregated. Native ticket-child CPU was 46–92 ms.
- Launcher sampled working-set peak was 21.697 MB cold and 16.179–17.547 MB on
  hits. Native ticket-child samples reported 16 MB when available; two warm rows
  explicitly lack resource samples. These are not process-family peak RAM.
- Available host memory went from 186,191,832 to 183,286,320 KiB. Host disk free
  went from 798,309,105,664 to 797,454,786,560 bytes. Peers were active: neither
  difference is attributable to this pilot. The owned artifact retained
  **26,476,198 logical bytes**, including a 26,471,936-byte executable; compiler
  cache growth and total I/O were not measured.
- Human minutes and total team model tokens were not observable as complete
  counters. Do not substitute estimates or count only the primary agent. Two
  bounded workers supplied evidence review and independent readback; this work
  produced one evidence deliverable and no production code improvement.
- Setup and publication are additional work, not included in the query ratio.
  The required preflight `fak-sync repo --shift-left` reported **19m0.381 s**,
  including a long admission queue. It passed boundary/scrub/consumer checks,
  preserved peer dirt, and reported a partial dead-ref reap due to peer Git locks.
  Do not treat those warnings as permission to remove locks or peer work.

The [environment receipt](environment.json) retains before, after, and capacity
during work measurements without attributing shared-host changes to the pilot.

## Failed attempts and evidence limits

1. Broad `docs` lane arbitration refused. The exact evidence-directory keyword
   lane was then admitted and journaled as disjoint; no held tree was overridden.
2. First `fak-flow start` refused dirty shared-root state. The documented
   `--allow-dirty-root` option provisioned the dedicated sparse worktree at the
   explicit base, preserving all peer changes. Start only reads the pinned trunk
   object and creates an independent worktree; it proceeded while preflight sync
   was queued, which later completed successfully.
3. An eight-run ordinary-versus-pinned warm pilot had matching output and selected
   the same artifact, but both repository HEADs moved. Retain its [raw timings](excluded-moving-heads-pilot.json)
   and exclude its apparent speedup from matched conclusions. Replace that design
   with the fully pinned first-use/hit pilot above.
4. An inspected live 81.513 s `fak-sync` manifest was evicted from the shared cache
   before archival. Do not present its reconstruction as a raw receipt. The
   retained pilot manifest above owns this report's current phase calculations.
5. HBL-12 documents 104.0→3.1 s warm no-op builds, but its cited raw scratch file
   was not found. TS-24's 21× result is controlled fixture pre-exec latency.
   The reported 117–269 s operator rebuild range was not sourced; HBL-11 instead
   records 137.1/179.2/293.6 s. None establishes full development throughput.

See [HBL-11](../../tickets/host-build-lease/HBL-11-delta-build-budget.md),
[HBL-12](../../tickets/host-build-lease/HBL-12-vcs-stamp-tax.md),
[TS-24](../../tickets/claude-session-time-sinks/TS-24-fak-run-cold-build-latency-under-load.md),
and the [ranked alternatives](alternatives.md).

## Acceptance and next owned step

Documentation acceptance requires parsing and recomputing the retained JSON,
checking the native operation/lease bindings, checking links and whitespace,
and independent claim-vs-artifact readback. Code tests are UNRUN for this
documentation-only change; this does not settle outstanding full-tier release debt.

Next experiment: in this same evidence scope, measure one frozen useful change
from intake through independent acceptance and publication, including failed
attempts, model usage if available, and integration waits. Compare existing
ordinary flow with a prewarmed immutable tool bundle after the peer's process
selectiveness work publishes. Preserve its required tests and source checks;
count one accepted change per arm. The peer chat owns build/Git/process fixes,
so do not duplicate them here. No architecture rewrite is justified by this pilot.
