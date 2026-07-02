---
title: "Cutting the per-tool-call DOS hook cost in fak fleets"
description: "A measured proposal to reduce per-agent-turn latency from spawned dos-kernel hooks via matcher-scoping and an in-process adjudication transport."
---

# Proposal: cut the per-tool-call DOS hook cost (upstream `dos-kernel`)

**Status:** local matcher mitigation applied (2026-06-25); kernel cache/transport
work remains a proposal · **Target repo:** `anthony-chaudhary/dos-kernel` ·
**Filed from:** fleet-public host-slowness diagnosis (2026-06-21)

## The problem, measured

A 32-core / 256 GB fleet workstation "felt slow" while **idle** — CPU ~3–13%,
disk 99.85% idle, 227 GB RAM free, no throttle, no paging. Nothing was
resource-starved. The cost was **per-agent-turn latency**, not load.

The fleet wires the DOS kernel as three Claude Code hooks (`.claude/settings.json`):

```json
"PreToolUse":  [{ "hooks": [{ "command": "dos hook pretool  --workspace ." }] }],
"PostToolUse": [{ "hooks": [{ "command": "dos hook posttool --workspace ." }] }],
"Stop":        [{ "hooks": [{ "command": "dos hook stop     --workspace ." }] }]
```

Two properties make this the dominant felt cost:

1. **Every hook is a process spawn.** `dos`'s own honest KPI (`docs/cli-reference.md`)
   measures the same `Fold` decide over two transports:

   ```
   in-process adjudication p50 : ~1,300 ns
   spawned-hook        p50     : ~6,000,000–50,000,000 ns   (process-per-decide)
   → ~5,000–39,000x
   ```

   That KPI is the kernel's own in-repo microbench (a warm Linux `Fold` decide).
   The cost a fleet host actually pays is **higher** — measured at the hook
   boundary on the Windows dev host with
   [`tools/dos_hook_bench.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/dos_hook_bench.py): `pretool` p50 ≈
   **300 ms** and `posttool` p50 ≈ **270 ms**, a **~0.57 s per-tool-call floor**,
   **synchronous** (it blocks the turn). Cold process startup dominates: the
   kernel's *internal* work is p50 **2.2 ms** (`.dos/metrics/observations.jsonl`),
   so **>99 %** of the wall-clock is spawn overhead — and **99.87 %** of those
   spawns are `passthrough` no-ops that bought no verdict. The order-of-magnitude
   gap from the 6–50 ms KPI is exactly the cold-spawn tax the in-process transport
   (fix #3) removes; treating 6–50 ms as the per-call floor undercounts the felt
   cost by ~10×.

2. **No `matcher` → it fires on every tool**, including read-only ones
   (`Read`/`Grep`/`Glob`/`TodoWrite`) that mutate nothing and need no gate. In a
   typical session reads are a large share of calls, so a large share of the
   spawn tax buys nothing.

   **Local mitigation now applied:** `.claude/settings.json` scopes the
   `PreToolUse` DOS hook to mutating tools plus DOS MCP calls:
   `Bash|Write|Edit|MultiEdit|NotebookEdit|mcp__.*dos.*`.

Multiply by N concurrent sessions on the host and the machine *feels* sluggish
while every gauge reads idle — death by a thousand 30 ms spawns.

> Local mitigation already shipped in fleet-public: an orphan reaper for the
> *process* side of the sprawl (`tools/proc_resource_guard.py --reap-orphans`,
> see `docs/perf-runaway-guard.md`). This proposal is the *latency* side, which
> only the kernel can fix.

## Three fixes, lowest-risk first

### 1. `matcher`-scope the hooks (config-only, no kernel change)

Stop spawning `dos hook` on read-only tools. In `.claude/settings.json`:

```json
"PreToolUse":  [{ "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit|mcp__.*dos.*",
                  "hooks": [{ "command": "dos hook pretool --workspace ." }] }]
```

- **Win:** eliminates the spawn on (often the majority of) tool calls.
- **Cost:** the kernel no longer *observes* read-only calls. If any DOS decision
  (arbitration, lane accounting) depends on seeing reads, this narrows it.
  **Kernel ask:** publish the minimal tool set DOS must observe so operators can
  scope the matcher without losing soundness. Until then this is operator-gated.

### 2. TTL-cached hook decisions (small kernel change)

Most consecutive `pretool` calls in a turn resolve against an **unchanged**
workspace (same git `HEAD`, same lane policy, same leases). Have `dos hook`
memoize its verdict keyed on a cheap fingerprint and short-circuit:

```
key   = (workspace, HEAD, lane-policy-mtime, lease-epoch)
value = verdict, cached for --cache-ttl (e.g. 2s) under .dos/hookcache
```

- **Win:** the repeated-spawn cost collapses to a stat + file read when nothing
  changed; first call per change still does full work. Keeps the spawn model
  (no harness change) — purely a fast path inside `dos hook`.
- **Risk:** staleness. Bound it with a short TTL **and** invalidate on any write
  the kernel itself gates (it already sees those via `posttool`), so the cache
  can never outlive a state change it authored.

### 3. In-process / async hook path (the real ceiling)

The KPI proves the in-process boundary is **5,000–39,000× faster**. Two shapes:

- **Async fire-and-forget `posttool`.** `posttool` records what already happened;
  it does not need to block the next tool call. Let it return immediately and
  flush its marker on a background thread (or batch-flush on session pause). This
  alone removes one of the two synchronous spawns per call.
- **In-process decide for `pretool`.** Expose the `Fold` decide as a resident
  endpoint (a per-workspace daemon `dos hook` talks to over a local socket, or a
  harness-native in-process callback) so the gate is the ~1.3 µs path, not a
  cold process start. This is the headline win and the larger change.

## Recommended sequencing

1. **Done locally (operator):** #1 is applied in `.claude/settings.json` for
   PreToolUse. Kernel should still publish the "must-observe" tool list so other
   workspaces can scope this matcher without guessing.
2. **Near (kernel):** ship #2 — biggest win per line of kernel change, no harness
   dependency, fully backward-compatible (cache miss == today's behavior).
3. **Later (kernel + harness):** #3 async `posttool`, then resident `pretool`.

## Acceptance criteria

- A microbench like the existing `fak bench` KPI but at the **hook** boundary:
  median wall-clock of `pretool`+`posttool` per tool call, before vs after.
  **Shipped:** [`tools/dos_hook_bench.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/dos_hook_bench.py) —
  `python tools/dos_hook_bench.py` reports `pretool`/`posttool` p10/p50/p90/max,
  the per-tool-call p50 floor, and the passthrough fraction from the observation
  log. It is read-only and SKIPs cleanly when `dos` is off PATH, so it is the
  reproducible witness for the before/after of any of the three fixes above.
- Soundness unchanged: the gate's verdict on a gated write is **identical**
  cached vs uncached and matcher-scoped vs full (replay the frozen trace).
- No new failure mode when the cache/daemon is absent — it must degrade to the
  current spawn-per-decide path, never to "fail open."
