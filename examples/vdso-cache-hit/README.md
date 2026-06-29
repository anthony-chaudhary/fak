# fak kernel — tool vDSO cache-hit demo (`by=vdso` and world-version invalidation)

**A repeated read-only tool call is answered with no engine and no round-trip — served
from the kernel's content cache — and a write-shaped call between two reads invalidates
that cache so the next read runs fresh.** This is the runnable walkthrough for the
`by=vdso` column and the `vdso_hits=N` line that `fak run --trace` already prints, but
that no example explained: *why* a call hits, *what* bumps the world-version, and *how*
to see the soundness property (a hit equals a fresh call) with your own eyes.

```
  fak run --trace cache-friendly.json        fak run --trace invalidating.json
  ┌──────────────────────────────┐           ┌──────────────────────────────┐
  │ [0] get_reservation by=monitor│  fresh    │ [0] get_reservation by=monitor│  fresh
  │ [1] get_reservation by=vdso   │  HIT  ◀── │ [1] book_reservation by=monitor│  WRITE → world++
  └──────────────────────────────┘           │ [2] get_reservation by=monitor│  MISS (fresh)
        summary: vdso_hits=1                  └──────────────────────────────┘
                                                    summary: vdso_hits=0
```

Same two reads on both sides. The only difference is the write in the middle of the
right-hand trace — and that single mutation flips the second read from a cache HIT
(`vdso_hits=1`) to a fresh MISS (`vdso_hits=0`).

## What the tool vDSO is

The tool vDSO is the agentic analogue of the kernel vDSO that serves `gettimeofday()`
from userspace without a syscall: a **3-tier local fast path** that answers a tool call
with no engine and no remote round-trip. The tiers are consulted cheapest-first
(`internal/vdso/vdso.go`):

1. **tier 1 — pure registry.** The result is a pure function of the args (e.g.
   `calculate`). Gated on `readOnlyHint`+`idempotentHint`, and **re-checked, not
   trusted** — the kernel recomputes, it does not take the hint's word for it.
2. **tier 2 — content-addressed cache.** Keyed on `(tool, sha256(canonical args),
   world-version)`, filled from completion events, evicted LRU. This is the tier this
   demo exercises: the second identical read hits it.
3. **tier 3 — static table.** Canned answers for genuinely static tools.

A hit is reported in the `by=` column as `by=vdso`; a fresh call shows the rung that
served it (`by=monitor` for an ordinary read). The run summary counts the tier-1/2/3
hits together as `vdso_hits=N`.

## Run it

```bash
# with a prebuilt binary (no build):
FAK_BIN=/path/to/fak ./examples/vdso-cache-hit/run.sh

# or let run.sh build ./cmd/fak (needs Go on PATH):
./examples/vdso-cache-hit/run.sh
```

Or just call `fak run --trace` directly on either file — there is no server, no model,
and no network; it is a pure offline replay:

```bash
fak run --trace examples/vdso-cache-hit/cache-friendly.json
fak run --trace examples/vdso-cache-hit/invalidating.json
```

Windows users: run the `.sh` launcher from WSL or Git Bash, or call `fak run --trace`
directly from any shell.

## What you are looking for

**`cache-friendly.json`** — two identical read-shaped calls:

```
[ 0] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 1] get_reservation_details      verdict=ALLOW     by=vdso      status=OK

summary: submits=2 vdso_hits=1 engine_calls=1 denies=0 transforms=0 quarantines=0
```

The line to read is **`vdso_hits=1`**, and the **`by=vdso` on call `[1]`**. Call `[0]`
is the first read — the cache is empty, so it runs fresh (`by=monitor`) and its result
*fills* the tier-2 cache. Call `[1]` is the same read, so its content key now resolves
and it is served from cache.

**`invalidating.json`** — the same two reads with a write between them:

```
[ 0] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 1] book_reservation             verdict=ALLOW     by=monitor   status=OK
[ 2] get_reservation_details      verdict=ALLOW     by=monitor   status=OK

summary: submits=3 vdso_hits=0 engine_calls=3 denies=0 transforms=0 quarantines=0
```

Here the line to read is **`vdso_hits=0`**. Call `[1]` is `book_reservation`, a
write-shaped completion (`readOnlyHint=false`, `destructive=true`). It **bumps the
world-version**. So when the identical read arrives at `[2]`, its old cache key — bound
to the *pre-write* world-version — is unreachable. The read misses and runs fresh
(`by=monitor`). `vdso_hits=0`.

A captured run of both is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## Why each call hits or misses — the soundness property

The cache key is `(tool, sha256(canonical args), world-version)`. Two things follow:

- **Content keys are arg-order-independent.** The args JSON is canonicalized (sorted
  keys, normalized number spelling) *before* it is hashed, so `{"a":1,"b":2}` and
  `{"b":2,"a":1}` produce the same key and the same hit. The cache keys on *what the
  call asks for*, not on how the JSON happened to be spelled.
- **A write-shaped completion bumps the world-version, which invalidates every prior
  key at once.** The world-version is a monotone consistency epoch; it is part of every
  tier-2 key. A `destructive`/non-read-only completion does `world++`, so every entry
  cached under the old version becomes unreachable. The post-write read cannot find its
  old entry — it is forced to run fresh.

That second rule *is* the soundness invariant: **a cache hit equals a fresh call.**
The vDSO would rather miss than serve a value the world has moved past. The demo makes
this visible as the `1 → 0` difference in `vdso_hits` between the two traces: the only
thing that changed is the write, and the write is exactly what makes the stale hit
impossible.

The Go witnesses for these properties live in `internal/vdso`:

| property | test |
|---|---|
| a hit equals a recompute (tier-1 soundness) | `TestUnit38_SoundnessTier1EqualsRecompute` |
| a write-shaped completion bumps the world and invalidates a cached read | `TestUnit28_WriteCompletionBumpsWorld`, `TestUnit28_BumpWorldInvalidates` |
| arg-order-independent canonical content keys | `TestUnit26_27_Tier2CacheAndCanonicalization` |
| same route always invalidated on write | `TestScope_Soundness_SameRouteAlwaysInvalidated` |

Run them with `go test ./internal/vdso`.

## Honesty note: vDSO is upside, never the headline

These two traces are **deliberately cache-favorable** — they were hand-built so the
repeated read is identical down to the args, which is the best case for the content
cache. The shipped smoke trace `testdata/tau2/tau2-smoke.json` is favorable in the same
way (~50% hits).

Real agent workloads are not like this. The measured addressable purity on the real
`tau2-airline` workload is about **0.7%** — far below a useful hit threshold, because
real tool calls rarely repeat with byte-identical args inside one unbumped world. So the
tool vDSO is an **UPSIDE secondary, never the headline**: when reuse is there, it is free
and sound; when it is not, the kernel pays nothing for having looked. `fak` never gates a
claim on the vDSO hit rate, and neither should you read `vdso_hits` as a throughput
number — it is reported, not promised.

## Cross-links

- [`../../CLAIMS.md`](../../CLAIMS.md) §"Tool vDSO (3-tier local fast path)" — the
  shipped capability and its honesty fence (the `~50%`/`~0.7%` framing above is theirs).
- `internal/vdso/vdso.go` — the package doc states the soundness invariant verbatim.
- [`../../testdata/tau2/tau2-smoke.json`](../../testdata/tau2/tau2-smoke.json) — the
  larger smoke trace where `vdso_hits=6` first appears; it also contains a live
  invalidation (a `book_reservation` write strands a later `get_reservation_details`
  re-read into a fresh call).

## Files

| file | what it is |
|---|---|
| `README.md` | this file |
| `cache-friendly.json` | two identical reads; the second HITS the tier-2 cache (`vdso_hits=1`) |
| `invalidating.json` | the same two reads with a write between them; the second MISSES (`vdso_hits=0`) |
| `run.sh` | replays both traces and prints the two `vdso_hits` lines side by side |
| `EXAMPLE-OUTPUT.md` | a captured run of both traces |
