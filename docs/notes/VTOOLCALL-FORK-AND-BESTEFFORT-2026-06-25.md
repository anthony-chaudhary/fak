# vToolcall, Forked: Best-Effort Serve Tiers And Owning The Loop

Date: 2026-06-25

This is the forward-looking sequel to
[`VTOOLCALL-MATERIALIZED-VIEW-2026-06-25.md`](VTOOLCALL-MATERIALIZED-VIEW-2026-06-25.md).
That note says fak can only serve cached tool results as if the tool ran when fak
owns dispatch. On the proxy path, the harness consumes tool calls and runs tools
outside fak, so fak cannot safely synthesize the result after the model has already
asked for it.

## Consistency Axis

A usable "best effort" story needs an explicit consistency level on the call, not an
implicit shortcut. Thread it as `c.Meta["consistency"]`:

- `STRICT`: run or refuse; never serve a synthetic result.
- `BOUNDED_STALE`: serve only when the vDSO entry has a witness key and a valid
  world-version bound.
- `BEST_EFFORT`: serve when the cached answer is useful and the caller accepts
  staleness as a quality tradeoff, not a correctness dependency.
- `SPECULATIVE`: serve provisionally and require a later promotion or correction
  before any dependent write-shaped effect can commit.

This is the database read-consistency dual for tool calls. The level is a policy
input and an audit field, not a hidden cache mode.

## Capability Matrix

The proxy remains the universal gate, but it is the weakest topology for vToolcall:

- `/v1/messages` or `/v1/chat/completions` proxy: fak can adjudicate and quarantine,
  but cannot make an outside harness consume a fak-synthesized result as if the tool
  ran.
- `fak_syscall` / in-kernel dispatch: fak owns the before-consumption boundary, so a
  vDSO hit can become the materialized tool result.
- forked or native harness loop: fak can make dispatch the only tool path, which turns
  the prior "impossible on proxy" verdict into a normal kernel capability.

## Fork Direction

The fork is not about softening the un-ring problem. It relocates the barrier:
after-consumption repair becomes a before-consumption write barrier that fak owns.
That still needs a net-new suspend-and-resume turn primitive. Today's `gateTurn`
terminates; it does not suspend a turn, wait for a provisional result to promote,
and resume the same turn safely.

License also matters. The clean path is either:

- build a minimal native TUI on the shipped `internal/agent` seam, keeping the module
  Apache-uniform and single-binary, or
- host opencode as a server-side harness where fak owns the dispatch edge.

Do not fork a source-available, commercial-restricted harness into the Apache module.

## Honest Current State

The loop fak would extend today is the benchmark A/B comparator (`RunArm`), not a
production agent loop. The right portfolio move is:

- keep the proxy as the universal gate;
- lead the fork with the native build path;
- carry the consistency-level field through the call metadata before claiming any
  best-effort serve mode;
- build suspend/resume before claiming speculative tool-result serving across writes.
