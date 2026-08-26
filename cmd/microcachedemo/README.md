# microcachedemo — one native kernel makes repeated swarm work local

`microcachedemo` is the reuse half of the all-in-one micro-agent story. It runs
32 logical agents through the real `kernel.Kernel`, `adjudicator.Adjudicator`,
and `vdso.VDSO` in one process. Every agent asks eight read-only questions drawn
from the same four-query public corpus.

A normal per-agent implementation sends all 256 calls upstream. fak sends the
first four, fills its content-addressed vDSO from real `EvComplete` events, and
serves the remaining 252 locally. The same spine also proves the two boundaries
that keep fleet sharing safe:

- a denied payment action is refused before the engine sees it;
- a public tool explicitly marked shareable hits across agents, while an
  undeclared private tool remains keyed by `vdso.MetaPrincipal`, so agent B
  cannot read agent A's cached result.

No key, model, GPU, subprocess, browser, or network is required. Go 1.26 or
newer is the only prerequisite. With the toolchain and module cache already
available, the selfcheck completes in a few seconds (2.6 seconds in the captured
Windows run); a cold toolchain download or first compile can take longer. Its
fixed corpus has no clock or randomness, so the reported counts and selfcheck
verdict are deterministic. Success exits 0 and an invariant failure exits
nonzero.

## Run it

```bash
go run ./cmd/microcachedemo
go run ./cmd/microcachedemo -selfcheck
go run ./cmd/microcachedemo -json
```

The exact selfcheck transcript is in [EXAMPLE-OUTPUT.md](EXAMPLE-OUTPUT.md).

## What you see

The default deterministic render separates observed engine calls and vDSO hits
from modeled output tokens, then reports the policy and tenancy assertions:

```text
FAK MICRO-CACHE - one shared kernel turns a swarm into four upstream calls
fleet       32 agents x 8 calls = 256 identical-work opportunities
engine      256 -> 4 calls (252 local vDSO hits; 98.4% upstream work avoided)
generation  46080 -> 720 modeled output tokens (45360 avoided)
safety      denied tool reached engine 0 time(s)
tenancy     public cross-agent hit = true; private A/B engine calls = 1/1
VERDICT     native fak shares public repeated work fleet-wide, keeps private reads principal-scoped, and blocks unsafe work before compute
PROOF       go run ./cmd/microcachedemo -selfcheck
```

## Honest comparison

The baseline is tuned in the ways that matter for a micro-agent host: one process,
one shared engine client, no artificial setup delay, and no repeated prompt in the
measurement. Its only missing capability is fak's shared content-addressed result
cache; therefore `256 -> 4` isolates reuse rather than comparing against a naive
process-per-agent strawman.

The **98.4%** is observed engine-call avoidance on this fixed repeated-query
corpus. The token row is a model (`180` generated tokens per engine response)
shown separately and labeled in JSON; it is not a provider bill or latency claim.
`-selfcheck` requires exactly four engine calls, 252 native vDSO hits, identical
answers, zero denied-action engine calls, a public cross-agent cache hit, and one
engine call per principal for the private lookup.

This demo does not claim provider-level cache behavior, end-to-end latency, or
workload-wide savings. It proves the native in-process reuse, policy, and
principal-isolation invariants only for the fixed local corpus.

Together with [`../microfleetdemo`](../microfleetdemo/README.md), this gives two
complementary runnable spines: bounded context/residency for many long-lived tiny
agents, and fleet-wide reuse of repeated safe work.
