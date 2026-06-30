# fak kernel — federated "what changed" feed + revoke walkthrough

**One agent's adjudicated actions become another agent's queryable history — and a
peer can _refute_ a witness it no longer trusts, with the refutation propagating to
every consumer.** Two `fak serve` instances exchange a change feed; a revoke on one
invalidates a previously-trusted witness and shows up in _both_ feeds. This is the
**federated-trust surface** — a verifiable, cursor-drained ledger of cross-agent
changes plus a peer-signed refutation, something no production agent stack offers.

```
  demo.py                          fak serve A (:8431)          fak serve B (:8432)
    │                                   │                            │
    ├─POST /v1/fak/syscall ────────────▶│  update_booking COMPLETES  │
    │   (a write-shaped tool)           │  → mutation on A's feed     │
    │                                   │                            │
    ├─GET  /v1/fak/changes ───────────▶│  feed: [mutation seq=1]     │
    ├─GET  /v1/fak/changes ──────────────────────────────────────────▶│  feed: []  (independent!)
    │                                   │                            │
    ├─POST /v1/fak/syscall ──────────────────────────────────────────▶│  cancel_order COMPLETES
    │   (a different write on B)        │                            │  → mutation on B's feed
    │                                   │                            │
    ├─POST /v1/fak/revoke  bbb222 ─────▶│  refute B's witness         │
    │                                   │  → revocation on A's feed,  │
    │                                   │    trust_epoch 0 → 1        │
    │                                   │                            │
    └─POST /v1/fak/revoke  bbb222 (replay)────────────────────────────▶│  → SAME revocation on B's
                                        │                            │    feed, trust_epoch 0 → 1
```

## The two routes

| Route | What it does |
|---|---|
| `GET /v1/fak/changes` | the cross-agent "what changed" feed — a bounded, **cursor-drained** stream of typed write `mutation`s and `revocation`s on _this_ instance, each carrying a monotone `seq` (the cursor), a `world_ver` (consistency clock) and a `trust_epoch` (integrity clock). `?since=N` returns only events after cursor N; `0` (or omitted) returns the whole retained tail. POST `{"since":N}` is equivalent. |
| `POST /v1/fak/revoke` | refute a poisoned external world-state witness (a git commit / blob hash / lease epoch). Every pooled entry admitted under it is stranded, future re-admission under it is refused, and the refutation is broadcast on this instance's feed with an advanced `trust_epoch`. Body `{"witness":"<id>"}`; an empty witness is a `400`. |

A `mutation` is produced when a **write-shaped tool completes** (`StatusOK`) — so the
demo's policy allow-lists the write tools it drives, because a denied call never
completes and never lands on the feed. A `revocation` is produced by a `POST
/v1/fak/revoke` call.

## The change-feed exchange — and what "federation" really means here

The load-bearing, honest fact this demo is built around: **each `fak serve` instance
observes its own process-global coherence bus.** A write driven through A appears on
A's feed; it does **not** auto-gossip to B over the network. Step 2 of the walkthrough
shows exactly this — B's feed is empty right after A's write.

So "federation" is not a hidden consensus protocol. It is the host (or a peer)
**draining one instance's feed by cursor and replaying** the relevant event onto
another. The demo does this explicitly for the load-bearing case: it reads the
refutation off A's feed and replays it onto B, so the **same revocation appears in
both feeds**. That replay _is_ the peer-trust mechanism — the routes give you the
verifiable, cursor-ordered ledger and the refutation primitive; the host wires them
between instances. Stating it this way avoids overclaiming a consensus layer fak does
not implement.

## The revoke → invalidate flow

A revoke is a **peer-signed refutation, not a silent delete.** It does not erase the
mutation that is already on the feed; it _adds_ a `revocation` event and advances the
`trust_epoch`, so a consumer reading the feed forward sees "this witness was refuted at
epoch 1" and can re-plan / evict accordingly. The history stays intact and auditable —
the refutation is a forward-going trust statement, recorded, never a quiet erasure.

The `evicted` count in the revoke response is the number of pooled entries the
refutation stranded **in that instance's cache**. It is legitimately `0` when no entry
happens to be pooled under that witness in the instance you revoke on (e.g. the demo's
mock-engine path doesn't populate the gateway's tier-2 pool) — the `trust_epoch`
advance and the feed broadcast are what carry the refutation, independent of the local
eviction count.

## Why this is distinctive

A verifiable cross-agent change feed plus a peer refutation is something no production
agent stack offers. Today, when agent A acts, agent B has no first-class, ordered,
machine-drainable record of _what A changed_ — and certainly no way for B to learn that
a fact A relied on has since been **refuted**. fak makes both first-class on the wire:
the feed is the "what changed," the revoke is the "stop trusting this," and the
`trust_epoch` is the monotone clock that lets a peer order the two.

## Honest scope — what this is NOT

- **Not global consensus.** There is no quorum, no leader, no total order across
  instances. Each feed is a local, append-only ledger; cross-instance agreement is
  whatever the host's drain-and-replay achieves.
- **Not Byzantine fault tolerant.** A revoke is a trust _statement_, not a proof; it
  assumes a cooperating host that replays it. A malicious peer that simply refuses to
  replay a revocation will not learn of it — this is a peer-to-peer trust feed, not a
  protocol that defends against an adversarial participant.
- **Eventual and verdict-scoped.** Propagation happens when the host drains and
  replays; until then the two feeds differ (as Step 2 shows on purpose). The
  consistency you get is eventual, and scoped to the witnesses/verdicts actually
  exchanged.
- **A revoke is forward-going.** It invalidates _future_ re-admission and refutes the
  witness going forward; it does not rewrite history or un-happen the prior mutation.

## Prerequisites

- **Go** (to build `fak`) **or** a prebuilt binary via `FAK_BIN`, and **Python 3**
  (stdlib only).
- **No model.** Both instances run `--engine mock`; the feed and the refutation are
  pure kernel state, so every load-bearing assertion fires deterministically.
Expected runtime: the mock-engine run completes in seconds after the binary build.

## Run it

```bash
./examples/federated-changes/run.sh              # build, serve x2 (with the demo policy), run, teardown
./examples/federated-changes/run.sh --no-color   # plain output
FAK_BIN=./fak ./examples/federated-changes/run.sh  # use a prebuilt binary
```

`run.sh` starts two kernels (ports `8431` and `8432` by default; override with
`FAK_PORT_A` / `FAK_PORT_B`) and tears down everything _it_ started on exit. Windows
users: run the `.sh` launcher from WSL or Git Bash; the demo itself is plain `fak serve`
plus stdlib Python.

## What you see

> **Reading the output:** a `✓` means _the kernel observation matched expectation_ — a
> `✓` on the propagation row means the refutation A issued was visible on B's feed too.

```
  ✓ A: a write lands on A's change feed              update_booking -> mutation seq=1 world_ver=1
  ✓ B: A's write did NOT auto-propagate (per-instance feed)   B feed has 0 event(s), none of them A's write
  ✓ B: B's own write lands on B's feed               cancel_order -> mutation on B
  ✓ A: revoke refutes the witness on A (trust_epoch advances)  evicted=0 trust_epoch=1
  ✓ B: the refutation propagates A -> B (revocation in BOTH feeds)   B trust_epoch=1
  ✓ cursor drain is monotone (nothing re-delivered past the cursor)   drained B since cursor=2 -> 0 new event(s)
  ✓ empty-witness revoke is refused (400)            HTTP 400: revoke requires a non-empty witness

summary: federated change-feed + revoke walkthrough passed
```

Full captured run, including the raw wire JSON for each step:
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: build kernel → serve **two** instances with the demo policy → run the demo → teardown |
| `demo.py` | the demo itself (fak-native HTTP client, stdlib only); CI-usable exit code |
| `federated-policy.json` | the capability floor: allow-lists the write-shaped tools (`update_booking`, `cancel_order`) so they COMPLETE and land on the feed |
| `EXAMPLE-OUTPUT.md` | a captured run, including the raw wire JSON for every step |

Related: the route handlers and the coherence bus in
[`../../internal/gateway/coherence.go`](../../internal/gateway/coherence.go) and
[`../../internal/gateway/http.go`](../../internal/gateway/http.go); the vDSO mutation /
revocation primitives in
[`../../internal/vdso/scope.go`](../../internal/vdso/scope.go) and
[`../../internal/vdso/revoke.go`](../../internal/vdso/revoke.go). The gateway tests that
pin this surface:
[`../../internal/gateway/coherence_test.go`](../../internal/gateway/coherence_test.go)
(`TestChanges_CapturesWriteMutation`, `TestRevoke_EvictsAndPublishesToFeed`) and the
vDSO refutation soundness witnesses
[`../../internal/vdso/revoke_test.go`](../../internal/vdso/revoke_test.go)
(`TestRevoke_EvictsAllConsumersOfWitness`, `TestRevoke_RefusesReAdmitUnderRefutedWitness`,
`TestRevoke_PublishesOnCoherenceBus`).
