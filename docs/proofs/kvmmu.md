---
title: "fak proof: kvmmu KV-span bijection invariant"
description: "Proof that fak's KV MMU keeps logical positions one-to-one with physical cache slots under append and evict, addressing each named span exactly."
---

# A2 · kvmmu

`kvmmu` is the write-time bridge that turns `ctxmmu`'s LOGICAL quarantine verdict
("these bytes may not enter the context window") into a MECHANICAL one — eviction of
the offending result's K/V span from every layer of the kernel-owned attention cache,
so the model is physically incapable of attending to it. It is a pure consumer of two
shipped primitives — the detector's `Admit` decision and `model.KVCache.Evict` — joined
by a small span LEDGER (`[]*Segment`, each holding `{ID, From, Len, KV}`) that records
which contiguous cache positions each appended unit occupies. Its regime is **A —
algebraic / structural**: "correct" does not mean a number is within an error bound, it
means two structural invariants hold over the live cache. (1) **Bijection over live
spans:** the map from live logical positions to physical KV slots stays one-to-one and
onto under arbitrary append/evict sequences — no two logical positions alias one slot,
no survivor slot is lost. (2) **Exact span addressing:** a named span (by logical id)
maps to exactly its recorded `[From, From+Len)` bytes and exactly its derived metadata —
addressing keyed to the named segment, never to raw position or token pattern. The
witnesses assert these structurally (by-id evict counts, ledger renumber) and
bit-exactly (post-evict next-token logits `max|Δ|=0` vs an independent prefill of the
survivors alone), with a non-vacuity control (`poison-vs-never > 0`).

All witnesses below ran natively on the macOS fleet node (Darwin/arm64, go1.26 — the
Windows/WSL test machinery in `CLAUDE.md` does not apply here). `go vet
./internal/kvmmu/` exits 0.

---

## THEOREM 1 — KV mapping is a bijection over live spans

**THEOREM.** For any sequence of segment appends and span evictions, the mapping from
live logical positions to physical KV-cache slots is a bijection: no two distinct live
logical positions alias the same slot, and no live survivor slot is lost. Concretely,
after evicting a contiguous span `[from, from+n)` the surviving K/V at each new index
`i` is byte-identical to a cache that prefilled the survivors alone (each survivor
re-RoPE'd from its pre-RoPE raw at new position `i`), and the kvmmu ledger's
`Segment.From` offsets stay consistent with that compaction.

**REGIME.** A — algebraic / structural (round-trip + bit-exact invariant).

**PROOF.** The slot bijection is enforced by `model.KVCache.Evict`
(`fak/internal/model/kv.go:60`). It slices the evicted `[from,end)` columns out of every
layer's `K`/`Kraw`/`V` (`kv.go:77`), compacts `c.pos` (`kv.go:84`), then for every
survivor whose original absolute position `c.pos[i] != i` re-derives `K[i]` from the
pre-RoPE `Kraw[i]` with a SINGLE rotation at new position `i` (`kv.go:85`). RoPE is
linear in position (angle = `pos·inv_freq`), so re-rotating from old `p` to new `p'` is
exactly `RoPE(p'-p)`; `V` is unrotated and needs no fix. The result is byte-identical to
a cache that never saw the span — i.e. every survivor is present exactly once, at the
contiguous new index, with the correct rotation; nothing is duplicated or dropped, which
is the bijection. The kvmmu ledger mirror is `Context.evict`
(`fak/internal/kvmmu/kvmmu.go:198`): it calls `Cache.Evict(seg.From, seg.Len)`
(`kvmmu.go:199`), then shifts every later segment's `From` down by `seg.Len`
(`kvmmu.go:202`) and zeroes the evicted `seg.Len`, keeping each live `Segment`'s
`[From,From+Len)` in exact correspondence with the compacted physical slots. The
middle-then-tail evict in the witness (with `len(C) != len(B)`) makes any stale offset
detectably misfire, and the `max|Δ|=0` compare against an independent
`prefill(survivors)` is the bit-exact bijection check.

**WITNESS.**
```
(go test ./internal/kvmmu/ -count=1 -timeout 120s \
  -run 'TestLedgerRenumberAfterMiddleEvict|TestWriteTimeEvictEqualsNeverSaw' -v)
```
`TestLedgerRenumberAfterMiddleEvict` evicts MIDDLE `B` (len 5) then tail `C` (len 2),
then asserts the `A+D` distribution equals `prefill(A+D)` with `max|Δ|=0`.
`TestWriteTimeEvictEqualsNeverSaw` evicts the poison span write-time and asserts the
next-token logits are bit-identical to never-saw, with the non-vacuity control
`poison-vs-never = 3.257e-01 > 0`.

**VERDICT.** PROVEN — 2026-06-20. Both tests PASS (native Darwin/arm64, go1.26; package
run 0.256s, `go vet` exit 0). Logged: `max|Δ| evict-vs-never = 0.000e+00 (want 0) ;
poison-vs-never = 3.257e-01 (want >0)`.

**DOS.** bound at ship.

---

## THEOREM 2 — span addressing is exact

**THEOREM.** A named span maps to exactly its bytes: evicting a segment by logical id
removes exactly its recorded `[From, From+Len)` cache positions (no more, no fewer) and
invalidates exactly the derived cache metadata whose `ParentKV` is that segment's `KV`
identity — and the addressing is keyed to the named segment's recorded span/identity,
not to span position or token content (an identical token span with different content/id
is addressed independently).

**REGIME.** A — algebraic / structural (contract + invariant).

**PROOF.** A segment records its own span and identity at append: `Context.Append`
(`fak/internal/kvmmu/kvmmu.go:118`) stamps `From=Cache.Len()`, `Len=len(ids)`, and a
content-derived `KV=cachemeta.FromKVPrefix(...).ID` (`kvmmu.go:126`). By-id eviction
(`Context.Quarantine`, `kvmmu.go:186`) scans for the segment whose `ID` matches and is
not already `Held`, then evicts EXACTLY its recorded `[From,Len)` via `Cache.Evict`
(`kvmmu.go:199`); the witness asserts the returned count equals `len(b)` and that a prior
middle evict already renumbered the target's `From` (via `kvmmu.go:202`), so the address
used is the correct shifted span, not a stale one. Exactness against POSITION/CONTENT
confusion: `AdmitResult` (`kvmmu.go:163`) evicts only when the gate reading the result
BYTES returns `VerdictQuarantine`, so an identical token span carrying benign bytes is
addressed as a distinct, admitted segment. Exactness against DERIVED metadata:
`Context.invalidateReferences` (`kvmmu.go:212`) and `externalEntryReferencesKV`
(`kvmmu.go:227`) invalidate exactly the entries whose attention-index reference or
external residency matches the evicted `KV` identity, and `PlanExternalInvalidations`
(`kvmmu.go:200`) emits directives only for those.

**WITNESS.**
```
(go test ./internal/kvmmu/ -count=1 -timeout 120s \
  -run 'TestLedgerRenumberAfterMiddleEvict|TestEvictionIsContentDrivenNotPositional|TestQuarantineInvalidatesTrackedAttentionIndex|TestQuarantinePlansExternalEngineInvalidations' -v)
```
`TestLedgerRenumberAfterMiddleEvict`: `Quarantine("B")` returns exactly `len(b)` and
renumbers `C.From` to `len(a)`. `TestEvictionIsContentDrivenNotPositional`: identical
token span with benign body is `Allow`ed, not evicted. `TestQuarantineInvalidatesTracked-
AttentionIndex`: only the poison segment's `attention_index` is invalidated; the
unrelated one stays live. `TestQuarantinePlansExternalEngineInvalidations`: exactly the 2
entries whose identity matches the evicted span become directives.

**VERDICT.** PROVEN — 2026-06-20. All four tests PASS (native Darwin/arm64, go1.26).

**DOS.** bound at ship.

---

## Scoped-out / upgrade path

The runtime witnesses prove the bijection and exact-addressing invariants for
sequential append/evict orderings. They do NOT prove data-race freedom or aliasing under
CONCURRENT eviction (two goroutines evicting overlapping spans). Per `00-METHOD.md §6`,
`kvmmu` is one of the three named concurrency-critical leaves whose aliasing edge would
be strictly dominated by a **Gobra** separation-logic proof (memory safety + data-race
freedom). That is SCOPED-OUT here (needs a JVM+Viper+Z3 toolchain and per-function
specs) and recorded as the upgrade path, not a passed gate.
