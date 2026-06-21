# A4 · recall

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/recall/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

`internal/recall` makes a **completed** agent session queryable **without replaying it**.
A finished session is persisted as a *core image*: a small page table (`manifest.json` —
roles, content descriptors, digests, quarantine/clearance state, a frozen world-version
marker) over a content-addressed swap device (`cas.json` — the raw tool-result bytes keyed
by their sha256 address). A follow-up question demand-pages only the working set it touches
through a fresh trust gate, instead of re-executing the whole history. **"Correct" for
recall is regime A (algebraic / structural):** (1) a query against the reloaded image
returns the *same answer as replaying the session* — benign bytes come back **verbatim**,
nothing benign is silently dropped and nothing poisoned is falsely recalled; and (2) the
assembly is **deterministic and input-driven** — same (session, query, k) yields the same
working set every time. Below, each obligation is stated, argued against the real code
(file:line), and either discharged by a deterministic witness actually run here or honestly
left OPEN.

---

## Theorem 1 — query against a completed session == replaying it (verbatim, no false recall, nothing missed)

**THEOREM.** For a completed session persisted as a core image, a query against the
reloaded session returns the **same answer as replaying** the session: every benign page
resolves **byte-identical** to the bytes the session recorded (verbatim re-output), and the
assembled working set contains no page the recording did not hold and **excludes every
quarantined/poison page** (no false recall) while keeping benign pages reachable (nothing
silently missed).

**REGIME.** A — structural / round-trip + invariant.

**PROOF.** The persisted CAS stores each recorded tool result under its sha256 content
address (`Record`: `r.cas[digest] = body`, `recall.go:125`–`126`; `Digest`,
`recall.go:450`). `Load` re-hashes every blob and **refuses the whole image** unless each
hashes to its key (`recall.go:250`–`254`), so the swap device a query reads is bit-identical
to what was recorded. `Resolve(step)` (`recall.go:273`) looks the bytes up by digest
(`recall.go:284`) and returns them via `append([]byte(nil), body...)` — a verbatim copy with
**no rewrite** (`recall.go:296`, `:302`). So the answer served for a benign page *is* the
recorded bytes — the formal reduction of "same answer as replay" once the swap device is
content-addressed. `Recall` (`recall.go:376`–`403`) walks the **ordered** page table, skips
quarantined and tombstoned pages, ranks the rest by extractive descriptor overlap, and
re-resolves each candidate through the same gate — so a poisoned slice can never enter the
window and a benign, on-topic page is reachable.

The witnesses assert exactly this:
- `TestBenignPageRoundTripsByteIdentical` — `string(got) == benignAccount` after
  persist + reload into a **fresh** `Session` (its own CAS + gate).
- `TestSessionIsSelfContained` — the same equality after the recorder is dropped, so it
  cannot depend on the producing process.
- `TestRecallWorkingSetExcludesPoison` — the account page ranks first and **no** slice
  contains injection/secret bytes.
- `TestQuarantinedDescriptorCarriesNoPoison` — even the *index descriptor* of a sealed page
  carries none of its bytes.
- `TestContextChangeTombstoneSuppressesRecallButKeepsAuditBytes` — a tombstoned page is
  suppressed from recall while its audit bytes survive byte-identical.

*Scope note (honesty):* no test replays the **entire** transcript and diffs a full query
transcript token-for-token against it. The witnessed claim is the strictly
stronger-per-page byte-identity of each resolved page plus the deterministic exclusion set —
which is what "same answer as replay" reduces to for a content-addressed image.

**WITNESS.**
```
go test ./internal/recall/ -count=1 -timeout 120s \
  -run 'TestBenignPageRoundTripsByteIdentical|TestSessionIsSelfContained|TestRecallWorkingSetExcludesPoison|TestQuarantinedDescriptorCarriesNoPoison|TestContextChangeTombstoneSuppressesRecallButKeepsAuditBytes' -v
```

**VERDICT.** **PROVEN** — 2026-06-20 (macOS arm64, go1.26, native). All five green:
`--- PASS: TestBenignPageRoundTripsByteIdentical`, `--- PASS: TestRecallWorkingSetExcludesPoison`,
`--- PASS: TestContextChangeTombstoneSuppressesRecallButKeepsAuditBytes`,
`--- PASS: TestQuarantinedDescriptorCarriesNoPoison`, `--- PASS: TestSessionIsSelfContained`;
`ok github.com/anthony-chaudhary/fak/internal/recall 0.233s`. Full package also green.

**DOS.** bound at ship.

---

## Theorem 2 — recall is deterministic and input-driven

**THEOREM.** For a fixed `(loaded session, query, k)`, the assembled working set — its
membership, order, and bytes — is identical on every invocation, depending **only** on the
inputs (page table, persisted CAS, query string, clearance/revocation state) and on nothing
nondeterministic (no RNG, wall-clock, network, or output-affecting map-iteration order).

**REGIME.** A — determinism / structural.

**PROOF.** The mechanism is deterministic **by construction.** `Recall`
(`recall.go:376`–`403`) iterates `s.Manifest.Pages`, an **ordered** slice; scores each with
`overlap` (`recall.go:484`–`500`), a pure function returning an order-**independent** count
over `tokenize(descriptor)` (`recall.go:478`–`482`, pure string ops); and ranks with
`sort.SliceStable` (`recall.go:389`), whose tie-break preserves page-table order — so there
is no map-iteration dependence in the output order. `Resolve`/`reScreen` (`recall.go:273`,
`:331`) consult only the loaded CAS, the `cleared` map, and the vdso revocation ledger, all
deterministic for fixed inputs; there is no `rand`, `time`, or network call. The lone map
read inside `overlap` (the query token set) only answers membership and never drives output
order.

So the theorem is **true-looking** — but **no existing test asserts it.** A grep over
`internal/recall/*_test.go` for `determin|replay|twice|stable|idempot` finds only a
non-asserting comment (`readmission_fold_test.go:29`); no test invokes `Recall` twice on the
same inputs to assert byte-and-order equality, nor property-checks it across generated
queries. A green package run exercises `Recall` once and never re-checks run-to-run
stability. Per the honesty rule this is **OPEN**, not PROVEN.

**TO CLOSE.** Add `TestRecallIsDeterministic` (stdlib-only): record the airline image, load
it once, call `s.Recall(ctx, q, k)` **twice** (ideally over several queries and/or a
`testing/quick` property), and assert the two `[]Slice` results are equal in length and in
per-index `Step`/`Role`/`Descriptor`/`Bytes`; optionally assert that re-`Load`-ing the same
dir yields an identical sequence. That witness promotes this to PROVEN.

**WITNESS (closing test does not yet exist).**
```
go test ./internal/recall/ -count=1 -timeout 120s -run 'TestRecall' -v
```

**VERDICT.** **OPEN** — 2026-06-20. Mechanism pure/deterministic
(`sort.SliceStable` + order-independent `overlap`, no RNG/clock/network; `recall.go:376`–`500`),
but no deterministic witness asserts run-to-run equality.

**DOS.** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **recall-deterministic-input-driven** → ✅ PROVEN by `TestRecallIsDeterministicAcrossRepeatedCalls, TestRecallIsIdenticalAcrossIndependentReloads, TestRecallDependsOnlyOnQueryAndK, TestRecallWorkingSetNeverContainsQuarantinedBytes`. Recall is a pure function of (page table, persisted CAS, query, k) plus clearance/revocation state, with no RNG/clock/network/map-order dependence. Witnessed by four asserting tests over a non-trivial corpus (7 benign pages incl. score ties on the token 'refund' that force sort.SliceStable to tie-break by INPUT order, plus 2 quarantined pages). (a) Repeated-call determinism: 64 Recall calls on one Session yield a byte-identical fingerprint capturing membership+order+role+descriptor+raw bytes. (b) Map-iteration-order independence: 48 INDEPENDENT reloads of the SAME on-disk manifest.json/cas.json -- each Load builds a fresh Go map for Manifest.Cleared (Go randomizes map iteration order per map/process) -- all yield the identical fingerprint, so no map-order leaks into output. (c) Input-driven: two byte-equal Sessions agree on the same query; changing only the query deterministically reselects a different, itself-stable working set (query is a genuine input). (d) Safe invariant: across queries sharing the poison page's tokens, no quarantined page step ever enters the working set. Mechanism confirmed in recall.go: Recall iterates s.Manifest.Pages (a slice -> deterministic order) and ranks via sort.SliceStable; Resolve/reScreen/tokenize/overlap are pure over inputs. No RNG used (none needed). Whole package go test passes with the new file present.
