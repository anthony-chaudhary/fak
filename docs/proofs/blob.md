# A6 · blob

The `blob` module (`fak/internal/blob`) is the v0.1 default backend behind every `abi.Ref`: a concurrency-safe, content-addressed (sha256) in-memory blob store. It implements the three seams the frozen ABI leaves open — `abi.Resolver` (Put bytes → Ref, materialize bytes ← Ref), `abi.RegionBackend`, and `abi.PageOutBackend`. Small payloads (`len ≤ InlineMax = 256`) ride inline on the Ref; larger payloads land in the CAS map keyed by their digest. **"Correct" for this module is algebraic (regime A): the address-and-materialize map must be a faithful round trip (`Resolve∘Put = id` on bytes, byte-identical, no aliasing), and content addressing must be honest — the digest IS the sha256 of the content, so byte-identical payloads collapse to one stored blob and one address.** Both properties are load-bearing: the vDSO tier-2 cache and the context-MMU page-out path share one `Default` store, and dedup is what makes that sharing free.

---

## Theorem A6.1 — content-address put/get round-trips byte-identical

**THEOREM.** For every payload `b`, `Resolve(Put(b))` returns bytes byte-identical to `b`, on both branches of the size split: the inline path (`len(b) ≤ InlineMax = 256`, bytes carried on the Ref, CAS untouched) and the CAS-backed blob path (`len(b) > 256`, bytes stored under the digest). The returned slice is an independent copy, never an alias of the stored buffer.

**REGIME.** A — algebraic / structural (round-trip / involution; witness taxonomy §3.4).

**PROOF.** `Put` (`store.go:54-72`) computes `d = Digest(b)` and sets `r.Len = len(b)`. On the inline branch (`store.go:57-61`) it copies `b` verbatim into `r.Inline` via `append([]byte(nil), b...)` and returns `RefInline` — no transform, no CAS touch. On the CAS branch (`store.go:62-71`) it stores a fresh copy of `b` under `d` and returns a `RefBlob` carrying only the digest. `Resolve` (`store.go:75-91`) inverts this: for `RefInline` it returns a copy of `r.Inline` (`store.go:77-78`); for `RefBlob`/`RefRegion` it looks up `s.blobs[r.Digest]` and returns a copy (`store.go:79-87`). No byte-mutating transform exists on either path and the maps hold exact copies, so `Resolve∘Put = id` on bytes. Both copy-out sites use `append([]byte(nil), …)`, so the returned slice cannot alias the stored buffer.

**WITNESS.** `go test ./internal/blob/ -count=1 -timeout 120s -run 'TestPutSmallInlineRoundTrip|TestPutLargeBlobRoundTrip' -v` (run from `fak/`). `TestPutSmallInlineRoundTrip` (`store_test.go:24`) asserts `bytes.Equal(got, small)` over `n ∈ {0, 1, 100, 256}` — exercising the `InlineMax` boundary — plus `Kind==RefInline` and `Stats==(0,0)` (CAS untouched). `TestPutLargeBlobRoundTrip` (`store_test.go:68`) uses `len = InlineMax+1 = 257`, asserts `Kind==RefBlob`, `Inline==nil`, `bytes.Equal(got, large)`, and `Stats==(1,0,1)`. `TestResolveIsACopy` (`store_test.go:107`) proves the no-alias clause (mutating the resolved slice does not corrupt the store). `TestDefaultStoreViaABI` (`store_test.go:274`) confirms the round trip survives the registered `abi.Resolver` seam.

**VERDICT.** PROVEN — 2026-06-20. `PASS: TestPutSmallInlineRoundTrip`, `PASS: TestPutLargeBlobRoundTrip`, `PASS: TestResolveIsACopy`, `PASS: TestDefaultStoreViaABI`; `ok github.com/anthony-chaudhary/fak/internal/blob`. Run natively (macOS arm64 fleet node, go1.26).

**DOS.** bound at ship.

---

## Theorem A6.2 — identical content dedupes to one digest (the address IS the hash)

**THEOREM.** Two `Put`s of byte-identical content — even from distinct backing arrays — produce the same digest (the address is the sha256 of the content, not of the slice identity) and leave exactly one physical blob in the CAS, recording a dedup hit on the second `Put`.

**REGIME.** A — algebraic / structural (invariant / property: content addressing + conservation; witness taxonomy §3.5).

**PROOF.** `Digest` (`store.go:46-49`) is `hex(sha256.Sum256(b))` — a pure function of the content, so identical bytes yield an identical address regardless of which slice holds them. `Put` keys the CAS by that digest (`store.go:54`); the dedup branch (`store.go:65-69`) does `if _, ok := s.blobs[d]; ok { s.hits++ } else { s.blobs[d] = copy }`, so a second `Put` of already-present content takes the hit branch and stores nothing new — the map holds exactly one entry per distinct content. Since both returned Refs carry `r.Digest = d` (`store.go:56`), `r1.Digest == r2.Digest` holds by construction. The witness's use of a distinct backing array for the duplicate rules out any pointer-based shortcut, pinning the property to content addressing.

**WITNESS.** `go test ./internal/blob/ -count=1 -timeout 120s -run 'TestContentDedup|TestDefaultStoreViaABI' -v` (run from `fak/`). `TestContentDedup` (`store_test.go:202`) Puts `large = makeBytes(1024, 13)`, then `dup = append([]byte(nil), large...)` (a **distinct** backing array), and asserts: after Put#1 `Stats==(1,0,…)`; after Put#2 `puts==2` **and** `hits==1`; `r1.Digest == r2.Digest`; and `len(s.blobs)==1` read under the lock (exactly one physical blob). `TestDefaultStoreViaABI` (`store_test.go:274`) re-Puts an identical payload through the process-wide `Default` store and asserts the dedup-hits counter increments by exactly 1 across the ABI seam.

**VERDICT.** PROVEN — 2026-06-20. `PASS: TestContentDedup`; `ok github.com/anthony-chaudhary/fak/internal/blob`. Run natively (macOS arm64 fleet node, go1.26).

**DOS.** bound at ship.

---

### Notes / non-claims

- The CAS is **in-memory** (`map[string][]byte`); durability, eviction, and GC of the store are out of scope for these two obligations and are not witnessed here.
- Dedup is asserted on the **CAS path only**. Inline payloads (`len ≤ 256`) are never deduplicated — each inline `Put` returns its own copied bytes on the Ref and deliberately does not touch the CAS (witnessed by the `Stats==(0,0)` assertion in `TestPutSmallInlineRoundTrip`). This is by design (avoid a store round-trip on the hot path), not a gap in the theorem, whose dedup clause is scoped to stored blobs.
- Collision-resistance of sha256 itself is a crypto (regime C) property, not asserted here; A6.2 claims only that the implementation addresses-by-content-hash and collapses equal content, which the witness checks directly.
