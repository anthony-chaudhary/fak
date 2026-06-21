# C2 · deletioncert

The `deletioncert` package mints and verifies a **DeletionCertificate**: a single, portable, re-checkable receipt that binds a bit-exact KV-cache eviction to the tamper-evident audit journal that recorded it. It is a pure fold — the caller supplies every fact (the eviction count and span from `model.KVCache.Evict`, the `max|Δ|=0` equivalence measurement, the hash-chained anchor row from `internal/journal`, and the integrity/trust epoch from `internal/vdso`), and the package does nothing but (a) bind them into one ed25519-signed pre-image and (b) re-verify that pre-image against a journal verifier. "Correct" here is **regime C (crypto / integrity)**: the certificate must be *unforgeable* (any post-issue field edit is detectable), it must *fail closed* (a non-zero drift, an absent/rewritten anchor row, a relabeled subject, or a missing journal verifier all yield `Valid=false`), and a *genuine* mint must round-trip back to `Valid=true`. The package is a leaf that imports only `crypto`/`encoding` stdlib, so its witnesses are deterministic (fixed ed25519 seed, an in-test `fakeJournal`; no RNG, clock, or network).

> Regime note (00-METHOD.md §1): `deletioncert` is regime **C** and additionally *depends on* an N-claim (`max|Δ|=0`). This doc discharges the C-binding and the C-verification; the underlying numerical `max|Δ|=0` deletion-equivalence measurement itself is the `model` eviction-parity obligation and is referenced here only as a signed string Verify checks for zero.

---

## THEOREM 1 — the certificate binds count + span + equivalence + anchor row + integrity epoch

**THEOREM.** A `DeletionCertificate` cryptographically binds, under one ed25519 detached signature, the eviction count (`EvictedCount`), the evicted span (`Span{From,Len}`), the `max|Δ|=0` bit-exact equivalence claim (`Equivalence.{MaxAbsDelta,RunID,Claim}`), the hash-chained journal anchor row (`Anchor{Seq,PrevHash,Hash,ResultDigest}`), and the integrity/trust epoch (`TrustEpoch`). Mutating any one bound field after issue is detectable; and `Verify` additionally enforces the equivalence-is-zero rung and re-derives the anchor row from the journal, so the anchor binding is re-checkable, not merely asserted.

**REGIME.** C — crypto / integrity (tamper-evidence + collision-binding of the bundled facts).

**PROOF.** The signed pre-image is `canonicalBytes(c)` — the certificate marshaled by `encoding/json` (sorted keys) with `Signature` cleared (`deletioncert.go:279`). Every theorem field is a member of `struct Certificate` (`deletioncert.go:105`): `EvictedCount` (`:115`), `Span` (`:114`), `Equivalence` (`:117`), `Anchor` (`:118`), `TrustEpoch` (`:120`) — all covered by the marshal, so all covered by the signature; `PublicKey` is covered (it is the trust root), only `Signature` is excluded (`deletioncert.go:280`). `Mint` signs that pre-image (`deletioncert.go:153`). `Verify` recomputes `canonicalBytes` over the cert with `Signature` blanked and runs `ed25519.Verify` (`deletioncert.go:206`), so any post-issue mutation of a covered field flips the verdict to `signature mismatch`. Beyond raw coverage, `Verify` semantically enforces the bound facts: the equivalence rung rejects any non-zero `MaxAbsDelta` (`deletioncert.go:221`), and the anchor rung re-derives `(prevHash,hash)` for `Anchor.Seq` via the `JournalVerifier` and refuses an absent/broken or mismatched row (`deletioncert.go:235`), with `Subject==Anchor.ResultDigest` pinning *which* data (`deletioncert.go:250`). The binding-by-signature is witnessed exhaustively by `TestTamperDetected` (`deletioncert_test.go:114`), whose mutation table flips exactly the theorem's fields — `evicted_count`, `span_from`, `span_len`, `trust_epoch`, `anchor_seq`, `anchor_hash`, `equiv_run` (plus `witness`, `scope`, `subject`, `issued_at`) — asserting `Valid==false` *and* `SignatureOK==false` for each. The equivalence and anchor rungs are independently witnessed green by `TestNonBitExactRejected`, `TestAnchorAbsent`, and `TestAnchorHashMismatch`.

**WITNESS.**
```
go test ./internal/deletioncert/ -count=1 -timeout 120s -v \
  -run 'TestTamperDetected|TestNonBitExactRejected|TestAnchorAbsent|TestAnchorHashMismatch'
```

**VERDICT.** **PROVEN** (2026-06-20, macOS arm64 node, go1.26 native). All 11 `TestTamperDetected` subtests PASS; `TestNonBitExactRejected`, `TestAnchorAbsent`, `TestAnchorHashMismatch` PASS. Package: `ok github.com/anthony-chaudhary/fak/internal/deletioncert 0.201s`.

**DOS.** bound at ship.

---

## THEOREM 2 — mint→verify accepts a genuine certificate; any tampered field is rejected

**THEOREM.** `Verify(Mint(priv, c))` accepts a genuine certificate — `Valid=true` with every rung green (`SignatureOK`, `AnchorOK`, `AnchorBound`, `SubjectBound`, `EquivalenceOK`) — and `Verify` rejects any certificate with a tampered field, a non-zero equivalence drift, an absent/mismatched journal anchor, a relabeled subject, or a missing journal verifier. The positive round-trip holds and every negative case fails closed.

**REGIME.** C — crypto / integrity (round-trip `verify∘mint = accept`, witness kind §3.4, plus unforgeability negatives §3.7).

**PROOF.** `Mint` stamps schema/defaults, derives `Subject` from `Anchor.ResultDigest` when empty, embeds the public key, clears `Signature`, and signs `canonicalBytes` (`deletioncert.go:133`). `Verify` runs four ordered rungs — signature over the canonical pre-image (`deletioncert.go:195`), `MaxAbsDelta==0` (`deletioncert.go:221`), anchor re-derivation + hash binding (`deletioncert.go:235`), `Subject==ResultDigest` (`deletioncert.go:250`) — and sets `Valid=true` only when all pass (`deletioncert.go:256`). The positive direction is witnessed by `TestMintVerifyRoundTrip` (`deletioncert_test.go:63`): mint `baseCert()` against a deterministic key and a matching `fakeJournal{7:{prev7,hash7}}`, then assert `Valid && SignatureOK && AnchorOK && AnchorBound && SubjectBound && EquivalenceOK && SelfAttested`. The negative direction is witnessed by `TestTamperDetected` (11 field mutations → `Valid=false`, `SignatureOK=false`), `TestNonBitExactRejected` (drift `1e-6` → `EquivalenceOK=false`, signature still OK), `TestAnchorAbsent` (empty journal → `AnchorBound=false`), `TestAnchorHashMismatch` (wrong hashes → `AnchorBound=false`), `TestNilVerifierFailsClosed` (`jv=nil` → `Valid=false`), and `TestSubjectRelabelRejected` (`Subject != ResultDigest` → `SubjectBound=false` despite a valid signature — the position-not-subject hole closed by `ResultDigest`). All ran green here.

**WITNESS.**
```
go test ./internal/deletioncert/ -count=1 -timeout 120s -v \
  -run 'TestMintVerifyRoundTrip|TestTamperDetected|TestNonBitExactRejected|TestAnchorAbsent|TestAnchorHashMismatch|TestNilVerifierFailsClosed|TestSubjectRelabelRejected'
```

**VERDICT.** **PROVEN** (2026-06-20, macOS arm64 node, go1.26 native). `TestMintVerifyRoundTrip`, `TestSubjectRelabelRejected`, `TestTamperDetected` (all 11 subtests), `TestNonBitExactRejected`, `TestAnchorAbsent`, `TestAnchorHashMismatch`, `TestNilVerifierFailsClosed` all PASS. `PASS / ok github.com/anthony-chaudhary/fak/internal/deletioncert 0.337s` (verbose run).

**DOS.** bound at ship.

---

### Note on what is NOT proven here (honesty rule, 00-METHOD.md §4)

- The signature is **self-attesting** in v1: it proves *integrity* (no field was edited) but not *independence* (you still trust the issuer minted honestly). `TestExternalAnchorClearsSelfAttested` confirms the `SelfAttested` flag flips when an `ExternalAnchor` is present, but `Verify` treats that anchor as *advisory* (`deletioncert.go:95-96`) — it does not yet validate an RFC-3161 / CT-log proof. The path to a third-party-verifiable receipt (validating `ExternalAnchor.Proof`) is **OPEN** and named.
- The underlying `max|Δ|=0` measurement is verified here only as a *signed string equal to zero*; the numerical deletion-equivalence itself is the `model` eviction-parity obligation, out of scope for this leaf.
- `EvictedCount` is, by the package's own doc (`deletioncert.go:115`), a self-report from `Evict`, not a witnessed cache delta. The certificate binds the *reported* count tamper-evidently; it does not independently re-count the cache.
