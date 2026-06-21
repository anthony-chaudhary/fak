# A3 · cachemeta

`internal/cachemeta` is the **payload-free metadata layer** for fak's reuse planes.
It lowers a reuse event — a KV prefix, an importable/signed KV manifest, a GLM-5.2 DSA
attention index, a live-engine KV-residency transfer (offload/restore/route/migrate), a
vDSO tool-result key, a recall context page, a memory view, or a provider prompt-cache
record — into a single `Entry` carrying a stable `EntryID` (digest · media-type · length ·
unit) plus the derivation, validity, security, residency, and coherence axes a hit must
match. It never holds the payload. For this module **"correct" is algebraic (regime A)**:
the identity key is a *deterministic, injective* fold over the binding axes; lowering an
event and reading it back *preserves the binding* (a matching lookup recovers a serveable
HIT / the recorded typed verdict); and *collision and eviction are well-defined* — distinct
bindings never alias to one key, a partial-match never serves the wrong entry (it yields a
typed MISS/FAULT, failing closed), and the external-invalidation cascade is a deterministic,
deduplicated set. All witnesses below ran **green natively** on this macOS node
(go1.26 darwin/arm64; the Windows/WSL machinery in the root `CLAUDE.md` does not apply here).

---

## Theorem A3.1 — the binding key is deterministic and axis-sensitive

**THEOREM.** For a fixed set of binding axes, the cache key is a deterministic function:
`ManifestBindingDigest` / `AttentionIndexBindingDigest` return the *same* hex digest on
repeat, and changing *any* covered axis changes the digest (so the key is injective on the
binding tuple).

**REGIME.** A — structural / determinism + injectivity.

**PROOF.** `ManifestBindingDigest` (`manifest.go:49`) and `AttentionIndexBindingDigest`
(`attention_index.go:58`) construct a `sha256.New()`, `writeField` each binding axis
followed by a `0x00` separator in a **fixed declared order**, and hex-encode the sum. The
only inputs are the struct fields; there is no map iteration over the digested set (layers
are ranged over an ordered slice, `attention_index.go:73`), no wall-clock, no RNG. A pure
fold over an ordered field list is by construction same-input → same-output. The `0x00`
field separator makes the fold injective on the *tuple* of fields (it rules out the
`"ab"+"c"` vs `"a"+"bc"` aliasing a naive concatenation would admit). The token digest
`DigestTokenIDs` (`cachemeta.go:645`) is the same shape over big-endian-encoded ids.

**WITNESS.**
```
go test ./internal/cachemeta/ -count=1 -timeout 120s -run \
  'TestManifestBindingDigestIsDeterministicOverBindingAxes|TestAttentionIndexDigestIncludesIndexShareLayerSet' -v
```
`TestManifestBindingDigestIsDeterministicOverBindingAxes` (`manifest_test.go:22`) asserts
`d1 == d2` on repeat **and** that `ModelID`/`Precision`/`PositionConvention` changes each
flip the digest. `TestAttentionIndexDigestIncludesIndexShareLayerSet`
(`attention_index_test.go:153`) asserts a different layer set yields a different digest.

**VERDICT.** **PROVEN** — 2026-06-20, both PASS. *Residual (honest):* the manifest key has
a direct same→same equality-on-repeat assertion; the attention-index key has only a
*sensitivity* witness (different layers → different digest). Its idempotence-on-repeat is
structurally true (pure ordered fold) and corroborated indirectly by
`TestFromAttentionIndexRecordsDSAPlaneAndIndexShareConsumers`, which recomputes
`DigestTokenIDs(tokens)` and compares it to the lowered `prefix_digest` label — but it is
not pinned by a dedicated `d1==d2` test. A one-line `AttentionIndexBindingDigest(idx) ==
AttentionIndexBindingDigest(idx)` assertion would close that last gap.

**DOS.** bound at ship.

---

## Theorem A3.2 — the kv-transfer / attention-index round-trip preserves the entry

**THEOREM.** Lowering an event into an `Entry` and reading it back preserves the binding:
`FromKVTransfer → KVTransferVerdict` recovers the recorded outcome as the correct typed
verdict (ok→HIT, fault→FAULT(`residency_fault`), missed→MISS(`restore_miss`)); and
`FromAttentionIndex`/`FromKVManifest` followed by a *matching-axis*
`AttentionIndexLookup`/`CheckResidentClaim` recovers a serveable HIT preserving plane,
handle, parents, and reason.

**REGIME.** A — round-trip / involution of the binding (not the payload; cachemeta is
payload-free by design).

**PROOF.** The round-trip is *lower* then *read-back*. For KV-transfer the carried datum is
the outcome string written to `Labels["outcome"]` (`kvtransfer.go:100`); `KVTransferVerdict`
switches on `KVTransferOutcome(e.Labels["outcome"])` (`kvtransfer.go:122`), so
`verdict(lower(t))` equals the typed form of `t.Outcome` exactly, with the default arm
mapping any unrecognized outcome to `MISS(absent)` — fail-closed. For attention-index and
manifest the binding is recovered by equality match: `AttentionIndexLookup`
(`attention_index.go:143`) and `CheckResidentClaim` (`manifest.go:92`) compare every
request/claim axis against the candidate/manifest axis and return `Hit(e)` only when all
agree; a request built from the entry's own axes therefore yields HIT, and `Hit()` sets
`Handle = e.ID` (`cachemeta.go:262`), so the recovered handle equals the lowered identity.
`ParentKV` round-trips into `Coherence.Parents` and is recovered by
`AttentionIndexReferences` (`attention_index.go:194`).

**WITNESS.**
```
go test ./internal/cachemeta/ -count=1 -timeout 120s -run \
  'TestFromKVTransferRecordsResidencyTransition|TestKVTransferRestoreFaultIsTypedNotSilent|TestKVRouteAndMigrateDirectionsSupported|TestAttentionIndexLookupRequiresPrefixDecisionAndCausality|TestAttentionIndexLookupBindsParentAndQualityBudget|TestCheckResidentClaimHitsOnVerifiedExactBinding|TestFromAttentionIndexRecordsDSAPlaneAndIndexShareConsumers' -v
```

**VERDICT.** **PROVEN** — 2026-06-20, all PASS. KV-transfer round-trips outcome + residency
tier/owner/from-tier/to-tier; the attention-index/manifest lower→lookup recovers HIT with
the correct plane (`attention_index` / `kv_artifact`), serveable handle, parents, and the
non-semantic-proof reason.

**DOS.** bound at ship.

---

## Theorem A3.3 — collision / eviction behavior is well-defined

**THEOREM.** Distinct bindings produce distinct digests (no false hit); a lookup whose axes
disagree returns a typed MISS/FAULT rather than serving the candidate (fail-closed); and the
external-invalidation cascade for a poisoned KV span is deterministic and deduplicated —
each engine object emitted at most once, referencing attention indexes invalidated with the
KV, provider telemetry excluded, empty poisoned id → no directives.

**REGIME.** A — structural (injectivity + fail-closed match + idempotent fold).

**PROOF.** Two parts. **(a) Key injectivity:** the digest folds are injective on the field
tuple (null-separated, Theorem A3.1), so two distinct bindings get distinct keys and the
package never indexes by a truncated/partial key. **(b) Match safety:** serving requires
*every* axis to match — `AttentionIndexLookup` (`attention_index.go:147`–`188`) and
`CheckResidentClaim` (`manifest.go:92`–`108`) return `Hit` only after all guards pass and
otherwise return a typed `Miss`/`Fault`, so a near-collision (some axes equal) never serves
the wrong entry; unknown/zero branches fail closed (`KVTransferVerdict` default → `Miss(absent)`,
`kvtransfer.go:129`; an unsigned manifest → `Fault(unsigned_artifact)`, `manifest.go:94`).
**Eviction determinism:** `PlanExternalInvalidations` (`external_invalidation.go:32`) gates
on `poisonedKV.Valid()`, iterates entries in slice order, and a `seen` set keyed on
`(kind,digest,mediatype,length,unit)` (`external_invalidation.go:37`) makes the output a
*set* — idempotent under duplicate inputs — while provider-plane telemetry is explicitly
skipped (`external_invalidation.go:57`) so cost-only metadata is never mistaken for an
evictable engine handle.

**WITNESS.**
```
go test ./internal/cachemeta/ -count=1 -timeout 120s -run \
  'TestAttentionIndexLookupRequiresPrefixDecisionAndCausality|TestCheckResidentClaimRefusesBindingMismatch|TestCheckResidentClaimRefusesUnsignedArtifact|TestKVTransferVerdictRefusesUnknownOutcome|TestPlanExternalInvalidationsDropsRemoteKVAndReferencingAttentionIndex|TestPlanExternalInvalidationsRejectsEmptyPoisonedKV|TestManifestBindingDigestIsDeterministicOverBindingAxes' -v
```
The lookup tests assert mismatched decision/prefix/model/parent → typed MISS, non-causal →
FAULT, incomplete candidate → FAULT, binding/length mismatch → FAULT(`manifest_mismatch`),
unsigned → FAULT(`unsigned_artifact`), unknown outcome → MISS(`absent`). The invalidation
tests assert exactly the remote KV + its referencing attention index are emitted, telemetry
dropped, and an invalid poisoned id → no directives.

**VERDICT.** **PROVEN** — 2026-06-20, all PASS.

**DOS.** bound at ship.

---

### Reproduce the whole module

```
go test ./internal/cachemeta/ -count=1 -timeout 120s
```
Ran green here (`ok github.com/anthony-chaudhary/fak/internal/cachemeta`). The dedicated
per-theorem `-run` filters above are the exact witnesses; the DOS column for each is bound
to the shipping commit at release via `dos commit-audit` / `dos verify` against the fleet
repo root.
