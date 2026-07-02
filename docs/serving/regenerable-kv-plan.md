---
title: "fak Regenerable KV: text is the source, cache is a build artifact"
description: "Design doc treating fak's KV cache as a regenerable artifact rebuilt from durable transcript text, so a model rollout becomes a backfill, not a cold start."
---

# Regenerable KV — the text is the source, the cache is a build artifact

Regenerable KV treats the KV cache not as durable memory but as a rebuildable build artifact derived from durable transcript text, under the identity `KV = prefill(model, tokenize(text))` — the text is the cheap, model-agnostic source of truth, and the KV is an expensive, model-bound derivation you can evict and recompute. fak already ships the pieces this rests on: the deterministic suffix-only text-to-KV regen path on the live per-turn loop, a content-addressed tool-result store that survives process death, and a `SourceDigest` field naming the text each KV span was prefilled from. This page is the design plan for the layers on top — persistence across a model rollout, a fleet-shared text corpus, and a backfill scheduler — which are unbuilt today; it turns a model rollout, the event that invalidates the entire KV cache at once, from a synchronized cold-start cliff into a background prefill-backfill. It asserts no benchmark number: the regen-versus-keep cost is the same A/B "turn-tax" the existing bench harnesses measure, and a throughput claim would need that harness, not this doc.

> **Design decision doc** for treating the KV cache as a *regenerable build artifact*
> derived from durable transcript **text**, rather than as durable memory in its own
> right. The one fact underneath it: `KV = prefill(model, tokenize(text))`. The text is
> cheap, model-agnostic, and is the actual source of truth; the KV is an expensive,
> model-bound derivation you can throw away and rebuild. This is the axis that makes a
> **model rollout** survivable — the event that today invalidates 100% of the KV cache
> at once becomes a background *prefill-backfill* instead of a synchronized cold-start
> cliff.
>
> **Scope.** Design plus an honest fence. fak already ships the deterministic text→KV
> regen primitive on the live per-turn path, a content-addressed tool-result store that
> survives process death, and the `SourceDigest` field that names the regen input. The
> persistence-across-rollout, fleet-shared-corpus, and backfill-scheduler layers that
> turn those primitives into a rolling cache migration are **unbuilt (GAP)**. Every mark
> below carries a `file:line` anchor checked against the working tree on 2026-06-23;
> line numbers drift, so re-anchor with [§10](#10-how-to-re-verify).
>
> **No benchmark number is asserted.** The regen-vs-keep cost is the same A/B "turn-tax"
> the bench harnesses already measure (`internal/swebench/cost.go:17`,
> `internal/webbench/geometry.go:105`); a throughput claim needs the harness, not this doc.
>
> *Sibling axis:* `docs/serving/polymodel-prefill-share-plan.md` (host many models, share
> the prefill, decode one). This doc is the *time* axis of the same cache: that plan shares
> a prefix across co-resident models *now*; this one rebuilds a model's whole cache *across
> a version change*.

## 1. TL;DR — the decision

1. **Adopt the transcript text as the durable substrate; treat KV as a rebuildable
   artifact.** Store the cheap, model-agnostic thing (the token stream and the source
   bytes behind it). Regenerate the expensive, model-bound thing (the KV span) on demand.
   The cache stops being something you must preserve and becomes something you can always
   reconstruct.
2. **The cache is already labelled "not the source of truth."** A KV hit in `cachemeta` is
   stamped `performance_material_not_proof` and lives on a dedicated `PlaneKVArtifact`
   plane (`internal/cachemeta/manifest_test.go:108-113`). The repo already draws the line
   this doc formalizes — the KV is performance material; the *text* is the proof.
3. **A model rollout is the one event that kills the entire cache at once.** A HIT
   requires an exact match across a **nine-axis binding tuple** — ModelID, TokenizerID,
   AdapterID, Precision, PositionConvention, Producer, SpanDigest, Tokens,
   IntegrityChecksum (`internal/cachemeta/manifest.go:107-116`). Bump any axis and every
   cached span misses. The text survives every axis bump.
4. **Regen is lossless within a fixed numeric environment, distributionally equivalent
   outside it.** On the same silicon, SIMD tier, and quant, the proven `Clone`/`Evict`/
   prefix-reuse rungs make a regenerated KV byte-identical to a kept one. Across hardware,
   CPU-feature tier, or deployment quant the contract is argmax-exact plus a sub-tolerance
   drift, by design. Serving needs only the latter, so regen is *operationally* lossless —
   but the doc never claims "byte-identical and indistinguishable."
5. **Backfill leans on the prefill/decode asymmetry; it is not free.** Backfill is prefill
   (compute-bound, fan-outable); live serving is decode (HBM-bandwidth-bound, serial). The
   pressures lean apart, but on one GPU they share the SMs and the weight-read bus, so the
   honest framing is "runs on otherwise-idle prefill capacity, throttled below a
   live-decode SLO," not "~free." Making that true needs a two-class scheduler that does
   not exist yet.
6. **What lands now is the principle plus the primitives it stands on.** The deterministic
   text→KV regen path is live; a content-addressed, poison-re-screened tool-result store
   that survives process death is shipped (per-session); `SourceDigest` names the regen
   input. The migration apparatus on top is sequenced in [§8](#8-the-child-map--sequencing),
   not claimed here.

## 2. The reframe — KV is a build artifact, text is the source

In a normal serving stack the word "cache" means the KV cache: the per-layer key/value
tensors prefill produces. That object has three properties that make it a poor candidate
for durable memory.

- **It is model-bound, not data-bound.** `CheckResidentClaim` refuses to serve a cached
  span unless the caller's claim matches the manifest on all nine binding axes
  (`internal/cachemeta/manifest.go:107-116`). The KV bytes mean nothing to a model with a
  different weight set, tokenizer, adapter, precision, or position convention.
- **It is large.** A KV span is the materialized attention state for every token across
  every layer; the manifest exists precisely so a planner can reason about a span
  "without paging in terabytes of KV" (`internal/cachemeta/manifest.go:19-22`).
- **It is a pure function of its input.** `KV = prefill(model, tokenize(text))`. Nothing
  in the KV is unrecoverable from the text that produced it, given the model.

The transcript text is the mirror image: data-bound (the same `Read` of the same file is
the same bytes for everyone), small (the manifest already carries a `SourceDigest`, "digest
of the source text the KV was prefilled from", `internal/cachemeta/manifest.go:23`), and
the thing the KV is derived from. So the durable store should key on the *source*
(`SourceDigest`), and each KV span (`SpanDigest`) should be a per-binding-tuple derivation
the system is free to evict and rebuild.

The repo already half-says this. The cache marks its own hits
`performance_material_not_proof` and isolates them on `PlaneKVArtifact`
(`internal/cachemeta/manifest_test.go:108-113`). The piece missing is a consumer: today
`SourceDigest` is folded into the binding digest (`manifest.go:60`) and surfaced as a label
(`manifest.go:187`), but nothing keys a durable text store off it or regenerates KV from it.
It is the right field with no one reading it.

## 3. "Regen" — and what "lossless" actually means

Regen is the most ordinary operation in the stack: re-run prefill over the stored text.
It is already the live per-turn path. `InKernelPlanner.Complete` looks up the longest
cached prefix, clones its KV with `SessionFromPrefix`, and re-prefills *only the divergent
suffix* with `s.Prefill(ids[matched:])` (`internal/agent/inkernel_planner.go:213-242`). The
RadixAttention prefix tree that backs it is on by default and keys on token-ids, which is
model-agnostic at the data-structure level (`internal/agent/inkernel_planner.go:84-89`;
`internal/radixkv/radixkv.go`). The benchmark harnesses already treat full regen as a
baseline arm — arm **A**, "re-prefill the WHOLE context every turn"
(`internal/swebench/cost.go:17`, `internal/webbench/geometry.go:105`) — so the cost of
regen versus keeping the cache is exactly the A/B turn-tax those harnesses report.

The one claim to state carefully is **losslessness**, because the adversarial pass refuted
its strong form. Bit-exactness is genuine but *regime-relative*:

- **Within one numeric environment it is bit-for-bit.** Parallelism reorders which core
  computes a row, never a single dot-product's reduction order, so `Clone == recompute` and
  `prefix-reuse == full-prefill` on the same box with the same settings
  (`TestKVPrefixReuseMatchesRecompute`, `TestPrefillBatchedMatchesSerial`,
  `TestKVQuarantineEqualsNeverSaw`). At this scope "deterministic regen" is real.
- **Across hardware, CPU-feature tier, or quant it is not.** The attention dot product has
  scalar, AVX2, and AVX-512 implementations chosen at runtime, whose reduction trees differ;
  the test suite asserts the scalar path bit-identical but holds the SIMD path only to
  `1e-4`, and the ARM64 decode kernel is documented as "NOT bit-identical," with a contract
  of "argmax-exact + cosine." Cross-quant prefill (f32 vs Q8 vs Q4K) produces different KV
  by design — which is exactly *why* Precision and PositionConvention are required match
  axes in the binding tuple.

The resolution: **serving does not require byte-identity, only argmax/distributional
equivalence.** So regen is operationally lossless even where it is not bit-exact. State it
as "lossless within a fixed numeric environment; distributionally equivalent across
hardware, tier, and quant" — never "indistinguishable from a kept cache." This also draws a
sharp line for the integrity story in [§6](#6-the-text-tier-is-a-governance-surface): KV-byte
hashing is a valid tamper check *only* inside one regime; across regimes the integrity oracle
must hash the text, not the KV.

## 4. The rollout cliff, and the rolling migration that replaces it

Here is the event the whole design is for. When you roll out a new model, every cached KV
span fails the nine-axis HIT gate at once. A KV-only cache does not degrade — it falls off a
cliff. Hit-rate goes to zero the instant the new model takes traffic, prefill load spikes,
and tail latency spikes with it, until the cache re-warms organically from live requests. The
cliff is synchronized across the whole fleet because every node bumped the same `ModelID` on
the same deploy.

If the durable substrate is the text, the cliff becomes a schedule. The text survived the
rollout untouched, so the new model's cache can be rebuilt from it:

- **Lazy regen on miss** is already the mechanical behavior. A cold tree means every lookup
  misses and the planner full-prefills. Nothing new is required for correctness; the new
  model simply pays arm-A cost until its tree warms. `FAK_INKERNEL_RADIX_BUDGET` is the
  natural definition of the "hot set" that warming fills.
- **Eager backfill** replays the N hottest transcripts against the new model *before or
  during* cutover, so the model launches warm. This is blue-green for caches: stand up the
  new model's KV from the shared text corpus, shift traffic once it is warm, retire the old
  model's spans. It is speculative — without a fleet model of which sessions will actually
  return, eager backfill can prefill abandoned work and lose to lazy. The honest default is
  lazy-on-miss with eager backfill gated on a return-probability predictor (an open question,
  [§9](#9-non-goals--honest-fence)).

The interaction with the exact-ModelID barrier is the subtle part, and it is a *complement*,
not a violation. Text-regen never serves model A's KV bytes to model B; it re-derives B's own
KV from the shared text, so the barrier stays exactly as strict as it is today. The barrier
guarantees correctness; the text tier makes the cost of obeying it (a cold cache on every
model change) cheap to pay.

**This is trajectory-replay, re-pointed from policy to model.** The replay engine already
shipped (the trajectory-replay program scores K policies against one recorded run by
deterministic replay). Point the same machinery at a model axis and "replay the recorded
transcripts against model B" *is* the backfill — provided the recorded run carries the text,
not just model-A's KV. The honesty gate transfers intact: you may validate B-regen against
B-recompute, never against A's stored KV, because cross-regime KV differs by construction.

## 5. What generalizes — the binding tuple is the invalidation set

Model rollout is the headline, but it is one axis of a tuple. The cache binding is
`(ModelID, TokenizerID, AdapterID, Precision, PositionConvention, Producer, SpanDigest,
Tokens, IntegrityChecksum)` (`internal/cachemeta/manifest.go:107-116`). Every axis is a
change that invalidates KV but leaves the text valid, so every axis is a regen trigger:

- **Tokenizer change** — but with a trap. The radix tree keys on token-ids, which are
  tokenizer-specific. A model-plus-tokenizer co-rollout invalidates the *key space*, forcing
  a re-tokenize before re-prefill. So the durable key must be the **text-digest plus
  TokenizerID**, and the regen path must be `bytes → re-tokenize → prefill`, not
  `token-ids → prefill`. Keying the durable store on token-ids would silently reproduce the
  exact cliff the design exists to avoid.
- **Context-window extension** (8k → 128k) changes the RoPE base, so old KV is wrong under
  the new position convention; the text is fine, regen rebuilds it.
- **Quantization change** produces different KV bytes by design (it is a required match
  axis); the text is invariant.
- **Adapter / precision / producer** swaps are the same shape.

The unifying statement: **text is the migration-invariant substrate.** Anything in the
binding tuple can move; the text does not. The relationship to the polymodel plan's
cross-model share is that the two are complementary halves. For a *declared-compatible*
family — same `Family` and byte-identical `PrefixDigest`, the `CanShare` decision
(`internal/polymodel/polymodel.go:543-549`) — the KV bytes are genuinely identical, so a
share can skip backfill entirely. Backfill is the answer for the *incompatible* jump (new
tokenizer, new family, new quant) where the bytes truly differ and must be recomputed.

## 6. The text tier is a governance surface

Promoting text to the durable-cache-source tier raises its stakes, because a tool-result
corpus is not inert. It holds file contents, command output, secrets, and whatever an
adversary managed to write. Two consequences follow, and fak already has the primitives for
both — at process scope.

A **content-addressed, poison-re-screened, death-surviving store already exists.**
`recall.Persist` writes a self-contained `manifest.json` + `cas.json` image
(`internal/recall/recall.go:262-281`); a reloaded `Session` resolves against its own private
CAS with a fresh gate, the explicit "survives death" proof
(`internal/recall/recall.go:287-296`); `Load` verifies every CAS entry against its digest
address, so a tampered swap fails closed (`:298-301`); and `reScreen` re-folds `canon.Scan`
plus the kernel's registered ResultAdmitter chain on every page-in, so a session recorded
under a weak gate is re-screened by every detector the fleet ships now
(`internal/recall/recall.go:404-433`). That last property is exactly the right shape for the
text tier: regen re-runs prefill over stored bytes, so page-in is a natural re-screen point,
and a stricter post-rollout policy may legitimately quarantine text the old regime admitted.

The **blast radius grows when the corpus is shared.** A poisoned text entry, persisted once,
re-materializes into *every future model's* backfilled cache — a single bad tool result
becomes a fleet-wide cache contaminant on the next rollout. Persisted secrets and injection
markers sit sealed-but-present in `cas.json`, and the sealed-bytes extractive-leak path is an
acknowledged open question in the recall leaf. So the fleet-shared tier needs a digest-keyed
quarantine authority consulted *before* any backfill-prefill, layered on the existing
re-screen and revocation machinery — not a free extension of the per-session store.

## 7. Capability-honesty table

Legend: **[SHIPPED]** real & proven · **[PARTIAL]** real but incomplete · **[SEAM-ONLY]**
field/seam exists, no consumer · **[GAP]** absent.

| Capability | Status | Anchor |
|---|---|---|
| Deterministic suffix-only text→KV regen is the LIVE per-turn path | **[SHIPPED]** | `internal/agent/inkernel_planner.go:213-242` (Lookup→`SessionFromPrefix`→`Prefill(ids[matched:])`) |
| RadixAttention prefix tree wired into the live planner, on by default, keyed on token-ids | **[SHIPPED]** | `internal/agent/inkernel_planner.go:84-89`; `internal/radixkv/radixkv.go` |
| Bit-exact `Clone` / `Evict` / prefix-reuse == full recompute (within one numeric regime) | **[SHIPPED]** (regime-scoped) | `internal/model/kv.go:123-152`; `TestKVPrefixReuseMatchesRecompute`, `TestKVQuarantineEqualsNeverSaw`, `TestPrefillBatchedMatchesSerial` |
| Prefill compute-bound/fan-out vs decode bandwidth-bound/serial asymmetry | **[SHIPPED]** (documented + witnessed) | `internal/model/parallel.go:51-72`; `internal/polymodel/polymodel.go:222-231` |
| KV hit marked `performance_material_not_proof` on a dedicated artifact plane | **[SHIPPED]** | `internal/cachemeta/manifest_test.go:108-113` |
| Nine-axis exact-binding HIT gate (the invalidation set) | **[SHIPPED]** | `internal/cachemeta/manifest.go:107-116` |
| Content-addressed tool-result store that survives process death, re-screened on page-in | **[PARTIAL]** (per-session / private only) | `internal/recall/recall.go:262-281`, `:287-296`, `:298-301`, `:404-433` |
| `SourceDigest` names "the source text the KV was prefilled from" | **[SEAM-ONLY]** (field exists, no regen consumer) | `internal/cachemeta/manifest.go:23` (folded into binding digest `:60`, surfaced as label `:187`) |
| KV-spill transfer concept (the L2 tier) | **[SEAM-ONLY]** | `internal/cachemeta/kvtransfer_test.go` (`Direction: KVOffload`) |
| Cross-model verdict-layer share decision (`CanShare`) | **[SEAM-ONLY]** (decision fn shipped, not wired into HIT path) | `internal/polymodel/polymodel.go:543-549`; polymodel plan rung #534 |
| Durable L2 (KV spill) or L3 (text) tier *across process restarts* | **[GAP]** | radixkv is process-scoped; no serialization across restart |
| Backfill / warm / eager / blue-green / regen-on-rollout machinery | **[GAP]** | confirmed absent by grep — no `backfill`/`warm`/`eager`/`prefetch`/`blue-green` site |
| Fleet-shared cross-session / cross-agent text corpus | **[GAP]** | recall CAS is per-session; vDSO dedup is process-local |
| Two-class scheduler fencing backfill-prefill off live serving | **[GAP]** | `polymodel.Schedule` is strictly serial; only cost fn is `DecodeBandwidthBytes`; no `PrefillBandwidthBytes`, no occupancy model |
| Fleet-wide quarantined-digest denylist consulted before backfill | **[GAP]** | reScreen + revocation are process-local |
| Device-HAL (`backend != nil`) prefix reuse / backfill | **[GAP]** (CPU-session-only today) | tree disabled when `backend != nil` (`internal/agent/inkernel_planner.go:55,205,238`) |

## 8. The child map / sequencing

Ordered so each rung stands on a shipped one. None of these land in this doc, and none are
filed as issues yet (the one overlap with existing tracking is the polymodel plan's
cross-model-share rung #534, noted at R8).

- **R1 — give `SourceDigest` a consumer: a join key.** Make `SourceDigest` the bridge between
  recall's `cas.json` (text bytes) and cachemeta's set of `(binding-tuple → KV span)`
  materializations. Stands on the field (`manifest.go:23`) and the text store
  (`recall.go:262-281`). Smallest possible first move; today the field is dead.
- **R2 — durable L3: a fleet-export path for recall.** Promote the per-session private CAS to
  a shared, content-addressed corpus keyed by `(text-digest, TokenizerID)`. Stands on the
  death-surviving image (`recall.go:287-296`). Carries the privacy work: extend the
  `AccessPolicy` tenancy axis from KV manifests to the text corpus, and make IFC taint survive
  content-addressed dedup.
- **R3 — regen-from-text driver (lazy-on-miss across the process boundary).** On a cold tree,
  hydrate from L3: `bytes → re-tokenize → planner.Prefill`. Stands on the live regen path
  (`inkernel_planner.go:213-242`) and R2. This is the trajectory-replay engine re-pointed
  model→cache, reusing the recorded-run format if it carries text and not only KV.
- **R4 — eager hot-set backfill plus a canary policy.** Replay the N hottest L3 transcripts
  against the new model before or at cutover. Stands on R3 and `FAK_INKERNEL_RADIX_BUDGET` as
  the hot-set definition. Needs a fleet return-probability predictor, or it loses to lazy.
- **R5 — a two-class backfill scheduler.** A priority lane, quantum cap, or device partition
  that fences backfill-prefill off live work and yields SMs to live decode under an SLO.
  Stands on `polymodel.Schedule`, which needs a `PrefillBandwidthBytes` and an occupancy model
  added. **This is the rung that makes the "idle headroom" claim true rather than asserted.**
- **R6 — a cross-regime integrity oracle.** Where KV-byte hashing no longer holds (across
  hardware/tier/quant), the oracle hashes the *text* (`SourceDigest`), re-screens, and trusts
  deterministic prefill within the new regime, with a stated and tested semantic-equivalence
  bound (argmax-exact, `max|Δ| < tolerance`) promoted to a first-class migration verdict.
- **R7 — a fleet-wide quarantine authority.** A digest-keyed denylist in cachemeta consulted
  before any backfill-prefill lowering, layered on the existing re-screen and revocation.
  Stands on `recall.reScreen` (`recall.go:404-433`).
- **R8 (parallel) — wire `CanShare` into the cachemeta HIT path.** For compatible-family
  rollouts this skips backfill entirely. Stands on the shipped decision fn
  (`polymodel.go:543-549`); tracked as the polymodel plan's #534. Complementary to R3–R4.

## 9. Non-goals / honest fence

- **Not claiming byte-identical regen.** "Lossless" is regime-relative: same architecture,
  SIMD tier, quant, and kernel-pinning env vars. Across those the contract is distributional
  (argmax-exact plus sub-tolerance drift), and serving needs no more. Do not assert
  "indistinguishable from a kept cache."
- **Not claiming backfill is "~free."** It leans on a different bottleneck than live decode,
  but on one GPU it contends for SM occupancy and the weight-read HBM bus. The free-lunch
  framing holds only under R5's scheduler; until then the magnitude of contention is
  unquantified, and a large hot-set is a real sustained compute and bandwidth load.
- **Not claiming the migration apparatus exists.** There is no durable text or KV tier across
  restarts, no backfill or blue-green code, no fleet-shared corpus, and `SourceDigest` has no
  regen consumer. The whole rolling-migration story is a PLAN.
- **Not claiming "model-agnostic text."** The substrate is weights-agnostic but
  *tokenizer-bound* (keys are token-ids), and the stored transcript folds prior assistant
  turns that the old model sampled. It is faithful to a model-A-authored transcript, not a
  neutral source of truth — acceptable for cache correctness, but state it plainly.
- **Not claiming tool results "recur identically across sessions/agents."** Only the
  read-only, idempotent, entity-nameable, non-secret, non-revoked, world-version-current
  subset is reusable, and even the existing dedup (vDSO) is process-local. A universally
  identical, freely shareable corpus is a design motivation, not implemented behavior.
- **Device-backend migration is out of scope today.** The device-HAL path never engages the
  prefix tree, so backfill is a CPU-session story until the device path grows its own prefix
  cache.
- **Not weakening the ModelID barrier.** Text-regen sidesteps it by re-derivation; it does
  not relax the nine-axis HIT gate. (Worth a guard before durable text amplifies the blast
  radius: confirm the live radix reuse site re-checks ModelID on a same-process model switch.)

## 10. How to re-verify

```bash
rg -n 'SourceDigest|performance_material_not_proof' internal/cachemeta      # text-as-source field + the "not proof" stamp
rg -n 'claim.ModelID != manifest.ModelID' internal/cachemeta/manifest.go    # the nine-axis exact-binding HIT gate
rg -n 'SessionFromPrefix|Prefill\(ids\[matched:\]\)' internal/agent/inkernel_planner.go  # the live suffix-only regen path
rg -n 'func \(c \*KVCache\) (Clone|Evict)' internal/model/kv.go             # the bit-exact fork + rollback
rg -n 'cas.json|survives death|func \(s \*Session\) reScreen' internal/recall/recall.go  # the death-surviving, re-screened text store
rg -n 'a_naive|re-prefill the WHOLE' internal/swebench internal/webbench    # regen-vs-keep is the measured A/B turn-tax
rg -n 'func CanShare|PrefixDigest' internal/polymodel/polymodel.go          # the compatible-family share that skips backfill
rg -n 'backfill|blue.*green|prefetch'                                       # expect NO hits — the GAP is real
```

---

*Sibling axis:* `docs/serving/polymodel-prefill-share-plan.md` (host many, decode one). ·
*Reused engine:* the trajectory-replay program, re-pointed policy→model. · *Text store:*
`internal/recall`. · *Source-of-truth field awaiting a consumer:*
`cachemeta.KVManifest.SourceDigest`.
