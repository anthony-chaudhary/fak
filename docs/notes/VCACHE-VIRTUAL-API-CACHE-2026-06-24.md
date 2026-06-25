# vCache — A Virtual API Cache over Providers We Don't Control

**Date:** 2026-06-24
**Status:** concept / design note (pre-spec)
**Relation to tree:** active control plane proposed *on top of* `internal/cachemeta` (tier-1, passive) and generalizing the single-session prefix-coherence shaper (`cachemeta.ShapeGLMTurnSegmentWitnessed` + the `internal/agent/glm_coherence.go` message↔segment bridge) to a multi-request, multi-provider regime.

> **Audit provenance (2026-06-24).** Every `cachemeta` symbol named below was grounded against `internal/cachemeta` at HEAD, and the §5/§10 numbers were independently re-derived and checked against current public provider docs. Three corrections from that pass are folded in: (1) there is **no `CacheLane` type** in the tree — the single-session work it referred to is the `cachemeta` coherence layer + `glm_coherence.go` shaper, named precisely throughout; (2) the Anthropic **minimum cacheable prefix is per-model and currently 1024 tokens** for Opus 4.8 / Sonnet 4.6 / Sonnet 4.5 (4096 belongs to Haiku 4.5), so the old "4096 (Opus)" rows are corrected; (3) `RecommendLayout` is **advisory-only today** (it returns a recommendation, it does not apply it) — "enforce it" is therefore net-new vCache work, not a tweak to existing behavior. The economics (break-even, Zipf `s≈1.74`, chain-horizon ceilings, the 300× recall loss) all reproduced to the decimal.

---

## 0. The one-sentence idea

> We cannot tell a provider *"cache this."* So we make our caching unit (a `cachemeta` entry — call it a **vBlock**) map onto whatever caching the provider **will** do on its own, by controlling the **order, shape, and timing** of the requests we send — and we keep a **manifest** that lets us *rebuild and reorder* a series of requests to recall any unit on demand.

vCache is virtual memory for a remote KV cache we can observe but not address. Our **virtual page** is a vBlock; the **physical frame** is a warm prefix sitting in some provider backend's KV cache; the **page table** is the vCache manifest; **page-in** is replaying a prefix chain; **the MMU** is the request scheduler that arranges requests so the provider serves the prefix from its cache instead of re-prefilling it.

---

## 1. Why this is even a problem (the asymmetry)

A local engine (SGLang/vLLM/llama.cpp) lets `cachemeta` name an *exact* KV span (`KVManifest`, `ExactSpanTarget`) and evict it precisely. Providers give us **none of that addressability**:

| Capability | Local engine | Anthropic | OpenAI | Gemini | DeepSeek |
|---|---|---|---|---|---|
| "Cache this span" command | yes (write KV) | **yes** — `cache_control` breakpoints | no | **yes** — `CachedContent` w/ handle | no |
| Query "is X warm?" | yes | no (only `cache_read_input_tokens` *after* a call) | no (`cached_tokens` after) | partial (handle exists) | no (after) |
| Evict X | yes | no (TTL only) | no | yes (delete handle) | no |
| Pin / TTL control | yes | **partial** — 5m or 1h | no (auto idle) | yes (set TTL) | no |
| Address by content | yes (digest) | **no — by byte-prefix only** | by byte-prefix | by handle | by byte-prefix |

The whole field splits into two regimes, and vCache must serve both behind one abstraction:

- **Explicit regime** (Anthropic, Gemini-explicit): there *is* a "cache this" verb. vCache's job is to use it well (breakpoint placement, TTL choice, pre-warm) — the `cachemeta` coherence shaper (`ShapeGLMTurnSegmentWitnessed`) + `glm_coherence.go` already do the single-session, per-turn version (place/move one cache break when a witnessed span goes stale).
- **Implicit regime** (OpenAI, DeepSeek, Gemini-implicit, most OpenAI-compatible self-host): **no verb at all.** The provider caches the literal token *prefix* of whatever you happen to send, on whatever backend you happen to land. This is the hard case the goal is really about, and the rest of this note is mostly about it.

The unifying observation: **even the explicit providers are, underneath, doing prefix caching.** `cache_control` is just a hint about *where* to cut the prefix. So if vCache can drive the implicit regime by shaping prefixes, the explicit regime is the same machine with an extra, cheaper lever. **Assemble a correct prefix chain and default caching works for free** — that is the goal's closing line, and it's the load-bearing insight.

---

## 2. The vBlock and the manifest (what we map)

A **vBlock** is the unit of cacheable work. In the goal's example: 10,000 units × ~10 tokens. In practice a vBlock is a `cachemeta.Entry` whose identity already carries every axis that must match for provider reuse:

```
vBlock identity  (re-using cachemeta.ProviderCache / KVManifest axes)
  ├─ content digest         (SourceDigest / SpanDigest)
  ├─ ModelID                (cache is model-scoped — switching model = cold)
  ├─ TokenizerID            (tokenizer drift ⇒ different tokens ⇒ different prefix)
  ├─ SerializerID           (deterministic serialization hash — see §6)
  ├─ Vary axes              (Endpoint, ReasoningMode — SILENT cache-breakers, already in provider.go)
  └─ position in a prefix DAG  (NEW — the recall plan)
```

The **manifest** is the page table. For each vBlock it records:

1. **The canonical prefix path** — the exact ordered token sequence (or chain of parent vBlocks) whose concatenation reconstructs this unit's request prefix, byte-for-byte.
2. **Provider binding** — which provider/model/endpoint/shard-affinity-key this unit was last warmed against.
3. **Warmth belief** — last-warmed timestamp, observed `cached_tokens`, estimated TTL, a decaying probability that it is still resident (§7). This is a `cachemeta.Lifecycle` record pointed at `TierProvider`.

The manifest is the *only* durable state vCache owns. Like `cachemeta`, **it stores no payloads** — it stores *how to make the provider reproduce them.*

### The prefix DAG

Provider prefix caching is **linear**: the cache is on the literal token prefix, so cacheable work forms **chains** (or trees) where each node shares a common prefix with its parent.

```
ANCHOR  P0  (shared system prompt + retrieved docs + few-shot,  ≥ Mₘᵢₙ tokens)
  │
  ├── u1   (P0 ++ unit-1 query, 10 tok)
  ├── u2   (P0 ++ unit-2 query)         ← siblings: all share P0, differ in the 10-tok tail
  ├── …
  └── u_k

CHAIN form (when units genuinely depend on each other):
  P0 → P0+u1 → P0+u1+u2 → P0+u1+u2+u3 → …
  (cache prefix grows; recall of u_k needs u1..u_{k-1} warm)
```

Two shapes, two economics:
- **Star** (one anchor, many independent 10-tok suffixes): the *anchor* is the cache unit. Warm P0 once, pay 10,000 × (cheap anchor read + 10-tok fresh prefill). This is the dominant, easy win and exactly what providers reward.
- **Chain** (each unit builds on the last): the *growing prefix* is the cache unit. Recall of unit K requires the chain 1..K−1 to be warm. This is where ordering, TTL, and the 20-block lookback bite (§8).

**Design rule:** model the **anchor**, not the 10-token unit, as the cacheable thing. A lone 10-token unit can *never* be cached (it's ~100× below the minimum cacheable prefix — §4). Units are cheap suffixes that ride a warm anchor.

---

## 3. Architecture — the active loop above cachemeta

`cachemeta` is **passive**: `FromProviderCache` *records* a hit after the fact (`AdmissionDefer`, "provider_cache_telemetry"); `prefix_stability.go` *lints* a recorded session offline; `lifecycle.go` *models* state given known transitions. None of it *causes* a cache to warm.

vCache is the **closed loop** that turns those passive primitives into control:

```
                          ┌───────────────────────────── vCache control plane (NEW, tier ≥2) ──────────────────────────────┐
   work units ───────────►│  Canonicalizer ──► Planner ──► Scheduler ──► Warmer ──► [PROVIDER] ──► Telemetry ──► Governor   │
   (vBlocks)              │     (§6)            (§2 DAG)     (§9)         (§5)                       fold        (§7 belief)  │
                          │       │                │           │           │                         │            │         │
                          │       └── uses ────────┴───────────┴───────────┴─────────── reads ───────┘            │         │
                          └──────────────────────────────────┬──────────────────────────────────────────────────┘         │
                                                              ▼                                                             │
   ┌──────────────────────────── cachemeta (tier-1, passive, EXISTING) ─────────────────────────────────────────┐         │
   │  prefix_stability.Diverge/RecommendLayout │ ProviderCache/FromProviderCache │ Lifecycle/TierTTL │ KVManifest │◄────────┘
   └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| vCache component | Job | Built on (cachemeta) |
|---|---|---|
| **Canonicalizer** | Make the cacheable prefix byte-stable; hoist volatile content to the tail | `Diverge`, `LintPrefixLayout`, `RecommendLayout`, `SerializerID` |
| **Planner** | Build/maintain the prefix DAG; choose anchors; decide star vs chain | `KVManifest` identity axes |
| **Warmth model** | Probabilistic belief "is this prefix resident on the shard I'll hit?" | `Lifecycle` + `TierTTL` aimed at `TierProvider` |
| **Warmer** | Issue the warming request (the primitive — §5) | (new; provider adapter) |
| **Scheduler** | Topological recall order, send-one-then-fan, rate-limit budget | `AccessRatePerSec` for ranking |
| **Governor** | TTL policy: pin (heartbeat) vs lazy-rebuild vs evict | `Advance`/`Touch`/`MoveTo` |
| **Telemetry fold** | After every real call, update belief from `cached_tokens` | `FromProviderCache`, `SavingsSplit` |

Critically, vCache **never** lets a provider hit become a *trust* claim — it inherits `ProviderCacheVerdict` ("cost_latency_only"). A warm prefix is performance material, never proof a result may be re-served. (`provider_guard.go`'s double-count guard stays in force: provider savings are a distinct metric from local reuse.)

---

## 4. The warming primitive — how you "cache this" without a cache verb

### 4a. Explicit regime: `max_tokens: 0` pre-warm

Anthropic gives the clean primitive the goal gropes toward ("individual requests with token max 1"). A `max_tokens: 0` request **runs prefill** — writing the KV cache at your `cache_control` breakpoint — and returns immediately: `content: []`, `stop_reason: "max_tokens"`, **0 output tokens billed**, normal cache-write charge on `cache_creation_input_tokens`. It is strictly better than the old `max_tokens: 1` decode-a-token trick (no token to discard, intent unambiguous).

Rejected combinations to encode as guards: `stream: true`, thinking enabled, `output_config.format`, `tool_choice` of `tool`/`any`, and inside Batches. Breakpoint goes on the **last block shared with the real request** (the anchor), never on the placeholder user turn.

### 4b. Implicit regime: the decode-1 prefill

No verb exists, so we **manufacture a prefill**: send the anchor as the prompt with `max_tokens: 1` (the floor). The provider prefills the whole prompt to produce one token — and *that prefill is what populates its prefix cache*. We throw the one token away. This is the goal's "token max 1 for decode."

Caveats that become design rules (hardened by the red-team panel, §11):
- You **pay full input price** for the warm prefill. So a dedicated warm only pays off under the break-even gate (§10.1).
- Some backends only persist cache **after a successful generation** — `max_tokens:1` is safer than `0` here; verify per provider by probing (§7).
- The warm prompt's prefix must be **byte-identical** to the real request's prefix (§6) or you warm the *wrong* entry and believe you saved when you didn't.
- The cache is **not readable until the first response begins streaming** — so warm-then-immediately-fan-out **races**. The fix is the send-one-then-fan protocol (§8).

### 4c. Once a correct prefix chain exists, default caching just works

The most important consequence: after the Canonicalizer + Planner have produced a **byte-stable, correctly-ordered anchor**, you often don't need *any* dedicated warm at all. The **first natural request** warms the anchor; every subsequent sibling reads it. Dedicated warming is then only for (a) **latency** (cut first-request TTFT before a known burst) or (b) **heartbeat** (keep alive across an idle gap longer than the TTL). That is the whole point of "assemble a correct prefix chain and default caching can work too."

---

## 5. The quantitative core (the "what does N have to equal?" questions)

> First-pass derivations below; an independent derivation panel (workflow `vcache-design-foundations`) cross-checks every number — reconciliation in §10.

Provider constants used (Anthropic confirmed; others approximate):

| | read mult `r` | write mult `w` | min prefix `Mₘᵢₙ` | block `B` | TTL `T` |
|---|---|---|---|---|---|
| Anthropic 5m | 0.1× | 1.25× | **1024** (Opus 4.8 / Sonnet 4.6 / Sonnet 4.5) · 4096 (Haiku 4.5) | breakpoint | 5 min |
| Anthropic 1h | 0.1× | 2.0× | same | breakpoint | 60 min |
| OpenAI | ~0.5× | ~1.0× (auto) | ~1024 | 128 | ~5–10 min (≤1 h hard) |

> **Correction (audit):** the minimum cacheable prefix is published **per model, not per family.** As of 2026-06-24 Opus 4.8, Sonnet 4.6, and Sonnet 4.5 are all **1024**; the **4096** tier is **Haiku 4.5**, not Opus. The earlier "4096 (Opus) / 2048 (Sonnet 4.6)" was wrong and over-stated the uncacheable-prefix wall by ~4× for the flagship models. Re-probe per model on each bump (§7) — these are version-dependent.

### 5.1 Break-even: when is a *dedicated* warm worth it?

A dedicated warm is an **extra** request you'd otherwise not send. Let the warm cost `c_w·P` and each later read cost `r·P`; the no-warm alternative is `k` misses at `1·P` each.

- **Explicit provider, dedicated `max_tokens:0` warm** costs a write `w·P`. But note: on these providers the *first real request* warms anyway. So a dedicated warm does **not** save tokens — it adds a write. Its payoff is **latency**, or surviving a TTL gap. The *natural* cache (first real request writes, rest read) breaks even at:

  `N > (w − r)/(1 − r)` → **5 min: N ≥ 2; 1 h: N ≥ 3** (matches Anthropic's own statement). ✓

- **Implicit provider, dedicated decode-1 warm** is a full prefill `1·P` you wouldn't otherwise pay. It converts `k` future misses (`1·P` each) into reads (`r·P`):

  `1 + k·r < k  ⟹  k > 1/(1 − r)`
  - OpenAI (`r≈0.5`): **k > 2 → k ≥ 3 reuses-before-TTL.**
  - Anthropic-discount (`r=0.1`): **k > 1.11 → k ≥ 2.**

**vCache gate:** only spend a dedicated warm on a prefix whose **expected reuse before TTL exceeds break-even** (≥3 for auto/OpenAI, ≥2 for Anthropic-discount). Otherwise let the first natural request warm it.

### 5.2 Coverage: "maybe 7 units is enough for 85%"

Model prefix popularity as **Zipf(s)** over `V` distinct prefixes. Volume covered by warming the top-`N` is the generalized-harmonic ratio:

`coverage(N) = H(N,s) / H(V,s),   H(n,s) = Σ_{i=1..n} i^(−s)`

Backing out the goal's intuition — solve `H(7,s)/H(V,s) = 0.85`:

| s (concentration) | top-7 coverage (V≈10³) | interpretation |
|---|---|---|
| 1.0 (classic Zipf) | ~0.35 | flat-ish → 7 anchors nowhere near enough |
| 1.5 | ~0.72 | concentrated |
| **~1.74** | **~0.85** | **the goal's "7 → 85%" world** (panel-exact: s\*=1.74, ≈V-independent over 10³–5×10³) |
| 2.0 | ~0.92 | very concentrated |

So **"7 units ≈ 85%" presumes a steep workload (s ≈ 1.8)** — a few dominant anchors (one system prompt, a handful of retrieved-doc bundles). That is realistic for agent fleets but **not universal**. The actionable law:

**Measure `s` (the concentration) before trusting vCache.** Rank vBlocks by `frequency × size × reuse-density` and warm only the **head** — the *cache working set*. On a flat workload (`s ≈ 1`) the head is huge and vCache barely helps; on a steep one (`s ≳ 1.5`) a tiny pinned set captures most of the volume. This is `AccessRatePerSec`-driven ranking applied to the provider tier.

### 5.3 Chain horizon: tiny units and the minimum-prefix wall

With unit size `u = 10` tokens and provider minimum cacheable prefix `Mₘᵢₙ`:

`cache-horizon C₀ = ⌈Mₘᵢₙ / u⌉` units before **any** caching is possible.

| `Mₘᵢₙ` | C₀ (units of 10 tok) | meaning |
|---|---|---|
| **1024** (Opus 4.8 / Sonnet 4.6 / 4.5) | 103 | first ~103 chained units are *uncacheable* — the common case |
| 2048 | 205 | (no current Anthropic model; OpenAI-family / generic) |
| **4096** (Haiku 4.5) | 410 | |

With block quantization `B = 128` (OpenAI), the cache then grows in `B/u = 12.8`-unit steps and the **un-cached tail can be up to `B/u ≈ 12` units**. Consequence and rule:

**A 10-token unit cannot be cached alone — ever.** It must ride a shared anchor of `≥ Mₘᵢₙ` tokens, or be aggregated with peers into a `≥ Mₘᵢₙ` super-block, and chains should be **cut on B-aligned boundaries** so the tail isn't stranded. The goal's "n=7" only makes sense at the **anchor** granularity, not the 10-token-unit granularity.

### 5.4 TTL heartbeat vs lazy rebuild

Keeping a prefix warm across window `W` costs `⌈W/T⌉` heartbeat prefills (`P` each). Lazy rebuild pays the recall cost `R` per miss; misses ≈ arrivals after an idle gap > `T`. Let `λ` = arrival rate for the prefix.

- **`λT ≥ 1`** (≥1 request per TTL window): natural traffic keeps it warm. **Do nothing** — don't pin, don't pre-warm.
- **`λT < 1`:** pin only if continuous heartbeat (`P` per `T`) beats on-demand rebuild (`R` per miss):

  `pin iff  λ·R > P/T  ⟺  λ > P/(R·T)`

  When recall is cheap (`R ≈ P`, a star anchor) this never beats lazy → **lazy rebuild.** When recall is expensive (`R ≫ P`, a deep chain that must be replayed) the tip is worth pinning even at sparse arrival → **heartbeat the tip.**

**Governor policy:** `hot (λT≥1) → ride natural traffic`; `valuable-but-sparse with expensive recall → heartbeat-pin the tip`; `everything else → lazy rebuild`; `cold → evict from manifest`. The 1h-TTL lever trades a `2×` write for `12×` fewer heartbeats — prefer it for bursty traffic with long idle gaps. *(§10 reframes the pin/lazy cutoff precisely: on pure dollars lazy weakly dominates; pin only when latency value L or rate-limit shadow price μ tip it — `λT > ln((W5+μ+L)/L) ≈ 1`.)*

### 5.5 Rate-limit feasibility: how big a warm set is sustainable

Tier gives `R` RPM and `X` TPM; real traffic uses `R_real`, `X_real`. Each warm of prefix `P` costs 1 request and `P` input tokens.

`warms/min = min(R − R_real, (X − X_real)/P)`
`sustainable warm-set = warms/min × T(min)`
`RPM/TPM crossover at P* = (X − X_real)/(R − R_real)`

Worked example, half-utilized tier `R=4000, X=400000`, so `R_avail=2000, X_avail=200000`, `P*=100` tokens. Any realistic anchor (`P ≥ 1024`) is therefore **TPM-bound**:

| anchor `P` | warms/min | sustainable warm anchors (×5 min) |
|---|---|---|
| 4,096 | ~48 | ~244 |
| 16,384 | ~12 | ~61 |
| 65,536 | ~3 | ~15 |

**Rule:** budget warms inside headroom, prioritize by the §5.2 ranking, and **degrade by warming fewer anchors (lower hit rate), never by 429-ing.** "Reasonable numbers" = on a mid-tier key you can keep a few hundred 4k anchors hot, or a couple dozen 64k contexts — pick anchor size accordingly.

---

## 6. Canonicalization — the load-bearing discipline

Everything fails if the cacheable prefix is not byte-stable. `prefix_stability.go` already gives the offline linter; vCache makes it a **pre-flight gate** on every manifest entry before it is trusted as warm.

Silent invalidators to canonicalize away (the prefix must be deterministic to the byte):
- **Render order** is `tools → system → messages`; a breakpoint on the last system block caches tools+system together.
- **No volatile content in the prefix** — timestamps, UUIDs, request IDs, per-session/user IDs. `RecommendLayout` *computes* the hoist of `SegVolatile` to the tail but is **advisory-only today** (it returns a `LayoutRecommendation`; nothing applies it). vCache's net-new job is to **apply** that recommendation as a pre-flight rewrite, not just report it.
- **Deterministic serialization** — sorted JSON keys, stable tool ordering, no set iteration. This is what `SerializerID` pins.
- **Vary axes are part of identity** — `Endpoint` and `ReasoningMode` (and `model`) silently break the cache; `provider.go` already folds them into the digest, so a mode switch reads as a *distinct* warm, not an invisible miss.
- **Tokenizer drift** — same text → different tokens across model versions → different prefix. The manifest entry is bound to `TokenizerID`; a model bump cold-starts the whole warm set.

**The dangerous failure** the Canonicalizer exists to prevent: *manifest says HIT, provider says MISS.* You pay full price and your savings accounting believes you won. The defense is **verify-then-trust**: after every real call, fold `cache_read_input_tokens`; if a believed-warm entry reads 0, mark the belief broken and diff the rendered prefix bytes to find the invalidator (`Diverge` / `FirstDivergeTokenOffset` localizes it — `ProviderCache.FirstDivergeAt` is the field they populate).

---

## 7. The warmth belief model — control without observability

On implicit providers you cannot ask "is X warm?"; you only learn warmth **after** spending a real request (`cached_tokens`). vCache therefore runs an **open-loop estimator with feedback**, not a known-state machine. It reuses `cachemeta.Lifecycle` but reinterprets the transitions as *beliefs*:

- Each manifest entry carries a `Lifecycle` at `TierProvider` with `TierTTL[TierProvider] = provider TTL`.
- `Advance` decays belief by TTL (resident → expiring → expired) **on the clock**, since we can't see eviction.
- `Touch` (a real call that *did* read cache) **revives** belief to resident and resets the TTL clock — this is the only ground truth we get.
- A believed-warm call that reads **0** demotes belief immediately (broken canonicalization, eviction, or a shard miss — §8).

**Calibration by probing.** Provider TTL/min-prefix/discount are version-dependent and undocumented for some providers. vCache **probes**: send a known anchor, wait, re-send, read `cached_tokens`, and fit `T`, `Mₘᵢₙ`, and `r` per (provider, model, endpoint). The result feeds `providerTTLMillis` and the §5 constants instead of hard-coding them.

---

## 8. Recall — rebuild and reorder a series of requests

To recall unit `K` from a chain, vCache reconstructs the request from the manifest and **replays the prefix chain in dependency order** so the provider serves `P0..K-1` from cache and freshly prefills only the last `u` tokens. The hazards (red-team §11) and the protocol:

- **Ordering race.** Firing the whole chain in parallel lets dependents overtake the prefix that isn't cached yet (cache readable only after the first response *starts streaming*). → **Send-one-then-fan:** send the anchor (or chain tip), await the **first streamed token**, *then* fan out the dependents — they read the cache the first one just wrote.
- **20-block lookback.** Anthropic's breakpoint walks back ≤20 content blocks; a turn that adds >20 blocks (agentic tool spam) silently misses. → place an **intermediate breakpoint every ~15 blocks** in long chains.
- **Chain longer than the provider keeps warm.** Recall cost grows with depth; if replaying the chain costs ≥ a fresh full prefill, recall is **net-negative.** → a **cost gate**: refuse a chain-rebuild when `replay_cost ≥ fresh_prefill_cost`; just send the unit cold.
- **Partial warmth.** Only a prefix of the chain survived. `Diverge` / `FirstDivergeTokenOffset` tells you exactly how much; replay only from the first cold node.

Recall is therefore a **topological-order scheduling problem over the DAG**, gated by the §5.4 (pin vs rebuild) and §10.1 (break-even) economics, with the send-one-then-fan barrier between every fan level.

---

## 9. Cross-shard routing & multi-tenancy (the leakiest assumption)

Providers run many backends; the KV cache is **per-server**. A prefix warm on shard A is cold on shard B, and your next request may route to B. Mitigations and limits:

- **Affinity hints:** OpenAI `prompt_cache_key`, Anthropic prefix-hash routing — set them **consistently across a whole chain** so chained requests land on the same warm shard. Partial: load-balancer rehash under autoscale still strands warm sets.
- **Concurrency cap per shard-key** to avoid a thundering herd onto one hot backend.
- **Treat the warm set as best-effort replicated** — a backend recycle silently evaporates it; belief decay (§7) absorbs this.
- **Security/privacy:** shared provider caches are a **timing side-channel** — cache-hit latency reveals whether *some* prefix is already warm (possibly another tenant's). And warming sensitive content has **data-retention** implications: a warmed prefix can carry a minimum-retention floor and may be unavailable to zero-data-retention (ZDR) orgs (some models/features impose a multi-day retention window for cached or stored context — **verify the exact policy per model before relying on it; treat the retention floor as a per-model constant to probe, not a documented constant**). → **never warm secrets**, treat hit-timing as observable, and keep warm-content under the same access policy `KVManifest.AccessPolicy` already demands.

---

## 10. Reconciliation with the derivation panel

Five independent derivations (workflow `vcache-design-foundations`, Opus 4.8 × 10 agents, 850k tokens) cross-check §5. Result: **§5.1, §5.3, §5.5 confirmed to the decimal; §5.2 sharpened; §5.4 reframed.**

| §5 law | verdict | correction / sharpening |
|---|---|---|
| **5.1 break-even** | ✅ confirmed | `k > w/(1−r)` → 5 min **k≥2**, 1 h **k≥3** (= Anthropic's own `N>(w−r)/(1−r)`). Dedicated warm `k>1/(1−r)`. **New:** on auto-cache providers the first *real* request warms for **free** (no write line-item), so dedicated-warm break-even is `k*=0` and savings are `P·k·(1−r)` from k=1 — **never spend a decode-1 warm on OpenAI/DeepSeek; just order requests so the first real one warms.** |
| **5.2 coverage** | ✏️ sharpened | "7→85%" implies **s\*=1.74** exactly (I estimated ~1.8), and it's ≈V-independent over 10³–5×10³. At s=1, top-7-of-1000 = **34.6%**; at s=0.8, **19.7%**. To cover 90%: s=1.5 needs N=**51** (0.5% of V=10⁴) but s=0.8 needs N=**6381** (64% of V) — a **125× swing from skew alone.** **New corollary:** when s≤1 the scheme is structurally defeated; the right move is not to warm the tail but to **manufacture skew** — canonicalize/aggregate so popularity *concentrates* onto fewer anchors. vCache can *create* the distribution it exploits. |
| **5.3 chain horizon** | ✅ confirmed (labels corrected) | `C₀=⌈M/u⌉` = **103 / 205 / 410** units (u=10, M=1024/2048/4096). The arithmetic is exact; the **model→M binding was wrong** and is fixed above — the common Anthropic case is M=1024 → **C₀=103** (Opus 4.8 / Sonnet 4.6 / 4.5), and 4096→410 is **Haiku 4.5**, not Opus. Tail ≤ B−1 = **127 tokens (~13 units)** stranded at any instant; eligibility is a **cliff at C₀** (0% → 99.4%) then →99.9%. Append-only is mandatory (any mid-chain byte change re-writes all downstream). |
| **5.4 TTL** | 🔁 reframed | My "pin iff λR > P/T" is the *expensive-recall* special case. Canonical result: **on pure cache dollars, lazy rebuild *weakly dominates for all λ*** (gap = `W5·e^{−λT}·P` → 0 only as λT→∞). Pinning is justified **only** by latency value `L` (cold-TTFT off the critical path) or rate-limit shadow price `μ`: **pin iff `λT > ln((W5+μ+L)/L)`** ≈ **0.81 → ~1 req/TTL** (L=1, μ=0), rising to ~1.2–1.5 under rate pressure. And **1 h opt-in is 7.5× cheaper to *hold* warm** than 5 min+heartbeat (2·P vs 15·P per hour) for W≥10 min prefixes reused ≥3×/h. |
| **5.5 rate-limit** | ✅ confirmed | `N* = min(R−R_real, (X−X_real)/P)·T`; crossover `P* = (X−X_real)/(R−R_real) ≈ 100 tok` on a 4000/400k tier → **every realistic anchor is TPM-bound.** P=4k → **500** anchors (5 min) / **6000** (1 h); P=64k → **31 / 375**. 1 h TTL buys 12× the set for the same headroom. Below P\*, you're RPM-starved → pack tiny units behind one shared prefix. |

**Net:** the goal's instinct is right *at the anchor granularity* — a steep workload (s≈1.74) lets ~7 anchors cover ~85%, each anchor warmed once and read cheaply — but every number lives or dies on **measuring s and the per-model minimum**, and on **billing at the uncached price until a hit is telemetry-confirmed** (§11).

---

## 11. Failure modes & design rules (adversarial)

Five skeptics each tried to break one mechanism. The findings collapse into **four cross-cutting laws** plus one **headline correction to the goal's own recall idea.**

### 11.0 The headline correction — recall-by-rebuild is the *wrong* tool for tiny-unit recall

The goal imagines recalling a unit by *"rebuilding and reordering a series of requests."* The economics say: **for a single ~10-token unit pulled from a long cold/warm chain, this almost always loses.**

> Replaying a 30 k-token prefix to rebuild one 10-token unit costs `0.1×·30 000 = 3000` token-equivalents of *read* to avoid `1.0×·10 = 10` token-equivalents of *fresh prefill* — a **300× loss.** Recall-by-rebuild is net-negative whenever `prefix_tokens·r > unit_tokens`, i.e. **essentially always for the 10-token target.**

So the **cost gate refuses almost every single-unit chain rebuild — and that is correct.** Recall-by-rebuild only wins in the **opposite** regime: a **large warm prefix amortized across *many* sibling units recalled *together* while the entry is still hot** — i.e. the **star/anchor + fan-out** shape, not the deep-chain + single-unit-recall shape. The reorder/rebuild machinery is real but **gated off by default** and reserved for amortized fan-out. The 80% win is §13-M2 (anchors), not chains.

### 11.1 Law A — Warmth is a *belief*, never a boolean; correctness never depends on it

The lethal failure is not a cache *miss* (cheap, visible) — it's **"manifest says HIT, provider says MISS."** A `datetime.now()` in the prefix makes every recall a cache *write*; the manifest still reports the unit warm, so **you book 90% savings while paying 1250%.**

- **Rule A1 — verify-then-trust.** A vBlock is "warm" only when `cache_creation_input_tokens>0` (explicit) or a real request reports `cached_tokens>0` (auto). A 200 response *never* marks it warm. Reconcile **every** call; demote + alarm + byte-diff on any believed-warm entry that reads `cache_read=0`.
- **Rule A2 — correctness never depends on warmth.** The full prefix that reconstructs a vBlock must **always be re-sendable**; a hit is *only ever* a cost/latency win. **Eliding resent context because "the provider has it" is banned** — eviction is invisible and unguaranteed, so eliding converts a cost bug into **silent data corruption** (the model answers over a truncated prefix). This single rule downgrades the one *fatal* failure to merely *expensive.*
- **Rule A3 — budget at the uncached price by default;** cache savings are a *realized rebate* from confirmed telemetry, never a pre-credited plan or SLO. Robust to every false-warm without changing behavior.

### 11.2 Law B — Fingerprint the *wire bytes*, scoped by the full identity tuple

The whole scheme rests on byte-identical prefixes, and the manifest must key on what's *actually sent*, not the logical object.

- **Rule B1 — manifest key = `hash(exact serialized prefix bytes in render order tools→system→messages, up to each breakpoint)`**, *never* a hash of the pre-serialization data structure.
- **Rule B2 — scope the key by `(model_id, tokenizer_epoch, tool_set_hash, breakpoint_layout, ttl, provider_surface)`.** Any differing element is a *hard* miss. This is what catches the **fatal** silent killers: a model switch (incl. **refusal-fallback** Fable 5→Opus 4.8 mid-request), a tool-set delta (tools at position 0 nuke everything), tokenizer drift (same text crosses the per-model min-prefix line — 1024 tok on current Opus/Sonnet — one way, not the other), and cross-surface routing (first-party warm invisible to a Bedrock recall; auto-caching unsupported there).
- **Rule B3 — canonical JSON everywhere** (sorted keys, fixed separators, pinned escaping, NFC, LF, no BOM) and **replay assistant/tool turns as the raw bytes the model emitted** — never `json.loads→json.dumps` round-trip (kills the Unicode/escaping/trailing-newline class).
- **Rule B4 — freeze the prefix:** all volatile content (timestamps, request/trace/session/tenant IDs, conditional/A-B sections) goes *after* the last breakpoint, or into a post-history `role:"system"` message (Opus 4.8). `cachemeta.RecommendLayout` already computes the hoist — **enforce it, don't just report it.**
- **Rule B5 — size in real tokens on the *recall* model** (`count_tokens`), not bytes/`tiktoken`; mark cacheable only above that model's min with margin.

### 11.3 Law C — The warming primitive has an irreducible race and an observability gap

- **Rule C1 — `max_tokens:0` only on Anthropic, breakpoint on the last *shared* block** (system/tools), never top-level auto-cache, never on the placeholder user turn (top-level keys the cache to the placeholder → the real request never matches). **Fatal foot-gun.** Forbidden with `stream`, thinking, `output_config.format`, `tool_choice tool/any`, Batches → for those, fall back to a real decode-1 warm; **Batches gets no explicit warm.**
- **Rule C2 — send-one-then-fan, gated on the first *streamed content delta*** (not HTTP 200, not `message_start`). Release the N−1 dependents only after the write is observably readable. **Note the tension:** this needs `stream:true`, which is *incompatible* with the `max_tokens:0` pre-warm — **choose one path per provider.**
- **Rule C3 — intermediate breakpoints every ~15 blocks** to stay inside the 20-block lookback, but **capped at 4** per request (Anthropic). A chain needing >4 anchors (>~60 blocks) **can't** place them all → the fix is **aggregation** (collapse tiny units into one ≥-min parent), not deeper chaining. On auto-cache providers there is *no* breakpoint lever at all.

### 11.4 Law D — On implicit providers you run open-loop; calibrate, don't assume; and routing is a roulette

- **Rule D1 — per-provider models, not one abstraction.** Anthropic (explicit, 4 breakpoints, `r=0.1`, 2-req break-even) and OpenAI (implicit, 128-tok-quantized, `r≈0.5`, break-even further out) get *different* break-even, recall, and routing logic.
- **Rule D2 — calibrate by probing, offline.** Empirically measure each provider's real min-prefix, increment, discount `r`, and TTL (replay at 30 s/2 m/5 m/10 m/30 m/1 h) and cross-shard hit rate (bursts ±`prompt_cache_key`). Re-run on a cadence; **the doc numbers are a starting hypothesis, not ground truth.** Charge probe traffic against an LRU budget so **measuring warmth doesn't evict what you believe warm** (observer-perturbs-state).
- **Rule D3 — affinity is a *bias*, never a guarantee; namespace it per-tenant per-chain.** Set `prompt_cache_key = hash(tenant‖chain)` consistently across warm + every recall; keep the Anthropic prefix byte-identical + append-only. **Autoscale rehash invalidates the whole warm-set at once** (and affinity makes that *correlated* across all keys behind one shard) — detect via correlated `p(hit)` collapse and **re-warm, don't keep issuing cheap recalls that now all miss**; cap warming-burst rate so vCache doesn't **self-trigger** the rehash that invalidates its own set. Pin egress **region** into the key.
- **Rule D4 — never warm secrets/PII/regulated content into a provider prefix cache.** Two reasons: **retention** (the secret sits in KV for at least the TTL, replicated opaquely across shards, outside your deletion control — and some models/features impose a **minimum-retention floor** that can exceed the TTL and that **ZDR orgs cannot opt out of**; treat that floor as a per-model policy to verify, not a fixed number) and the **cross-tenant timing side-channel** (cache-hit latency is a membership oracle: "has anyone recently processed string X?"). Classify every prefix; secrets take the no-cache or explicit-cache-with-deletion path; no secret byte precedes a breakpoint; add jitter if vCache ever reads hit-timing itself.
- **Rule D5 — prefer explicit caching where it exists.** Gemini `CachedContent` (store-once + handle + caller TTL + a *deletion primitive*) escapes shard roulette entirely — route long-lived or sensitive vBlocks there by preference.

### 11.5 Residual risk (honest floor)

Even with every rule: on auto-cache providers vCache **cannot** achieve closed-loop control — the only state channel is one rounded integer (`cached_tokens`) reported *after* a paid request, against a distributed, multi-tenant, nondeterministic plant whose policy the provider changes silently. So a **non-zero, time-varying false-warm miss rate is structural**, indistinguishable from shard-miss / eviction / below-min without per-call reconciliation; calibration is least accurate exactly under high load (when caching matters most); and **invisible regret** from false-*cold* beliefs (savings offered but never claimed) is unmeasurable by construction. The cross-tenant timing channel is a **provider** property vCache can refuse to amplify but cannot close. **Design posture: be *correct and cheaper-in-expectation*, never *dependent on a hit.*** Keep the cold path always correct and always affordable; the moment warmth gates correctness or hard-commits an SLO, one eviction turns a probabilistic hint into a confident lie.

---

## 12. What's new vs what exists (the build boundary)

**Reuse unchanged:** `KVManifest`/`Entry` identity; `prefix_stability.{Diverge,LintPrefixLayout,RecommendLayout,AnalyzeStability}`; `ProviderCache`/`FromProviderCache`/`ProviderCacheVerdict`; `lifecycle.{Lifecycle,TierTTL,Advance,Touch}`; `SavingsSplit` double-count guard; `AccessPolicy` access control.

**Net-new (the vCache plane, tier ≥2):**
1. **Manifest with a prefix DAG** — the recall plan (parent chain per vBlock), not just identity.
2. **Warmer** — the `max_tokens:0` / decode-1 request issuer + per-provider capability adapter (explicit vs implicit).
3. **Warmth belief estimator** — `Lifecycle` at `TierProvider` driven by `cached_tokens` feedback + TTL decay + probing calibration.
4. **Scheduler** — rate-limit warm-budgeting, topological recall ordering, send-one-then-fan barrier.
5. **Governor** — pin/lazy/evict policy from §5.4 + §5.5.
6. **Canonicalizer as a gate** — enforce `RecommendLayout`, not just report it.

**Generalizes the single-session coherence shaper:** today's `cachemeta.ShapeGLMTurnSegmentWitnessed` + `glm_coherence.go` (`SegmentsFromMessages` / `ApplyBreakToMessages` / `GLMCoherenceShaper`) is the single-session, intra-request prefix-break placer — it shapes one turn's messages so a stale span breaks the provider cache exactly where it must. vCache is the multi-request, multi-provider, TTL-governed superset; the existing `SegmentKind` classifier (`SegStable` / `SegToolSchema` / `SegVolatile` / `SegToolResult` / `SegMessage` / `SegSealed`) and the cache-stability gate become vCache's per-anchor canonical form.

### Prior art in the tree & related issues

vCache is the active control plane over a passive substrate that mostly already exists. It does **not** duplicate these — it consumes them:

- **#218 `feat(gateway): Prompt Caching Features [F-002]`** — the existing prompt-caching umbrella. vCache is the *active-loop* generalization; #218's wire-level `cache_control` forwarding (`gateway/messages.go`) is the surface vCache's Warmer drives.
- **#425 `cache: vDSO first-class cachemeta event emission + per-tool witness adapters`** — the telemetry-fold dependency. vCache's warmth-belief estimator (§7) needs `cached_tokens` lifted as a first-class `cachemeta` event; M1 builds on #425.
- **#41 `feat(serving): per-worker prefix-residency index + cache-aware power-of-two routing`** — the *local-engine* analog of §9 cross-shard routing. vCache's affinity-key routing is the provider-tier sibling; share the residency-index shape.
- **#201 / #204 / #203 (docs)** — open doc bugs that assert prompt-cache TTL provider-neutrally and assert window thresholds without derivation. §5's per-provider constants table + §5.4 derivation are the citable answer; closing those should reference this note.

---

## 13. Phased plan

1. **M1 — Observe & calibrate.** Probe each provider for `T`, `Mₘᵢₙ`, `r`; build the warmth-belief estimator on top of `FromProviderCache`. No warming yet — just predict hits and measure prediction error.
2. **M2 — Star anchors.** Canonicalizer-gated anchors + first-natural-request warming + sibling fan-out. The 80% win, lowest risk. Enforce `RecommendLayout`.
3. **M3 — Dedicated warming.** `max_tokens:0` (explicit) / decode-1 (implicit) under the §10.1 break-even gate; send-one-then-fan.
4. **M4 — Chains & recall (gated OFF by default — see §11.0).** Prefix DAG, topological replay, 20-block intermediate breakpoints, cost-gated rebuild. The cost gate refuses single-unit chain rebuilds (≈always net-negative for 10-token units); enable rebuild *only* for a large warm prefix amortized across many siblings recalled together while hot. For most of the imagined "rebuild/reorder" workload the correct action is **fresh prefill**, not rebuild.
5. **M5 — Governor.** Heartbeat-pin vs lazy-rebuild vs evict; rate-limit warm-budget scheduler; affinity-key routing.

## 14. Open questions

- How stable is per-provider shard affinity in practice? (Determines whether chains are viable or only stars.)
- Does the implicit-regime decode-1 reliably populate cache on OpenAI/DeepSeek, or only after real generation? (Probe in M1.)
- Tokenizer-drift blast radius on model bumps — can we re-warm incrementally or is it always a cold restart?
- Is there a safe, provider-agnostic way to *detect* eviction earlier than the next miss? (Timing side-channel is observable but ethically/operationally fraught.)
