# INNOVATIONS INDEX — fak's innovations, concepts, and learnings, in one map

The durable, refreshable catalog of *what fak invented or assembled* — distinct from
the doc map ([`llms.txt`](../llms.txt)), the repo map ([`INDEX.md`](../INDEX.md)), the
naming disambiguators ([`docs/glossary.md`](glossary.md) / [`docs/fak/concept-glossary.md`](fak/concept-glossary.md)),
and the per-capability claim ledger ([`CLAIMS.md`](../CLAIMS.md)). Those answer
*where is the doc / file / name / claim*. This answers **what is the innovation, what
general concept does it embody, where does it live, is it shipped, and has it been
generalized for reuse.**

> fak's own honesty leads here: a 29-claim prior-art audit scored **0/29 novel**
> ([`CLAIMS.md`](../CLAIMS.md)). The contribution is the **assembly** — one fused,
> fail-closed, witness-gated kernel where the tool call is an in-process syscall —
> and **one invariant carried at every scale**. Read this index with that frame: the
> parts are old; the wiring and the invariant are the thing.

Ledger snapshot (regenerate from [`claims-salience-register`](claims-salience-register.md)):
**160 claims — 143 `[SHIPPED]` · 7 `[SIMULATED]` · 10 `[STUB]`.** Status tags below
are drawn from that ledger; `MIXED` means a package spans rungs.

---

## Part 1 — The concept spine

Everything in the catalog hangs off one structure and one invariant.

**The invariant (one sentence).** *A decision no participant can move by narrating a
number* — evidence the claimant did not author is the only admissible truth, a claim
is untrusted until witnessed, and a refusal carries a token from a closed set.

**The grid (two axes, not one ladder)** — from
[`engineering-is-building-loops.md`](explainers/engineering-is-building-loops.md):

- **Vertical = scale** (how much of the stack is one address space): tool-call → turn
  → session → fleet → RSI, and below the tool call: decode → forward → KV cache →
  compute HAL → (borrowed) hardware. fak owns from the KV/decode loop up through RSI
  in one in-process kernel.
- **Orthogonal = invariants that recur at every scale**: trust/witness ·
  cost/economy · memory/durability · observability/feedback · human-governance. The
  distinctive crossing point: *most scales, same invariant, one kernel.*

**Two structural shapes the spine proves** (the patterns every good primitive obeys):

1. **The verification ladder** — smallest-sufficient-rung adjudication, escalate only
   on `INDETERMINATE` ([`verification-ladder-doctrine.md`](notes/verification-ladder-doctrine.md)).
2. **The two lenses** — the same primitive is a security control *and* an
   optimization, on the same code path ([`EXPLAINER-trust-floor-two-lenses`](notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md)).
   This is why "safe and fast for the same reason" is structurally possible.

Companion decompositions: [`EXPLAINER-what-is-an-agent`](notes/EXPLAINER-what-is-an-agent-2026-06-24.md)
(the *parts* of an agent) · [`cross-platform-spine`](explainers/cross-platform-spine.md)
(the *deployment* axis) · [`hardware-portability`](explainers/hardware-portability.md)
(the *hardware-depth* axis).

---

## Part 2 — Innovation catalog by subsystem family

Curated to the load-bearing innovations. The full leaf roster is in
[`dos.toml`](../dos.toml) `[lanes.trees]`; per-claim status is in
[`CLAIMS.md`](../CLAIMS.md). `Home` is the `internal/<pkg>` (or doc) it lives in.

### A. Safety / kernel / adjudication

| Innovation | Concept it embodies | Home | Status |
|---|---|---|---|
| Tool call as in-process syscall (`Submit`/`Reap`) | model proposes, kernel disposes — adjudicate before any engine/network | `kernel`, `abi` | SHIPPED |
| Default-deny reference monitor, 12-reason closed vocabulary | provable refusal only; deny never reaches the engine | `adjudicator` | SHIPPED |
| Deny-as-value with disposition routing | a refusal is a first-class, routable value (RETRYABLE/WAIT/ESCALATE/TERMINAL) | `adjudicator`, `abi` | SHIPPED |
| Runtime JSON policy manifest (version-tagged) | configure the floor by editing one reviewable file, never forking the kernel | `policy` | SHIPPED |
| IFC taint + sink-gating (StampGate/ScopeCeiling/SinkGate) | source-stamped taint refuses a tainted→sink flow before dispatch | `ifc`, `engine` | SHIPPED |
| Egress floor / secret + norm screens | result-side admit gate; evasion catch 0→20/24, 0 new FPs | `normgate`, `egressfloor` | SHIPPED |
| Tool vDSO (3-tier local fast path) | answer a repeated read-only call locally, no engine round-trip | `vdso` | MIXED (real-trace hit-rate `[SIMULATED]`) |
| Pre-flight ladder + grammar rung | graduated parse→schema verification before admit | `preflight`, `grammar` | MIXED (rung-2/3 + schema→mask `[STUB]`) |
| Architest layered-DAG floor + boundary lint | a feature is a leaf, not a core edit; the ABI is additive-only | `architest`, `boundarylint` | SHIPPED |
| Capability-floor on every inter-agent message | a per-message floor (`gateSend`/`gateRecv`), not just tool calls | `a2achan` | SHIPPED |
| De-obfuscating canonicalizer | base64/hex/homoglyph/bidi/zero-width normalize-then-scan before any decision | `canon`, `normgate` | SHIPPED |
| Plan control-flow integrity | refuse a tool call that deviates from the operator-approved call-graph | `plancfi` | SHIPPED |
| Kernel-authored provenance / secret gate | close the model-forged-trust hole; quarantine credential-shaped results | `provenance`, `secretgate` | SHIPPED |

### B. Context / cache / memory

| Innovation | Concept it embodies | Home | Status |
|---|---|---|---|
| Context-MMU write-time result gate | poison/secret quarantined before the model sees it; survives the session boundary + re-screen | `ctxmmu` | SHIPPED |
| KV-MMU quarantine→eviction bridge | the text verdict becomes a mechanical K/V eviction | `kvmmu` | SHIPPED |
| Content-addressed bit-exact KV eviction (RoPE-linear) | evict a mid-run span to `max|Δ|=0` vs never-having-seen it | `model` (kvcache) | SHIPPED |
| Signed deletion certificate | attest a span/page set was destroyed, offline-verifiable | `deletioncert` | SHIPPED |
| Context planner (ctxplan) | O(1) resident view over lossless history; a miss is a demand-page fault, not a lost fact | `ctxplan` | SHIPPED (live-loop seam off by default) |
| Session core-dump + context debugger | a finished session is a page table over a CAS swap device | `recall`, `cdb` | SHIPPED |
| Portable session image | pack/unpack `.faksession`; quarantine moat survives offload across hosts/models | `sessionimage`, `snapshot` | SHIPPED |
| Durability-class promotion gate ("context is not memory") | classify turn/session/durable at write time; promote only durable | `ctxmmu`, `recall` | MIXED (enforce mode; bitemporal/TTL `[STUB]`) |
| Cache-prefix-preserving compaction (byte-splice) | shed middle turns, keep the cached prefix byte-identical (95.4% shed on 142k) | `agent`, `gateway` | SHIPPED (real-traffic cascade unverified) |
| RadixAttention prefix trie | local token-trie into KV spans; reuse-through-edge-split bit-identical | `radixkv` | SHIPPED |
| cachemeta / vCache family | model a provider prefix cache you don't own as virtual pages + warmth belief + governor | `cachemeta`, `vcachegov`, `vcachechain` | MIXED (governor SHIPPED; live transport `[STUB]`) |
| Materialized-view-over-lossless-history | tool result / context served as a view; closed invalidation contract | `vdso`, `ctxplan` (vToolcall design) | MIXED (design + vDSO path) |
| Trajectory + simhash observability | typed per-turn `Turn` record + dependency-free cosine top-k scorer seam | `trajectory`, `simhash` | SHIPPED |
| L3 disaggregated cache | `Ref`→page-key resolver + durability-gated, digest-verified, scope-checked admission | `l3region`, `gateway` (l3share) | SHIPPED (live transport `[STUB]`) |

### C. Model / compute / forward

| Innovation | Concept it embodies | Home | Status |
|---|---|---|---|
| In-kernel forward pass, kernel-owned KV cache | the decode loop + KV are in-process kernel objects (greedy, synchronous) | `model` | SHIPPED (SmolLM2 bit-exact; other families argmax-exact) |
| Compute HAL (whole-op `Backend`) | add CUDA/Vulkan/Metal by registration, not a forward-loop edit; CPU=Reference, devices=Approx | `compute`, `metalgemm` | SHIPPED |
| Resident Q4_K/Q6_K/Q5_K + AWQ device GEMM | activation-aware + k-quant device kernels; kills mixed-quant Q8 fallback | `model`, `metalgemm` | MIXED (per-backend; some Apple/CUDA-gated) |
| GGUF loader + tokenizer + grammar | load any GGUF; tokenizer-aware constrained decoding seam | `ggufload`, `tokenizer`, `grammar` | MIXED (schema→token compiler `[STUB]`) |
| Multi-GPU tensor parallelism seam | Megatron column/row shard + 4-collective HAL; bit-exact vs single-device on CPU | `model` (dist) | MIXED (real device communicator hardware-gated) |
| Run-by-name model registry | `fak run ornith:9b` friendly-name → model-ref | `modelreg`, `polymodel`, `advmodel` | SHIPPED |

### D. Serving / engine / routing / scheduling

| Innovation | Concept it embodies | Home | Status |
|---|---|---|---|
| One-binary governed-serving gateway | OpenAI/Anthropic/MCP wires + floor + quarantine + audit in one static Go binary | `gateway`, `engine` | SHIPPED |
| Per-aspect model routing + ensembles | route an *aspect* (tool call / sub-query / step), not the whole request; vote/best_of/concat/all_reduce | `modelroute` | MIXED (decision SHIPPED; live dispatch `[STUB]`) |
| Native admission / priority / fairness scheduler | token-budget + max-seqs admit, AGING no-starvation, trust→403/shed→429 | `gateway`, `session` | SHIPPED (live `/metrics` fold + 429 wire remaining) |
| Session = serving admission (one machine, two altitudes) | the turn boundary is the admit/preempt/resume quantum | `session`, `loopmgr` | MIXED (unification epic #912) |
| Cross-machine lease visibility | `refs/fak/locks/*` lease object (owner/TTL/target) | `leaseref`, `gpulease` | SHIPPED (fencing token gap) |
| Capacity / residency preflight | auto-fit context to host DRAM/VRAM; refuse-free sizing | `residency`, `ctxresidency`, `headroom` | MIXED |
| Engine-cache invalidation adapter | translate cachemeta directives into SGLang/vLLM prefix-cache reset/evict | `enginecache` | SHIPPED (seam) |

### E. Fleet / RSI / governance / witness / observability

| Innovation | Concept it embodies | Home | Status |
|---|---|---|---|
| RSI keep-bit (non-forgeable) | keep only on gain ∧ suite-green ∧ truth-clean, all self-measured | `shipgate`, `rsiloop` | SHIPPED (one demo tunable wired) |
| Witness / commit-audit | corroborate a claimed effect against evidence it didn't author; `diff-witnessed` vs `subject-only` | `witness` | SHIPPED |
| Provenance + decisions audit log | git-notes readback of adjudication records; WITNESSED vs OBSERVED labels | `provenance`, `journal`, `rungobs` | SHIPPED |
| Scorecard control pane (folds 18→1 debt) | every score re-derived from disk/toolchain; `--check` CI ratchet | `tools/scorecard_control_pane.py` | SHIPPED |
| Net-true-value standard | a gain is reported only if it survives the 6-question rubric | [`net-true-value`](standards/net-true-value.md) | SHIPPED (doctrine; `claim-check` verb `not yet`) |
| Witness-gated issue-dispatch loop | spawn under cap → ship #N → per-SHA audit → close; `closure_rate = TRUE/(TRUE+CLAIMED)` | `docs/dispatch-loop.md`, `steward`, `harvest` | SHIPPED |
| Prediction-vs-reality dojo | predict → run → measure → eval → calibrate; never over-claim | `dojo` | SHIPPED (self-improving loop = #1021) |
| Run-it-all-night collection | "most valuable datum on THIS box now" → durable ledger | `nightrun` | SHIPPED |
| Rule synthesis (the loop that grows the vocabulary) | mine the refusal log → propose the next structural adjudicator rule → prove it model-free | `rulesynth` | SHIPPED |
| Journal-gardening stewards | single-invariant validators over the decision journal (secret-in-context, vDSO-soundness, …) | `steward` | SHIPPED |
| Benchmark suites (turn-tax / fan-out / web / SWE / AgentDojo) | model-agnostic fleet benchmarks; every number traced + fenced | `turnbench`, `webbench`, `swebench`, `agentdojo` | MIXED (some modeled/host-gated) |

### F. Coordination / multi-agent

| Innovation | Concept it embodies | Home | Status |
|---|---|---|---|
| Multi-agent coordination protocol (RFC, D-007) | every coordination act is an adjudicated synthetic tool call on the same floor | `docs/multi-agent-coordination-protocol.md` | SHIPPED (spec) |
| In-kernel agent-to-agent channel | message passing under the capability floor | `a2achan`, `comm` | MIXED (durable cross-process `[STUB]`) |
| Shared task record + shared-state ladder | executable JSON envelopes; 5-rung shared/durable/disaggregated vocabulary | `sharedtask`, `taskmgr` | SHIPPED |
| Wave coordination (cohort / topology) | collectives mapped honestly (agent layer ≠ tensor layer) | `cohort`, `agenttopo` | SHIPPED |
| One-sided shared window | kernel-adjudicated `MPI_Win`-style window over an `abi.Ref` (vocabulary, not RDMA) | `region` | SHIPPED |
| Deterministic DAG workflow engine | map-reduce / fan-out / explicit dep DAG; CPU-correct, no model needed | `workflow` | SHIPPED |

---

## Part 3 — General primitives & learnings (the reusable concepts)

The cross-cutting concepts that transcend any one subsystem — the candidates for a
reusable agent grammar. `Expressed` = how far each has been lifted out of
fak-specific code into a domain-free form.

| General concept | Essence | Expressed | Lives in |
|---|---|---|---|
| Witness discipline | a decision no participant can move by narrating a number | well | `witness`, `recall`, `dos_verify` |
| Closed structured-refusal vocabulary | every denial cites a token from a closed, checkable set | well | `policy`, `dos.toml [reasons.*]`, `dos_refuse_reasons` |
| Net-true-value standard | report a gain only if it survives the 6-question rubric | well | [`net-true-value`](standards/net-true-value.md) |
| The two lenses (safe == fast) | one primitive is a security control *and* an optimization | well (doctrine) | [`trust-floor-two-lenses`](notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md) |
| Verification ladder | smallest-sufficient-rung; escalate on `INDETERMINATE` | doctrine only | [`verification-ladder-doctrine`](notes/verification-ladder-doctrine.md) |
| Readiness / surface-ceiling ladder | maturity rung gated by third-party evidence; benchmark can't pose as product | partially (#582) | `tools/product_scorecard.py`, [readiness ladder note](notes/CONCEPT-DOS-READINESS-VERDICT-LADDER-2026-06-26.md) |
| Materialized-view-over-lossless-history | the resident set is a view; a miss is a demand-page fault | fak-specific | `ctxplan`, `vdso` |
| Durability-class promotion gate | classify truth-duration at write time; promote only durable | fak-specific | `ctxmmu`, `recall` |
| Content-addressed provable deletion | evict bit-exact + attest with a certificate | fak-specific | `kvmmu`, `deletioncert` |
| Taint / IFC sink-gating | tainted→sink flow refused at adjudication | fak-specific | `ifc`, `engine` |
| One-sided screen + witnessed-loss polarity | an additive screen may only tighten; a wrong proposal costs a fault, never a loss | fak naming only | `wirescreen`, `ctxmmu` |
| Prediction-vs-reality calibration | back-test a projection against telemetry before defaulting it on | partially | `dojo`, `resume` Backtest |
| Per-aspect routing + ensembles | route an aspect, not the request; ensemble+reduction is a plan | partially | `modelroute` |
| Provider-cache-as-virtual-pages | model a cache you don't own; Law A1/A2 safety + amortized governor | partially | `vcachegov`, `vcachechain` |
| Disjoint-lease admission | concurrency keyed on disjoint file-trees | well | `dos_arbitrate`, `dos.toml [lanes]` |
| Claim-salience partition | LIVE vs PARKED, no-loss (`[SHIPPED]` vs `[SIM]/[STUB]`) | well | `dos.salience.partition` |

---

## Part 4 — The programming grammar (for ourselves and for other agents)

The general primitives above are converging on a domain-free **grammar** — nouns,
verbs, a closed vocabulary — that any agent fleet can adopt by configuration without
forking fak. The seed is **DOS** (the trust substrate fak dogfoods on its own repo).

- **Nouns:** lane · lease · reason token · witness · verdict · claim · ladder rung · scope.
- **Verbs (shipped):** `arbitrate` · `verify` · `audit` · `review` · `refuse` /
  `check_reason` · `recall` · `resolve` · `status` · `doctor` · `answer`.
- **The next verbs (under-expressed concepts):** readiness · verification-ladder ·
  promote · context-contract · calibrate · taint-check · route · claim-check.

The **normative standard** — the closed nouns, the verb signatures, the lift recipe as
MUST clauses, the `G6` one-sided-screen polarity predicate, and the per-verb conformance
checklist a second implementation is checked against — is
[`docs/standards/agent-grammar.md`](standards/agent-grammar.md). The full design — what is
already lifted, the recipe for a correct lift, and the `G1–G9` backlog — is in
[`CONCEPT-AGENT-PROGRAMMING-GRAMMAR`](notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md).
The actioning epic groups the lifts as child issues.

---

## Part 5 — Honest ledger & authorities

- [`CLAIMS.md`](../CLAIMS.md) — every capability with one machine-checked tag.
- [`claims-salience-register.md`](claims-salience-register.md) — LIVE (143) vs PARKED (17), no-loss.
- [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) — every number, traced to commit + artifact.
- [`STATUS.md`](../STATUS.md) — what's shipped and on the critical path.
- [`docs/notes/CHARTER.md`](notes/CHARTER.md) — the ten principles + the alignment scorecard.

---

## Part 6 — Keeping this index durable (refresh procedure)

This index is hand-curated, re-derived from authoritative sources — not generated, so
it never drifts silently from a stale generator. Refresh it when the tree moves under
it:

1. **Roster** — diff `dos.toml [lanes.trees]` against Part 2; a new leaf lane is a
   candidate innovation row.
2. **Status** — re-read the counts from [`claims-salience-register`](claims-salience-register.md)
   (regenerate with `python tools/claims_salience_register.py`); update Part 0 snapshot
   and any row whose tag moved.
3. **Concepts** — scan `docs/notes/CONCEPT-*` and `docs/explainers/*` for a new
   general primitive; add a Part 3 row with its `Expressed` level.
4. **Grammar** — when a `G#` lift ships a verb, move it from "next verbs" to "shipped"
   in Part 4 and update the grammar note + epic.
5. **Cross-check** — the entries here must agree with `CLAIMS.md` tags; an `[SHIPPED]`
   row here that is `[STUB]` there is a bug in this index, not a claim.

Cadence: refresh on each major subsystem ship or at least per release. This page is
the innovations counterpart to [`INDEX.md`](../INDEX.md) (repo map) and
[`llms.txt`](../llms.txt) (doc map).
