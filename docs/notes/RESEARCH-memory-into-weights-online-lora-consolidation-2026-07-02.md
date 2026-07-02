---
title: "RESEARCH: projecting memory back into the weights — a witnessed, sleep-cadence LoRA consolidation loop"
description: "SOTA triage of online/continual LoRA consolidation of agent memory into model weights (TMEM, PEAM, SEAL, sleep-time compute, LoRA Without Regret, rejection-sampling fine-tuning, multi-adapter serving, unlearning limits), a minimal six-stage spine that reuses fak's existing witness/quarantine/durability/shipgate seams and trains nothing inside the kernel, and a 44-issue ladder (MW-01..MW-44) to take it to production."
slug: memory-into-weights-online-lora-consolidation
keywords:
  - parametric memory
  - LoRA consolidation
  - continual learning
  - rejection-sampling fine-tuning
  - sleep-time compute
  - adapter hot-swap
  - machine unlearning
date: 2026-07-02
---

# Projecting memory back into the weights — a witnessed, sleep-cadence LoRA consolidation loop

> **Deliverable for:** the "memory into weights" goal — research SOTA on an automatic
> "online" LoRA/PEFT loop, define a minimal spine that can be built first and actually
> work, then the 30–50 issues that make it production-useful.
>
> **Verdict: adopt as a governed capability at the gateway boundary — and only there.**
> fak referees a sleep-cadence consolidation loop (witnessed traces → rejection-filtered
> SFT data → QLoRA adapter → held-out keep-bit → engine-side hot-swap → provenance
> ledger) and **never trains itself**: the episodic ledger stays canonical, the adapter
> is a rebuildable derived view, deletion is deterministic *retrain-without* (not
> unlearning), and the adjudication path stays structurally independent of any adapter.
> Everything in this note is design + prior art; **nothing here is shipped** (see the
> honest boundary).

## The question

fak's memory story today is entirely non-parametric: durable truths live in the
episodic store (`internal/recall` pages, memory files, the verdict ledger), context is
managed as views over canonical bytes (`internal/memview`), and the model's weights
never change. The field's critique of exactly this architecture is now explicit: a
prompt-space memory is "a memo, not true memory" — the agent can look up what it saw
but its policy is unchanged by experience
([arXiv:2604.27707](https://arxiv.org/abs/2604.27707)). The question this note answers:
what is the smallest *automatic* loop that periodically consolidates fak's witnessed
experience into a LoRA adapter on the local model it fronts, without violating the
doctrines that make fak a trustworthy referee?

## SOTA readout (2025–2026)

The field has converged on a **dual-store, cultivate-then-consolidate** architecture:
fast episodic external memory plus a periodic (or online) LoRA-based consolidation
channel into weights — the AI analog of sleep. The load-bearing results, and what fak
takes from each:

| Source | One-line | What fak takes | What fak does differently |
| --- | --- | --- | --- |
| TMEM ([arXiv:2606.04536](https://arxiv.org/abs/2606.04536)) | Self-evolving parametric memory: distilled supervision absorbed into fast LoRA weights via lightweight online updates, even intra-episode | The dual-store + consolidation-channel framing; LoRA subspace anchored to principal directions of base weights aids recall | v0 stays sleep-cadence and gated; intra-episode fast weights are a fenced follow-on (MW-42), not the spine |
| PEAM ([arXiv:2605.27762](https://arxiv.org/abs/2605.27762)) | Sleep-inspired offline consolidation into per-category adapters; parameter isolation vs cross-category forgetting; decides *which* traces become parametric competence and *when* | The consolidate-then-serve cadence; per-lane adapter isolation (MW-34) | Admission is a witness/durability gate, not a category heuristic |
| "Memo, not memory" ([arXiv:2604.27707](https://arxiv.org/abs/2604.27707)) | Position: retrieval-only memory leaves the policy unchanged; add a second pathway from the episodic store to weights | The thesis itself | fak's answer keeps raw bytes canonical (`memview` contract); the adapter is a *view* over the ledger, never the fact |
| SEAL ([arXiv:2506.10943](https://arxiv.org/abs/2506.10943)) | The model writes its own "self-edits" (training data + update directives); RL-rewarded by downstream gain; suffers catastrophic forgetting under repeated edits | The inner/outer loop split; eval-gated updates | The student never authors its own admission: data selection is decided by kernel witnesses, not self-edits ([self-edit search extension](https://arxiv.org/abs/2601.14532) is triaged under MW-43, likely defend/cite not adopt) |
| Sleep-time compute ([arXiv:2504.13171](https://arxiv.org/abs/2504.13171), Letta) + "LMs Need Sleep" ([arXiv:2606.03979](https://arxiv.org/abs/2606.03979)) | Idle-time consolidation; alternate active/sleep phases; distill fragile short-term memory into stable knowledge with replay and "dreaming" | The cadence trigger (idle window / nightrun) and the replay mixture | Consolidation output is accepted only by a non-forgeable keep-bit, never by construction |
| LoRA Without Regret ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/lora/)) | LoRA matches full fine-tuning in the post-training regime when applied to *all* linear layers, ~10× the full-FT learning rate, 1/r prefactor | The trainer defaults, verbatim (MW-09) | Nothing — adopted as-is; reproduced in [TRL](https://huggingface.co/docs/trl/en/lora_without_regret) |
| "LoRA Learns Less and Forgets Less" ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)) + replay literature | LoRA is **not** a forgetting cure; even minimal replay of anchor data significantly stabilizes retained capability | The anchor/replay mixture (MW-10) and a committed forgetting floor (MW-15) | The floor is a baseline the gate *enforces* RSI-style, not a reported metric |
| Rejection-sampling fine-tuning (RFT; step-level variant [arXiv:2605.10674](https://arxiv.org/abs/2605.10674)) | The standard agent-trace recipe: roll out, keep only verified-success trajectories, SFT on the survivors; failed-trace steps are salvageable later | The filter shape | Success at the floor is a **kernel witness** (diff-witnessed `dos commit-audit`, `SuiteGreen`), never a judge-model opinion; judge models may only *narrow* the kept set |
| Regimes ([arXiv:2606.10241](https://arxiv.org/abs/2606.10241)) | An auditable, held-out-gated improvement loop with deterministic replay | Confirmation of the held-out-gate posture | fak already owns this pattern (`shipgate.Evaluate`); reuse it, don't reinvent it |
| WikiBigEdit ([arXiv:2503.05683](https://arxiv.org/abs/2503.05683)) / lifelong knowledge editing | Point-fact editing at scale hits hard limits; LoRA-merge interpolation drifts | Edit-vs-retrain honesty | Point facts ride the episodic store forever; weights get *skills and priors*, not facts with expiry dates |
| Unlearning surveys ([arXiv:2510.25117](https://arxiv.org/abs/2510.25117), [arXiv:2503.01854](https://arxiv.org/abs/2503.01854)) | Unlearning is suppression, not deletion; benign "relearning through fine-tuning" recovers forgotten content | The negative result, as a design axiom | Deletion is a deterministic **rebuild-without** off the data manifest (MW-27); unlearning is explicitly rejected as a deletion mechanism |
| S-LoRA ([repo](https://github.com/S-LoRA/S-LoRA)), LoRAX ([repo](https://github.com/predibase/lorax)), [vLLM](https://docs.vllm.ai/en/stable/features/lora/), [SGLang](https://sgl-project.github.io/advanced_features/lora.html), [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | Multi-adapter serving with runtime load/unload is commodity: vLLM `/v1/load_lora_adapter` (behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING`), SGLang `/load_lora_adapter` (+ overlap loading, pinning), llama.cpp `/lora-adapters` (+ per-request `lora` field), LoRAX ~200 ms just-in-time loads | Engine-side hot-swap endpoints (MW-20); PEFT→GGUF via `convert_lora_to_gguf.py` | fak stays **out** of adapter serving — it orchestrates and witnesses the swap. The `docs/industry-scorecard/models.md` "multi-LoRA hot-swap = out of scope" claim stands until MW-20/21 ship with witnesses |
| Trajectory multi-LoRA stack ([field notes, 2026](https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning)) | Continuous training with in-place LoRA reload into the inference engine while other tenants keep decoding | Proof the train→swap seam can run as a continuous service | v0 swaps only at gate promotions, never mid-turn (MW-23) |

**What the SOTA agrees on** (and the spine below encodes): (1) dual store — episodic
stays authoritative, weights get the distilled residue; (2) sleep cadence — consolidate
in idle windows, not on the request path; (3) rejection filtering — only verified
successes become training signal; (4) gate + rollback — held-out eval before promotion,
instant revert after; (5) LoRA-without-regret config makes the trainer boring; (6)
forgetting needs an explicit replay/anchor mechanism; (7) parametric deletion doesn't
work, so provenance and rebuildability must be designed in from day one.

## Why fak is unusually well-positioned — and the one doctrine it must answer

Every stage of the SOTA recipe has a **shipped fak seam** where other stacks must build
trust machinery from scratch:

- **The RFT success filter is the witness ledger.** SOTA pipelines filter traces with a
  judge model; fak filters with non-forgeable, action-level fact the model cannot
  author — a diff-witnessed `dos commit-audit`, `SuiteGreen`/`TruthClean` from
  `shipgate`, a `dos verify` SHIPPED verdict. No verdict ⇒ gate→0 ⇒ the trace is not
  training data (fail closed).
- **The training-data firewall is the quarantine.** Memory poisoning → weight poisoning
  is the new attack surface this feature opens. fak's existing posture answers it
  structurally: tainted/quarantined spans (`internal/kvmmu` quarantine, `memview`
  taint) are excluded from harvest *by construction*, with a test proving a poisoned
  tool result cannot reach a training file (MW-03) — not by a classifier the attacker
  can argue past.
- **The admission policy is the durability axis.** `CONTEXT-IS-NOT-MEMORY.md` decides
  context-vs-memory by truth-duration; the same axis decides memory-vs-weights. Weights
  cannot selectively expire, so: only `durable` truths (and corroborated dispositions
  per `internal/ctxmmu/disposition.go`, #1598) may consolidate; `bounded` truths only
  if the rebuild cadence is shorter than their validity window; `turn`/`session` never.
  **Adapter TTL = min truth-duration of its training set.**
- **The adapter is a memview.** Under the `MEMORY-VIEW-CONTRACT`, a lossy derivative
  never hashes to the source digest and never becomes a canonical fact. An adapter is
  exactly that: a typed, lossy, materialized view over the trace ledger, carrying
  provenance (data-manifest digest), invalidated when its sources are revoked, and
  re-entering adjudication like any other view. This one framing yields the deletion
  story for free.
- **The promotion gate is the RSI keep-bit.** `shipgate.Evaluate` already encodes the
  house rule: keep only on a strictly measured gain from derived witnesses; a scorecard
  can explain a metric but cannot flip a REVERT. The adapter gate is the same shape
  with two metrics: strict gain on the target battery AND no anchor-suite regression
  beyond ε.
- **The deletion story is the KV-eviction analogy, told honestly.** At the context
  layer fak evicts a span and proves the cache bit-identical (`max|Δ| = 0`). At the
  weight layer that proof is unavailable — so the equivalent is *rebuild-without*:
  revoke spans in the manifest, deterministically retrain, swap. The v0 witness is
  manifest-level (the revoked span is provably absent from the rebuild's inputs), not
  byte-level; bitwise-reproducible training is its own tracked issue (MW-12).

**The doctrine this design must answer: layer-1 disinterest.**
`RESEARCH-three-layers-of-agent-optimization` states fak deliberately leaves the
skill/weights layer empty — "a model author cannot be a disinterested referee." The
resolution is a strict referee/student split:

1. **fak never trains itself.** The adjudication path — permission gate, verdict
   model, quarantine — runs on no adapter, ever, and the syscall-LLM stub stays a
   stub. A test asserts adjudication output is identical with and without any adapter
   loaded (MW-30).
2. **The trainer is userspace, not kernel.** Training is a pinned external sidecar
   (PEFT/TRL) that fak launches, gates, and witnesses — fak owns admission, the
   keep-bit, the swap, and the ledger; it authors no gradient.
3. **The student never grades itself.** Data admission, gate metrics, and promotion
   are all read from kernel-owned witnesses the tuned model cannot author. This is the
   same answer the span-credit notes give ("what outcome gate stops the model rewarding
   its own preferred context") lifted one layer up.

Tuning fak's *own* router/micro-scout (the `MICRO-SCOUT-NATIVE-ROUTING` aspiration) is
a genuinely different governance problem — the referee tuning part of itself — and is
explicitly fenced out to MW-44.

## The minimal spine (v0): six stages, every one on an existing seam

One box, one local model (e.g. the Qwen the Mac/Windows dogfood already serves via
vLLM or `llama-server`), one adapter lane. "Online" here means **automatic on a sleep
cadence** (idle-window or nightly trigger, plus a minimum-new-data floor) — not
per-turn weight updates; that honesty matters and is repeated in the boundary section.

```
   trigger            harvest              admit                train
 idle window ──► session journal ──► 4 gates: witness, ──► QLoRA sidecar
 or nightrun       + verdict ledger     taint, durability,     (all-linear, 1/r,
 + data floor      + memory files       freshness              anchor replay mix)
                                          │                        │
                                          ▼                        ▼
                                    SFT JSONL +             adapter.safetensors
                                    data manifest           + config digest
                                    (content-addressed)          │
                                                                 ▼
   ledger  ◄──────  swap  ◄──────────  gate
 adapter digest =   engine endpoint    frozen held-out battery, base vs adapter
 H(base, manifest,  load + alias       keep-bit: strict target gain AND
 config, trainer)   flip; rollback     anchor regression ≤ ε; states
 revoke ⇒ rebuild   = flip back        SHADOW → CANARY → LIVE
```

1. **HARVEST** — walk the session journals and verdict ledger; collect candidate spans
   and memory-file facts. Seams: the session journal, `dos commit-audit`,
   `abi.Verdict.Meta` (the OPEN map — any new per-span tag rides it additively, per the
   durability-seam pattern; the closed enums never move), `internal/recall` pages.
2. **ADMIT** — four fail-closed gates: **witness** (only spans from turns closed by a
   non-forgeable witness), **taint** (quarantined/tainted spans structurally excluded),
   **durability** (`durable` + corroborated dispositions only), **freshness** (memory
   files re-verified at harvest time; stale memory is never baked in). Output: loss-
   masked SFT JSONL (assistant spans only, invariant-prefix only per the
   `exact | bounded@i` off-policy fence) + a content-addressed data manifest.
3. **TRAIN** — shell out to a pinned QLoRA sidecar with LoRA-without-regret defaults
   (all-linear targets, modest rank, 1/r prefactor, ~10× full-FT LR), a fixed anchor/
   replay mixture, and a recorded seed. fak's Go binary gains no training code.
4. **GATE** — run the frozen held-out battery (a fixed dojo/bench subset) base-vs-
   adapter. Keep-bit = strictly measured gain on the target metrics AND no anchor
   regression beyond ε, `shipgate.Evaluate`-pattern. Promotion states:
   `SHADOW → CANARY → LIVE`; any breach ⇒ REVERT.
5. **SWAP** — load via the engine's own endpoint (vLLM `/v1/load_lora_adapter`, SGLang
   `/load_lora_adapter`, llama.cpp `/lora-adapters`) and flip a fak routing alias
   (`qwen3-8b+mem@v7`). Rollback is an alias flip; the base model is never touched; no
   swap mid-turn.
6. **LEDGER** — record `adapter_digest = H(base_digest, manifest_digest, config_digest,
   trainer_version)`, signed. The audit question "which sessions shaped these weights?"
   has an exact answer. A revocation (deletion request, discovered poisoning, expired
   `bounded` truth) triggers a deterministic rebuild-without and a swap.

**v0 non-goals** (each is a tracked follow-on, not a silent gap): no RL on traces (off-
support past divergence), no intra-episode fast weights, no model-authored self-edits,
no training of fak's adjudicator or router, no provider-hosted models (you cannot LoRA
a frontier API — the routing split is honest: consolidation applies to self-hosted
engines only), no multi-adapter composition, no cross-tenant data mixing.

## The production ladder — 44 issues (MW-01…MW-44)

Ready to file as one epic + children (house pattern, cf. #860 → #861–#867). **P0 =
the nine spine issues**; the spine is demonstrable when all nine land.

### Track A — harvest & data ledger (MW-01…MW-08)

- **MW-01 (P0)** `fak consolidate harvest`: walk session journals + verdict ledger, emit loss-masked SFT JSONL + content-addressed data manifest. Acceptance: manifest lists every source span by digest; re-run is byte-identical on unchanged inputs.
- **MW-02 (P0)** Witness gate: admit only spans from turns closed by a non-forgeable witness (diff-witnessed `dos commit-audit`, `SuiteGreen`, `dos verify` SHIPPED); missing verdict ⇒ excluded. Acceptance: fail-closed test.
- **MW-03 (P0)** Taint gate: quarantined/tainted spans structurally excluded. Acceptance: a test proving a poisoned tool result cannot reach a training file.
- **MW-04** Durability gate: `durable` + corroborated dispositions (#1598) only; `bounded` admitted only when rebuild cadence < validity window; `turn`/`session` never. Acceptance: adapter TTL computed as min truth-duration of its manifest.
- **MW-05** Freshness gate: memory-file facts re-verified at harvest (`dos_recall`-style); stale facts excluded with a structured reason.
- **MW-06** Dedupe + per-source caps: no single session/repo dominates the mixture; log what was dropped (no silent truncation). Follow-on: step-salvage from failed traces (SRFT/EEF-shape) once the plain recipe works.
- **MW-07** Example schema: system/tools/turns serialization, loss masking on assistant spans, invariant-prefix fence with the `exact | bounded@i` divergence witness carried per example.
- **MW-08** Secret/PII scrub rung: pre-train scan; scrub decisions recorded in the manifest (a scrubbed example is a different record — selector = producer).

### Track B — trainer (MW-09…MW-13)

- **MW-09 (P0)** Pinned QLoRA sidecar (PEFT/TRL container or venv): all-linear targets, 1/r prefactor, LoRA-without-regret LR defaults, recorded seed; emits adapter + config digest. fak shells out; no training code in the Go binary.
- **MW-10** Anchor/replay mixture: fixed general-capability anchor set mixed at a fixed ratio; anchor-set digest in the manifest.
- **MW-11** Data-floor refusal: refuse to train below a minimum admitted-example count with a structured reason (`DATA_FLOOR`), wired to the `dos` refusal vocabulary.
- **MW-12** Reproducibility witness: record seed + library versions; measure and document the nondeterminism floor; state honestly that the v0 rebuild witness is manifest-level, not byte-level.
- **MW-13** Multi-tenancy: one adapter lane per workspace; cross-tenant mixing refused; concurrent train requests arbitrated (`dos_arbitrate`-shape).

### Track C — eval gate & forgetting (MW-14…MW-19)

- **MW-14 (P0)** Held-out gate: frozen battery, base-vs-adapter, keep-bit = strict target gain AND anchor regression ≤ ε; `shipgate.Evaluate` pattern (a scorecard can explain, it cannot flip a REVERT).
- **MW-15** Forgetting floor: committed anchor baseline (general capability + refusal behavior); breach ⇒ REVERT, CI-enforceable like `rsiloop -mode track`.
- **MW-16** Memorization probe: canary strings planted in training data must not be emitted verbatim; secret-leakage probe on the gated adapter.
- **MW-17** Promotion states: `SHADOW` (logged, not served) → `CANARY` (N% of turns or one seat) → `LIVE`; each transition needs its own witness; `snfitness.go` shadow-first posture is the reference.
- **MW-18** Battery versioning: battery digest recorded in every keep-bit; changing the battery is a new referee and resets comparability.
- **MW-19** Tool-grammar drift gate: malformed-call rate and repair rate must not regress (the kernel's own gateway metrics are the witness).

### Track D — serving & swap (MW-20…MW-25)

- **MW-20 (P0)** Engine adapter orchestration: capability probe + load/unload drivers for vLLM (`/v1/load_lora_adapter`), SGLang (`/load_lora_adapter`), llama.cpp (`/lora-adapters`, PEFT→GGUF conversion step).
- **MW-21 (P0)** Alias routing: `<base>+mem@vN` as a first-class routed model name; promotion/rollback = alias flip with an audit line.
- **MW-22** In-kernel parity: complete #291 wiring so the reference engine can apply the same adapter (MERGE/DYNAMIC paths in `internal/model/lora.go`) — parity witness vs the fronted engine.
- **MW-23** Swap safety: never mid-turn; drain or idle-slot handling (llama.cpp global-scale caveat); per-request adapter field where the engine supports it.
- **MW-24** KV-cache correctness across swap: define which cached prefixes an adapter swap invalidates; bit-exactness claims do not transfer across weight changes — witness the invalidation rule.
- **MW-25** Composition policy: default-deny multi-adapter composition (workspace + persona) until interference is measured.

### Track E — provenance, deletion, integrity (MW-26…MW-31)

- **MW-26 (P0)** Adapter ledger: signed record `H(base, manifest, config, trainer_version)`; query answers "which sessions shaped these weights"; adapters are content-addressed artifacts.
- **MW-27 (P0)** Deletion = rebuild-without: revocation list ⇒ deterministic retrain excluding revoked spans ⇒ swap; witness = revoked span provably absent from the rebuild manifest. Unlearning documented as suppression-only and rejected as the deletion mechanism.
- **MW-28** Trace-poisoning threat model: content that survives to a *witnessed-success* trace (attacker gets malicious text into a landing commit) still reaches training data — mitigations: source-trust weighting, corroboration floor for new sources, per-run distribution-shift anomaly check on the admitted set.
- **MW-29** Artifact integrity: signature verification before any engine load; unsigned adapter refused with a structured reason (`UNSIGNED_ADAPTER`).
- **MW-30** Referee independence test: adjudication output provably identical with and without any adapter loaded; the verdict path pins base weights by digest.
- **MW-31** Compliance surface: per-data-class retention policy; exportable per-adapter manifest for audit; deletion SLA = one rebuild cadence.

### Track F — routing & fleet (MW-32…MW-35)

- **MW-32** Honest routing split: consolidation exists only for self-hosted engines; provider-hosted models are out of scope by construction — document at the routing seam, not in marketing.
- **MW-33** Fleet rollout: staged promotion across seats/replicas (the vLLM multi-replica caveat is real — runtime adapter load is per-replica); one-seat canary first.
- **MW-34** Adapter granularity: per-lane vs per-workspace adapters; measure interference before choosing; training jobs get a concurrency-class budget.
- **MW-35** Cadence scheduler: idle-window/nightrun trigger + minimum-new-data floor + budget cap (the `BUDGET-TRIGGERED-SESSION-RESET` pattern); a skipped run logs a structured reason.

### Track G — observability & ops (MW-36…MW-40)

- **MW-36** Consolidation dashboard: per-version target gains, anchor drift, admitted/excluded counts by gate, promotion state (grafana surface exists).
- **MW-37** Refusal vocabulary: `DATA_FLOOR`, `GATE_REGRESSION`, `UNSIGNED_ADAPTER`, `STALE_MEMORY`, `TAINTED_SOURCE`, `OFF_SUPPORT` — closed vocabulary, `dos_check_reason`-verifiable.
- **MW-38** Cost accounting: train-cost vs measured benefit per adapter version; the loop must pay for itself and say so with the same honesty as the single-use-cache-entry-is-a-loss rule.
- **MW-39** Adapter GC: retire superseded adapters (manifests kept — rebuildability survives GC); LRU on engine adapter slots.
- **MW-40** End-to-end demo + doc refresh: laptop-class round trip (harvest → train → gate → swap → answer changes) reproducible from a fresh clone; update `docs/industry-scorecard/models.md` only when the witness exists.

### Track H — research follow-ons (MW-41…MW-44)

- **MW-41** Span-credit as example weighting: couple #860's exact-eviction LOO reward as per-example weights — **only after** #866 returns `CORRELATE`; until then attention-derived weights are telemetry.
- **MW-42** Intra-episode fast weights (TMEM-shape): full adopt/defend/cite triage; default posture is defend-the-gate (an un-gated weight write is the same class of hazard as an un-gated tool call).
- **MW-43** Model-authored self-edits (SEAL-shape): triage; prior expectation is cite + defend, not adopt — the student authoring its own curriculum crosses the admission boundary.
- **MW-44** Tuning fak's own router/micro-scout: fenced as a separate governance problem (the referee tuning part of itself); requires its own doctrine note before any code.

## Honest boundary (`not yet`, with the missing witnesses)

- **Nothing in this note is shipped.** fak today has zero training code; the in-kernel
  LoRA core (`internal/model/lora.go`) is apply/merge only and itself carries an open
  fence (#291). Evidence: the tree; `docs/industry-scorecard/models.md` marks
  multi-LoRA hot-swap out of scope. Next checkable step: MW-01 (harvest) — it needs
  only the journal and the verdict ledger, both shipped.
- **"Online" is sleep-cadence batch, not per-turn updates.** Calling a nightly gated
  loop "online learning" would be an overclaim; the honest term is *automatic
  sleep-cadence consolidation*. Per-turn fast weights are MW-42, unproven here.
- **The deletion witness is manifest-level.** Bitwise-reproducible retraining across
  GPU stacks is not assumed (MW-12 measures the floor). What v0 proves: the revoked
  span is absent from the rebuild's inputs — not that the new adapter is byte-identical
  to a never-saw-it counterfactual.
- **The forgetting floor is a gate, not a solution.** LoRA does not prevent forgetting;
  replay mitigates; the committed anchor baseline only *detects and refuses* regression.
- **Frozen traces are off-support past divergence.** SFT on the invariant prefix of
  witnessed-success trajectories is the sound primitive; anything RL-shaped on recorded
  traces inherits the 0/1=0 importance-ratio fence and is out of v0.
- **Witnessed success can still carry poisoned text** (MW-28). The witness gate proves
  the *outcome* landed; it does not prove every byte of the trace is benign. The taint
  gate covers what quarantine caught; MW-28 covers what it didn't.

## Forbidden overclaims (downstream rule)

Any doc, commit, or dashboard touching this feature must not say:

- "The agent learns online" — until per-turn updates exist, say *sleep-cadence
  consolidation*.
- "Memory now lives in the weights" — the episodic ledger stays canonical; the adapter
  is a lossy derived view under the `memview` contract.
- "Deleted via unlearning" — only rebuild-without counts as deletion.
- "The model improved" without naming adapter digest + data manifest + gate record
  (battery digest, base-vs-adapter deltas, promotion state). Absent those, it is
  telemetry, not improvement — the same producer/verifier/promotion-state discipline
  the span-reward notes impose.
- "fak trains models" — fak referees a governed trainer; the adjudication path is
  untrained and adapter-independent by test (MW-30).

## Sources

**Primary (external):**
[TMEM, arXiv:2606.04536](https://arxiv.org/abs/2606.04536) ·
[PEAM, arXiv:2605.27762](https://arxiv.org/abs/2605.27762) ·
["Memo, not memory", arXiv:2604.27707](https://arxiv.org/abs/2604.27707) ·
[SEAL, arXiv:2506.10943](https://arxiv.org/abs/2506.10943) ·
[Self-edit strategy search, arXiv:2601.14532](https://arxiv.org/abs/2601.14532) ·
[Sleep-time compute, arXiv:2504.13171](https://arxiv.org/abs/2504.13171) ·
[LMs Need Sleep, arXiv:2606.03979](https://arxiv.org/abs/2606.03979) ·
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) ([TRL reproduction](https://huggingface.co/docs/trl/en/lora_without_regret)) ·
[LoRA Learns Less and Forgets Less, arXiv:2405.09673](https://arxiv.org/abs/2405.09673) ·
[Step Rejection Fine-Tuning, arXiv:2605.10674](https://arxiv.org/abs/2605.10674) ·
[Regimes, arXiv:2606.10241](https://arxiv.org/abs/2606.10241) ·
[WikiBigEdit, arXiv:2503.05683](https://arxiv.org/abs/2503.05683) ·
[Unlearning survey, arXiv:2510.25117](https://arxiv.org/abs/2510.25117) ·
[Unlearning survey, arXiv:2503.01854](https://arxiv.org/abs/2503.01854) ·
[S-LoRA](https://github.com/S-LoRA/S-LoRA) ·
[LoRAX](https://github.com/predibase/lorax) ·
[vLLM LoRA docs](https://docs.vllm.ai/en/stable/features/lora/) ·
[SGLang LoRA serving](https://sgl-project.github.io/advanced_features/lora.html) ·
[llama.cpp server `/lora-adapters`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) ·
[GGUF-my-LoRA](https://huggingface.co/blog/ngxson/gguf-my-lora) ·
[Trajectory multi-LoRA stack](https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning)

**Local source files:**
`docs/CONTEXT-IS-NOT-MEMORY.md` · `docs/MEMORY-LAYERS-EXPLAINER.md` ·
`docs/notes/MEMORY-VIEW-CONTRACT-2026-06-26.md` · `docs/rsi-loop.md` ·
`docs/notes/RESEARCH-three-layers-of-agent-optimization-2026-06-24.md` ·
`docs/notes/RESEARCH-span-credit-signal-menu-2026-06-26.md` ·
`docs/notes/RESEARCH-span-reward-novelty-boundary-2026-06-26.md` ·
`docs/notes/RESEARCH-progress-advantage-rl-posttraining-triage-2026-06-28.md` ·
`docs/supported/engines.md` · `docs/industry-scorecard/models.md` ·
`internal/model/lora.go` (#291) · `internal/ctxmmu` (durability, #82; disposition,
#1598) · `internal/memview` (#904) · `internal/rsiloop` + `shipgate.Evaluate` ·
`internal/model/span_reward_shadow.go` (#860/#866)

*Method note: SOTA claims above were gathered from multiple independent web sweeps
(parametric memory, self-adaptation loops, serving infrastructure, training regime,
data recipe, unlearning) and cross-checked against the repo's existing doctrine notes
before any design commitment; repo claims were read from the tree, not recalled.*
