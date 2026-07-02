# A native micro-scout for model routing — running a 135M–1.5B classifier in-process (2026-07-01)

**Question.** Can fak run a micro model (~125M / 0.5B / 1B params) *locally* to make model-routing
decisions — and when is that relevant?

**Answer.** Yes — and almost every part already exists in-tree. fak has the routing spine with a
deliberately injectable scout seam, a native inference engine that already loads and serves exactly
this size class on CPU, an alias/pull path for one-command model acquisition, and measured numbers
showing the size class is fast *on CPU specifically*. The one missing piece is the binding: an
in-process `modelroute.Classifier` backed by the in-kernel engine instead of a remote HTTP call.
This note is the survey + design; the binding itself is a named follow-on (issue filed), not yet built.

---

## 1. What already exists (survey, with pointers)

**The routing spine** (`internal/modelroute`, epic #595): `Route(Subject) → Decision` (per-aspect,
first-match, fail-closed), ensembles via `Combine`, versioned JSON manifests, `fak route` +
`fak routebench`. The scout seam shipped and closed as #599:

- `scout.go` defines `Classifier` / `ClassifierFunc` — the cheap-model call is injected, exactly like
  `judge.go`'s `Scorer`, so no engine enters the leaf's import graph.
- `ScoutRoute` returns a `ScoutOutcome` carrying **Before** (static-signals route) and **After**
  (scouted route) plus `Changed()` — the scout's cost is priced and observable, never a hidden
  latency headline.
- The label vocabulary is closed (`Complexity` ∈ low/medium/high, `""` = unset); an
  out-of-vocabulary label is a fail-loud error.
- The manifest already reserves the shape: `Plan.Scout` names a scout model per rule, and
  `examples/routing-presets/scout-then-route.json` is the classify-before-you-route preset.

**The only live binding today is remote.** `cmd/fak/commit_review.go` (#1185) binds
`ClassifierFunc` → `agent.NewHTTPPlanner` — an HTTP model call. Nothing binds the scout to the
in-kernel engine.

**The native engine can already run this size class, in-process, daemon-less:**

- `fak run smollm2 "…"` loads SmolLM2-135M into the in-kernel engine via
  `agent.NewInKernelPlanner` (`cmd/fak/run_model.go`) — alias-aware (`internal/modelreg`),
  hf:// pull-on-demand (`internal/hfhub`), GGUF or safetensors.
- `internal/modelladder` auto-detects exactly the ladder this note is about
  (SmolLM2-135M → Qwen2.5-0.5B → 1.5B → 3B) from disk and lazily loads + quantizes + memoizes
  each rung — built for the demos, reusable as the scout's model registry.
- `internal/model` exposes logits directly (`logitlens.go`), supports LoRA (`lora.go`), and the
  supported-arch gate (`arch_support.go`) covers the llama-family GQA path (SmolLM2) and the
  Qwen2.5 family used by the ladder.

**Measured, in-repo numbers for the size class** (docs/HARDWARE-MATRIX.md, BENCHMARK-AUTHORITY.md):

- SmolLM2-135M decode: ~100–120 tok/s (RTX 4070, CUDA Graph); **862 agg tok/s** batched Q8 @ b512.
- On the agent-host box (Ryzen 9950X), the **CPU leads the RX 7600 Vulkan path 7.2× at 135M**
  (Vulkan Q8 24.6 tok/s ⇒ ≈175 tok/s implied on CPU) — the device path is launch-bound on tiny
  models. The micro-scout is a *CPU-native* workload; no GPU needs to be present or reserved.
- Qwen2.5-1.5B Q8 single-stream CPU: 27.9 tok/s (~36 ms/token).
- RadixAttention prefix reuse: **86.7% hit rate** measured on agent loops. A scout's system prompt +
  policy text is a *fixed prefix*, so after the first call each classification re-prefills only the
  subject tail and decodes ~1 token — marginal cost in the low tens of milliseconds on CPU.

## 2. Why *local* is the point (not just an optimization)

1. **The scout reads the raw prompt before routing.** Classify-first means the scout sees exactly
   the bytes the residency PDP exists to protect. A remote cheap-model scout is an egress of the
   full subject *before* adjudication; a local micro-scout keeps classify-first compatible with
   epic #595's load-bearing wiring contract (route **before** adjudicate, `ToolCall.Engine` set
   pre-submit) without widening egress at all. The surveyed routers (RouteLLM, Martian, NotDiamond,
   OpenRouter…) all ship the prompt to *their* classifier; a router that is also the trust boundary
   has to be able to classify on-host. This is the fak-native argument — structural, not a speed claim.
2. **Cost/quota.** A scout is an extra model call per routed aspect (ScoutOutcome charges it
   honestly). Remote, that multiplies per-request spend and eats seat/429 budget
   (`internal/ratelimit`, `internal/attemptbudget` reality); local, the marginal cost is ~0 tokens
   billed and no seat pressure.
3. **Latency.** Industry bands for the routing decision: rules <1 ms, embedding lookups ~5 ms,
   ML-classifier routers 50–100 ms, a remote LLM scout call 500–2000 ms. An in-process 135M
   classify (cached prefix + 1 token) lands in the classifier band while keeping LLM-grade
   label quality — and `<200 ms` is what Arch-Router markets as production-grade.

## 3. External SOTA (honest positioning)

- **RouteLLM** (LMSYS, arXiv 2406.18665): learned binary router (BERT + causal-LM classifier
  variants) over preference data; headline 85%/45%/35% cost cuts are pair- and benchmark-specific.
  Precedent: a sub-1B classifier is enough to route.
- **Arch-Router-1.5B** (Katanemo, arXiv 2506.16655): a **Qwen2.5-1.5B fine-tune, GGUF published**,
  93.17% preference-routing accuracy, <200 ms decisions. Its scheme — policy descriptions in the
  prompt, model emits a policy identifier, mapping policy→model lives *outside* the model — maps
  1:1 onto fak's manifest (`Rule` names ≈ policies; no retrain to add models). Because it is the
  same architecture family as the ladder's 1.5B rung, it is **candidate weights, not just a
  candidate idea** — expected to load through the existing Qwen2.5-1.5B paths (hf Q8 / GGUF), though
  loading it in-kernel is an unverified follow-on (see §5).
- **vLLM Semantic Router**: ModernBERT *encoder* classifier — out of scope for fak's causal-only
  engine; honest non-goal unless an encoder forward is ever added.
- The 2026 production pattern is layered: static rules for the obvious cases → a classifier for the
  ambiguous middle → a cascade with escalation for the tail. fak has the first layer (manifest) and
  the third's skeleton (#2022 microagent tier-routing, `internal/spec` drafter/verifier); the local
  micro-scout is the missing middle layer.

## 4. Design — the native binding (follow-on)

- **Where it lives:** the dispatch layer (`cmd/fak`, like `commit_review.go`) — *not* inside
  `internal/modelroute`. The scout.go doctrine stands: the leaf owns the wiring, the classifier owns
  the model call, and no engine import enters the spine.
- **Shape:** `bindNativeScout(modelRef) modelroute.ClassifierFunc` — resolve via `modelreg`
  (pull-on-demand), load via the `fak run` loaders or `modelladder.Registry` (memoized: load once,
  classify many), wrap in a `ClassifierFunc` that renders the subject into a fixed classification
  prompt. Default model: `modelladder.SmallestPresent` (135M today).
- **Closed vocabulary by construction — score labels, don't parse text.** Instead of free decode +
  parse, run one forward pass over the prompt and compare the logits of the candidate label tokens
  ("low"/"medium"/"high"), i.e. constrained single-token classification. The engine already exposes
  logits (`logitlens.go`). `ScoutLabel.Valid()`'s fail-loud check becomes structurally unreachable,
  the output is deterministic at temp 0, and the decode cost is exactly 1 token.
- **"If and when relevant" — gate the scout on ambiguity.** Never scout unconditionally: invoke only
  when (a) the matched rule names a `Plan.Scout`, or (b) the static route fell through to the
  fail-closed default (`Decision.Matched == false`). `ScoutOutcome.Changed()` is the witness that
  the call earned its cost; `fak routebench` gets a scout arm to measure changed-rate + latency
  offline before any live default flips.
- **Escalation ladder (#2022):** the same binding gives the cheap-first cascade its classifier —
  low → local/cheap model with an escalate-on-failure witness, per-aspect.
- **The learned loop (#600) becomes fully local:** DecisionRecords + outcome telemetry are a
  training corpus; with `internal/model/lora.go` in-tree, fak can eventually fine-tune its *own*
  scout on its *own* routing outcomes without the corpus ever leaving the host —
  `internal/advmodel` (linear classifier over harvested verdicts, fail-closed, default-off) is the
  in-repo precedent one rung down.

## 5. Feasibility verdict + named fences

**Feasible now** for SmolLM2-135M and Qwen2.5-0.5B/1.5B (arches proven in-engine, weights already
laddered/aliased; CPU is the *preferred* device at this size). Fences, each a checkable next step:

1. **`arch-router` alias not seeded** — modelreg's rule requires a verified HEAD on the exact
   resolve URL first (`hf://katanemo/Arch-Router-1.5B.gguf/…`); outbound HF from the agent host
   timed out during this survey, so seeding is *not yet* done. Verify + seed + `fak run arch-router`
   smoke is the witness.
2. **No native binding yet** — `ClassifierFunc` → `InKernelPlanner` (+ label-token scoring) is
   unbuilt; test = fixed-stub parity + a `smollm2` smoke classify.
3. **Latency unbenchmarked in fak** — the ~10s-of-ms claim above is derived from measured decode
   rates + Radix hit rates, not a measured scout round-trip; the routebench scout arm is the witness.
4. **Encoder routers out of scope** — causal engine only; a non-goal, stated.
5. **135M label quality unknown on fak's subjects** — Arch-Router's 93% is on *its* benchmark; the
   right first fak metric is `Changed()`-rate + downstream outcome deltas on the routebench corpus,
   never a borrowed number.

## 6. Related

Epic #595 (spine; scout child #599 CLOSED) · #600 (telemetry → learned routing) · #2022
(cheap-model tier routing + escalation) · #1725 (multi-harness routing) · `docs/model-routing.md`
(the spine's own doc — its roadmap still lists scout wiring as open) · `internal/spec`
(drafter/verifier — the other consumer of a resident micro model).
