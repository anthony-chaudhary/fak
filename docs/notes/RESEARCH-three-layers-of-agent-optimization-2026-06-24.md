---
title: "Three layers of agent optimization: skill, trajectory, substrate"
description: "A survey that separates three things people lump together when they say they are 'optimizing an agent': the skill (what it should do), the trajectory (how any run actually unfolds, regardless of skill), and the hardware substrate (what is kept hot). Grounded in fak's own code, and honest about where the clean story breaks — the popular claim that trajectory tricks turn a weak agent into a strong one is true only on a live re-run, and false for a frozen trajectory."
slug: three-layers-of-agent-optimization
keywords:
  - agent optimization
  - skill optimization
  - trajectory optimization
  - hardware co-design
  - KV cache residency
  - prompt optimization
  - off-policy evaluation
  - inference-time compute
date: 2026-06-24
---

# Three layers of agent optimization: skill, trajectory, substrate

When someone says they are "optimizing an agent," they usually mean one of three different
things that happen to share the word. This note separates them, surveys the research that
lives in each, and is deliberate about where the tidy three-layer picture stops being true.

The three layers, in one line each:

1. **Skill** — optimize *what the agent is supposed to do*. You rewrite the intent: the
   prompt, the program, the constitution, the reusable skill library, or the weights. This
   layer is allowed to change what counts as a good answer.
2. **Trajectory** — optimize *the realized execution path*, abstract over which skill
   produced it. You take the run as a sequence of states, actions, and results and bend it
   toward a better-scoring path by search, verification, refinement, stitching, or context
   reconstruction. The skill is held fixed; the path is the object.
3. **Substrate** — optimize *what is physically resident and cheap*. You decide which KV
   spans, weights, and prefixes stay hot and which get evicted, demoted, quantized, or
   recomputed. The intent and the path are held fixed; only the cost of running them moves.

The litmus that sorts any technique onto a layer: if making the component "better" could
change *which* answer is considered correct, it is layer 1; if it only changes *how* a
fixed answer is reached, it is layer 2; if it only changes *where the bytes live and how
fast they are*, it is layer 3.

| Layer | Mutates | Takes as given | Time-scale | Canonical research |
|---|---|---|---|---|
| 1 · Skill | The objective: prompt, program, constitution, skill library, weights | Nothing (it redefines "good") | Amortized (train / author once) | APE, OPRO, DSPy/MIPRO, TextGrad, GEPA, Voyager, DreamCoder, Constitutional AI, Agent Skills |
| 2 · Trajectory | The executed state-action-result sequence; which spans are in context | The skill's intent; the substrate's cost | Per-instance (per session / per turn) | Decision/Trajectory Transformer, ToT/LATS, self-consistency, PRM800K, Reflexion, STaR, doubly-robust OPE |
| 3 · Substrate | Physical residency: KV placement, eviction, tier, precision | Intent and path shape | Per-token (hardware-bound) | FlashAttention, PagedAttention/vLLM, RadixAttention, H2O, StreamingLLM, Quest, KIVI, speculative decoding |

The rest of this note walks each layer, then spends most of its length on the three
*differences* between them, because that is where the interesting and contestable claims
live.

## Layer 1 — optimizing the skill

A skill optimizer searches over the task specification itself: the instructions, the
demonstrations, the decomposition into subroutines, the success criterion. The defining
move is that it may change what the agent is trying to do.

The research splits into three families:

- **Automatic prompt and program optimization.** APE (2022) framed instruction design as
  black-box search. OPRO (2023) made the LLM the optimizer, feeding it past prompts with
  their scores. DSPy/MIPRO (2023–24) compile a typed pipeline, jointly tuning instructions
  and few-shot demonstrations against a metric. TextGrad (2024) backpropagates
  natural-language "gradients" through a compound system. GEPA (2025) evolves prompts by
  reflecting on full execution traces and keeping a Pareto front, and reports beating RL
  fine-tuning at a fraction of the rollouts. PromptBreeder and EvoPrompt (2023) run
  evolutionary search where the mutation operators are themselves LLM-generated.
- **Skill and library learning.** Voyager (2023) has an agent write, verify, and store
  executable skills in a growing library, then compose them. DreamCoder (2020) is the
  principled ancestor: a wake-sleep loop that abstracts reusable subroutines into a learned
  DSL, reshaping the hypothesis space itself. Agent Workflow Memory (2024) mines reusable
  sub-routines from past trajectories and injects them back as memory.
- **Behavior specification.** Constitutional AI (2022) writes behavior as an explicit
  document the model critiques itself against, so the reward *is* the spec. Anthropic's
  Agent Skills (SKILL.md, 2025) package procedural knowledge as model-readable folders
  loaded on demand.

The open problem that haunts this layer is Goodhart. A spec optimizer tunes to a small
judged eval, so a "win" can be metric memorization that fails to transfer. When the judge
is itself an LLM, the loop is circular and wants a disinterested verifier it does not have.

## Layer 2 — optimizing the trajectory

Layer 2 treats the execution path as a first-class object, independent of the generator
that emitted it. Three mechanism families bend the path:

- **Classical trajectory optimization and control.** iLQR/DDP and direct collocation /
  MPC iterate a candidate path toward a local optimum and re-solve each step under
  receding-horizon control. This is the deterministic ancestor of "optimize the realized
  path, not the controller."
- **Sequence-model and offline-RL stitching.** Decision Transformer (2021) conditions on a
  desired return and assembles a high-return path; Trajectory Transformer (2021) plans by
  beam search over a learned model, stitching high-value segments out of mostly mediocre
  logged trajectories. This is the deepest analog of "recombine good sub-paths from bad
  ones."
- **Inference-time search and verification.** Self-consistency and best-of-N (2022)
  marginalize over many sampled paths; Tree of Thoughts / LATS / MCTS (2023) search a tree
  of partial paths with a value estimate; process reward models (PRM800K, 2023) supervise
  each step; Reflexion and Self-Refine (2023) retry conditioned on self-critique; STaR /
  ReST / RFT (2022–23) keep only verified-correct trajectories and distill them.

### The convergence claim, told honestly

The seductive version of this layer is: *a bad skill becomes similar to a good skill at
execution time, because the optimizations are a normalizer.* The harness carries the weak
agent. It is partly true and the popular strong form is false, and the difference is worth
being precise about, because it is the whole reason a skill-agnostic serving layer can or
cannot work.

What is genuinely true: a stronger *harness* narrows the outcome gap between a weak and a
strong base model. ReAct, Reflexion, SWE-agent's agent-computer interface, and
repeated-sampling (Brown et al., 2024, "Large Language Monkeys") all lift task success at
fixed weights. Best-of-N with a verifier can take a weak model to strong-model coverage.

The catch is the word "execution time." Every one of those wins is a **live re-decode**.
Best-of-N draws N fresh samples. A skill library re-prompts the model so it decodes again.
Reflexion runs the model a second time on the critique. None of them re-optimizes a single
*frozen* trajectory into a strong outcome for free. Offline stitching only recombines
segments that already exist in the logged data, and a weak skill's trajectory does not
contain a strong skill's actions as observed transitions, so the strong path is off-support.

fak makes this concrete with its own machinery. The trajectory-replay spine
(`internal/turnbench/policyreplay.go`) scores K candidate policies against one recorded run
as model-free kernel replays, collapsing a product (K full agent+model runs) into a sum
(one recording plus K replays). It is sound only up to the first point a candidate policy
would have changed what the model *saw*. Past that divergence the trajectory forks, and
resolve-rate becomes, in the file's own words, "fiction the frozen trace cannot produce."
The honesty gate (`ope.go`, the `exact | bounded@i` witness) exists precisely because for a
deterministic policy the importance ratio at a divergence degenerates to 0/1 = 0: you
cannot reweight a weak trajectory into a strong one, you can only bound how wrong the
counterfactual is, with an interval that widens as the divergence deepens.

The decisive counterexample is in the repo. On a GPU-server SWE-bench Verified run
(`docs/benchmarks/SWEBENCH-VERIFIED-GPU-SERVER-RESOLVE-COMPARE.md`), the *same* local model
on the *same* instance resolves it through raw SGLang and is driven to an empty patch
through the fak gateway, because the trust floor denied a call the model never saw and the
run looped to its step limit. A harness change did not carry the agent toward a good
outcome. It forked the trajectory and the outcome went to zero. The harness *decides* the
outcome; it does not monotonically improve it.

So the refined claim: harness optimization can narrow a weak model's outcome gap, but only
by changing what the model re-decodes on a live run, not by re-optimizing a frozen path.
Trajectory-level re-scoring is skill-agnostic only in the *verdict mechanism* (a pure
function of tool, args, result, and policy) and is sound only on the prefix where the new
policy would not have changed what the model observed.

## Layer 3 — hardware co-design: what to keep hot

Every turn maintains a working set (the KV cache, the weights, cached prefixes) that has to
live somewhere on a hierarchy from SRAM through HBM, DRAM, far memory, disk, to recompute.
Layer 3 governs that residency. The defining act is eviction-and-placement under finite
fast memory.

- **The cache primitives.** FlashAttention (2022) is IO-aware exact attention that keeps the
  working tile in SRAM and never materializes the score matrix. PagedAttention/vLLM (2023)
  treats KV like OS virtual memory in non-contiguous blocks. RadixAttention/SGLang (2024)
  keeps every request's KV in a radix tree keyed by token ids, discovers the longest shared
  prefix, and recomputes only the suffix.
- **Informed eviction.** H2O (2023) found that a few "heavy hitter" tokens carry most
  attention mass and keeps those plus a recent window. StreamingLLM (2023) found "attention
  sink" tokens that let a model stream to unbounded length with bounded KV. Scissorhands,
  SnapKV, FastGen, and Quest (2024) refine the keep-oracle to be access-pattern- and even
  query-conditioned. KIVI (2024) demotes precision instead of presence.
- **The cluster frontier.** Speculative decoding (2023) and disaggregated prefill/decode
  with KV-aware routing (Mooncake, DistServe, and the productized smart routers) make
  residency a cluster property: route a request to the replica already holding its prefix.

fak's substrate mirrors this with a stricter guarantee. `internal/radixkv` rebuilds the
RadixAttention prefix tree over kernel-owned KV primitives and adds policy eviction (evict a
named poisoned span regardless of recency), with `internal/kvmmu` turning a quarantine
verdict into a bit-exact span eviction (re-RoPE the survivors so the result is byte-identical
to never having seen the span). `internal/cachemeta` carries the tier model and the
demote-instead-of-evict cost comparison that a blind LRU cache cannot make. `internal/polymodel`
is the host-many-share-prefill-decode-one substrate, with speculative accept arithmetic.

## The three differences (the part that matters)

The layers are easy to name and easy to over-unify. Here is where each pair actually
differs, and where the clean story leaks. All three refinements below survived an adversarial
pass that tried to break them.

### (a) Skill vs trajectory — changing the target is not normalizing the path

A skill edit can change which answer is correct. A trajectory optimizer takes the objective
as fixed and bends the realized path toward it. That is a real type difference: a skill
change can *relocate the optimum* and is exposed to Goodhart and reward-hacking in a way a
pure cost reduction is not.

But "categorically incommensurable" is too strong. Off-policy evaluation is the entire field
whose job is to score a changed policy against data logged under a different one. fak's
policy-replay shows the relationship is graded, not a wall: two different policies can both be
`exact` on the same trace (they differ only on calls that trajectory never made), and a
changed target is commensurable on the invariant prefix at near-zero model cost, partially
commensurable on the divergent suffix via a bounded estimate, and genuinely incommensurable
only past the divergence frontier. The honest framing is a coupled, graded commensurability
with a measurable frontier, and skill is simply the axis *most* likely to move the target,
so its gains need a divergence or faithfulness witness before they are trusted.

### (b) Trajectory vs hardware — "should be present" vs "is hot"

There is a real producer/consumer relationship here, and it is tempting to call it a duality.
The trajectory layer (`internal/ctxplan`) is a query optimizer that decides which spans
*should* be resident: a budgeted knapsack over a lossless store, where a miss is a cheap
demand-page, not a lost fact. The substrate decides which model-bound KV bytes *are* hot,
where a miss is a full re-prefill. The O(1)-context economics result
(`docs/explainers/o1-context-window-economics.md`) even gives the exchange rate: a freshly
reconstructed window beats a warm prefix cache exactly when the window is smaller than the
cache's effective discount, and the crossover fraction (0.118) equals the measured cache
multiplier (0.118).

The honest correction is that this is a **complement, not an adjoint**. "Should be present"
is lossless and model-agnostic; "is hot" is model/tokenizer/position-bound KV bytes that are
garbage under a different binding. The two live on different lattices with different miss-cost
units, so neither is the dual of the other. Two further corrections: informed eviction does
beat blind LRU, and it does so *losslessly* in fak (demote-not-evict weighing stage cost
against recompute cost), but the H2O/SnapKV/Quest family is a *third, orthogonal* lever —
lossy intra-sequence compression that trades fidelity for a fixed budget — not the residency
dual it is often cited as. And both levers are upstream-dominated by prefix *stability*: in
measured agentic serving, moving one mutated byte (a per-request UUID) from the head to the
tail took the cache hit rate from 0.3% to 87%, which dwarfs any eviction-policy tuning.

### (c) Why all three, and where the stack analogy breaks

The appealing unification is a compiler: source (authored prompt) to IR optimization passes
(context-layout planner) to ISA and cache hierarchy (KV mechanism). The co-design half of
that picture is right and is where the live work is. One quarantine verdict can propagate
from a source-level taint, through a planner decision, into a bit-exact span eviction, a
single decision crossing all three layers. The planner's optimal window is pinned to the
cache's discount structure. Co-design, not any single layer, is the frontier.

The clean-stack half is where fak refutes its own analogy by design. A compiler's value is
that the layers are *decoupled* behind a stable ISA: an IR pass can ignore cache-line sizes
for correctness. fak deliberately *fuses* the safety boundary and the reuse boundary into one
write-time gate (the same code is both injection containment and paging into the cache).
There is no stable ISA: KV reuse is intra-model-only and re-contracts every model release,
so the bottom layer behaves like a backend recompiled each release, not a fixed target. And
the axes are incommensurable: bit-exact forgetting is a *chosen, paid-for* property, not a
hard external constraint, so cost, latency, and correctness do not reduce to one number. The
source-to-transform-to-mechanism *gradient* is real; the clean three-layer *stack* is not.

## Related concepts: a placement inventory

Almost every adjacent idea is a claim about one of the three layers. Sorted by what it
mutates:

- **Test-time / inference-time compute scaling** (o1-style; Snell et al., 2024) shifts effort
  from the skill layer (bigger train) to the trajectory layer (longer, searched inference).
- **The Bitter Lesson** (Sutton, 2019) is the prior that general compute over search beats
  hand-crafted skill. Read carefully it is layer-neutral: it could equally justify pouring
  compute into bigger models (skill) or into trajectory and substrate.
- **Amortized vs per-instance optimization** is the time-scale axis itself: skill is
  amortized (train once), trajectory is per-instance (per-session search), substrate is
  per-token.
- **Memory and retrieval** straddle. RAG (2020) and MemGPT/virtual context (2023) shape
  context per turn (trajectory); KV-as-memory and semantic caching live in the substrate.
  The cleanest failure they expose is using *salience* (matters now) as a *durability* (stays
  true) trigger, so a fire alarm gets remembered and a lifelong preference gets dropped.
- **Caching theory** (LRU, LFU, Belady-optimal, TTL, 1966 onward) is the substrate's native
  theory. The agentic twist is eviction by *meaning or legality*, not only recency, with
  Belady clairvoyance impossible because future legality is unknown.
- **Sufficient statistics and the information bottleneck** are the deep "what to keep"
  question. A bounded context window is a bet that it is a sufficient statistic for the next
  turn, which is why a lossless store plus a bounded planner is the honest hedge.
- **Speculative execution, prefetch, branch prediction** are the substrate's borrowed
  vocabulary. The limit is that physical coherence keeps bytes identical but cannot know a
  `git push` invalidated a cached `git status`; semantic coherence is the kernel's job.
- **Recursive self-improvement / STaR-style loops** try to close the skill loop. They are
  the one layer fak deliberately does not touch.

The four axes that actually separate the layers, table-ready: *mutability target* (weights /
context / KV bytes), *time-scale* (amortized / per-instance / per-token), *commensurability*
(FLOPs / tokens / bytes-resident), and *ownership* (model lab / harness / serving engine).

## Where fak sits, and the open question

fak is, on purpose, a layer-2-and-3 system with layer 1 left empty. Its stated moat is
"structural disinterest as referee": it optimizes the trajectory and the substrate for *any*
skill and authors none, on the bet that a model author structurally cannot be a disinterested
referee. The strongest case for the bet is that substrate and legality optimizations
(provable forgetting, legal reuse) do not commoditize as models improve, unlike scaffolding.
The strongest case against is that skill-agnosticism forecloses the *tightest* co-design path
(for example, "this task only ever touches these tools, so pin their KV hot"), so disinterest
is bought at the cost of the deepest cross-layer optimization the thesis describes.

The most important open question is not conceptual but a wiring fact. On the flagship route
(`fak guard -- claude`, the Anthropic passthrough), the model reads the original request
bytes verbatim (`req.Raw`, `internal/gateway/messages.go`), while every byte-level layer-2
rewrite targets `req.Messages`, which is never re-serialized on that route. So the
write-time admitters `ctxmmu` and `normgate`, and the `ctxplan` view planning behind them,
are mechanism-proven but **dormant** on the live wire; only the information-flow taint
high-water mark and the adjudication of proposed tool calls are actually live there. (The
window-economics tools `tools/ctxcost.py` and `tools/ctxwin.py` are offline analyses, not
live rewrite paths.) Most of the layer-2 and layer-3 packages (`ctxplan`, `radixkv`,
`kvmmu`, `polymodel`) are also absent from the default config's link set. The whole co-design
story currently rests on a single unbuilt seam: an outbound `req.Raw` transform (the
`ctxplan` #555/#556 gateway-wiring item) that collides head-on with the `cache_control`
prefix preservation the passthrough was built to protect. Until that seam exists, fak proves
the mechanisms and does not yet run them on the model-facing path.

---

*Method: this note is the synthesis of a parallel research survey (six research sheets, four
adversarial verdicts, one synthesis pass) cross-checked against fak's own code and
benchmarks. Every framing claim here is the refined form that survived an adversarial pass;
all four original strong claims came back "partial," and the refinements are stated above
rather than the overclaims. External works are cited by name and year; internal claims point
at the file or doc that grounds them.*
