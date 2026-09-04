---
title: "The fak Learning Path — a prerequisite-ordered course"
description: "A linear, prerequisite-based curriculum across every fak concept: 99 courses in six levels, from \"what is fak\" to landing an optimization in the kernel. Join at the level that matches your background and walk straight through."
---

# The fak learning path

*This page owns one job: teaching fak's ideas in prerequisite order. It is for a reader who
wants to understand the system, not evaluate it ([README](README.md)), install it
([GETTING-STARTED](GETTING-STARTED.md)), route a task ([START-HERE](START-HERE.md)), or look
a page up by name ([INDEX](INDEX.md)). Nothing here is required to run fak.*

fak is a lot of ideas stacked into one binary: an addressable KV cache that keeps long
sessions cheap to hold warm, right-model-per-call routing, and a pure-Go in-kernel model
(preferring Qwen3.8 for native-performance work per AGENTS.md) —
and, riding along on the same write-time checkpoint, a default-deny capability floor, a
write-time result quarantine, and the honesty discipline that keeps every claim checkable.
This page turns all of it into one **linear, prerequisite-ordered curriculum** — a course
catalog, not a doc dump.
Each course points at the doc that already teaches it; the value added here is the
**order** and the **prerequisites**, so you always have the background a page assumes
*before* you open it.

> **Want the integrated system story first?** Take the
> [8-module flagship course](docs/courses/end-to-end-inference-agent-harness-memory.md) to
> follow one request across native inference, the agent harness, policy, context control,
> durable memory, observability, and proof. Then use this 99-course catalog to enter at the
> right prerequisite level or deepen a subsystem. The course is a guided end-to-end route;
> this page remains the prerequisite-ordered concept front door.

**You do not have to start at the beginning.** Find the row in
[Find your starting point](#find-your-starting-point) that matches your background, start
at that course, and walk forward. The catalog is a strict prerequisite order — every
course's *hard* prerequisites are lower-numbered courses — so reading top-to-bottom never
lands you on a concept whose prerequisite you have not met yet.

99 courses, six levels (100 → 600), from "what is fak" to landing an optimization into
the kernel. The readings are the docs you would read anyway; the path is what stops you
reading them in the wrong order.

> New to the project entirely? The fastest taste of the payoff is one offline pass that
> prints the token/turn savings from the shared prefix — `go run ./cmd/fak agent --offline`
> (see **FAK 104**); the same run also prints the safety A/B, which
> [`README.md`](README.md#try-fak) frames as the
> security-side boundary proof. Either way, come back here and start at **FAK 101**. Just
> want to install and run? [`GETTING-STARTED.md`](GETTING-STARTED.md) is the install-and-run
> page and [`START-HERE.md`](START-HERE.md) routes a job you already have; this page is the
> *concept* front door.

## Learn without mysteries

Do not memorize fak's nouns first. Use the same five-step loop whenever you want to
understand or change a process:

1. **Predict** one observable result in plain language.
2. **Run** the smallest real command that can prove you wrong.
3. **Locate** the winning rule in the output and then in the named config or source.
4. **Adjust one input** while keeping everything else fixed.
5. **Rerun** and explain why the result changed. If you cannot, the mechanism is still a
   mystery; follow the winning rule one level deeper before adding machinery.

That loop separates three questions that are easy to conflate:

- **What happened?** Read the verdict or measurement.
- **Why did it happen?** Read the winning rung and reason; `DEFER` means “this rung did
  not decide,” not “allow.”
- **What can I change?** Change the input owned by that winning rung, not an unrelated
  downstream setting.

### Five-minute policy lab: predict, inspect, adjust

This lab needs no key, model, or GPU. It uses the real policy fold rather than a diagram.
Start by predicting all three outcomes, then run:

```powershell
fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}" --explain
fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}" --explain
fak preflight --policy examples/customer-support-readonly-policy.json --tool mystery_action --args "{}" --explain
```

The expected winners are deliberately different:

| Tool | Result | Plain-language reason | Policy input to inspect |
|---|---|---|---|
| `refund_payment` | `DENY / POLICY_BLOCK` | The manifest names this tool in `deny`. | `deny.refund_payment` |
| `search_kb` | `ALLOW` | The tool matches the `search_` allow prefix. | `allow_prefix` |
| `mystery_action` | `DENY / DEFAULT_DENY` | No rule grants the unknown tool. | `posture: fail_closed` |

The same method holds when the input is unfriendly. These are argument-rule cases, not
global claims that every tool needs arguments or that every call has one fixed size cap:

| Input class | One input change | Expected result | Winning rule |
|---|---|---|---|
| Empty required argument | Remove a value required by a positive path constraint. | `DENY / POLICY_BLOCK` | The `allow_glob` rule cannot prove that a missing value is contained. |
| Oversized string argument | Grow one scalar past its policy-declared byte bound. | `DENY / OVERSIZE` | The matching `max_bytes` argument rule. |
| Malformed quote-wrapped argument | Open a quote in a constrained value without closing it. | `DENY / MALFORMED` | Canonicalization fails closed and returns a bounded repair hint. |
| Hostile denied argument | Make one scalar match its policy-declared deny expression. | `DENY / POLICY_BLOCK` | The matching `deny_regex` rule; the witness names the rule, not the input. |

`TestMysteryFreeAdjustmentEdgeAdversarial` drives all four rows through the real
adjudicator and checks that both learning documents keep the same table. Run the full
edge/adversarial witness with:

```powershell
go test ./internal/adjudicator -run 'Edge|Adversarial' -v
```

Now change exactly one policy fact in a temporary copy and predict the new result:

```powershell
$p = Get-Content examples/customer-support-readonly-policy.json | ConvertFrom-Json
$p.allow += "mystery_action"
$tmp = Join-Path $env:TEMP "fak-learning-policy.json"
$p | ConvertTo-Json -Depth 10 | Set-Content $tmp
fak preflight --policy $tmp --tool mystery_action --args "{}" --explain
```

The last verdict should be `ALLOW`: the same call now has one affirmative policy rule.
The original manifest is untouched. To continue investigating, use the trace's winner:
`adjudicator.Adjudicator` leads to [`internal/adjudicator`](internal/adjudicator), while the
manifest format and safe editing workflow live in
[`docs/fak/policy-guide.md`](docs/fak/policy-guide.md). This is the pattern for every later
course: predict → run → locate → adjust one thing → rerun. For a cross-process map of the deciding evidence and owned knobs, use the [`mystery-free adjustment atlas`](docs/fak/mystery-free-adjustment-atlas.md).

## How to read a course

Each course is one entry shaped like a syllabus line:

- **Prerequisites** — *hard* dependencies. These will block this course's lab or
  checkpoint if you skip them, and they are always lower-numbered, so they sit above this
  course in the catalog.
- **Background** — *context* prerequisites: helpful framing you can defer. Skipping them
  costs you some "why", not the ability to do the lab.
- **You'll be able to** — the concrete skills the course certifies.
- **Read** — the canonical doc(s). This is the actual course material.
- **Lab** — a command you can run (most need no key, model, or GPU) or a hands-on task.
- **Checkpoint** — answer it (or do it) to certify yourself before moving on. If you can
  clear a level's checkpoints, you have met the `assumed_passed` bar for the next level.

Honesty carries through the whole catalog: where a number is **SIMULATED** or a proof is
**OPEN**/**REFUTED**, the checkpoint says so. The headline multipliers are stated against
the *naive* baseline and the *tuned-SOTA* baseline separately, never blended — see
**FAK 605**.

## Find your starting point

Start at the course in the **Start** column, then follow the **Route** straight through
to the destination. The route already lists every hard dependency in between, in order —
so you can join mid-catalog without hitting a wall. Anyone can also just start at
**FAK 101** and read every course in number order.

| Your background | Start | Route (in order) → destination |
|---|---|---|
| Total newcomer — knows what an AI agent and a tool call are, nothing else | **FAK 101** | FAK 101 → FAK 102 → FAK 103 → FAK 104 → FAK 105 |
| App dev who only calls an LLM API and wants governance with minimal agent rewrite | **FAK 101** | FAK 101 → FAK 102 → FAK 103 → FAK 104 → FAK 105 → FAK 207 → FAK 301 → FAK 310 → FAK 501 → FAK 502 → FAK 503 → FAK 511 |
| Platform / SRE who already runs vLLM or SGLang in production | **FAK 201** | FAK 201 → FAK 103 → FAK 207 → FAK 301 → FAK 303 → FAK 304 → FAK 310 → FAK 501 → FAK 502 → FAK 503 → FAK 504 → FAK 505 → FAK 507 → FAK 314 → FAK 510 → FAK 535 |
| Security engineer who already knows prompt injection, default-deny, reference monitors | **FAK 105** | FAK 105 → FAK 207 → FAK 103 → FAK 301 → FAK 302 → FAK 303 → FAK 304 → FAK 305 → FAK 306 → FAK 307 → FAK 308 → FAK 309 → FAK 310 → FAK 311 → FAK 312 → FAK 313 → FAK 314 → FAK 315 → FAK 318 |
| ML-systems / kernel hacker who wants the in-kernel model and compute HAL | **FAK 201** | FAK 201 → FAK 205 → FAK 207 → FAK 210 → FAK 401 → FAK 521 → FAK 522 → FAK 523 → FAK 524 → FAK 525 → FAK 526 → FAK 404 → FAK 405 → FAK 406 → FAK 527 → FAK 528 → FAK 529 → FAK 530 → FAK 532 |
| Memory / RAG engineer focused on what fak persists, forgets, and reuses | **FAK 202** | FAK 202 → FAK 203 → FAK 201 → FAK 205 → FAK 207 → FAK 301 → FAK 303 → FAK 310 → FAK 316 → FAK 307 → FAK 407 → FAK 409 → FAK 402 → FAK 401 → FAK 412 → FAK 413 → FAK 414 |
| Compliance / audit / governance engineer (journal, provenance, deletion, honesty discipline) | **FAK 105** | FAK 105 → FAK 207 → FAK 103 → FAK 301 → FAK 303 → FAK 310 → FAK 311 → FAK 312 → FAK 313 → FAK 314 → FAK 315 → FAK 317 → FAK 404 → FAK 405 → FAK 406 → FAK 411 → FAK 601 → FAK 602 → FAK 606 → FAK 614 → FAK 307 → FAK 616 |
| Contributor / autonomous agent landing an optimization into the kernel | **FAK 207** | FAK 207 → FAK 208 → FAK 209 → FAK 210 → FAK 614 → FAK 615 → FAK 616 → FAK 617 |

> The **Route** is the *hard-dependency* path. You can read the context prerequisites
> noted on each course later (or never) without breaking a lab.

## The level ladder

Read the ladder performance-first: L400's cache reuse and the in-kernel model are the
headline payoff, and the L300 security floor rides the same write-time checkpoint rather
than sitting on a separate path.

```
L100  Orientation .................. what fak is, the one idea, the two gates      (start cold)
  |
L200  Foundations .................. KV cache, context != memory, content addressing,
  |                                  the frozen ABI, the proofs method
  +--> L300  Security Core ......... the in-process default-deny floor + the write-time wall
  +--> L400  Performance Core ...... cache reuse, addressable eviction, the scaling laws
            |
            +--> L500  Serving / Integration / In-Kernel Model
                       run & harden the gateway, repoint one base URL, the pure-Go model + HAL
                       |
                       +--> L600  Mastery .. benchmarks, the honesty discipline, extend the kernel
```

Each level states the courses it assumes you can already pass. If you can clear those
checkpoints, you are qualified to start there.

| Level | Theme | Assumes you can pass |
|---|---|---|
| **L100 — Orientation** | The plain category, the syscall framing, the two gates, the recurring vocabulary, and how to prove the boundary is real in two minutes. | — (start cold) |
| **L200 — Foundations** | The handful of mechanisms every later claim rests on: the KV cache, context-vs-memory durability, the four memory layers, content addressing, the frozen ABI, and the proofs method. | FAK 101, FAK 102, FAK 103, FAK 104, FAK 105 |
| **L300 — The Security Core** | The reference monitor, the policy lifecycle, the rungs (preflight, plan-CFI, witness, stewards, rate-limit, escalation), the write-time result gate, canonicalization, IFC, provenance, durability, and code-linting at the same boundary. | FAK 105, FAK 207 |
| **L400 — The Performance Core** | Why agents stress the cache, prefill-elimination economics, the addressable/bijective KV-MMU, RadixAttention reuse, the vDSO, durable session recall, and the first-order scaling law (incl. cache legality and residency). | FAK 201, FAK 205, FAK 310 |
| **L500 — Serving, Integration, and the In-Kernel Model** | Running and hardening the gateway (`fak serve`, `fak l3-serve` / `fak l3serve`), the gateway drop guarantee, repointing existing agents at one base URL, the framework cookbook, the pure-Go in-kernel model + compute HAL with oracle parity (preferring Qwen3.8 for native performance per AGENTS.md), and the GPU lease. | FAK 105, FAK 301, FAK 304, FAK 310 |
| **L600 — Mastery** | Honest baselines and the benchmark authority, the fleet/web/parity results, the AgentDojo red-team, the claims ledger and status gates, the additive ABI + architest, the RSI ship-gate, the three-gate leaf pattern, and the dispatch loop. | FAK 207, FAK 208, FAK 209, FAK 210 |

---

## The catalog

## L100 — Orientation: what fak is and the one idea

**Theme.** The plain category, the syscall framing, the two gates, the recurring vocabulary, and how to prove the boundary is real in two minutes.

**Who joins here.** A total newcomer, or anyone who has never seen fak. You only need to know what an AI agent is, what a tool call is, and roughly what a model server (vLLM, llama.cpp) does. Start here if any of fak's one-liners ('untrusted program', 'two gates', 'security == reuse') are not yet obvious to you.

| Course | Hard prerequisites |
|---|---|
| **FAK 101** — What fak Is: One Binary Between Agent and Tools | — |
| **FAK 102** — The Core Move: Untrusted Program, Tool-Call-as-Syscall (and the Word List) | **FAK 101** |
| **FAK 103** — The Parable and the Two Gates | **FAK 102** |
| **FAK 104** — The Convergence: Security Boundary == Reuse Boundary | **FAK 103** |
| **FAK 105** — Adoption Rungs and the 2-Minute Honest Proof | **FAK 104** |

### FAK 101 — What fak Is: One Binary Between Agent and Tools

**Prerequisites:** —

**You'll be able to:**
- State in one sentence what fak is and name one thing it explicitly is NOT (it is not a faster model server)
- Name two of the four questions fak owns that a token engine leaves open
- Build the single binary and print its version

**Read:** [`README.md`](README.md), [`START-HERE.md`](START-HERE.md), [`docs/FAQ.md`](docs/FAQ.md)

**Lab:**
```bash
go run ./cmd/fak version          # confirm the single binary builds and prints its version
go run ./cmd/fak version modules -top 10  # version-everything: per-module rev+date over the tree (the fak-module-versions/1 ledger, 410 modules)
```

**Checkpoint:** In one sentence, state what fak is and name one thing it is explicitly NOT. Name two of the four questions fak owns that token engines leave open.

### FAK 102 — The Core Move: Untrusted Program, Tool-Call-as-Syscall (and the Word List)

**Prerequisites:** **FAK 101**

**You'll be able to:**
- Reframe the model as an untrusted program and each tool call as a syscall on a controlled path
- Explain why an in-process default-deny check differs structurally from a pre-tool hook or a second 'is this safe?' model
- Pin the recurring vocabulary: preflight (before-gates) vs inflight (during-state) vs prefill (KV economics), plus adjudicator/fold/rung/monitor/admit
- Run a denied call and read the DEFAULT_DENY verdict

**Read:** [`docs/concepts-and-story.md`](docs/concepts-and-story.md), [`docs/fak/faq.md`](docs/fak/faq.md), [`docs/glossary.md`](docs/glossary.md), [`README.md`](README.md)

**Lab:**
```bash
go run ./cmd/fak preflight --tool create_user --args '{"_positional":["alice"]}'  # adjudicated as a syscall -> DENY DEFAULT_DENY
```

**Checkpoint:** Explain why putting the check ON the same in-process call path (default-deny) is structurally different from a pre-tool hook or an LLM judge. Then disambiguate preflight vs inflight vs prefill, and say what 'the lever was never wired up' means concretely.

### FAK 103 — The Parable and the Two Gates

**Prerequisites:** **FAK 102**

**You'll be able to:**
- Map the night-shift-clerk parable onto fak mechanisms: locked drawer, screened notes, imperfect screener
- Name the two independent gates (the lock/capability floor and the wall/quarantine) and what each protects against
- Explain why the detector on top is treated as evadable by design and why that does not weaken the floor

**Read:** [`docs/concepts-and-story.md`](docs/concepts-and-story.md), [`docs/fak/faq.md`](docs/fak/faq.md), [`README.md`](README.md)

**Lab:**
```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args '{}'  # the lock: DENY POLICY_BLOCK
```

**Checkpoint:** Name the two gates and what each protects against (effect vs. context entry). Why is an attacker beating TWO gates harder than fooling one classifier, and why is the detector deliberately treated as evadable?

### FAK 104 — The Convergence: Security Boundary == Reuse Boundary

**Prerequisites:** **FAK 103**

**You'll be able to:**
- Explain how one write-time gate is simultaneously a security act and an optimization act
- State the two honest fences on the convergence (which workload, which metric it does NOT win)
- Run one offline pass that prints both the safety A/B and the token/turn savings

**Read:** [`README.md`](README.md), [`docs/concepts-and-story.md`](docs/concepts-and-story.md)

**Lab:**
```bash
go run ./cmd/fak agent --offline  # one run prints the safety A/B AND the token/turn savings from the same boundary
```

**Checkpoint:** Explain how one write-time gate is both security and optimization. State the two honest fences: which workload it is a win for, and which metric (raw GPU throughput) it does NOT win.

### FAK 105 — Adoption Rungs and the 2-Minute Honest Proof

**Prerequisites:** **FAK 104**

**You'll be able to:**
- List the three adoption rungs (front your model / offline kernel / fused in-kernel model) least-to-most committed and pick a starting rung
- Identify which rung unlocks the reuse win and the self-host fence on it
- Run the 2-minute proof (a structural DENY and an ALLOW) and read the headline numbers against SOTA, not a strawman

**Read:** [`docs/fak/tutorial.md`](docs/fak/tutorial.md), [`README.md`](README.md), [`START-HERE.md`](START-HERE.md)

**Lab:**
```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args '{}' && go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args '{}'
```

**Checkpoint:** List the rungs least-to-most committed and say which most adopters should start at and why. Run the proof and report both verdicts; state what ~60x compares against vs ~4x, what is SIMULATED, and what the prior-art audit scored (0/29-novel) and what the contribution actually is.

---

## L200 — Foundations: the load-bearing mechanisms

**Theme.** The handful of mechanisms every later claim rests on: the KV cache, context-vs-memory durability, the four memory layers, content addressing, the frozen ABI, and the proofs method.

**Who joins here.** Someone comfortable with the orientation framing who wants the underlying mechanics. Join here if you already know fak is a governing binary and want to understand the KV cache, content-addressed stores, and how the repo proves things before you touch the security or performance cores.

**Assumes you can already pass:** **FAK 101**, **FAK 102**, **FAK 103**, **FAK 104**, **FAK 105**.

| Course | Hard prerequisites |
|---|---|
| **FAK 201** — What a KV Cache Is and Why Reuse Is Always a Prefix | **FAK 105** |
| **FAK 202** — Context Is Not Memory: The Truth-Duration Axis | **FAK 105** |
| **FAK 203** — Why Memory Systems Get Promotion Backwards | **FAK 202** |
| **FAK 204** — The Four Layers of Agent Memory | **FAK 201** |
| **FAK 205** — Content-Addressed Blob Store (CAS) | **FAK 201** |
| **FAK 206** — cachemeta: Payload-Free Binding Keys | **FAK 205** |
| **FAK 207** — The Proofs Method: Theorem, Witness, Verdict, DOS | **FAK 105** |
| **FAK 208** — The Frozen Additive-Only ABI and Registry Seams | **FAK 207** |
| **FAK 209** — architest: Layered DAG, Tier Rules, and Hot-Path Hygiene | **FAK 208** |
| **FAK 210** — The Reference/Approx Correctness Contract | **FAK 207** |

### FAK 201 — What a KV Cache Is and Why Reuse Is Always a Prefix

**Prerequisites:** **FAK 105**

**You'll be able to:**
- Explain why token i's K/V depends only on tokens 0..i and why causality forces reuse to be a prefix
- Predict that a change at position N invalidates everything from N on
- Run the offline prefix-divergence script and watch longest-common-prefix reuse climb on an append-only loop

**Read:** [`docs/explainers/kv-cache-agentic-context.md`](docs/explainers/kv-cache-agentic-context.md), [`docs/glossary.md`](docs/glossary.md)

**Lab:**
```bash
Run the offline prefix-divergence script from the doc: feed it JSONL of {"turn": i, "tokens": [...]} per line and watch the longest-common-prefix reuse climb toward 100% on an append-only loop.
```

**Checkpoint:** Explain why token i's K/V depends only on tokens 0..i, and why that causality forces reuse to be a prefix rather than an arbitrary mid-sequence span. Then state the prefill-vs-prefix distinction the glossary pins.

### FAK 202 — Context Is Not Memory: The Truth-Duration Axis

**Prerequisites:** **FAK 105**
  ·  **Background:** **FAK 201**

**You'll be able to:**
- Distinguish context from memory by truth-duration, not size, recency, or location
- Sort facts into context-only vs memory-worthy using verb/tense cues
- Explain why two surface-identical facts can be different durability classes

**Read:** [`docs/CONTEXT-IS-NOT-MEMORY.md`](docs/CONTEXT-IS-NOT-MEMORY.md)

**Lab:**
```bash
List 5 facts you'd tell an assistant today and sort each into context-only (let it expire) vs memory-worthy (durable), then state the verb/tense cue that decided each.
```

**Checkpoint:** Explain why "it's raining here now" and "I live somewhere it rains" are the same surface fact but different durability classes, and which one must never be promoted to memory.

### FAK 203 — Why Memory Systems Get Promotion Backwards

**Prerequisites:** **FAK 202**

**You'll be able to:**
- Show that overflow, recency, salience, and explicit-save are all proxies for 'relevant to now' (i.e. context, not durability)
- Name the single root cause shared by 'the ephemeral promoted' and 'the durable dropped'
- Diagnose a write trigger by the present-moment proxy it actually measures

**Read:** [`docs/CONTEXT-IS-NOT-MEMORY.md`](docs/CONTEXT-IS-NOT-MEMORY.md)

**Lab:**
```bash
For each of overflow/summarization, recency, salience scoring, and explicit user-save, write one sentence naming the present-moment proxy it measures and one ephemeral fact it would wrongly promote.
```

**Checkpoint:** Name the single root cause shared by 'the ephemeral promoted' and 'the durable dropped' failures, and why it is one bug, not two.

### FAK 204 — The Four Layers of Agent Memory

**Prerequisites:** **FAK 201**
  ·  **Background:** **FAK 205**

**You'll be able to:**
- Separate routing (where), addressing (name), fusion (zero-copy arena), and semantics (mutate/isolate/attribute/gate) as four distinct problems
- Apply the one-line test (is this true of a frozen single-writer cache that merely moved/named/co-located?) to classify a claim
- Place fak in the semantics layer and explain why it does not compete on raw throughput

**Read:** [`docs/MEMORY-LAYERS-EXPLAINER.md`](docs/MEMORY-LAYERS-EXPLAINER.md)

**Lab:**
```bash
Apply the one-line test to five sentences (e.g. 'two readers share one cell by digest', 'evict a poisoned span from the middle and survivors stay byte-correct') and label each routing/addressing/fusion vs semantics.
```

**Checkpoint:** Using the Docker<->Kubernetes analogy, explain why 'a KV router is not a better memory MMU' and which layer fak occupies.

### FAK 205 — Content-Addressed Blob Store (CAS)

**Prerequisites:** **FAK 201**

**You'll be able to:**
- Explain why making the address the sha256 of the bytes gives free dedup and a faithful Ref backend
- Show why byte-identical Puts from distinct arrays collapse to one digest while the inline path is not deduped
- State what is in-scope vs out-of-scope (durability, GC, collision-resistance)

**Read:** [`docs/proofs/blob.md`](docs/proofs/blob.md)

**Lab:**
```bash
go test ./internal/blob/ -count=1 -timeout 120s -run 'TestPutSmallInlineRoundTrip|TestPutLargeBlobRoundTrip|TestContentDedup' -v
```

**Checkpoint:** Explain why two Puts of byte-identical content from DISTINCT backing arrays collapse to one blob with one digest, and why the inline path (len<=256) is deliberately NOT deduped.

### FAK 206 — cachemeta: Payload-Free Binding Keys

**Prerequisites:** **FAK 205**

**You'll be able to:**
- Explain why a deterministic, injective fold (null-separated sha256) over binding axes guarantees no false hit
- Show why the 0x00 separator rules out 'ab'+'c' vs 'a'+'bc' aliasing
- Explain why a partial-axis match yields a typed MISS/FAULT rather than a wrong serve, and why provider telemetry is excluded from invalidation

**Read:** [`docs/proofs/cachemeta.md`](docs/proofs/cachemeta.md)

**Lab:**
```bash
go test ./internal/cachemeta/ -count=1 -timeout 120s -run 'TestManifestBindingDigestIsDeterministicOverBindingAxes|TestCheckResidentClaimRefusesBindingMismatch|TestPlanExternalInvalidationsDropsRemoteKVAndReferencingAttentionIndex' -v
```

**Checkpoint:** Why does the 0x00 field separator make the fold injective on the tuple? Explain how a near-collision (some axes equal) yields a typed MISS/FAULT rather than a wrong serve.

### FAK 207 — The Proofs Method: Theorem, Witness, Verdict, DOS

**Prerequisites:** **FAK 105**

**You'll be able to:**
- Distinguish the four verdicts (PROVEN / REFUTED / OPEN / SCOPED-OUT)
- Explain why a structurally-deterministic function with no repeated-call test stays OPEN, not PROVEN
- Explain what dos commit-audit adds on top of a green witness

**Read:** [`docs/proofs/00-METHOD.md`](docs/proofs/00-METHOD.md), [`docs/proofs/README.md`](docs/proofs/README.md)

**Lab:**
```bash
go test ./internal/architest ./internal/abi ./internal/adjudicator ./internal/shipgate
```

**Checkpoint:** Distinguish the four verdicts and explain why a structurally-deterministic function with no repeated-call test stays OPEN rather than PROVEN, plus what dos commit-audit adds on top of a green witness.

### FAK 208 — The Frozen Additive-Only ABI and Registry Seams

**Prerequisites:** **FAK 207**

**You'll be able to:**
- Name the only sanctioned way to add a new admission rung or engine (a new package + one Register*() call)
- Explain why renumbering an existing VerdictKind fails TestABIGoldenFreeze while appending a new value does not
- Explain why a shared spine that changes breaks every dependent worker in a multi-session tree

**Read:** [`ARCHITECTURE.md`](ARCHITECTURE.md), [`EXTENDING.md`](EXTENDING.md), [`docs/proofs/abi+architest.md`](docs/proofs/abi+architest.md)

**Lab:**
```bash
go test ./internal/abi/ -run 'TestABIGoldenFreeze|TestClosedReasonVocabulary' -v
```

**Checkpoint:** Name the only sanctioned way to add a new admission rung or engine, and explain why a renumber of an existing VerdictKind fails the golden freeze while appending a new value does not.

### FAK 209 — architest: Layered DAG, Tier Rules, and Hot-Path Hygiene

**Prerequisites:** **FAK 208**

**You'll be able to:**
- State the five tiers (root -> foundation -> mechanism -> composer -> integrator) and what an upward import produces
- Explain why the decision-path packages must never import os/exec
- Explain why the architest gate is build-tag-blind

**Read:** [`docs/proofs/abi+architest.md`](docs/proofs/abi+architest.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`SUBSYSTEM-CHECKS.md`](SUBSYSTEM-CHECKS.md)

**Lab:**
```bash
go test ./internal/architest/ -run 'TestNoUpwardImports|TestHotPathHasNoExec|TestEveryPackageDeclaresTier' -v
```

**Checkpoint:** State the five tiers and explain what failure a leaf importing a higher-tier package produces, and why a spawned subprocess on the decide path would kill the in-process syscall thesis.

### FAK 210 — The Reference/Approx Correctness Contract

**Prerequisites:** **FAK 207**

**You'll be able to:**
- Explain why Reference is held to max|delta|=0 plus the argmax oracle while Approx is held to argmax-exact plus a declared logit-cosine threshold
- Explain why a CUDA or quant backend declares Approx, not Reference
- Explain what RequireReference(b) prevents

**Read:** [`EXTENDING.md`](EXTENDING.md), [`docs/proofs/00-METHOD.md`](docs/proofs/00-METHOD.md)

**Lab:**
```bash
go test ./internal/compute/
```

**Checkpoint:** Explain why a CUDA backend declares Approx not Reference, and what RequireReference(b) prevents.

---

## The staged parts

To keep each stage a bounded read, the path ships as this page (overview plus
L100–L200) plus five staged parts under `docs/learning/`. The course numbers,
prerequisites, and checkpoints are continuous across the parts:

- [L300 — The Security Core](docs/learning/security-core.md) — the reference monitor, the policy lifecycle, and the enforcement rungs.
- [L400 — The Performance Core](docs/learning/performance-core.md) — why agents stress the cache and the addressable-eviction answer.
- [L500 — Serving, Integration, and the In-Kernel Model](docs/learning/serving-integration.md) — running and hardening `fak serve` (and `fak l3-serve` / `fak l3serve`), repointing real agents, and the in-kernel model (preferring Qwen3.8 per AGENTS.md).
- [L600 — Mastery](docs/learning/mastery.md) — benchmarks, honesty discipline, and extending the kernel.
- [The shipped-surface appendix](docs/learning/appendix-shipped-surface.md) — the wrap-up plus the full operator/contributor/package map.

