---
title: "The fak Learning Path — a prerequisite-ordered course"
description: "A linear, prerequisite-based curriculum across every fak concept: 98 courses in six levels, from \"what is fak\" to landing an optimization in the kernel. Join at the level that matches your background and walk straight through."
---

# The fak learning path

fak is a lot of ideas stacked into one binary: a default-deny capability floor, a
write-time result quarantine, an addressable KV cache, a pure-Go in-kernel model, and
the honesty discipline that keeps every claim checkable. This page turns all of it into
one **linear, prerequisite-ordered curriculum** — a course catalog, not a doc dump.
Each course points at the doc that already teaches it; the value added here is the
**order** and the **prerequisites**, so you always have the background a page assumes
*before* you open it.

**You do not have to start at the beginning.** Find the row in
[Find your starting point](#find-your-starting-point) that matches your background, start
at that course, and walk forward. The catalog is a strict prerequisite order — every
course's *hard* prerequisites are lower-numbered courses — so reading top-to-bottom never
lands you on a concept whose prerequisite you have not met yet.

98 courses, six levels (100 → 600), from "what is fak" to landing an optimization into
the kernel. The readings are the docs you would read anyway; the path is what stops you
reading them in the wrong order.

> New to the project entirely? The fastest taste is the 2-minute boundary proof in
> [`README.md`](README.md#see-it-in-2-minutes-no-key-no-model-no-gpu), then come back here
> and start at **FAK 101**. Just want to install and run? [`START-HERE.md`](START-HERE.md)
> and [`GETTING-STARTED.md`](GETTING-STARTED.md) are the install front doors; this page is
> the *concept* front door.

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
| Contributor / autonomous agent landing an optimization into the kernel | **FAK 207** | FAK 207 → FAK 208 → FAK 209 → FAK 210 → FAK 614 → FAK 615 |

> The **Route** is the *hard-dependency* path. You can read the context prerequisites
> noted on each course later (or never) without breaking a lab.

## The level ladder

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
| **L500 — Serving, Integration, and the In-Kernel Model** | Running and hardening the gateway, the gateway drop guarantee, repointing existing agents at one base URL, the framework cookbook, the pure-Go in-kernel model + compute HAL with oracle parity, and the GPU lease. | FAK 105, FAK 301, FAK 304, FAK 310 |
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
go run ./cmd/fak version  # confirm the single binary builds and prints its version
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

## L300 — The Security Core: the in-process default-deny floor and the write-time wall

**Theme.** The reference monitor, the policy lifecycle, the rungs (preflight, plan-CFI, witness, stewards, rate-limit, escalation), the write-time result gate, canonicalization, IFC, provenance, durability, and code-linting at the same boundary.

**Who joins here.** A security engineer, or anyone who has the Foundations and wants the actual enforcement machinery. Join here if you already understand the KV cache, fail-closed/default-deny, the proofs method, and content addressing, and want to learn how fak adjudicates calls and quarantines results.

**Assumes you can already pass:** **FAK 105**, **FAK 207**.

| Course | Hard prerequisites |
|---|---|
| **FAK 301** — Policy in the Kernel: The First Flip | **FAK 103**, **FAK 207** |
| **FAK 302** — What the Capability Floor Does and Does NOT Bound | **FAK 301** |
| **FAK 303** — The Default-Deny Adjudicator and Closed Refusal Vocabulary | **FAK 301** |
| **FAK 304** — Policy Manifests: Dump, Edit, Check, Load | **FAK 303** |
| **FAK 305** — Preflight Ladder and Grammar Argument-Repair | **FAK 303** |
| **FAK 306** — Plan Control-Flow Integrity (plan-CFI) | **FAK 303** |
| **FAK 307** — The Require-Witness Rung: Effect Verification | **FAK 303** |
| **FAK 308** — Stewards and the Rate-Limit Governor | **FAK 303** |
| **FAK 309** — Graceful Deny: Escalation to a Declared safe_sink | **FAK 304** |
| **FAK 310** — Context-MMU: The Write-Time Tool-Result Gate | **FAK 301** |
| **FAK 311** — Gate Soundness (Regime D): Idempotence and No Gratuitous Mutation | **FAK 310** |
| **FAK 312** — canon: The De-Obfuscating Canonicalizer | **FAK 311** |
| **FAK 313** — normgate: Canonicalize-and-Rescan and Its Honest Limit | **FAK 312** |
| **FAK 314** — IFC: The Taint Lattice and Provenance-Keyed Non-Interference | **FAK 313** |
| **FAK 315** — Provenance: The Model Cannot Author Its Own Trust | **FAK 314** |
| **FAK 316** — Durability Classes and the Expire-by-Default Write Gate | **FAK 203**, **FAK 303**, **FAK 310** |
| **FAK 317** — Hash-Chained Tamper-Evident Audit Journal | **FAK 207** |
| **FAK 318** — codelint: Validating Agent-Written Code at the Same Boundary | **FAK 310** |

### FAK 301 — Policy in the Kernel: The First Flip

**Prerequisites:** **FAK 103**, **FAK 207**

**You'll be able to:**
- Explain why 'the model can't talk past the gate' and 'the default is closed' are properties of WHERE the code runs, not how smart the check is
- Distinguish a fail-closed in-process check from a fail-open out-of-process recognizer
- Sketch which tools in a sample floor are allow-listed and which irreversible ones are deliberately left off

**Read:** [`docs/explainers/policy-in-the-kernel.md`](docs/explainers/policy-in-the-kernel.md), [`POLICY.md`](POLICY.md)

**Lab:**
```bash
go run ./cmd/fak policy --dump  # read the floor; sketch which tools are allow-listed and which irreversible ones are left off (see TestFoldDefaultDenyEmptyPolicy / TestNoOsExecOnHotPath)
```

**Checkpoint:** Explain why 'the model can't talk past the gate' and 'the default is closed' are properties of one address space with no IPC, not of how smart the check is. Name the two independent gates an attacker must beat.

### FAK 302 — What the Capability Floor Does and Does NOT Bound

**Prerequisites:** **FAK 301**

**You'll be able to:**
- Distinguish structural enforcement (refusing a tool NAME) from heuristic detection (argument regex, result flagging)
- Show why allow-listing Bash permits Bash{rm -rf /} and why arg-regex denies are reword-evadable
- State the durable fix: keep irreversible tools off the allow-list

**Read:** [`docs/explainers/policy-in-the-kernel.md`](docs/explainers/policy-in-the-kernel.md)

**Lab:**
```bash
Given a policy that allow-lists Bash with an RE2 deny on 'rm -rf', invent three rewordings the regex would miss; then state the structural fix (don't allow-list the irreversible tool at all).
```

**Checkpoint:** Classify each as structural or heuristic: (a) refusing an unallowed tool name, (b) the capability deny on the call side, (c) flagging a poisoned result, (d) the result-side quarantine DECISION. State which is the evadable part.

### FAK 303 — The Default-Deny Adjudicator and Closed Refusal Vocabulary

**Prerequisites:** **FAK 301**

**You'll be able to:**
- Explain why an empty policy denies everything and why an arg predicate can never produce an Allow
- State the FoldRank of Deny vs Allow and what happens to an unknown verdict kind
- List several of the 12 reason codes and say which deny is the structural floor (DEFAULT_DENY) vs a policy-pattern deny (POLICY_BLOCK)

**Read:** [`docs/proofs/adjudicator.md`](docs/proofs/adjudicator.md), [`POLICY.md`](POLICY.md), [`examples/adjudication-demo/README.md`](examples/adjudication-demo/README.md)

**Lab:**
```bash
go test ./internal/adjudicator/ -count=1 -run 'TestEmptyPolicyDefaultDeny|TestDefaultPolicyUnknownToolDefaultDeny|TestArgPredicatesAreRestrictOnly' -v && fak policy --check policy.json
```

**Checkpoint:** Explain why an empty policy denies everything and why an arg predicate can never Allow. Name the FoldRank of Deny vs Allow, what happens to an unknown verdict kind, and why every deny must cite a code from the fixed vocabulary.

### FAK 304 — Policy Manifests: Dump, Edit, Check, Load

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain what makes the loader fail-loud (DisallowUnknownFields, unknown-reason abort) and why that prevents silently loosening the floor
- Show that dump -> check round-trips losslessly
- Ship different floors (coding agent, ops bot, support agent) against the same binary

**Read:** [`POLICY.md`](POLICY.md), [`docs/proofs/policy.md`](docs/proofs/policy.md)

**Lab:**
```bash
fak policy --dump > policy.json && fak policy --check policy.json && fak preflight --policy policy.json --tool delete_account --args '{}'
```

**Checkpoint:** What makes the loader fail-loud and why does that prevent silently loosening the floor? Show that dump->check round-trips losslessly.

### FAK 305 — Preflight Ladder and Grammar Argument-Repair

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain why a rung-0 deny stamps RungFailed=0 and never reaches rung 1
- Explain why the grammar rung Defers (not Denies) for a tool with no registered grammar
- Distinguish when the grammar rung Transforms vs Denies

**Read:** [`docs/proofs/preflight.md`](docs/proofs/preflight.md), [`docs/proofs/grammar.md`](docs/proofs/grammar.md)

**Lab:**
```bash
go test ./internal/preflight/ -count=1 -run 'TestRung0FailureNeverReachesRung1|TestNegativesRowFields' -v && go test ./internal/grammar/ -count=1 -run 'TestAdjudicatePositionalRepairable|TestAdjudicateNoGrammarDefers' -v
```

**Checkpoint:** Why does a rung-0 deny stamp RungFailed=0 and never reach rung 1? Why does the grammar rung Defer (not Deny) for a tool with no registered grammar, and when does it Transform vs Deny?

### FAK 306 — Plan Control-Flow Integrity (plan-CFI)

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain why plan-CFI is opt-in (Defers with no plan declared)
- State what a deviating call returns by default vs in strict mode
- Explain monotone pos advance in Sequence mode and the ROP-gadget analogy

**Read:** [`docs/proofs/plancfi.md`](docs/proofs/plancfi.md)

**Lab:**
```bash
go test ./internal/plancfi/ -count=1 -run 'TestDeviationEscalates|TestStrictModeDenies|TestSequenceMode|TestConformingCallDefers' -v
```

**Checkpoint:** Why is plan-CFI opt-in and what does a deviating call return by default vs in strict mode? Explain monotone pos advance in Sequence mode and the binary-CFI analogy for an exfil gadget inside an allowed task.

### FAK 307 — The Require-Witness Rung: Effect Verification

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Name the three resolver outcomes (Confirm/Refute/Abstain) and how the kernel folds each
- Explain why a missing git Abstain results in Deny/UNWITNESSED rather than Confirm or Refute
- Corroborate a claimed effect against evidence the agent could not author

**Read:** [`docs/proofs/witness.md`](docs/proofs/witness.md)

**Lab:**
```bash
go test ./internal/witness/ -count=1 -run 'TestAncestorClaim|TestGitMissingAbstains|TestUnparseableClaimAbstains|TestRealGitAncestor' -v
```

**Checkpoint:** What are the three resolver outcomes and how does the kernel fold each? Why does a missing git Abstain (Deny/UNWITNESSED) rather than Confirm or Refute?

### FAK 308 — Stewards and the Rate-Limit Governor

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain why a steward must abstain by default and carry an independently-authored witness
- Explain why check-then-consume ordering makes a denied call cost nothing
- Explain why the limiter is fail-open until configured and denies with RATE_LIMITED (a WAIT)

**Read:** [`docs/proofs/steward.md`](docs/proofs/steward.md), [`docs/proofs/ratelimit.md`](docs/proofs/ratelimit.md)

**Lab:**
```bash
go test ./internal/steward/ -count=1 -run 'TestSecretInContext|TestSweepAbstainingStewardNotReported' -v && go test ./internal/ratelimit/ -count=1 -run 'TestQuotaDeniesOverCap|TestDeniedCallConsumesNoBudget|TestInertUntilConfigured' -v
```

**Checkpoint:** Why must a steward abstain by default and carry an independently-authored witness? In the limiter, why is check-then-consume ordering what makes a denied call cost nothing, and why is it fail-open until configured?

### FAK 309 — Graceful Deny: Escalation to a Declared safe_sink

**Prerequisites:** **FAK 304**

**You'll be able to:**
- Explain why the escalation call itself is adjudicated (no side-channel un-sanctioned human-queue tool)
- Explain why the harness, not the kernel, must redact the escalation payload of a denied call
- Route a denied call to the policy's declared safe_sink with a redacted ticket

**Read:** [`examples/escalation-demo/README.md`](examples/escalation-demo/README.md)

**Lab:**
```bash
./examples/escalation-demo/run.sh   # build kernel -> serve policy -> catch deny -> route to declared sink -> redacted ticket
```

**Checkpoint:** Why is the escalation call itself adjudicated, and why must the harness (not the kernel) redact the escalation payload of a denied call?

### FAK 310 — Context-MMU: The Write-Time Tool-Result Gate

**Prerequisites:** **FAK 301**

**You'll be able to:**
- Name the three Admit verdicts (Allow / Quarantine / Transform) and which fires for clean, secret-bearing, and small JSON results
- Explain why ctxmmu is the dual of the call-side adjudicator (screening what comes back)
- Explain why PointerMax (2048) is deliberately less than OversizeBytes (4096)

**Read:** [`docs/proofs/ctxmmu.md`](docs/proofs/ctxmmu.md)

**Lab:**
```bash
go test ./internal/ctxmmu/ -count=1 -timeout 120s -run 'TestAdmit'
```

**Checkpoint:** Name the three Admit verdicts and state which fires for a 6KB clean log line, a body containing an API key, and a 200-byte JSON record. Why is PointerMax deliberately less than OversizeBytes?

### FAK 311 — Gate Soundness (Regime D): Idempotence and No Gratuitous Mutation

**Prerequisites:** **FAK 310**

**You'll be able to:**
- State the two soundness invariants: byte-identical round-trip on Allow, and idempotent page-out
- Explain why re-Admitting a quarantined stub returns Allow without incrementing the quarantine counter
- Identify which property a missing bytes.Equal assertion would leave un-witnessed

**Read:** [`docs/proofs/ctxmmu.md`](docs/proofs/ctxmmu.md), [`docs/proofs/normgate.md`](docs/proofs/normgate.md)

**Lab:**
```bash
go test ./internal/ctxmmu/ -count=1 -run 'TestProofPageOutIdempotent|TestProofBenignByteIdentical'
```

**Checkpoint:** Explain why re-Admitting an already-quarantined stub returns Allow and does not increment the quarantine counter (but DOES increment the total call counter). Which property would a missing bytes.Equal assertion leave un-witnessed?

### FAK 312 — canon: The De-Obfuscating Canonicalizer

**Prerequisites:** **FAK 311**

**You'll be able to:**
- Explain why Normalize is idempotent (the property of its output runes that guarantees a fixed point)
- Name one obfuscation family canon folds and the canonical view that catches it
- Explain why a lexical scan must run over the canonical view, not raw bytes

**Read:** [`docs/proofs/canon.md`](docs/proofs/canon.md)

**Lab:**
```bash
go test ./internal/canon/ -count=1 -run 'TestObfuscatedInjectionCaught|TestNormalizeUndoesObfuscation|TestNormalizeIdempotent_Deterministic' -v
```

**Checkpoint:** Why is Normalize idempotent (what property of its output runes guarantees Normalize(Normalize(x))==Normalize(x))? Give one obfuscation family canon folds and the specific view that catches it.

### FAK 313 — normgate: Canonicalize-and-Rescan and Its Honest Limit

**Prerequisites:** **FAK 312**

**You'll be able to:**
- State the superset theorem (canon flags every body the raw gate flags, plus more) and prove the easy direction informally
- Give an injection string normgate provably does NOT catch (a marker-free paraphrase) and explain why that is an honest limit, not a bug
- Explain why closing the lexical gap needs an IFC/semantic seam

**Read:** [`docs/proofs/normgate.md`](docs/proofs/normgate.md)

**Lab:**
```bash
go test ./internal/normgate/ -count=1 -run 'TestCanonInjectionSupersetOfRaw_Quick|TestParaphraseEvadesByDesign' -v
```

**Checkpoint:** State the superset theorem and prove the easy direction informally. Then give an injection string normgate provably does NOT catch and explain why that is recorded as an honest limit rather than a bug.

### FAK 314 — IFC: The Taint Lattice and Provenance-Keyed Non-Interference

**Prerequisites:** **FAK 313**

**You'll be able to:**
- Explain why the taint join must be a join-semilattice for the most-restrictive fold to be well-defined
- Trace how a marker-free paraphrase read from an external page still gets its follow-up send_email denied
- Explain declassification as the only sanctioned way tainted data reaches a sink

**Read:** [`docs/proofs/ifc.md`](docs/proofs/ifc.md)

**Lab:**
```bash
go test ./internal/ifc/ -count=1 -run 'TestParaphrasedExfilBlockedByProvenance|TestForgedSelfTrustCannotEvadeTaint|TestVDSOHitDoesNotLaunderTaint|TestAuthorizeEscape' -v
```

**Checkpoint:** Why must the taint join be a join-semilattice (monotone/commutative/associative/idempotent) for the most-restrictive fold? Trace how a marker-free paraphrase read from an external page still gets its follow-up send_email denied.

### FAK 315 — Provenance: The Model Cannot Author Its Own Trust

**Prerequisites:** **FAK 314**

**You'll be able to:**
- Name the two kernel-controlled facts Taint(c,r) consults and the field it deliberately never reads on a verdict path
- Explain why a forged Meta['provenance'] cannot mint trust and survives only as a forensic signal
- State the honest caveat in Theorem 2: which half of the no-drift claim rests on grep evidence

**Read:** [`docs/proofs/provenance.md`](docs/proofs/provenance.md), [`docs/proofs/ifc.md`](docs/proofs/ifc.md)

**Lab:**
```bash
go test ./internal/provenance/ -count=1 -run 'TestModelCannotAuthorTrust|TestTaintBySource|TestRegisterSourceIsHostAuthored' -v
```

**Checkpoint:** What two kernel-controlled facts does Taint(c,r) consult, and which field does it deliberately never read? Explain the honest caveat in Theorem 2: which half of the no-drift claim rests on grep evidence rather than a re-run-on-build assertion?

### FAK 316 — Durability Classes and the Expire-by-Default Write Gate

**Prerequisites:** **FAK 203**, **FAK 303**, **FAK 310**
  ·  **Background:** **FAK 204**

**You'll be able to:**
- Classify every value crossing into durable store as turn/session/bounded/durable at write time
- Justify why an un-classified observation must default to turn (expire), citing the asymmetric error costs
- Locate the attach point: an additive Verdict.Meta['durability'] tag on the ctxmmu Admit seam, fail-closed to 'turn', costing zero frozen-ABI surface
- State precisely what fak claims and does NOT claim vs the named prior art (Tulving, bitemporal SQL:2011, Zhang-Choi 2023, Springdrift, Zep, Cloudflare)

**Read:** [`docs/CONTEXT-IS-NOT-MEMORY.md`](docs/CONTEXT-IS-NOT-MEMORY.md)

**Lab:**
```bash
Trace the rung-1 bite test by hand: classify 'it's 3pm' and 'the user prefers afternoons' through the ctxmmu gate and state the durability class + promotion verdict each gets; then open internal/abi/types.go and confirm a 'durability' key on the OPEN Meta map does not move TestABIGoldenFreeze.
```

**Checkpoint:** Justify why the default for an un-classified observation must be 'turn' (expire) rather than a centered threshold, citing the asymmetry of the silent false-positive vs the recoverable false-negative; explain why an additive Meta tag (not a new VerdictKind) is the correct attach point; and state the one column where each prior-art system fails to gate on truth-duration at write time.

### FAK 317 — Hash-Chained Tamper-Evident Audit Journal

**Prerequisites:** **FAK 207**

**You'll be able to:**
- Walk through why mutating one content byte trips authenticity AND re-hashing trips the next row's continuity
- Distinguish tamper-evidence from tamper-prevention
- Explain how the durable-flush witness distinguishes per-Emit flush from flush-only-at-Close

**Read:** [`docs/proofs/journal.md`](docs/proofs/journal.md)

**Lab:**
```bash
go test ./internal/journal/ -count=1 -timeout 120s -run 'TestVerifyDetectsTampering|TestFileJournalReopensAndContinuesChain|TestPerWriteDurableFlush_VerifyWithoutCloseRecoversEveryEmittedRow' -v
```

**Checkpoint:** Walk through why mutating one content byte trips the authenticity check AND why re-hashing to cover it trips the next row's continuity check. Explain how the durable-flush witness distinguishes 'flushed per Emit' from 'flushed only at Close'.

### FAK 318 — codelint: Validating Agent-Written Code at the Same Boundary

**Prerequisites:** **FAK 310**
  ·  **Background:** **FAK 302**

**You'll be able to:**
- Explain why a write_file producing broken code is checkable at the same write-time boundary ctxmmu already runs
- Route a file to the language-server pack that owns its extension and parse/compile-check it
- Feed the parse/compile errors back so the model self-corrects, closing the coding-agent loop the SWE-bench story leans on

**Read:** [`docs/explainers/code-linting-at-the-kernel.md`](docs/explainers/code-linting-at-the-kernel.md)

**Lab:**
```bash
go test ./internal/codelint/ -count=1 -timeout 120s -run 'TestGoPackReportsParseError|TestPackForKnownAndUnknown|TestParseDiagnosticsGCCStyle|TestHasErrorAndSummaryOrdersErrorsFirst' -v
```

**Checkpoint:** Explain pack-by-extension routing and why a clean file yields no opinion while a semantic (not syntactic) error is ignored by the Go pack. State why feeding errors back at the write boundary is the concrete coding-agent payoff of the FAK 310 write gate, and how it underwrites the L600 SWE-bench coding-agent material.

---

## L400 — The Performance Core: cache reuse, addressable eviction, and the scaling laws

**Theme.** Why agents stress the cache, prefill-elimination economics, the addressable/bijective KV-MMU, RadixAttention reuse, the vDSO, durable session recall, and the first-order scaling law (incl. cache legality and residency).

**Who joins here.** An ML-systems or kernel-minded reader who has the Foundations KV-cache unit and the security write-time gate. Join here if you want the speed story and how it converges with the security boundary, rather than the enforcement details. Memory/RAG engineers continue here for the scaling laws after the durability gate.

**Assumes you can already pass:** **FAK 201**, **FAK 205**, **FAK 310**.

| Course | Hard prerequisites |
|---|---|
| **FAK 401** — How Agents Stress the KV Cache | **FAK 201** |
| **FAK 402** — Prefill Elimination and the A/B/C Cost Arms | **FAK 401** |
| **FAK 403** — The 10 SOTA Serving Optimizations and the Honest Baseline | **FAK 402** |
| **FAK 404** — Addressable KV Cache: Exact Span Removal (The Second Flip) | **FAK 310**, **FAK 401** |
| **FAK 405** — RadixAttention Prefix Reuse + LRU Eviction | **FAK 401** |
| **FAK 406** — KV-MMU: Addressable, Bijective Span Eviction | **FAK 405**, **FAK 404** |
| **FAK 407** — The 3-Tier Tool vDSO (Fast-Path Cache) | **FAK 205**, **FAK 307** |
| **FAK 408** — What the Semantics-Layer Vantage Unlocks | **FAK 204**, **FAK 406** |
| **FAK 409** — recall: Session Core-Dump That Survives the Boundary | **FAK 407** |
| **FAK 410** — contextq: On-Demand Context Materialization | **FAK 409** |
| **FAK 411** — ed25519 Deletion Certificates | **FAK 317**, **FAK 406** |
| **FAK 412** — The First-Order Scaling Law of Agents | **FAK 402**, **FAK 316** |
| **FAK 413** — Cache Legality: The Next Scaling Wall | **FAK 412** |
| **FAK 414** — Three Regimes and the Agent-City Saturation Points | **FAK 413** |

### FAK 401 — How Agents Stress the KV Cache

**Prerequisites:** **FAK 201**

**You'll be able to:**
- Explain why a broken cache turns a linear loop into a quadratic one in latency and dollars
- Show why caching matters far more at 239:1 input:output (agents) than at 2:1 (chat)
- Name the failure modes (eviction during tool latency, head-mutation, injected timestamps, unstable JSON) and the zero-infra fix

**Read:** [`docs/explainers/kv-cache-agentic-context.md`](docs/explainers/kv-cache-agentic-context.md)

**Lab:**
```bash
Take a prompt with a per-request UUID at the head; move it to the tail and re-run the LCP analysis to reproduce the 0.3% -> 87% hit-rate jump described in the doc.
```

**Checkpoint:** Explain why a changed file causes a visible cache miss (recompute) rather than a silently stale answer, and the one condition (result cache keyed on call args alone) under which staleness CAN go silent; give the fix (key on content version).

### FAK 402 — Prefill Elimination and the A/B/C Cost Arms

**Prerequisites:** **FAK 401**

**You'll be able to:**
- Distinguish arm A (naive re-send), arm B (per-agent KV, duplicated prefixes), and arm C (fak fused, one shared prefix)
- State when fak does NOT help (single-turn, zero shared context, tiny contexts)
- Read the 20-24x as vs naive, not vs a tuned baseline

**Read:** [`docs/prefill-elimination-explained.md`](docs/prefill-elimination-explained.md)

**Lab:**
```bash
go run ./cmd/fak swebench describe --difficulty <file>  (inspect live cost numbers); or read internal/swebench/cost.go to see how A/B/C token totals are computed.
```

**Checkpoint:** Distinguish arm B from arm C and state when fak does NOT help. Note that the 20-24x is vs naive, not vs a tuned baseline.

### FAK 403 — The 10 SOTA Serving Optimizations and the Honest Baseline

**Prerequisites:** **FAK 402**

**You'll be able to:**
- List which of the 10 optimizations fak marks IMPLEMENTED vs PARTIAL vs ENGINE-LEVEL and map each to its owning engine
- Name the three sources of the 1.5-4x-vs-tuned gain
- Name the three things the gain is explicitly NOT from (raw model speed, basic KV reuse, quantization)

**Read:** [`docs/explainers/sota-optimizations.md`](docs/explainers/sota-optimizations.md)

**Lab:**
```bash
From the SOTA table, list every optimization fak marks IMPLEMENTED vs PARTIAL vs NOT-FOCUSED/ENGINE-LEVEL, then map each to the engine that owns it (llama.cpp / vLLM / SGLang).
```

**Checkpoint:** When fak reports '1.5-4x vs tuned SOTA', name the three sources of the gain and the three things it is explicitly NOT from.

### FAK 404 — Addressable KV Cache: Exact Span Removal (The Second Flip)

**Prerequisites:** **FAK 310**, **FAK 401**

**You'll be able to:**
- Trace the four senses of 'addressable' (prefix / span / content / queryable-context) onto fak's status
- Explain why llama.cpp's K-shift drifts ~1e-6 while a single re-rotation from Kraw is exact
- State honestly that bit-exact span removal is proven on a synthetic model in internal/kvmmu but not yet wired into the live agent HTTP loop

**Read:** [`docs/explainers/addressable-kv-cache.md`](docs/explainers/addressable-kv-cache.md)

**Lab:**
```bash
Trace the four senses of 'addressable' onto fak's status; identify which test pins exact span removal (TestKVQuarantineEqualsNeverSaw, max|delta|=0).
```

**Checkpoint:** Explain why llama.cpp's K-shift drifts ~1e-6 while fak's single re-rotation from Kraw is exact, and why bit-exact span removal is proven on a synthetic model but NOT yet wired into the live fak agent HTTP loop.

### FAK 405 — RadixAttention Prefix Reuse + LRU Eviction

**Prerequisites:** **FAK 401**

**You'll be able to:**
- Explain why longest-prefix reuse + suffix prefill is bit-identical to a from-scratch prefill (logits/argmax match)
- Explain 'upward collapse': why removing a leaf can make its parent a new eviction candidate
- State the refcount-conservation invariant across a Lookup->Insert->Done cycle and why the root boundary lease is counted for a cold request

**Read:** [`docs/proofs/radixkv.md`](docs/proofs/radixkv.md)

**Lab:**
```bash
go test ./internal/radixkv/ -count=1 -timeout 120s -run 'TestReuseThroughSplitMatchesRecompute|TestLRUEvictsOldestRetainsHotAndLeased|TestLRUUpwardCollapse|TestRefcountConservationCycleNetsZero' -v
```

**Checkpoint:** Explain 'upward collapse' and state the refcount-conservation invariant (Sigma node.refs across a Lookup->Insert->Done cycle) and why the root boundary lease must be counted for a cold request.

### FAK 406 — KV-MMU: Addressable, Bijective Span Eviction

**Prerequisites:** **FAK 405**, **FAK 404**
  ·  **Background:** **FAK 206**

**You'll be able to:**
- State the two structural invariants (bijection over live spans; exact span addressing)
- Explain why eviction must be content/id-driven, not positional, and how RoPE re-rotation of survivors makes post-evict cache byte-identical to never-saw-it
- Identify what is explicitly SCOPED-OUT (concurrent-eviction data-race freedom, deferred to Gobra)

**Read:** [`docs/proofs/kvmmu.md`](docs/proofs/kvmmu.md)

**Lab:**
```bash
go test ./internal/kvmmu/ -count=1 -timeout 120s -run 'TestLedgerRenumberAfterMiddleEvict|TestWriteTimeEvictEqualsNeverSaw|TestEvictionIsContentDrivenNotPositional' -v
```

**Checkpoint:** State the two structural invariants and explain why eviction must be content/id-driven, not positional. What is explicitly SCOPED-OUT?

### FAK 407 — The 3-Tier Tool vDSO (Fast-Path Cache)

**Prerequisites:** **FAK 205**, **FAK 307**

**You'll be able to:**
- Trace the fixed lookup order (tier-1 pure recompute, tier-3 static, tier-2 cached)
- Name the four conditions that downgrade a tier-2 hit to a MISS
- Explain why the integrity epoch advances monotonically on a non-empty Revoke and is a no-op on an empty-witness Revoke

**Read:** [`docs/proofs/vdso.md`](docs/proofs/vdso.md)

**Lab:**
```bash
go test -run 'Unit25|Unit26_27|Unit28|Unit29|Unit34_Miss|Scope_Soundness' ./internal/vdso/ -count=1 -timeout 120s -v
```

**Checkpoint:** Trace the fixed lookup order and name the four distinct conditions that downgrade a tier-2 hit to a MISS. Explain why the integrity (trust) epoch advances monotonically on a non-empty Revoke and is a no-op on an empty-witness Revoke.

### FAK 408 — What the Semantics-Layer Vantage Unlocks

**Prerequisites:** **FAK 204**, **FAK 406**

**You'll be able to:**
- For each of the five optimizations (us filter, exact rewind/branch, transactional turn, structure-aware eviction, per-principal audit), name the structure it depends on
- Explain why a serving engine on an anonymous token stream cannot do bit-exact middle-eviction even with zero-copy read access to fak's arena
- Distinguish 'faster at the same thing' from operations structurally impossible without identity + state machine + owned arena

**Read:** [`docs/MEMORY-LAYERS-EXPLAINER.md`](docs/MEMORY-LAYERS-EXPLAINER.md)

**Lab:**
```bash
For each of the five optimizations, name the one piece of structure (identity, state machine, or owned-arena+Kraw) it depends on and check its SHIPPED/SEAM-SHIPPED tag in the doc.
```

**Checkpoint:** Explain why a serving engine sitting on an anonymous token stream cannot do bit-exact middle-eviction even with zero-copy read access to fak's arena (gate 3: Kraw is a write-time decision).

### FAK 409 — recall: Session Core-Dump That Survives the Boundary

**Prerequisites:** **FAK 407**
  ·  **Background:** **FAK 205**

**You'll be able to:**
- Explain what 'same answer as replay' reduces to for a content-addressed image (per-page byte-identity + deterministic exclusion set)
- Explain why Load refuses the whole image if any blob fails to re-hash to its key
- Explain how run-to-run determinism is witnessed against Go's randomized map iteration

**Read:** [`docs/proofs/recall.md`](docs/proofs/recall.md)

**Lab:**
```bash
go test ./internal/recall/ -count=1 -timeout 120s -run 'TestBenignPageRoundTripsByteIdentical|TestSessionIsSelfContained|TestRecallWorkingSetExcludesPoison|TestRecallIsDeterministicAcrossRepeatedCalls' -v
```

**Checkpoint:** Explain what 'same answer as replay' reduces to for a content-addressed image. Why does Load refuse the whole image if any blob fails to re-hash to its key, and how is run-to-run determinism witnessed against Go's randomized map iteration?

### FAK 410 — contextq: On-Demand Context Materialization

**Prerequisites:** **FAK 409**

**You'll be able to:**
- Explain why the unqualified byte-identity theorem is FALSE for the summary path and how it must be restated
- State the summary path's contract (FaithfulnessProbe==1.0 extractive prefix + reported Coverage)
- Name the five MaterializationVerdicts

**Read:** [`docs/proofs/contextq.md`](docs/proofs/contextq.md)

**Lab:**
```bash
go test ./internal/contextq/ -count=1 -timeout 120s -run 'TestMaterializeByteIdentical|TestMaterializationDeterministic' -v
```

**Checkpoint:** Why is the unqualified byte-identity theorem FALSE for the summary path, and how must it be restated? Name the five MaterializationVerdicts.

### FAK 411 — ed25519 Deletion Certificates

**Prerequisites:** **FAK 317**, **FAK 406**

**You'll be able to:**
- List the four ordered verification rungs and what each rejects
- State the three honest non-claims (self-attesting in v1, max|delta|=0 checked only as a signed string, EvictedCount is a self-report)
- Re-derive the journal anchor row to make the receipt re-checkable, not merely asserted

**Read:** [`docs/proofs/deletioncert.md`](docs/proofs/deletioncert.md)

**Lab:**
```bash
go test ./internal/deletioncert/ -count=1 -timeout 120s -run 'TestMintVerifyRoundTrip|TestTamperDetected|TestNonBitExactRejected|TestAnchorAbsent|TestSubjectRelabelRejected|TestNilVerifierFailsClosed' -v
```

**Checkpoint:** List the four ordered verification rungs and explain what each rejects. State the THREE honest non-claims.

### FAK 412 — The First-Order Scaling Law of Agents

**Prerequisites:** **FAK 402**, **FAK 316**
  ·  **Background:** **FAK 203**

**You'll be able to:**
- Write the law: agents x turns x working-set x reread rate x legality checks
- Explain why reread rate is the only safe term to attack, and only when legality permits
- Explain why the measured 60.3x session result is not a '60x faster model' but a deletion of duplicate setup re-reads

**Read:** [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)

**Lab:**
```bash
go run ./cmd/longctxbench  (compute the contention-free work floor; compare naive setup payments = agents x turns vs coherent = 1 per legal shared scope for a 5-agent x 50-turn workload)
```

**Checkpoint:** Explain why the measured 60.3x session result is NOT a '60x faster model' and which term in the scaling law it actually deletes.

### FAK 413 — Cache Legality: The Next Scaling Wall

**Prerequisites:** **FAK 412**

**You'll be able to:**
- State net reuse value = shared read hits - invalidation cost - stale-read risk, keyed on (digest, scope, world-version, taint)
- Distinguish physical (residency) coherence from semantic (legality) coherence
- Give an example where a hit passing every hardware coherence check is still the wrong answer (a git push invalidating cached git status)

**Read:** [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)

**Lab:**
```bash
Work Scenario B from the doc on paper: a byte-coherent hot KV span after a git push — state the two distinct failures (stale fact; cross-tenant leak) and which key field (world-version / scope) the coherence kernel uses to evict exactly that span.
```

**Checkpoint:** Distinguish physical (residency) coherence from semantic (legality) coherence and give one example where a hit passing every hardware coherence check is still the wrong answer.

### FAK 414 — Three Regimes and the Agent-City Saturation Points

**Prerequisites:** **FAK 413**

**You'll be able to:**
- Distinguish single-chat / long-session / agent-city regimes by bottleneck
- Compute a Qwen2.5-7B KV geometry and show a 100k-token cache is ~143x too big for L2
- Identify why the binding constraint at city scale is KV residency, not FLOPs, and name two meters that would prove a system scales

**Read:** [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)

**Lab:**
```bash
Reproduce the doc arithmetic for a Qwen2.5-7B geometry: compute KV bytes/token (2 x 28 x 4 x 128 x 2), a 100k-token cache size, and its ratio to A100 L2 (40MB) and one SM's SRAM (192KB).
```

**Checkpoint:** State which saturation point binds first at agent-city scale and why it is residency rather than compute; then name two meters that would prove a system actually scales.

---

## L500 — Serving, Integration, and the In-Kernel Model

**Theme.** Running and hardening the gateway, the gateway drop guarantee, repointing existing agents at one base URL, the framework cookbook, the pure-Go in-kernel model + compute HAL with oracle parity, and the GPU lease.

**Who joins here.** A platform/SRE who already runs vLLM, or an app developer who just calls an LLM API and wants governance with zero agent rewrite. Join here if you can take the security and performance cores as given and want to deploy, integrate, or understand the reference forward pass.

**Assumes you can already pass:** **FAK 105**, **FAK 301**, **FAK 304**, **FAK 310**.

| Course | Hard prerequisites |
|---|---|
| **FAK 501** — The fak serve Mental Model: One Binary, Four Tiers, Three Modes | **FAK 105**, **FAK 301** |
| **FAK 502** — Starting the Gateway: serve Flags and the Engine-vs-Upstream Axis | **FAK 501** |
| **FAK 503** — The HTTP API: OpenAI, Anthropic, fak-native, and MCP Surfaces | **FAK 502**, **FAK 310** |
| **FAK 504** — Hardening the Gateway: Bearer Auth, the Policy Floor, and Live Reload | **FAK 503**, **FAK 304** |
| **FAK 505** — Observability: Prometheus Metrics, JSON Access Log, X-Trace-Id | **FAK 503** |
| **FAK 506** — Tuning Timeouts and the serve Env Vars | **FAK 502** |
| **FAK 507** — Deploying the Gateway: Docker, Compose, Kubernetes, Bare Metal | **FAK 504**, **FAK 505** |
| **FAK 508** — Scaling and HA: Process-Local State and Sticky Routing | **FAK 507**, **FAK 407**, **FAK 314** |
| **FAK 509** — The MCP Tool-Result Wire: Refusal as a Value | **FAK 503**, **FAK 312** |
| **FAK 510** — Troubleshooting the Gateway and the fak CLI Verbs | **FAK 504** |
| **FAK 511** — The Integration Index: Repoint One Base URL | **FAK 503** |
| **FAK 512** — Claude Code / Anthropic API Through fak | **FAK 511** |
| **FAK 513** — OpenAI Codex / OpenAI SDK Through fak | **FAK 511** |
| **FAK 514** — Cursor via MCP or OpenAI Proxy | **FAK 511** |
| **FAK 515** — MCP One-Paste Setup and the fak_* Tools | **FAK 511**, **FAK 509** |
| **FAK 516** — Agent<->Kernel Architecture and the Frozen ABI Verdict Union | **FAK 511**, **FAK 208** |
| **FAK 517** — Framework Cookbook: Transparent Proxy (Mode A) vs Explicit Adjudication (Mode B) | **FAK 516**, **FAK 513**, **FAK 302** |
| **FAK 518** — Migration: Moving Existing Code by Repointing a Base URL | **FAK 516** |
| **FAK 519** — Multi-Language Client Code and Disposition-Aware Retry | **FAK 516**, **FAK 509** |
| **FAK 520** — The Adopter Playbook: Front-a-Model, Manual MCP, Embed-in-CI | **FAK 512**, **FAK 515** |
| **FAK 521** — GGUF Loading: Offsets, Dtypes, and Dequant Layout | **FAK 205** |
| **FAK 522** — Tokenizer: Lossless ByteLevel BPE With Oracle Parity | **FAK 521** |
| **FAK 523** — Normalization: RMSNorm, NormGain1p, and LayerNorm | **FAK 522** |
| **FAK 524** — RoPE: Rotary Position Embedding and Scaling Variants | **FAK 523** |
| **FAK 525** — Attention: Stable Softmax, Causal Mask, and the Attention Sink | **FAK 524** |
| **FAK 526** — MLP / SwiGLU+GeGLU, MoE Routing, and the Residual Stream | **FAK 525** |
| **FAK 527** — In-Kernel KV Cache: Slotting, Span-Exact Eviction, SWA, Prefix Reuse | **FAK 526**, **FAK 406** |
| **FAK 528** — Quantization: Q4_K/Q8_0/Q4_0 Dequant, AWQ, and Bit-Identical int8 SDOT | **FAK 521**, **FAK 526** |
| **FAK 529** — Forward-Pass Parity vs the HuggingFace Oracle | **FAK 527**, **FAK 528**, **FAK 210** |
| **FAK 530** — The Compute HAL Seam and Hardware Portability | **FAK 529**, **FAK 210** |
| **FAK 531** — Metal GPU GEMM Parity and the Stub-vs-Device Build | **FAK 530** |
| **FAK 532** — The Engine Seam: Determinism and Cache-Invalidation Binding | **FAK 529**, **FAK 206** |
| **FAK 533** — In-Kernel Model & Compute Env Knobs (FAK_* Engine Vars) | **FAK 502**, **FAK 528** |
| **FAK 534** — GPU Lease: Machine-Wide Mutual Exclusion for Model Residency | **FAK 533** |
| **FAK 535** — The Gateway Drop Guarantee: Fail-Closed on a Failed Adjudication | **FAK 510**, **FAK 314** |

### FAK 501 — The fak serve Mental Model: One Binary, Four Tiers, Three Modes

**Prerequisites:** **FAK 105**, **FAK 301**
  ·  **Background:** **FAK 302**, **FAK 403**

**You'll be able to:**
- Frame the deploy-stack-ownership claim: fak collapses the governance half of agent serving (API surface + capability gate + result containment + audit + auth) into ONE static binary that fronts, not replaces, a token engine — identical laptop to fleet
- Distinguish proxy mode (--base-url), in-kernel mode (--gguf, no --base-url), and offline mock
- Name the four escalating setup tiers (0 offline kernel, 1 front a model, 2 in-kernel synthetic, 2b real weights)
- Explain why Tier 2's in-kernel SmolLM2 is a reference forward pass and NOT a production chat server

**Read:** [`docs/explainers/one-binary-one-surface.md`](docs/explainers/one-binary-one-surface.md), [`GETTING-STARTED.md`](GETTING-STARTED.md), [`docs/fak/server-quickstart.md`](docs/fak/server-quickstart.md)

**Lab:**
```bash
go run ./cmd/fak run --trace testdata/tau2/tau2-smoke.json   # Tier 0: replay a trace through the kernel offline
```

**Checkpoint:** Draw the two-halves split (governance+gateway vs token engine) and explain why 'the laptop story and the fleet story are the same binary' — what changes is flags, not installed components. Then explain proxy vs in-kernel vs offline mock, and why Tier 2's in-kernel SmolLM2 is a reference forward pass and NOT a production chat server.

### FAK 502 — Starting the Gateway: serve Flags and the Engine-vs-Upstream Axis

**Prerequisites:** **FAK 501**

**You'll be able to:**
- Use the core serve flags (--addr, --provider, --base-url, --model, --gguf, --tokenizer, --engine, --stdio)
- Explain why --engine (serving /v1/fak/*) is a separate axis from --base-url (the upstream model)
- Predict what /healthz reports for the engine field in a Tier-1 proxy deployment

**Read:** [`docs/fak/server-config.md`](docs/fak/server-config.md), [`docs/fak/server-quickstart.md`](docs/fak/server-quickstart.md), [`GETTING-STARTED.md`](GETTING-STARTED.md)

**Lab:**
```bash
ollama serve & ; ollama pull qwen2.5:1.5b ; go run ./cmd/fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b ; curl -s http://127.0.0.1:8080/healthz
```

**Checkpoint:** Given a Tier-1 deployment, predict what curl /healthz returns for the engine field, and explain why your upstream model is reached only via /v1/chat/completions and not via /v1/fak/syscall.

### FAK 503 — The HTTP API: OpenAI, Anthropic, fak-native, and MCP Surfaces

**Prerequisites:** **FAK 502**, **FAK 310**

**You'll be able to:**
- Identify which endpoint to call across the four wire surfaces on one port
- Explain why a policy refusal returns HTTP 200 carrying a verdict (deny-as-value, not an error) and that SSE is synthesized from the finished turn
- Distinguish /v1/fak/adjudicate from /v1/fak/syscall and /v1/fak/admit

**Read:** [`docs/fak/api-reference.md`](docs/fak/api-reference.md), [`GETTING-STARTED.md`](GETTING-STARTED.md), [`docs/fak/server-config.md`](docs/fak/server-config.md)

**Lab:**
```bash
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate -H 'Content-Type: application/json' -d '{"tool":"refund_payment","arguments":{}}'   # observe verdict DENY in a 200 response
```

**Checkpoint:** Explain why a policy refusal returns HTTP 200 (not 4xx), what the fak response extension contains for a turn with a dropped tool call, and how /v1/fak/adjudicate differs from /v1/fak/syscall and /v1/fak/admit.

### FAK 504 — Hardening the Gateway: Bearer Auth, the Policy Floor, and Live Reload

**Prerequisites:** **FAK 503**, **FAK 304**

**You'll be able to:**
- Add dual-header bearer auth with --require-key-env and pin a fail-closed --policy floor
- Reload the policy live with POST /v1/fak/policy/reload without restarting or dropping warm vDSO/IFC state
- Explain why a non-loopback bind without a key still serves (with a warning) and why that is a hazard

**Read:** [`docs/serve-config.md`](docs/serve-config.md), [`docs/fak/server-config.md`](docs/fak/server-config.md), [`docs/fak/server-quickstart.md`](docs/fak/server-quickstart.md)

**Lab:**
```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)" ; fak policy --dump > policy.json ; fak policy --check policy.json ; fak serve --addr 0.0.0.0:8080 --base-url http://localhost:11434/v1 --model M --policy policy.json --require-key-env FAK_GATEWAY_KEY
```

**Checkpoint:** Set up auth + a custom policy, prove every route except /healthz now requires the token, then edit policy.json and reload it live with a single authenticated POST without restarting the process.

### FAK 505 — Observability: Prometheus Metrics, JSON Access Log, X-Trace-Id

**Prerequisites:** **FAK 503**

**You'll be able to:**
- Alert on fak_gateway_up, build_info, per-route latency/error rate, verdict counts, and startup-phase timings
- Correlate one request across logs/metrics/headers via X-Trace-Id
- Name which fields the access log deliberately never carries and why that lets you ship it to a SIEM

**Read:** [`docs/fak/observability.md`](docs/fak/observability.md), [`docs/fak/server-config.md`](docs/fak/server-config.md)

**Lab:**
```bash
curl -s http://127.0.0.1:8137/metrics | grep fak_gateway ; curl -si -H 'X-Trace-Id: my-req-42' http://127.0.0.1:8137/healthz | grep -i x-trace-id
```

**Checkpoint:** Write the PromQL for per-route p99 latency and per-route 5xx error rate, and explain which fields the access log deliberately never carries and why that lets you ship it to a SIEM safely.

### FAK 506 — Tuning Timeouts and the serve Env Vars

**Prerequisites:** **FAK 502**

**You'll be able to:**
- Size FAK_HTTP_*_TIMEOUT_S and FAK_PLANNER_TIMEOUT_S for a slow local CPU model vs a fast hosted upstream
- Explain why FAK_HTTP_WRITE_TIMEOUT_S must be >= FAK_PLANNER_TIMEOUT_S
- Explain what setting the write timeout to 0 does and why it is a slow-loris risk, plus the [5,3600] planner clamp

**Read:** [`docs/serve-config.md`](docs/serve-config.md), [`docs/fak/server-config.md`](docs/fak/server-config.md), [`docs/fak/advanced-topics.md`](docs/fak/advanced-topics.md)

**Lab:**
```bash
FAK_PLANNER_TIMEOUT_S=600 FAK_HTTP_WRITE_TIMEOUT_S=600 fak serve --addr 127.0.0.1:8080 --gguf model.gguf --policy policy.json
```

**Checkpoint:** Explain why FAK_HTTP_WRITE_TIMEOUT_S must be at least FAK_PLANNER_TIMEOUT_S, what setting the write timeout to 0 does and why it is a slow-loris risk on a network bind, and the [5,3600] clamp on the planner timeout.

### FAK 507 — Deploying the Gateway: Docker, Compose, Kubernetes, Bare Metal

**Prerequisites:** **FAK 504**, **FAK 505**

**You'll be able to:**
- Deploy the single static binary across four targets using the distroless nonroot image
- Walk the production-readiness checklist (auth on, policy pinned, intentional bind, sized timeouts, audit journal, non-root)
- Explain why /healthz is a valid readiness probe (no /readyz; GGUF loads before bind) and why readOnlyRootFilesystem is safe

**Read:** [`docs/fak/deployment-guide.md`](docs/fak/deployment-guide.md), [`docs/fak/server-quickstart.md`](docs/fak/server-quickstart.md)

**Lab:**
```bash
docker build -t fak:0.30.0 . ; docker run --rm -p 8080:8080 -e FAK_GATEWAY_KEY="$(openssl rand -hex 32)" fak:0.30.0 serve --addr 0.0.0.0:8080 --base-url http://host.docker.internal:11434/v1 --model qwen2.5:1.5b
```

**Checkpoint:** Walk the production-readiness checklist and justify each item; explain why /healthz is a valid readiness probe and why readOnlyRootFilesystem is safe for fak.

### FAK 508 — Scaling and HA: Process-Local State and Sticky Routing

**Prerequisites:** **FAK 507**, **FAK 407**, **FAK 314**

**You'll be able to:**
- Explain why the verdict path is stateless and replicates freely but the vDSO cache and per-trace IFC ledger are process-local
- Configure sticky-by-trace_id routing for IFC correctness
- Explain why scaling out dilutes the cross-agent vDSO hit rate and why rate-limit counters are per-process

**Read:** [`docs/fak/advanced-topics.md`](docs/fak/advanced-topics.md), [`docs/fak/observability.md`](docs/fak/observability.md)

**Lab:**
```bash
Configure an nginx upstream with `hash $http_x_trace_id consistent;` over three fak gateways and verify that all calls of one trace land on one replica.
```

**Checkpoint:** Explain why a multi-call IFC flow needs sticky routing by trace_id, why scaling out reduces the vDSO cross-agent hit rate, and why FAK_RATELIMIT_MAX_CALLS gives 'N per replica the trace touches' rather than a true fleet cap under round-robin.

### FAK 509 — The MCP Tool-Result Wire: Refusal as a Value

**Prerequisites:** **FAK 503**, **FAK 312**

**You'll be able to:**
- Explain why isError is always false even on a DENY (deny as successful adjudication)
- Given verdict.reason='SELF_MODIFY', derive the disposition class (RETRYABLE/WAIT/ESCALATE/TERMINAL)
- Name on which verdict kind repaired_arguments appears

**Read:** [`docs/mcp-tool-result.md`](docs/mcp-tool-result.md)

**Lab:**
```bash
Hand-write the SyscallResponse JSON a client would receive (a) when ctxmmu quarantines a secret-shaped result and (b) when canon repairs a path; verify each field against the tables in docs/mcp-tool-result.md.
```

**Checkpoint:** Why is isError false even on a DENY? Given verdict.reason='SELF_MODIFY', what disposition does kernel.Disposition derive, and on which verdict kind does repaired_arguments appear?

### FAK 510 — Troubleshooting the Gateway and the fak CLI Verbs

**Prerequisites:** **FAK 504**

**You'll be able to:**
- Diagnose port conflicts, OOM/model-load failures, GPU/CUDA/Vulkan errors, tokenizer fallbacks, and policy errors
- Use the debugging tools (/healthz, /metrics load phases, FAK_LOG=debug, --policy-check)
- Situate serve among the run/preflight/bench/policy/agent/recall/debug verbs that author and exercise the same capability floor

**Read:** [`docs/fak/server-troubleshooting.md`](docs/fak/server-troubleshooting.md), [`docs/cli-reference.md`](docs/cli-reference.md)

**Lab:**
```bash
fak serve --gguf models/qwen.gguf --policy-check   # validate model+policy load without binding a listener
```

**Checkpoint:** Given 'bind: address already in use', diagnose and fix it two ways; explain the troubleshooting step for a GGUF that embeds no usable BPE tokenizer (the offline-mock-planner fallback), and situate serve among the run/preflight/bench/policy verbs.

### FAK 511 — The Integration Index: Repoint One Base URL

**Prerequisites:** **FAK 503**

**You'll be able to:**
- Identify the one configuration value a team changes to route every proposed tool call through fak
- State what does NOT change (the agent code itself)
- Pick the right per-agent integration guide from the index

**Read:** [`docs/integrations/README.md`](docs/integrations/README.md)

**Lab:**
```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # expect DENY (POLICY_BLOCK); then --tool search_kb expecting ALLOW
```

**Checkpoint:** Given a team running LangChain against Ollama, name the one configuration value they change to route every proposed tool call through fak, and state what does NOT change.

### FAK 512 — Claude Code / Anthropic API Through fak

**Prerequisites:** **FAK 511**

**You'll be able to:**
- Point ANTHROPIC_BASE_URL at the gateway ORIGIN (not the /v1 path) and run the dogfood launcher
- Read the denial table and the _fak/fak response extension
- Predict the verdict for a dangerous call under the dogfood policy

**Read:** [`docs/integrations/claude.md`](docs/integrations/claude.md)

**Lab:**
```bash
./scripts/dogfood-claude.sh --probe "Reply with exactly the word: pong"  (Windows: .\scripts\dogfood-claude.ps1 --probe "say pong"); then ./fak preflight --tool Bash --args '{"command":"rm -rf /tmp/x"}' --policy examples/dogfood-claude-policy.json
```

**Checkpoint:** Explain why the Anthropic base URL is the gateway ORIGIN (http://127.0.0.1:8080) and not the /v1 path, and predict the verdict for git push origin master under the dogfood policy.

### FAK 513 — OpenAI Codex / OpenAI SDK Through fak

**Prerequisites:** **FAK 511**

**You'll be able to:**
- Set OPENAI_BASE_URL (or SDK base_url) to fak's /v1 origin with no code change
- Apply coding-agent policy patterns (code-review, safe-refactor, dry-run DevOps)
- Show the two-step migration from a direct OpenAI client

**Read:** [`docs/integrations/openai-codex.md`](docs/integrations/openai-codex.md)

**Lab:**
```bash
./fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model codellama:7b --policy examples/dev-agent-policy.json  &&  ./fak preflight --tool Bash --args '{"command":"git push origin main"}' --policy examples/dev-agent-policy.json
```

**Checkpoint:** Show the two-step change that adds the kernel boundary to an existing openai.OpenAI(api_key=...) client, and explain why the application code itself stays unchanged.

### FAK 514 — Cursor via MCP or OpenAI Proxy

**Prerequisites:** **FAK 511**

**You'll be able to:**
- Wire fak into Cursor as a native MCP server (ask-the-kernel) or as an OpenAI-compatible proxy
- Contrast ask-the-kernel with transparent-proxy and write the JSON config for each
- Decide when to choose MCP over the proxy integration

**Read:** [`docs/integrations/cursor.md`](docs/integrations/cursor.md)

**Lab:**
```bash
./fak policy --dump > cursor-policy.json  &&  ./fak policy --check cursor-policy.json  &&  ./fak preflight --tool read_file --args '{"path":"test.txt"}' --policy cursor-policy.json
```

**Checkpoint:** Describe when you would choose Cursor's MCP integration over the OpenAI-proxy integration, and what each gives you at the tool boundary.

### FAK 515 — MCP One-Paste Setup and the fak_* Tools

**Prerequisites:** **FAK 511**, **FAK 509**

**You'll be able to:**
- Run fak serve --stdio as an MCP server exposing fak_adjudicate, fak_syscall, fak_admit, fak_changes, fak_revoke
- Drop a .mcp.json at the project root and complete the stdio handshake
- Name which fak_* tool you call BEFORE running a tool vs AFTER

**Read:** [`examples/mcp/README.md`](examples/mcp/README.md), [`docs/integrations/adopter-playbook.md`](docs/integrations/adopter-playbook.md)

**Lab:**
```bash
python examples/mcp/verify.py   # PASS/FAIL, exit 0/1 — drives the real stdio transport: initialize, tools/list, git_push->DENY, git_status->ALLOW
```

**Checkpoint:** Name which fak_* tool you call BEFORE running a tool your own client executes vs which one you call AFTER, and state what each protects against.

### FAK 516 — Agent<->Kernel Architecture and the Frozen ABI Verdict Union

**Prerequisites:** **FAK 511**, **FAK 208**

**You'll be able to:**
- Name the six verdict kinds in the closed union
- Explain 'deny-as-value': which HTTP status a policy refusal carries and what an HTTP error status is reserved for
- Use the stable contract (gateway entry points, ToolCall struct, internal/abi/types.go) that every integration depends on

**Read:** [`docs/fak/agent-integration-architecture.md`](docs/fak/agent-integration-architecture.md)

**Lab:**
```bash
curl http://127.0.0.1:8080/v1/fak/changes?since=0  &&  curl -X POST http://127.0.0.1:8080/v1/fak/revoke -H 'Content-Type: application/json' -d '{"witness":"git-commit-abc123"}'
```

**Checkpoint:** Name the six verdict kinds in the closed union and explain what 'deny-as-value' means: which HTTP status does a policy refusal carry, and what is an HTTP error status reserved for?

### FAK 517 — Framework Cookbook: Transparent Proxy (Mode A) vs Explicit Adjudication (Mode B)

**Prerequisites:** **FAK 516**, **FAK 513**, **FAK 302**

**You'll be able to:**
- Give the smallest per-framework change for LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI (plus Semantic Kernel, Haystack, Griptape)
- Write the shared guarded() wrapper that adjudicates and admits (Mode B)
- Apply the honest scope (the floor bounds tool NAMES not arguments) and choose proxy vs explicit adjudication

**Read:** [`docs/fak/agent-framework-integration.md`](docs/fak/agent-framework-integration.md)

**Lab:**
```bash
fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b --policy policy.json  &&  curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate -H 'Content-Type: application/json' -d '{"tool":"refund_payment","arguments":{}}'
```

**Checkpoint:** For LangChain, give the Mode A one-line change AND the Mode B guarded() wrapper, and explain the honest-scope caveat about why you keep irreversible operations OFF the allow-list.

### FAK 518 — Migration: Moving Existing Code by Repointing a Base URL

**Prerequisites:** **FAK 516**

**You'll be able to:**
- Migrate LangChain, AutoGen, llama.cpp, or a direct OpenAI/Anthropic client by redirecting the base URL
- State the two invariants that hold for every migration (fak never executes your tools; a refusal is a 200 carrying a value)
- Diagnose the OpenAI vs Anthropic base-URL gotcha

**Read:** [`docs/fak/migration-guide.md`](docs/fak/migration-guide.md)

**Lab:**
```bash
fak serve --addr 127.0.0.1:8080 --provider openai --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model gpt-4o --policy policy.json  &&  fak preflight --policy policy.json --tool git_push --args '{}'
```

**Checkpoint:** A client gets 404 on /v1/v1/messages. Diagnose the cause and the fix, then state which two invariants hold for every migration.

### FAK 519 — Multi-Language Client Code and Disposition-Aware Retry

**Prerequisites:** **FAK 516**, **FAK 509**

**You'll be able to:**
- Call the fak-native one-POST-one-verdict surface from Python, JS/TS, Go, and Rust
- Read verdict.kind (never HTTP status alone) and branch on disposition to spend zero extra model turns
- Explain how the four dispositions change retry logic

**Read:** [`docs/fak/multi-language-examples.md`](docs/fak/multi-language-examples.md)

**Lab:**
```bash
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate -H 'Content-Type: application/json' -d '{"tool":"Bash","arguments":{"command":"rm -rf /tmp/x"}}'   # inspect verdict.kind / reason / disposition
```

**Checkpoint:** Given a DENY verdict, explain how the four dispositions (RETRYABLE, WAIT, ESCALATE, TERMINAL) change your client's retry logic, and state why you must read verdict.kind instead of the HTTP status code.

### FAK 520 — The Adopter Playbook: Front-a-Model, Manual MCP, Embed-in-CI

**Prerequisites:** **FAK 512**, **FAK 515**

**You'll be able to:**
- Run the bare-serve production loop (author policy, bind an auth-key env, start, check /healthz, repoint base URL)
- Serve all three shapes (A proxy, B stdio MCP, C offline CI gate) from one binary
- Explain why --require-key-env matters once the bind address is not loopback

**Read:** [`docs/integrations/adopter-playbook.md`](docs/integrations/adopter-playbook.md)

**Lab:**
```bash
fak policy --dump > policy.json  &&  fak policy --check policy.json  &&  export FAK_TOKEN=$(openssl rand -hex 32)  &&  fak serve --addr 0.0.0.0:8080 --provider openai --base-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b --policy policy.json --require-key-env FAK_TOKEN  &&  curl -s http://127.0.0.1:8080/healthz
```

**Checkpoint:** List the five ordered steps of the bare-serve loop (Shape A), and explain why --require-key-env matters once the bind address is not loopback.

### FAK 521 — GGUF Loading: Offsets, Dtypes, and Dequant Layout

**Prerequisites:** **FAK 205**

**You'll be able to:**
- Address each tensor's own byte window off the hot path and dequantize every block format to f32
- Map GGUF tensor names to HF names
- Compute an absolute FileOffset from an in-data offset and alignment, and explain why reading tensor i can never address tensor j's bytes

**Read:** [`docs/proofs/ggufload.md`](docs/proofs/ggufload.md)

**Lab:**
```bash
go test ./internal/ggufload/ -count=1 -timeout 120s -run 'TestReadParsesMetadataTensorDirectoryAndConfig|TestWeightSourceReadsAndDequantizesSimpleTensors' -v
```

**Checkpoint:** Given a tensor declared at in-data offset 64 with 64-byte alignment, compute its absolute FileOffset and explain why reading tensor i can never address tensor j's bytes. Why is the strict encode-then-read involution OPEN here?

### FAK 522 — Tokenizer: Lossless ByteLevel BPE With Oracle Parity

**Prerequisites:** **FAK 521**

**You'll be able to:**
- Convert text to/from token ids via a ByteLevel byte-to-unicode bijection and lowest-rank-first BPE merges
- Explain why BPE merge selection is deterministic (a pure function of symbols + merge ranks)
- Explain why the per-model pre-tokenizer dispatch (Qwen Split regex vs GPT-2 ByteLevel) is needed for oracle parity

**Read:** [`docs/proofs/tokenizer.md`](docs/proofs/tokenizer.md)

**Lab:**
```bash
go test -run 'TestEncodeSmallByteLevelBPEFixture|TestDecodePreservesSplitUTF8Bytes|TestQwenOracleGolden' -v ./internal/tokenizer/ -count=1 -timeout 120s
```

**Checkpoint:** Explain why BPE merge selection is deterministic and why the per-model pre-tokenizer dispatch is needed for oracle parity.

### FAK 523 — Normalization: RMSNorm, NormGain1p, and LayerNorm

**Prerequisites:** **FAK 522**

**You'll be able to:**
- Compute RMSNorm, Gemma's (1+w) gain, and mean-subtracting LayerNorm to their closed forms
- Explain why the sum-of-squares is kept scalar in-order so f32 forward rungs stay bit-reproducible
- State the approximate input magnitude at which the f32 sum-of-squares overflows

**Read:** [`docs/proofs/model-norm.md`](docs/proofs/model-norm.md)

**Lab:**
```bash
go test -run 'TestNormGain1p|TestLayerNormAxis|TestProofNormNumericallyStableLargeInputs' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Write the closed form RMSNorm computes and state why LayerNorm is shift+scale equivariant in the eps->0 limit. At roughly what input magnitude does the f32 sum-of-squares overflow?

### FAK 524 — RoPE: Rotary Position Embedding and Scaling Variants

**Prerequisites:** **FAK 523**

**You'll be able to:**
- Inject position by Givens-rotating each dim-pair by p*inv_freq and show attention depends only on (m-n)
- Apply llama3/yarn/longrope frequency rescaling
- Explain why the yarn/longrope attention-factor scale breaks per-pair norm preservation

**Read:** [`docs/proofs/model-rope.md`](docs/proofs/model-rope.md)

**Lab:**
```bash
go test -run 'TestProofRopePreservesPairNorm|TestProofRopeDotRelativePosition|TestRopeScalingLlama3' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Prove <R_m q, R_n k> depends on m,n only through (m-n), and explain why the yarn/longrope attention-factor scale breaks per-pair norm preservation (cos^2+sin^2=scale^2!=1).

### FAK 525 — Attention: Stable Softmax, Causal Mask, and the Attention Sink

**Prerequisites:** **FAK 524**

**You'll be able to:**
- Compute scaled-dot-product attention with a row-stochastic shift-invariant softmax
- Explain why the score loop makes causality structural rather than after-the-fact masking
- Derive the single-visible-score sink weight 1/(1+exp(sink-s))

**Read:** [`docs/proofs/model-attention.md`](docs/proofs/model-attention.md)

**Lab:**
```bash
go test -run 'TestAttentionSinkSoftmaxDropsSink|TestProofSoftmaxRowStochasticAndShiftInvariant|TestProofCausalStrictlyLowerTriangular' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Explain why the score loop `for j := lo; j <= t` makes causality structural rather than after-the-fact masking, and derive the single-visible-score sink weight.

### FAK 526 — MLP / SwiGLU+GeGLU, MoE Routing, and the Residual Stream

**Prerequisites:** **FAK 525**

**You'll be able to:**
- Compute the gated MLP down(act(gate(x))*up(x)) and top-k MoE weighted-sum routing
- Describe torch.topk's stable tie-break and NormTopKProb renormalization
- Name the four residual topologies (PreNorm/PostNorm/Sandwich/Parallel) and how each composes the sub-layer delta

**Read:** [`docs/proofs/model-mlp+residual.md`](docs/proofs/model-mlp+residual.md)

**Lab:**
```bash
go test -run 'TestMoEDenseNoOpIdentical|TestBlockTopologyComposition|TestMoERoutingHandComputed' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Describe MoE top-k routing including torch.topk's stable tie-break and NormTopKProb renormalization, and name the four residual topologies and how each composes the sub-layer delta.

### FAK 527 — In-Kernel KV Cache: Slotting, Span-Exact Eviction, SWA, Prefix Reuse

**Prerequisites:** **FAK 526**, **FAK 406**

**You'll be able to:**
- Correctly slot (layer,pos,head) and Evict byte-identically to never-having-seen a span
- Explain why eviction re-rotates each survivor's K from stored pre-RoPE Kraw in a SINGLE rotation
- Explain why the sliding window keys off pos[] rather than the slice index

**Read:** [`docs/proofs/model-kv.md`](docs/proofs/model-kv.md)

**Lab:**
```bash
go test -run 'TestStandardLayoutNoOp|TestKVQuarantineEqualsNeverSaw|TestSWAWindowMasksOldKeys|TestKVPrefixReuseMatchesRecompute' ./internal/model/ -count=1 -timeout 180s -v
```

**Checkpoint:** Explain why eviction re-rotates each survivor's K from stored pre-RoPE Kraw in a SINGLE rotation rather than composing two, and why the sliding window keys off pos[] instead of the slice index.

### FAK 528 — Quantization: Q4_K/Q8_0/Q4_0 Dequant, AWQ, and Bit-Identical int8 SDOT

**Prerequisites:** **FAK 521**, **FAK 526**

**You'll be able to:**
- Apply affine-correct dequant of GGUF k-quant and AWQ 4-bit formats
- Explain why the int8 SDOT reduction is bit-identical across SIMD lane orders (order-independent, no overflow)
- Distinguish what the AWQ 'matches reference' claim PROVES (affine self-consistency) from what is OPEN (no HF AutoAWQ fixture)

**Read:** [`docs/proofs/model-quant.md`](docs/proofs/model-quant.md), [`docs/explainers/awq-quantization.md`](docs/explainers/awq-quantization.md)

**Lab:**
```bash
go test -run 'TestQ4KDequantSuperBlockMatchesRef|TestQ4KReduceAsmMatchesScalar|TestProofAWQMatchesReference' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** State the AWQ dequant formula scale[o]*(code-8) and explain why the int8 SDOT reduction is bit-identical across SIMD lane orders. Which part of the AWQ claim is PROVEN and which is OPEN?

### FAK 529 — Forward-Pass Parity vs the HuggingFace Oracle

**Prerequisites:** **FAK 527**, **FAK 528**, **FAK 210**

**You'll be able to:**
- Reproduce PyTorch/HF hidden-state cosine ~1, per-position argmax, and greedy ids token-for-token on smollm2
- Explain why argmax-pin at every position is a stronger witness than a logit tolerance
- Read the honest ledger: PROVEN on llama, OPEN for other families, REFUTED for Qwen3.6 hybrid-GDN (diverges at token 3)

**Read:** [`docs/proofs/model-forward-parity.md`](docs/proofs/model-forward-parity.md)

**Lab:**
```bash
go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s -v
```

**Checkpoint:** Explain why argmax-pin at every position is a stronger witness than a logit tolerance, and describe the Qwen3.6 REFUTED finding (near-tie argmax flip at token 3) without conflating it with the llama PROVEN row.

### FAK 530 — The Compute HAL Seam and Hardware Portability

**Prerequisites:** **FAK 529**, **FAK 210**

**You'll be able to:**
- Name three of the seven baked-in hardware assumptions the internal/compute Backend interface neutralizes and the type that lifts each
- Explain why adding a GPU/NPU is a registration, not a fork of the hot loop
- Explain why only a Reference backend faces max|delta|=0 while every Approx faces argmax-exact + logit-cosine

**Read:** [`docs/explainers/hardware-portability.md`](docs/explainers/hardware-portability.md), [`docs/proofs/compute-gemm.md`](docs/proofs/compute-gemm.md)

**Lab:**
```bash
go test -run 'MatMul|Reduction|Q8|Correctness|Registry|Device' ./internal/compute/ -count=1 -timeout 120s -v
```

**Checkpoint:** Name three of the seven assumptions the seam neutralizes and the type that lifts each, and explain why only a Reference backend faces max|delta|=0 while every Approx faces argmax-exact + logit-cosine.

### FAK 531 — Metal GPU GEMM Parity and the Stub-vs-Device Build

**Prerequisites:** **FAK 530**
  ·  **Background:** **FAK 534**

**You'll be able to:**
- Match Apple-Silicon Metal GEMM (f16 MPS) to the f32 CPU reference within the half-precision error model
- Explain why the witness is err/scale<1% and logit-cosine=1.0 rather than a bit-compare
- Explain how mutually-exclusive build tags guarantee the stub introduces no numerical drift

**Read:** [`docs/proofs/metalgemm.md`](docs/proofs/metalgemm.md)

**Lab:**
```bash
CGO_ENABLED=1 go test -tags fakmetal -run 'MatMul|Reset' ./internal/metalgemm/ -count=1 -v   # (Apple Silicon only; default build: go build ./internal/metalgemm/)
```

**Checkpoint:** Explain why the Metal witness is err/scale<1% and logit-cosine=1.0 rather than a bit-compare, and how the mutually-exclusive build tags guarantee the stub introduces no numerical drift.

### FAK 532 — The Engine Seam: Determinism and Cache-Invalidation Binding

**Prerequisites:** **FAK 529**, **FAK 206**

**You'll be able to:**
- Explain why greedy decode makes Complete a pure function of (tool,args) (no RNG/clock)
- Bind enginecache invalidation directives to SGLang/vLLM resets
- Explain the fail-closed gate: why Invalidate errors BEFORE issuing any reset when RequiredScope==exact_span but the engine only supports whole-prefix reset

**Read:** [`docs/proofs/engine-seam.md`](docs/proofs/engine-seam.md)

**Lab:**
```bash
go test ./internal/modelengine/ -run 'TestDecodeIsDeterministicAndInputDriven|TestCompleteRunsRealDecode' -count=1 -v && go test ./internal/enginecache/ -count=1 -v
```

**Checkpoint:** Explain why greedy decode makes Complete a pure function of (tool,args), and describe the fail-closed gate when RequiredScope==exact_span but the engine only supports whole-prefix reset.

### FAK 533 — In-Kernel Model & Compute Env Knobs (FAK_* Engine Vars)

**Prerequisites:** **FAK 502**, **FAK 528**

**You'll be able to:**
- Tune GPU residency budget, Q4K/Q8 load format, matmul worker budget, SIMD tiers, and generation bounds
- Distinguish FAK_WORKERS vs FAK_BUDGET for matmul parallelism
- Separate the model-engine-env vars from the serve-config vars

**Read:** [`docs/model-engine-env.md`](docs/model-engine-env.md), [`docs/fak/server-config.md`](docs/fak/server-config.md), [`GETTING-STARTED.md`](GETTING-STARTED.md)

**Lab:**
```bash
FAK_Q4K=1 fak serve --addr 127.0.0.1:8137 --gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf --model qwen3.6-27b-q4k
```

**Checkpoint:** Explain what FAK_Q4K changes about the load/decode path for a Qwen3.6-27B model, how FAK_WORKERS vs FAK_BUDGET differ, and which FAK_* vars belong to model-engine-env vs serve-config.

### FAK 534 — GPU Lease: Machine-Wide Mutual Exclusion for Model Residency

**Prerequisites:** **FAK 533**

**You'll be able to:**
- Explain why at most one live holder machine-wide is required before two processes both try to make a model resident on the same GPU
- Explain the three regime-D properties: fail-closed-when-busy (no-wait), bounded wait-then-acquire, and crashed-holder reclaim via flock release on process exit
- Identify this as the operational precondition for Tier-2b real-weights serving (FAK 533) and Metal modelbench (FAK 531)

**Read:** [`docs/proofs/gpulease.md`](docs/proofs/gpulease.md)

**Lab:**
```bash
go test ./internal/gpulease/ -count=1 -timeout 120s -run 'TestNoWaitBusyThenFree|TestWaitTimesOut|TestWaitThenSucceed|TestReleaseOnProcessExit|TestReleaseIdempotent' -v
```

**Checkpoint:** Explain why a machine-wide flock guarantees at most one live holder, why a busy lease fails closed (no-wait) rather than racing, and how a crashed holder's lease is reclaimed without a manual unlock. State why this is the precondition for the real-weights modelbench path.

### FAK 535 — The Gateway Drop Guarantee: Fail-Closed on a Failed Adjudication

**Prerequisites:** **FAK 510**, **FAK 314**

**You'll be able to:**
- State the two regime-D theorems: a wire verdict equals the in-process kernel verdict (no network bypass), and a call that fails adjudication is dropped fail-closed
- Explain why the wire never carries an abi.Ref so a client cannot smuggle a pre-trusted CAS handle to skip the IFC / self-modify rungs
- Identify the honest gap (no single A==B DeepEqual test; parity rests on a matched pair plus the single-seam structural argument)

**Read:** [`docs/proofs/gateway.md`](docs/proofs/gateway.md)

**Lab:**
```bash
go test -run 'Verdict|Adjud|HTTPSyscall|DefaultDeny|DenyIsValue|FailsClosed' ./internal/gateway/ -count=1 -timeout 180s -v
```

**Checkpoint:** State the two gateway theorems and explain why buildCall minting its own tainted agent-scoped Ref (not accepting one off the wire) is what prevents a network bypass. Name the honest gap the proof discloses, and explain why this is the serving-side analogue of the security floor.

---

## L600 — Mastery: benchmarks, honesty discipline, and extending the kernel

**Theme.** Honest baselines and the benchmark authority, the fleet/web/parity results, the AgentDojo red-team, the claims ledger and status gates, the additive ABI + architest, the RSI ship-gate, the three-gate leaf pattern, and the dispatch loop.

**Who joins here.** A contributor or reviewer who has worked through the cores and serving. Join here if you want to read fak's numbers honestly, land an optimization that survives review, or operate the self-improvement and issue-dispatch loops.

**Assumes you can already pass:** **FAK 207**, **FAK 208**, **FAK 209**, **FAK 210**.

| Course | Hard prerequisites |
|---|---|
| **FAK 601** — The Claims Ledger: SHIPPED/SIMULATED/STUB and the 0/29-Novel Posture | **FAK 207** |
| **FAK 602** — STATUS, Subsystem Checks, and What a Passing Boundary Does NOT Prove | **FAK 601** |
| **FAK 603** — The Repro Packet: A No-Credential Offline Boundary Reproduction | **FAK 601**, **FAK 105** |
| **FAK 604** — The Fleet Benchmark Suite: Five Model-Agnostic Kernel Demos | **FAK 405**, **FAK 407** |
| **FAK 605** — Honest Baselines: Naive/Cold vs Tuned Warm-Cache, Measured vs Modeled | **FAK 604**, **FAK 403** |
| **FAK 606** — Benchmark-Authority: The Single Source of Truth Discipline | **FAK 605** |
| **FAK 607** — A/B Paired-Replay Isolation: Attributable Deltas | **FAK 604**, **FAK 407** |
| **FAK 608** — Metrics: Percentiles, KPIs, and the A/B Gate | **FAK 607** |
| **FAK 609** — WebVoyager Baselines and Baseline Stratification | **FAK 605** |
| **FAK 610** — fak vs vLLM / SGLang / llama.cpp / Provider KV Caching | **FAK 609**, **FAK 405** |
| **FAK 611** — The Hardware Matrix: Portability as a Correctness Claim | **FAK 606**, **FAK 530** |
| **FAK 612** — Local-vs-Frontier Parity: Three Axes, Never Blended | **FAK 303**, **FAK 607** |
| **FAK 613** — The AgentDojo Red-Team Threat Model and Two-Gate Defense | **FAK 303**, **FAK 315** |
| **FAK 614** — The RSI Ship-Gate: The Non-Forgeable Keep-Bit and the Self-Measured Loop | **FAK 207**, **FAK 210** |
| **FAK 615** — Extending fak: The Three-Gate Leaf Pattern | **FAK 209**, **FAK 210**, **FAK 614** |
| **FAK 616** — The Witness-Gated Issue-Dispatch Loop | **FAK 614**, **FAK 307** |

### FAK 601 — The Claims Ledger: SHIPPED/SIMULATED/STUB and the 0/29-Novel Posture

**Prerequisites:** **FAK 207**

**You'll be able to:**
- Assign exactly one tag (SHIPPED / SIMULATED / STUB) to a capability claim and justify it
- Explain what the 0/29-novel finding means for how fak frames its contribution (the assembly, not a novel primitive)
- Surface the honest ceilings (the ~100% evadable detector; baselines that are vs-naive not vs-tuned)

**Read:** [`CLAIMS.md`](CLAIMS.md), [`STATUS.md`](STATUS.md)

**Lab:**
```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ci.ps1
```

**Checkpoint:** Given a capability described as 'GPU backend witnessed real' vs 'token-per-watt telemetry', assign the correct tag to each and justify it; explain what the 0/29-novel finding means for how fak frames its contribution.

### FAK 602 — STATUS, Subsystem Checks, and What a Passing Boundary Does NOT Prove

**Prerequisites:** **FAK 601**

**You'll be able to:**
- Read STATUS.md and SUBSYSTEM-CHECKS.md with each check's explicit 'what it does not prove' column
- State what the tau2-smoke boundary-tax check proves and three things it does not
- Name the two real product gates (Phase 0 clean-node, Phase 1 non-reference 7-9B GPU parity)

**Read:** [`STATUS.md`](STATUS.md), [`SUBSYSTEM-CHECKS.md`](SUBSYSTEM-CHECKS.md)

**Lab:**
```bash
python tools\subsystem_check_audit.py --profile smoke --out-json fak\experiments\subsystem-checks\latest-smoke.json --out-md fak\experiments\subsystem-checks\latest-smoke.md
```

**Checkpoint:** State what the tau2-smoke boundary-tax check proves and at least three things it explicitly does not, and name the two real product gates.

### FAK 603 — The Repro Packet: A No-Credential Offline Boundary Reproduction

**Prerequisites:** **FAK 601**, **FAK 105**

**You'll be able to:**
- Run the four packet commands and state what each of the four witnesses proves
- State what the packet's Non-Claims section deliberately does NOT prove (detector recall, production readiness, fleet-scale)
- Put the smallest honest artifact in front of a skeptic

**Read:** [`docs/repro-packet.md`](docs/repro-packet.md)

**Lab:**
```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json && go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}" && go run ./cmd/fak agent --offline
```

**Checkpoint:** Run the four packet commands and state, from the output, what each of the four witnesses proves and what the packet's Non-Claims section says it deliberately does NOT prove.

### FAK 604 — The Fleet Benchmark Suite: Five Model-Agnostic Kernel Demos

**Prerequisites:** **FAK 405**, **FAK 407**

**You'll be able to:**
- Name the five demos (fan-out, turn-tax sweep, A/B + safety floor, RadixAttention hit rate, token accounting)
- For each demo, name the one kernel counter or ablation it reads
- Explain why none of them needs a GPU

**Read:** [`docs/explainers/fleet-benchmarks.md`](docs/explainers/fleet-benchmarks.md)

**Lab:**
```bash
go run ./cmd/fanbench -agent-max 1024 -grid log  # then: go run ./cmd/fleetbench -agents 50 -turns 50 -trials 24 -profile read-heavy -granularity resource
```

**Checkpoint:** Name the five demos and state, for each, the one kernel counter or ablation it reads. Explain why none of them needs a GPU.

### FAK 605 — Honest Baselines: Naive/Cold vs Tuned Warm-Cache, Measured vs Modeled

**Prerequisites:** **FAK 604**, **FAK 403**

**You'll be able to:**
- Report every multiple against BOTH a naive/cold reference and the best already-shipped warm baseline
- Never blend measured kernel events with modeled cost
- Explain which number survives contact with a tuned SGLang stack and why

**Read:** [`docs/explainers/fleet-benchmarks.md`](docs/explainers/fleet-benchmarks.md), [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md)

**Lab:**
```bash
go run ./cmd/ctxdemo -print  # read the same table's (refx)=35.5x cold column vs fak-win=1.1x warm column side by side
```

**Checkpoint:** Given the ctxdemo fleet-5x50 row (35.5x vs cold, 1.1x vs warm), explain which number survives contact with a tuned SGLang stack and why, and which half of a turntax result is measured vs modeled.

### FAK 606 — Benchmark-Authority: The Single Source of Truth Discipline

**Prerequisites:** **FAK 605**

**You'll be able to:**
- State the rule for adding/changing a benchmark number and the three pieces of evidence that must back it (source commit, JSON artifact, reproduce command)
- Trace a row to its cited artifact and confirm the field value
- Explain why a stale claim is tombstoned (e.g. 11.2x->5.3x), not removed, and what made the old number shrink

**Read:** [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md), [`docs/explainers/fleet-benchmarks.md`](docs/explainers/fleet-benchmarks.md)

**Lab:**
```bash
Pick any row in BENCHMARK-AUTHORITY.md (e.g. RadixAttention hit rate 86.7%) and trace it: open its cited JSON artifact and confirm the field value matches; run the row's reproduce command.
```

**Checkpoint:** State the rule for adding/changing a benchmark number and what three pieces of evidence must back it. Explain why the F1 tombstone (50x5 11.2x->5.3x) is kept, not removed, and what made the old number shrink.

### FAK 607 — A/B Paired-Replay Isolation: Attributable Deltas

**Prerequisites:** **FAK 604**, **FAK 407**

**You'll be able to:**
- State the two isolation invariants: only the toggled variable differs, and Net.TurnsSaved delta == VDSOHits exactly
- Explain why the happy-path control saving 0 matters
- Replay one frozen trace through a freshly-reset kernel twice toggling one lever

**Read:** [`docs/proofs/bench-ab-isolation.md`](docs/proofs/bench-ab-isolation.md)

**Lab:**
```bash
go test ./internal/turnbench/ -count=1 -run 'TestRun_VDSOAblationIsARealPathSwap|TestRun_HappyPathSavesNothing|TestStochastic_ZeroRateP50IsZero' -v
```

**Checkpoint:** Explain the two invariants the isolation proof discharges and why the happy-path control saving 0 matters.

### FAK 608 — Metrics: Percentiles, KPIs, and the A/B Gate

**Prerequisites:** **FAK 607**

**You'll be able to:**
- Show why pct(p)=sorted[int(p/100*(n-1))] is monotone non-decreasing in p (P50<=P99)
- Explain the identical-workload guard and the fail-closed gate at a zero baseline
- State the doc's two honest OPENs (one sample-set instance witnessed; KPI fold-equals-definition lives in bench.go)

**Read:** [`docs/proofs/metrics.md`](docs/proofs/metrics.md)

**Lab:**
```bash
go test ./internal/metrics/ -run 'TestHistPercentilesMonotonic|TestValidateWorkloadHash|TestComputeGate' -count=1 -timeout 120s -v
```

**Checkpoint:** Show why pct(p) is monotone non-decreasing in p. Then explain the doc's two honest OPENs.

### FAK 609 — WebVoyager Baselines and Baseline Stratification

**Prerequisites:** **FAK 605**

**You'll be able to:**
- Distinguish A/C (8.8-9.7x), B/C (1.0-1.10x), and A/B (8.8x worker-independent) on the 643-task WebVoyager set
- Identify which is the structural turn-tax and which is the marginal-vs-tuned win
- Explain why fak does not appear on the success-rate leaderboard (capability vs efficiency)

**Read:** [`docs/webbench-baselines.md`](docs/webbench-baselines.md)

**Lab:**
```bash
go run ./cmd/fak webbench describe --dataset testdata/webbench/sample-tasks.jsonl
```

**Checkpoint:** On WebVoyager, distinguish A/C, B/C, and A/B. Which is the structural turn-tax, which is the marginal-vs-tuned win, and why does fak not appear on the success-rate leaderboard?

### FAK 610 — fak vs vLLM / SGLang / llama.cpp / Provider KV Caching

**Prerequisites:** **FAK 609**, **FAK 405**

**You'll be able to:**
- Explain why a per-instance vLLM cache stores ~10x more tokens than fak for a 100-agent fleet
- Name the one capability (addressable/governance eviction) an opportunistic LRU radix cache structurally cannot offer
- Position fak honestly: matches SGLang's hit rate, does NOT win raw throughput, adds the cross-worker layer

**Read:** [`docs/fak-vs-alternatives-comparison.md`](docs/fak-vs-alternatives-comparison.md)

**Lab:**
```bash
go run ./cmd/radixbench -scale 1  # compare fak's hit rate against SGLang's published 50-99% band; note policy-eviction witness
```

**Checkpoint:** For a 100-agent / 100-issue fleet, explain why a per-instance vLLM cache stores ~10x more tokens than fak, and name the one capability that an opportunistic LRU radix cache structurally cannot offer.

### FAK 611 — The Hardware Matrix: Portability as a Correctness Claim

**Prerequisites:** **FAK 606**, **FAK 530**

**You'll be able to:**
- Explain why running the same correctness gates on four platforms (Metal, Vulkan, CUDA Ada+Ampere) is itself a result
- Distinguish which numbers may differ across boxes (live wall-clock) from those that must reproduce byte-for-byte (deterministic token-count/hit-rate)
- Inspect the machine-readable node catalog

**Read:** [`docs/HARDWARE-MATRIX.md`](docs/HARDWARE-MATRIX.md), [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md)

**Lab:**
```bash
python tools/bench_catalog.py show  # inspect the machine-readable node catalog (roles, runs, by-model indexes)
```

**Checkpoint:** Explain why running the SAME correctness gates on four hardware platforms is itself a result, and which class of numbers is allowed to differ across boxes and why.

### FAK 612 — Local-vs-Frontier Parity: Three Axes, Never Blended

**Prerequisites:** **FAK 303**, **FAK 607**

**You'll be able to:**
- Name the three never-blended axes (safety, cost, capability) and who delivers each
- Explain why a local model running fewer turns is not 'faster'
- Explain why the safety win (injection containment) is structural rather than alignment-probabilistic

**Read:** [`docs/explainers/local-vs-frontier-parity.md`](docs/explainers/local-vs-frontier-parity.md), [`SOTA-COMPARISON.md`](SOTA-COMPARISON.md)

**Lab:**
```bash
go -C fak run ./cmd/paritybench --local 'fak/experiments/parity/local-*.json' --reference-cards fak/experiments/parity/reference-frontier.json --reference claude-sonnet --out-md fak/experiments/parity/PARITY.md
```

**Checkpoint:** Name the three never-blended axes and who delivers each. Explain why a local model running FEWER turns is not 'faster', and why the safety win is structural rather than alignment-probabilistic.

### FAK 613 — The AgentDojo Red-Team Threat Model and Two-Gate Defense

**Prerequisites:** **FAK 303**, **FAK 315**

**You'll be able to:**
- Explain why detection-only shows ASR > 0 on paraphrased attacks while full-stack (capability floor + provenance IFC) holds at 0
- Identify which of the four compiled-loop arrows is intentionally NOT built (an RL generator) and why the generative expander is an honest stand-in
- Score Attack Success Rate against two independent gates under an adaptive attacker

**Read:** [`examples/agentdojo-redteam/README.md`](examples/agentdojo-redteam/README.md), [`docs/fak/security.md`](docs/fak/security.md)

**Lab:**
```bash
./examples/agentdojo-redteam/run.sh   # exit 0 iff full-stack ASR == 0 (every attack barred)
```

**Checkpoint:** Why does the detection-only defense show ASR > 0 on paraphrased attacks while full-stack holds at 0? Which of the four compiled-loop arrows is intentionally NOT built, and why is the generative expander an honest stand-in?

### FAK 614 — The RSI Ship-Gate: The Non-Forgeable Keep-Bit and the Self-Measured Loop

**Prerequisites:** **FAK 207**, **FAK 210**

**You'll be able to:**
- Explain why shipgate.Evaluate KEEPs only on strict metric gain AND green suite AND clean truth syscall
- Explain why the unexported keep-bit set only inside Evaluate makes 'no measurable win -> REVERT' forgery-proof
- Explain why the loop re-derives its baseline from latest main every run

**Read:** [`docs/rsi-loop.md`](docs/rsi-loop.md), [`docs/proofs/shipgate.md`](docs/proofs/shipgate.md)

**Lab:**
```bash
go run ./cmd/rsiloop -mode improve -repo . -baseline-ref main -candidates 6,8,8,10 -journal /tmp/rsi.jsonl
```

**Checkpoint:** Explain cycle 3 of the witnessed rsiloop run: why a candidate with a green suite AND a clean tree is still REVERTED, and why the loop re-derives its baseline from latest main every run.

### FAK 615 — Extending fak: The Three-Gate Leaf Pattern

**Prerequisites:** **FAK 209**, **FAK 210**, **FAK 614**

**You'll be able to:**
- Attach at a Register* seam, prove correctness with a deterministic witness, then prove a speed win via the non-forgeable keep-bit
- For a new quantization kernel, name the seam (internal/compute), the correctness class to declare, and the exact gate command that proves it earns its keep
- Explain why a contributor cannot land a plausible-but-wrong (gate 2) or correct-but-slower (gate 3) kernel

**Read:** [`EXTENDING.md`](EXTENDING.md), [`ARCHITECTURE.md`](ARCHITECTURE.md)

**Lab:**
```bash
python tools/extend_preflight.py
```

**Checkpoint:** For a new quantization kernel, name which seam it uses, which correctness class it should declare, and which exact gate command proves it earns its keep (the Gate 3 keep-bit from FAK 614).

### FAK 616 — The Witness-Gated Issue-Dispatch Loop

**Prerequisites:** **FAK 614**, **FAK 307**

**You'll be able to:**
- Trace the loop: route -> spawn one worker -> require an #N-cited commit -> bind commit to issue via dos commit-audit -> close only when re-verified per-SHA
- Explain why a resolved issue whose commit omits #N can never be witnessed-closed
- Explain how the loop guarantees the live-worker population can never exceed its cap

**Read:** [`docs/dispatch-loop.md`](docs/dispatch-loop.md)

**Lab:**
```bash
python tools/dispatch_status.py
```

**Checkpoint:** Explain why a resolved issue whose commit omits #N can never be witnessed-closed, and how the loop guarantees the live-worker population can never exceed its cap.

---

## You've finished the path

If you can pass the checkpoints through **FAK 616**, you can: stand up and harden the
gateway in front of any OpenAI- or Anthropic-compatible model; author and review a
capability floor; explain the write-time quarantine and the IFC taint lattice; read the
in-kernel model's forward pass and its oracle-parity ledger; tell an honest benchmark
from a strawman; and land a new optimization into the kernel through the three-gate leaf
pattern (**FAK 615**) — prove it correct, prove it faster, earn the keep-bit.

Where to go from there:

- **Contribute.** Pick up the leaf pattern (**FAK 615**) and the witness-gated dispatch
  loop (**FAK 616**); the contract is in [`EXTENDING.md`](EXTENDING.md) and
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Audit the honesty.** Re-run the repro packet (**FAK 603**,
  [`docs/repro-packet.md`](docs/repro-packet.md)) and check every number against
  [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) and the claims ledger
  [`CLAIMS.md`](CLAIMS.md).
- **Go deep on the math.** The per-module correctness proofs are the graduate seminar:
  [`docs/proofs/README.md`](docs/proofs/README.md).

Found a course whose reading no longer matches what the code does? That is a doc bug —
please [open an issue](https://github.com/anthony-chaudhary/fak/issues).

