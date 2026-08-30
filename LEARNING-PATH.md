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
sessions cheap to hold warm, right-model-per-call routing, and a pure-Go in-kernel model —
and, riding along on the same write-time checkpoint, a default-deny capability floor, a
write-time result quarantine, and the honesty discipline that keeps every claim checkable.
This page turns all of it into one **linear, prerequisite-ordered curriculum** — a course
catalog, not a doc dump.
Each course points at the doc that already teaches it; the value added here is the
**order** and the **prerequisites**, so you always have the background a page assumes
*before* you open it.

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
> [`README.md`](README.md#try-the-kernel-without-a-key-model-or-gpu) frames as the
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
- Mark why the high public cache number is just the frozen-trajectory ceiling, and the three axes that bend it toward 0% — flexibility, per-turn tool density, and cross-agent fan-out (and why fan-out is a fleet metric, not one agent's hit %)

**Read:** [`docs/explainers/kv-cache-agentic-context.md`](docs/explainers/kv-cache-agentic-context.md), [`docs/explainers/frozen-trajectory-cache-cliff.md`](docs/explainers/frozen-trajectory-cache-cliff.md), [`docs/explainers/context-tape-visuals.md`](docs/explainers/context-tape-visuals.md)

**Lab:**
```bash
Take a prompt with a per-request UUID at the head; move it to the tail and re-run the LCP analysis to reproduce the 0.3% -> 87% hit-rate jump described in the doc.
python tools/cache_curve.py compound   # watch the frozen 99% ceiling collapse along the flex + tool-density axes
python tools/context_tape.py trajectory <your-session>.jsonl --svg session.svg   # SEE the reused prefix dwarf the fresh tip, turn by turn, on YOUR own session (docs/explainers/context-tape-visuals.md)
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

**Read:** [`docs/proofs/vdso.md`](docs/proofs/vdso.md), [`docs/explainers/vdso-revoke-as-comm-revoke.md`](docs/explainers/vdso-revoke-as-comm-revoke.md)

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
docker build -t fak:0.34.0 . ; docker run --rm -p 8080:8080 -e FAK_GATEWAY_KEY="$(openssl rand -hex 32)" fak:0.34.0 serve --addr 0.0.0.0:8080 --base-url http://host.docker.internal:11434/v1 --model qwen2.5:1.5b
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

**Operator verb map:** when the troubleshooting path points at the wider CLI, use
`fak usage` for gateway/provider usage ledgers, `fak issue` for issue queue inspection,
`fak workflow-audit` for workflow-policy drift, `fak fleetcap` for fleet capacity,
`fak frontierswe` for FrontierSWE cache/score witnesses, `fak fused` for fused-turn
diagnostics, and `fak ablate-arm` when you need an explicit ablation arm in a comparison.

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
CGO_ENABLED=1 go test -run 'MatMul|Reset' ./internal/metalgemm/ -count=1 -v   # (Apple Silicon only; default build links Metal when cgo is enabled)
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
| **FAK 617** — Loops All the Way Down: The Durable Verified Loop, Loop Health, and Session Net-True | **FAK 614**, **FAK 616** |
| **FAK 618** — Navigating the Shipped Surface: Verb, Command, or Internal Leaf? | **FAK 209**, **FAK 617** |
| **FAK 619** — From Objective to Runtime Evidence and Retained Learning | **FAK 614**, **FAK 618** |

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
- Run the read-only issue-gardening pass, distinguish mechanical actions from review-only priority/area/ownership decisions, and name the current top backlog rot from the report
- Explain why a resolved issue whose commit omits #N can never be witnessed-closed
- Explain how the loop guarantees the live-worker population can never exceed its cap

**Read:** [`docs/dispatch-loop.md`](docs/dispatch-loop.md), [`.claude/skills/issue-triage/SKILL.md`](.claude/skills/issue-triage/SKILL.md), [`docs/SKILL-CONTEXT-MEMORY.md`](docs/SKILL-CONTEXT-MEMORY.md)

**Lab:**
```bash
python tools/issue_triage.py --markdown --out docs/_audits/issue-triage-YYYY-MM-DD.md
python tools/issue_triage.py --actions --out docs/_audits/issue-actions-YYYY-MM-DD.json
python tools/dispatch_status.py
```

**Checkpoint:** From the issue-triage report, name the largest current backlog gap
and the top three review-only P0/P1 rows. Then explain why a resolved issue whose
commit omits #N can never be witnessed-closed, how the loop guarantees the
live-worker population can never exceed its cap, and why an identical skill
invocation can be served as procedural-memory HIT rather than re-rendered.

### FAK 617 — Loops All the Way Down: The Durable Verified Loop, Loop Health, and Session Net-True

**Prerequisites:** **FAK 614**, **FAK 616**

**You'll be able to:**
- Place every fak mechanism on the five-ring loop ladder (tool-call → turn → session → fleet → RSI) and name the witness primitive each ring carries, plus the five orthogonal threads (trust, cost, memory, observability, governance)
- Distinguish the durable loop ledger (`fak loop run -- CMD`, which records a hash-chained `HeadBefore..HeadAfter` witness) and the verified driver (`fak loop drive`) from the hand-fed one-shot `rsicycle`, and say what a `dark-loop` state means in `fak loop health`
- Read a session's net-true verdict (HELPED / WASH / HURT) and explain why cost data alone (tokens, dollars) cannot grade whether a session *achieved* anything

**Read:** [`docs/explainers/engineering-is-building-loops.md`](docs/explainers/engineering-is-building-loops.md), [`docs/rsi-loop.md`](docs/rsi-loop.md), [`docs/fak/session-observability-rsi-loop.md`](docs/fak/session-observability-rsi-loop.md)

**Lab:**
```bash
go test ./internal/loopmgr/ ./internal/rsiloop/ ./internal/sessionobs/ -count=1 -timeout 120s
```

**Checkpoint:** Draw the five-ring ladder and name the witness primitive each ring carries (the adjudicator's provable refusal, ctxmmu's Clear+rescreen, recall's sealed page, the fleet's per-SHA `dos commit-audit`, the RSI keep-bit). Then explain why a `fak loop drive` turn that the model calls "done" still re-arms unless a dos witness agrees, and why a session that burned 200 turns and hit a STOP must grade HURT, not WASH, even though both spent tokens.

### FAK 618 — Navigating the Shipped Surface: Verb, Command, or Internal Leaf?

**Prerequisites:** **FAK 209**, **FAK 617**

**You'll be able to:**
- Start with a supported `fak <verb>` when operating the product, use a standalone
  `cmd/<name>` only for a bounded fixture, publisher, or compatibility lab, and open an
  `internal/<leaf>` when changing the pure contract behind either entry point
- Follow a request from the appendix's user-facing verb to its implementation leaf, then
  identify the witness surface that prevents the command's output from becoming a claim
- Distinguish read-only reports and dry-run plans from commands that launch, enroll,
  publish, reap, or otherwise mutate state before choosing a first probe

**Lab:**
```bash
go run ./cmd/fak help architecture
go run ./cmd/fak index query "architecture report" --json
go test ./internal/archreport -count=1
```

**Expected result:** Help supplies the operator contract, the index locates the owning
surface, and the package test exercises the contract without making an external change.
Use the three maps in the appendix to repeat that trace for an operations, model, or
curriculum task.

**Checkpoint:** For a new model-performance observation, explain why `fak model-observe`
is the normal operator door, `cmd/modelperfobs` is the bounded capture/report utility,
and `internal/modelperfobs` is the reusable measurement contract. Then name which layer
you would test after changing only the observation fold, and which command is safe to run
first when an entry offers both a report and a mutating action.

### FAK 619 — From Objective to Runtime Evidence and Retained Learning

**Prerequisites:** **FAK 614**, **FAK 618**

**You'll be able to:**

- Turn an objective into a bounded plan, preserve external-study provenance, and compile
  findings into explicit transfer candidates without confusing any of those steps with
  execution
- Inspect what the current binary can run before loading a model, then distinguish that
  preflight from actually starting the unified runtime
- Trace native-performance facts from bounded metrics and artifacts through SLO coverage
  into the scorecard that decides what the next improvement cycle must learn

The work-and-learning route has three distinct boundaries. `fak agentic` calls
`internal/agentic` to compile broad text into a deterministic, read-only, offline
expand → experiment → contract plan; its `fak ultracode` handoff is data, not a worker
launch. `fak study` stores and retrieves immutable content-addressed research receipts.
`fak learning-mesh compile --file LEDGER.json` then calls `internal/learningmesh` to
compare provenance-bearing mechanisms across declared hardware, framework, engine, and
baseline envelopes. A `COPY`, `ADAPT`, `BENCHMARK_ONLY`, `REJECT`, or `UNKNOWN` candidate
is still a candidate, never permission to change the product execution path.

Two discovery leaves support that route without becoming hidden authorities:
`internal/docsearch` loads the curated documentation map used by repository discovery,
while `internal/openviking` is an optional, bounded HTTP adapter for OpenViking search,
message, and commit operations. The local documentation map remains usable offline, and
an OpenViking result still needs the same study provenance and witness as any other
external observation.

The runtime route starts with `fak runtime-capabilities`, whose `internal/runtimecap`
fold separates “the binary runs,” “the governed control plane runs,” and “this requested
model backend is runnable.” Exact `--backend` requests fail closed; only an explicit
`--prefer-backend ... --fallback-policy local_cpu_degraded --cpu-envelope ...` posture
may select the portable CPU fallback, and remote placement must pass every declared gate
before payload load. `fak up` is the short product door for `fak serve`: it starts the
same gateway, flags, policy, metrics, and session lifecycle rather than a second server.
Behind optional runtime operations, `internal/dockerprocess` bounds Docker Compose calls
for the rich dashboard, and `internal/harnessserve` owns one adapter-provided loopback
model process with readiness, one-token probe, ownership, and bounded shutdown receipts.
Update automation stays separate: `internal/selfupdate` classifies `current`, `stale`,
`divergent`, or audit-only `attention` and supplies an explicit next command; it does not
silently replace a running runtime.

The native-performance route is another evidence pipeline, not one giant metric bag:

| Stage | Owning contract and relationship |
| --- | --- |
| Define and collect | `internal/nativeperfobscontract` freezes the Qwen3.8, `fak-native` signal set and its cardinality/freshness rules; `internal/nativeperfbackend` defines bounded per-backend Prometheus snapshots where unavailable values remain absent rather than zero. |
| Correlate and expose evidence | `internal/nativeperfcorrelation` replaces high-cardinality run, request, trace, and receipt IDs with a scrubbed bounded key; `internal/nativeperfartifact` maps that key to at most five public-safe, expiring receipt/profile/trace/report links. |
| Decide operational health | `internal/nativeperfslo` compares only matched `module@rev` + benchmark + Qwen3.8 model + backend envelopes and preserves `missing_evidence`; `internal/nativeperfcoverage` proves dashboards, contracts, PromQL, fixtures, and live receipts agree. |
| Select and learn | `internal/sweepcert` validates extrema, thresholds, censored edges, constraints, and point provenance across a declared sweep. `fak performance-rsi-scorecard` feeds versioned evidence to `internal/perfrsiscore`, which scores the complete improvement loop, names debt and the dominant bottleneck, and can compare a prior report; it does not claim raw model speed. |

**Lab:**

```bash
go run ./cmd/fak agentic --json --objective "turn one measured performance finding into a bounded learning cycle"
go run ./cmd/fak study search --limit 5 --store /tmp/fak-learning-study "native performance"
go run ./cmd/fak learning-mesh compile --file docs/_witnesses/issue-9839/mechanisms.json > /tmp/fak-learning-candidates.json
cmp /tmp/fak-learning-candidates.json docs/_witnesses/issue-9839/candidates.json
go run ./cmd/fak runtime-capabilities
go run ./cmd/fak up --help
go run ./cmd/fak performance-rsi-scorecard --input internal/perfrsiscore/testdata/complete.json --json > /tmp/fak-learning-performance-rsi.json
go test ./internal/agentic ./internal/dockerprocess ./internal/docsearch ./internal/harnessserve ./internal/learningmesh ./internal/nativeperfartifact ./internal/nativeperfbackend ./internal/nativeperfcorrelation ./internal/nativeperfcoverage ./internal/nativeperfobscontract ./internal/nativeperfslo ./internal/openviking ./internal/perfrsiscore ./internal/runtimecap ./internal/selfupdate ./internal/sweepcert -count=1
```

**Expected result:** The agentic plan reports `read_only=true` and `offline=true`; an
empty study store returns `[]`; the learning-mesh output matches its captured candidate
witness byte for byte; runtime capabilities emits `fak-runtime-capabilities/1` without
loading a payload; `fak up --help` prints the shared serve contract without starting a
listener; and the scorecard plus focused package tests complete locally. Put every
`fak study search` flag before the query because the query is positional.

**Checkpoint:** Given a new native benchmark finding, state where you would (a) retain
its external provenance, (b) decide whether it transfers to another envelope, (c) prove
the dashboard does not turn missing evidence into zero, and (d) measure whether the
performance-learning loop improved. Then explain why neither `fak agentic` nor
`fak runtime-capabilities` launches the work it describes.

---

## You've finished the path

If you can pass the checkpoints through **FAK 619**, you can: stand up and harden the
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

## Appendix — the full shipped surface (operator, contributor, and package map)

**Who this is for.** The courses above teach the load-bearing ideas end to end, but the
binary ships far more than 100 courses can each drill into: a fleet-operations toolbox, a
set of self-measuring scorecards, model/hardware benchmarks, and a stack of internal
leaves you will meet the moment you open the tree. This appendix is the orientation index
for that surface — one honest line per verb, standalone binary, and internal package,
each anchored to the code that implements it. It is a reference, not a lesson: skim it to
know a tool *exists*, then reach for `--help` or the cited source when you need it.

**Try any verb.** Every verb is reachable through the single binary; append `-h`/`--help`
or the JSON flag most carry:

```bash
go run ./cmd/fak <verb> --help      # each verb prints its own usage/subcommands
go run ./cmd/fak demo               # if you want a zero-flag proof first, start here
```

Deploying onto rented GPU hardware rather than fronting a hosted API? The operator
quickstart is [`docs/fak/neo-cloud-deploy.md`](docs/fak/neo-cloud-deploy.md).

### A. Running the gateway and the shared-trunk fleet

| Verb | What it does | Try |
| --- | --- | --- |
| `fak service` | Install/run fak as a system service (systemd on Linux, launchd on macOS) via `internal/systemservice`. | `fak service -h` |
| `fak watchdog` | Operator surface over the default OS-scheduled watchdog monitors (resume, supervisor, dos-dispatch, stale-work garden). | `fak watchdog status` |
| `fak schedscan` | Observability window onto the fleet's Windows Scheduled Tasks (FleetResumeWatchdog / FakFleetJanitor) beyond "is it Ready?". | `fak schedscan` |
| `fak host-crash` | Samples host crash / soft-fault signals over an interval and writes them for the crash-profile ledger. | `fak host-crash -h` |
| `fak host-relaunch-broker` | Drains the machine-control-plane→desktop relaunch request spool and launches Windows Terminal; `--validate` prints the argv without launching. | `fak host-relaunch-broker --validate` |
| `fak conpty` | Searchable status surface for the pwsh `0xE9` ConPTY FailFast crash class: resolves the ConPTY pair on PATH and compares its FileVersion to a known-good floor. | `fak conpty --json` |
| `fak stallscan` | Reads the churn signals (soft-fault storm, scheduler/syscall thrash, spawn bursts) that reveal a low-usage machine stall. | `fak stallscan` |
| `fak growthgate` | The standing-footprint twin of `stallscan`: classifies where the disk/IO went and emits a census + verdict. | `fak growthgate --json` |
| `fak git-maint` | Guarded, lock-aware "consolidate-never-prune" object-DB maintenance (multi-pack-index / commit-graph) the destructive-git guard otherwise only defers. | `fak git-maint -h` |
| `fak git-daily` | The scheduled daily tick that makes `git-maint` actually fold: reaps the orphaned locks the maintenance tiers correctly defer on, THEN consolidates; once-a-day deduped, so a coarse OS trigger is safe. | `fak git-daily --dry-run` |
| `fak clean-bins` | Safe, idempotent, witnessed prune of the stray `go build` binaries dropped at the module root. | `fak clean-bins` |
| `fak buildcheck` | Concurrency-safe compile check for a fleet editing one shared trunk; never drops a binary in the tree. | `fak buildcheck` |
| `fak ci-preflight` | Answers "is the committed trunk tip CI-buildable and gofmt-clean, and if not which files" without trusting the working tree. | `fak ci-preflight -h` |
| `fak go` | A poison-free `go` passthrough for the permanently peer-dirty shared trunk (builds in an isolated temp copy). | `fak go build ./...` |
| `fak validate` | Answers whether the committed tree builds, a distinct question from a live working-tree build. | `fak validate -h` |
| `fak trunk-build-probe` | Read-only diagnosis of whether the release gate's red trunk is a forgotten `git add`. | `fak trunk-build-probe -h` |
| `fak trunk-red` | Folds the pre-existing trunk-red witness ledger into the distinct shared breaks the build gate admitted over (a peer's red, not yours), worst-first. | `fak trunk-red -h` |
| `fak worktree` | `fak worktree worker prepare\|land\|reap` — the detached per-worker build-isolation worktree that lands its diff back on `main` under a lane lease. | `fak worktree worker -h` |

### B. Session, budget, and spend accounting

| Verb | What it does | Try |
| --- | --- | --- |
| `fak budget` | Inline per-task budget readout ("spent N, budget M, here's where it went") from real gateway usage records. | `fak budget --json` |
| `fak spend` | Cross-account spend rollup; gate-fails on unlabeled spend figures. | `fak spend -h` |
| `fak savings` | The Track-2 observed-$ cache-savings audit trio over the local savings ledger. | `fak savings audit` |
| `fak balance` | Night-balance readout: resume recovery-vs-stranding and gardening-vs-throughput side by side; exits non-zero on an underwater recovery budget. | `fak balance -h` |
| `fak footprint` | Always-sent MCP tool-schema floor scorecard: prices fak's registered `tools/list` floor offline. | `fak footprint -h` |
| `fak assume` | Thin shell over the `internal/assumecheck` kernel: gathers evidence for a launch-seat assumption, prints the verdict, maps it to an exit code. | `fak assume -h` |
| `fak sessionjournal` | Crash-survivable session-registration journal (open/beat/close JSONL) and its monitoring/resume sidecar. | `fak sessionjournal -h` |
| `fak wip` | Working-tree checkpoint/restore spine: a gc-safe snapshot of uncommitted tracked changes under `refs/fak/wip/<session>`. | `fak wip -h` |
| `fak idempotency` | Retry-safe executor for non-idempotent tool ops (create issue, push, ledger append) keyed by op + token. | `fak idempotency -h` |
| `fak goal-park` | Status/claim a durable goal under a supervisor lease (`internal/goalpark`). | `fak goal-park status -h` |
| `fak tasks` | Thin shell over `internal/taskgraph`: the shared task list as a typed table with lease-gated claims. | `fak tasks -h` |
| `fak multisubmit` | Multi-submission planner: once an issue is resolved once, plans cheap differentiated bonus takes on the same issue. | `fak multisubmit -h` |
| `fak cachesweep` | Sweeps the cache-savings knee: finds the threshold as a fraction of the infinite-cache ceiling and emits the sweep result. | `fak cachesweep --json` |

### C. Guard, audit, and safety surfaces

| Verb | What it does | Try |
| --- | --- | --- |
| `fak guard-audit` | Prunes repo-local guard audit journals only after a durable logvault mirror proves their bytes (`internal/guardaudit`). | `fak guard-audit prune -h` |
| `fak guard-commit-gate` | Claude Code PreToolUse boundary that binds a git-commit's stamp/paths (hook actuator). | `fak guard-commit-gate -h` |
| `fak guard-sessionstart` | Claude Code SessionStart hook actuator installed by `fak guard`. | `fak guard-sessionstart -h` |
| `fak guard-stops` | Folds the typed Stop-hook decision ledger into a tally (clean stops, bounded stand-downs, fail-open stops). | `fak guard-stops -h` |
| `fak guard-stops-slack` | Durable update-in-place Slack scoreboard feeder for the Stop decision ledger. | `fak guard-stops-slack -h` |
| `fak chatops` | The inbound READ-ONLY control door: drives fak from one Slack channel through a closed verb grammar and a fail-closed admin allowlist. It answers the read verbs (`help`, `ping`, `status`, `fleet`) inline and DECLINES the act verbs (`dispatch`, `resume`, `bench`, `halt`) — the mutating path detaches through guarded dispatch, so the door itself can never start or stop work. Channel text is only ever parsed as a closed verb, never executed. | `fak chatops --dry-run` |
| `fak knownbad` | Impure shell over `internal/knownbad`: record/match/claim/resolve blast-radius "known-bad tree" containment entries. | `fak knownbad report` |
| `fak blast` | `fak blast estimate` — blast-radius estimate for a change (the containment epic's planner). | `fak blast estimate -h` |
| `fak headless-lint` | Scans an agent's final-output text for operator-directed "pesky notes" ("do you want me to push?") — the sensor-side dual of choice-triage. | `fak headless-lint -h` |
| `fak conformance` | Standalone safety-conformance suite: re-adjudicates the shipped dogfood verdict matrix against the compiled kernel — a CI-gateable, third-party-runnable attestation. | `fak conformance` |
| `fak egresslist` | Maintenance surface for the bundled egress filter lists the adjudicator's egress rung compiles. | `fak egresslist refresh --dry-run` |
| `fak eve` | Eve integration bridge: mechanical security preflight over Eve's MCP/OpenAPI connections. | `fak eve -h` |

### D. Self-measuring scorecards, RSI loops, and contributor tooling

| Verb | What it does | Try |
| --- | --- | --- |
| `fak score` | Parent verb grouping the meta-scorecards / RSI loops; `fak score <name>` routes to each legacy `*-scorecard`/`*-score` handler. | `fak score -h` |
| `fak quality` | The missing-middle quality ladder: run one case through a reference path and an engine path, compare, score against a rubric, emit a replayable failure bundle. | `fak quality -h` |
| `fak mlp-score` | Witnessed grade of the "first lovable cut" for the all-in-one agent-runtime epic. | `fak mlp-score -h` |
| `fak antipattern-scorecard` | The unifying work-loss card: folds REDUNDANT_REWORK + UNWIRED_PKG + ORPHAN_FUNC into one `antipattern_debt`. | `fak antipattern-scorecard` |
| `fak checkpoint-scorecard` | Which long-running subsystems persist durable resumable WIP and expose a witnessed status surface. | `fak checkpoint-scorecard` |
| `fak checkpoint-debt-dispatch` | Files one deduped issue per un-checkpointed subsystem, capped. | `fak checkpoint-debt-dispatch -h` |
| `fak unwired-scorecard` | Which code-complete internal packages are wired into a default path vs orphaned (imported by no `.go`). | `fak unwired-scorecard` |
| `fak unwired-debt-dispatch` | Files one deduped issue per unwired package, capped. | `fak unwired-debt-dispatch -h` |
| `fak qa-process-debt-dispatch` | The E-testing-quality "is our test process honest?" fan-out (regression_catch). | `fak qa-process-debt-dispatch -h` |
| `fak mode-debt-dispatch` | Permission-regime fan-out: one deduped issue per un-lifted permission dial. | `fak mode-debt-dispatch -h` |
| `fak harness-debt-dispatch` | Routes the model-strength classifier's already-produced verdict into the backlog: keeps the REDUNDANT/HOBBLING harness scaffolds and plans at most `--cap` deletion issues. Dry-run mutates nothing; `--live` is required to file. | `fak harness-debt-dispatch -h` |
| `fak concept` | The conceptbench disambiguation surface: `fak concept position\|classify\|validate\|freshness\|admission` over the concept catalog. | `fak concept -h` |
| `fak negate` | The negation operator: `fak negate detect\|resolve\|reframe` over `internal/negframe`. | `fak negate detect -h` |
| `fak signals` | Plain-English behavioral signals (NL prompt + verdict schema + sample rate) judged over an agent's turns. | `fak signals -h` |
| `fak question-ledger` | Deterministic labeling authority for `docs/questions/asked.jsonl` that `/question-loop` defers to. | `fak question-ledger -h` |
| `fak refactor-verify` | Read-only proof that a god-split / code-motion refactor dropped no top-level declaration. | `fak refactor-verify -h` |
| `fak godsplit-plan` | Read-only, doc-comment-aware boundary + hazard planner for a behavior-preserving Go split (consumed by `/modularize`). | `fak godsplit-plan --json <file>` |
| `fak hwgate-lint` | Lints that no local-hardware blocker is declared as terminal, keeping the fleet hardware-portable. | `fak hwgate-lint` |
| `fak tier-calibrate` | Offline calibration fold over recorded tier decisions and witnessed outcomes; proposes threshold moves, mutates nothing live. | `fak tier-calibrate -h` |
| `fak dispatch-aging` | Read-only anti-starvation diagnostic: which ready issues are starving and in what pick order. | `fak dispatch-aging` |
| `fak dispatchlat` | Read-only dispatch-latency diagnostic. | `fak dispatchlat --json` |
| `fak execution-route` | Composes harness/model/session routing into one inspectable execution decision (`internal/executionroute`). | `fak execution-route -h` |
| `fak steer` | `fak steer prs` — folds the pending dev→release delta into operator-legible PR-sized units, worst-attention-first. | `fak steer prs -h` |
| `fak project` | The ProjectsV2 board control-pane fold: makes the board an operator-visible report/verdict/Slack-ready dimension. | `fak project -h` |
| `fak trajctl` | Declare/scope your own trajectory-corpus views (scope confinement for `trajquery`). | `fak trajctl declare -h` |
| `fak trajquery` | Query your own trajectory corpus with a small scoped SQL `SELECT`; the validator refuses any scope escape. | `fak trajquery -h` |
| `fak shadowgit` | Non-invasive per-step write ledger over a worktree via a separate git dir; attributes each step's changed files. | `fak shadowgit -h` |
| `fak wiki` | The fak-native, witness-verified repo-wiki core (structure + content). | `fak wiki -h` |
| `fak skill` | Queried skill loader: `fak skill query\|residency\|swap` over `.claude/skills` (+ the MCP resolver). | `fak skill query -h` |
| `fak demo` | The zero-flag 60-second offline proof: fak's scenario through the real kernel, one live verdict per call class. | `fak demo` |
| `fak init` | Emit a minimal, valid `fak.toml` deployment manifest. | `fak init` |
| `fak micro` | The native in-process Go microagent runtime front door. | `fak micro -h` |
| `fak doomloop` | Two-axis doom-loop guard: classifies live workers on effort vs verified progress and wires the reversible-first correction. | `fak doomloop -h` |
| `fak llms-full` | Generate the full `llms-full.txt` corpus digest from the git tree. | `fak llms-full -h` |

### E. Model and hardware benchmarks

| Verb | What it does | Try |
| --- | --- | --- |
| `fak macbench` | Apple-silicon decode-longgen + prefill-sweep benchmark driver. | `fak macbench -h` |
| `fak macfit` | Models Apple unified-memory capacity for many concurrent agents (`internal/macfit`). | `fak macfit -h` |
| `fak mac` | Crisp handle for the mac path (long form `fak claude-mac-fak`); both spellings route to one handler. | `fak mac -h` |
| `fak deepseekbench` | DeepSeek V4 Pro/Flash TTFT/TPOT/context-scaling scorecard (thin wire over `internal/deepseekbench`). | `fak deepseekbench -h` |
| `fak glm52-prefill-sweep` | GLM-5.2 pure-fak prefill-latency sweep driver; `--dry-run` prints the plan, `--endpoint` runs it live. | `fak glm52-prefill-sweep --dry-run` |
| `fak kvbm` | Replays a KV-block-manager trace and proves the #2666 validation shape; exit 1 unless proven. | `fak kvbm replay` |
| `fak bench-ingest` | Folds checked-in benchmark snapshot fixtures (Terminal-Bench, SWE-bench, FrontierSWE) into a provenanced `modelscore` registry, refusing any unprovenanced row. | `fak bench-ingest -h` |
| `fak microbench` | Turns "ultra-light memory/CPU, thousands of agents in one process" into a measured number with zero provider spend: boots the real `internal/microagent` host at N agents, reports RSS/agent against a guarded-CLI process-pair baseline, and appends the row as JSONL. | `fak microbench -h` |

### F. Standalone binaries under `cmd/`

Not everything is a `fak` sub-verb; a handful of engines and fixtures ship as their own
binary so they compose without importing the whole kernel.

| Binary | What it does |
| --- | --- |
| `cmd/agentdojoprobe` | Scores attacker-PROPOSED prompt injections through the real red-team stack: attack specs in on stdin, per-attack outcomes plus detection-only vs full-stack (IFC) ASR out. Exit 0 only if the full stack held on every in-scope attack. |
| `cmd/auditreceipt` | Verifies a model-route audit-receipt ledger (`fak-auditreceipt-verification/v1`) against its HEAD-bound cursor. |
| `cmd/codesearch` | Standalone front door to the code-intelligence engine: trigram regex/literal search, AST shape queries, call-graph traversal, and RRF-fused feature retrieval. |
| `cmd/crossauditcalibrate` | Calibrates a cross-audit run manifest against per-arm observation files. |
| `cmd/crossauditdogfood` | Folds cross-audit receipt envelopes (author/auditor/verdict) for dogfooding the cross-model audit. |
| `cmd/crossauditfixture` | Test fixture for the cross-audit ABI: PASS/FAIL exact contract comparison of a candidate against a base64 contract. |
| `cmd/customlintfixture` | Hostile-behavior test fixture for the custom-linter ABI. |
| `cmd/devfresh` | Scans docs for unpointed version claims (`docfreshrsi.ScanVersionClaims`); `--selfcheck` runs the built-in witness. |
| `cmd/ggufmeta` | Prints a GGUF checkpoint's metadata header — the CLI front door to `internal/ggufload`'s full-metadata export. `-json` emits the lossless, byte-identical snapshot for diffing two checkpoints; the default is sorted `key=value` lines, narrowable with `-grep`. |
| `cmd/negframescan` | Throwaway harness (not shipped) that exercises `internal/negframe` against real repo prose when `cmd/fak` is wedged. |
| `cmd/wdcheck` | Throwaway verifier for the `info_watchdog.go` pure mappers. |
| `cmd/wgscan` | Scans a tree via `internal/windowgate` for un-suppressed window-popping exec calls. |
| `cmd/zaitask` | Bounded non-streaming Z.AI task runner CLI (emits content + a usage receipt). |

### G. Internal package map — the leaves you will meet in the tree

Each leaf is a tiered, single-purpose fold (see **FAK 209** for the tier DAG). The map is
grouped by the job the leaf does, because that is how you will look one up — you will
arrive knowing "something supervises the host" or "something prices the cache", not the
package name. One honest line each.

#### G.1 — Fleet, host, and service supervision

| Package | Purpose |
| --- | --- |
| `internal/fleetspine` | Networking-aware self-discovery for the fleet-control pane: UDP-multicast heartbeats let machines find each other live on the LAN instead of only through the git-mediated machine-dir snapshot. |
| `internal/fleetbottleneck` | Ranks which class is the fleet's binding constraint right now — seats, throttle, resume backlog, host load, or auth — from one fleet snapshot. |
| `internal/fleetreap` | Bounded retention and footprint measurement for per-session fleet artifacts; never follows directories or removes a file outside the caller's explicit set. |
| `internal/fleetmemory` | The cross-agent lessons ledger and its write-time duplicate guard: a workaround one agent learns is published once with a trigger context and injected to any peer whose session matches it. |
| `internal/fleetverify` | A throwaway compile-verification of the `operator.go` fleet helpers, isolated from the churning `cmd/fak` so the loopfleet/loopmgr API usage type-checks with no duplicate-symbol risk. |
| `internal/ghexec` | Builds deadlined `gh` invocations: every construction carries a context deadline, disables interactive prompting and the update notifier, and suppresses the Windows console window — so a wedged `gh` cannot wedge its caller. |
| `internal/hostfault` | Names the HOST-level failure classes that destabilize the fleet without being a child tool process crashing: a failed Windows Update install, an orchestrator fault, a GPU live-kernel/TDR watchdog event, an app-termination dump. |
| `internal/hostplacement` | Deterministic multi-host worker placement: a per-host headroom heartbeat registry plus a pure function that picks the least-saturated, non-stale host to spill a dispatch worker onto, or stays local when none is eligible. |
| `internal/hostresurrect` | Turns a host-fault crash class plus the guard-session index into a bounded, rate-limited relaunch request, so a crashed box brings its sessions back without a relaunch storm. |
| `internal/linkstate` | The general "what state is my channel with a peer in right now?" record — a three-phase CLEAR / WORKING / WAITING model replacing a five-state vocabulary that left agents unsure which not-ready they were in. |
| `internal/loaddebounce` | Publishes a per-worker load signal only when it CHANGES, and coalesces a burst behind a short reset-on-every-change window so only the latest value survives to be emitted. |
| `internal/promalert` | Parses an Alertmanager webhook payload and renders it into compact Slack text — the inbound half of the Prometheus-alerts-to-Slack wiring that `fak slack alert` enqueues into the durable outbox. |
| `internal/scmbridge` | One desired-state / read-back contract shared by the Windows SCM control plane and the Scheduled-Task recovery bridge, so machine services, interactive agents, boot recovery, and crash recovery cannot diverge or double-launch. |
| `internal/servicespec` | The portable `fak.service.v1` desired-state and restart-semantics contract every service surface renders from: identity, command, cwd, environment references, readiness, restart/backoff, checkpoint/resume, dependencies, intentional stop. |
| `internal/servicelease` | The lease / generation / incarnation fencing layer over that contract: the durable facts that let a control plane decide whether a disconnected remote node still owns a leased workload, so a partition retry cannot create a second owner. |
| `internal/serviceledger` | The portable append-only observed-state event ledger for services: Windows crash rows, SCM state changes, systemd journal entries, launchd exits, and fak's own supervisor receipts folded into one correlated schema. |
| `internal/stallpage` | Turns stallscan's reboot high-water decision into a durable, deduped operator page. A reboot drops every live session, so the choice explicitly names operator approval; the publisher never kills a process or reboots the host. |
| `internal/terminalrisk` | Reports whether this box carries the Windows Terminal Direct2D render-crash risk (an AMD adapter plus a prior render crash) and can write the settings key that avoids it. |

#### G.2 — Loops, sessions, and worker isolation

| Package | Purpose |
| --- | --- |
| `internal/chatopsdetach` | Pure detached-execution decision kernel for chatops ACT verbs: ack-now / witnessed-completion / stall-escalation routing. |
| `internal/doomloop` | Two-axis doom-loop classifier: an effort-vs-verified-progress window → a closed verdict + reversible-first correction. |
| `internal/dormancysim` | The deterministic time-travel harness for the dormancy organs: one injectable clock threaded through the pure measure and rehydrate leaves, so a test fast-forwards an agent's dormancy from hours to months with zero real waits. |
| `internal/executionroute` | Composes harness, model, and session routing into one inspectable execution decision. |
| `internal/guardrotate` | The pure decision core for `fak guard`'s cooldown-aware seat selection: without it a bare `fak guard` launches against an account the launcher just watched bounce off its own cap. |
| `internal/guardsessions` | The local, queryable index of `fak guard` sessions — one appended row per launch (handle, trace id, wrapped agent, pid, cwd, journal path, start time), and prefix resolution to the one session it names. |
| `internal/lookahead` | The witness-gated Lesson core: a fork-rollout's outcome distilled into a lesson whose assertive authority is bounded by the witness rung its evidence earned — a self-report rollout may assert nothing at all. |
| `internal/looporphan` | The pure duplicate-loop-supervisor reaper: folds a process census of loop and drainer supervisors into a closed keep/reap plan that never strands live work. |
| `internal/loopunblock` | The generic head-of-line unblocker for any worklist-draining loop — the always-on rung that keeps a loop which has already selected its next move from stalling on a move it cannot make. |
| `internal/microagent` | Hosts many agent loops in one process: a worker pool sharing one in-process kernel gateway. |
| `internal/resumebackoff` | The pure resume-storm containment fold. |
| `internal/resumemetrics` | The process-global expvar surface for the resume/heal watchdog, so a renamed or rotated status ledger can no longer read as "zero activity" and be mistaken for a healthy-but-idle watchdog. |
| `internal/seatpark` | A bounded park-and-retry fold for the no-seat transient: consecutive parks plus a clock become SEAT_READY or SEAT_PARKED, instead of bursting preflight probes at a wall only a peer finishing can move. |
| `internal/sessionctl` | The control-op vocabulary spine of the out-of-band operator control plane: every shipped write op (pause, resume, cancel, throttle, budget, pace, priority, steer, redirect) named once with what "applied" means and how it refuses. |
| `internal/sessionread` | The read/query/observe-op vocabulary spine — the outbound twin of `sessionctl`: what each shipped read DISCLOSES, under whose right, on what evidence, and how it refuses when the read is illegal. |
| `internal/sessionledger` | The durable per-trace hash chain the gateway and the RSI loop append to at every turn boundary, as append-only JSONL — one bounded line per append instead of re-marshalling whole-state JSON on every write. |
| `internal/sessionreplay` | Freezes one turn's regime-conditioned harness decision into a checked-in, deterministically replayable regression fixture, so a mode/regime bug is reproducible with its ambient state pinned, not just its transcript. |
| `internal/sessionsearch` | Witnessed cross-session recall over the guard session journal: full-text recall whose relevance is MEASURED, so an index that quietly starts returning irrelevant hits is caught instead of trusted. |
| `internal/sessionsteer` | The steering and admission half of the zero-knob automatic-context doctrine: a content-free snapshot of a long session's context-value advice becomes one typed directive the guard hooks turn into a start-time rule and a stop-time persist decision. |
| `internal/supervisoragent` | The closed, payload-free INPUT CONTRACT a supervisor agent consumes. The meta-loop actor may CONSUME a witnessed signal and pick a move; it may never MANUFACTURE a health signal by reading transcripts. |
| `internal/workerworktree` | The per-worker git worktree isolation primitive the live dispatch spawn wires in, so N concurrent workers stop sharing one working tree, one index, and one build cache on the trunk. |
| `internal/worktreewitness` | Runs a command inside a transient detached worktree pinned at `origin/main`, so the verdict reflects the trunk tip and not a peer's dirty working tree. |

#### G.3 — Trunk hygiene, work-in-progress, and shared plumbing

| Package | Purpose |
| --- | --- |
| `internal/buildwitness` | A structural CI guard that fails when `cmd/fak` does not compile with the DEFAULT build tags — the recurring break where an uncommitted or tagless sibling file makes the package green only on its author's disk. |
| `internal/godfileceiling` | Ratcheting god-file LOC ceiling gate — the merge-time "no" that stops god-files from growing. |
| `internal/godsplitplan` | The doc-comment-aware boundary and hazard planner for a behavior-preserving Go split; the error-prone part is cutting at the right line, because a declaration's doc comment sits above its keyword. |
| `internal/refactorverify` | Proves a god-split or code-motion refactor dropped NO top-level definition — the question `go build` cannot answer, since it only catches a REFERENCED symbol that went missing. |
| `internal/jsonlledger` | The shared JSONL-ledger row helpers the report packages each used to copy-paste: `Parse` scans a ledger into typed rows, `LatestBefore` finds the newest prior row. |
| `internal/patchcommit` | Commits one explicitly supplied unified patch through a temporary git index — the hunk-scoped sibling of safecommit's path commit. |
| `internal/privatepath` | Resolves private operator artifacts outside the public checkout. |
| `internal/refutil` | Foundation-level helpers for materializing ABI refs. |
| `internal/tokencache` | The persisted, content-addressed backing store for clonescan's per-file tokenization, so an unchanged file is a file read instead of a re-lex — which is what lets the push rung and the CI duplicate job scale. |
| `internal/trunkbuildprobe` | Diagnoses *why* the release gate's ci-fast subset is red. The commonest cause is a coherence break: a commit lands a caller whose definition lives only in an uncommitted sibling file, so the whole tree builds on the author's disk. |
| `internal/wipattr` | The pure attribution core: for every dirty working-tree hunk it decides whether a session OWNS it (that session's checkpoint records the identical edit) or it is an ORPHAN a peer's sweeping stage could silently destroy. |
| `internal/wipfence` | Applies and removes the shared-trunk WIP build fence — a `//go:build wip_<slug>` first line that keeps not-yet-compiling work on the trunk without reddening the default build. Pure text: no git, no filesystem. |
| `internal/wiprecon` | The pure reconciliation core for a crashed session's orphaned checkpoint: DISCARD_WITNESSED (the delta already landed in HEAD), RECLAIM (unlanded but applies cleanly), or QUARANTINE (unlanded and conflicting). |
| `internal/wipref` | The pure core of `fak wip`: the ref-name grammar, the stamp encode/decode, and the status fold for the checkpoint ledger under `refs/fak/wip/<session>`, with zero git I/O of its own. |
| `internal/tuiplugin` | In-process extension seam for fak console panes (Register-from-init). |
| `internal/versionskew` | Turns "I can't tell which fak is running" into a structured, refusable version-skew verdict. |
| `internal/wiki` | The fak-native, witness-verified repo-wiki core (structure L1 + content). |
| `internal/xprobe` | A throwaway `Ping()` symbol used as the end-to-end fallback probe for buildcheck. |

#### G.4 — Guard, policy, and safety

| Package | Purpose |
| --- | --- |
| `internal/blastradius` | The pure join blast-radius containment is built on: given a broken package, which live leases and queued issues intersect its transitive dependency radius — so a hold stops being all-or-nothing guesswork. |
| `internal/egresslist` | Adblock-style site allow/block layer above the hardwired cloud-metadata egress floor and below the WebFetch research allowlist. |
| `internal/egressrefresh` | Re-fetches the bundled egress filter lists from their provenance URLs and rewrites the checked-in artifact + pinned checksum. |
| `internal/ghspam` | Reusable GitHub-comment abuse match families — the release-archive download lure and the fake patch/fix lure. Each needs two independent signals, so a genuine outsider bug report that merely says "fix" does not match. |
| `internal/guardaccuracy` | Folds a labeled command corpus through the real guard reversibility classifier and scores the boundary itself: the false-positive rate (benign calls escalated) and the false-negative rate (dangerous calls let through). |
| `internal/guardaudit` | Bounds repo-local guard audit journals only after a verified logvault mirror proves durability. |
| `internal/guardcompile` | Compiles one authoring-time model extraction into a review-only policy patch (data, not runtime enforcement). |
| `internal/guardcorpus` | Folds a guarded session's decision journal into the durable, policy-attributable guard-session dataset: one record per session plus replayable, redacted example rows that survive the journal reaper. |
| `internal/guardvars` | The canonical wire shapes shared by the `/debug/vars` producer (`internal/gateway`) and the `fak info` consumer, defined once so a new producer field can no longer drop silently on the floor. |
| `internal/knownenv` | The fleet-wide known-ENVIRONMENT-failure registry: an error-text needle and/or exit code mapped to a not-your-fault verdict. It matches by tool OUTPUT where `knownbad` matches by TREE. |
| `internal/market` | Validates discoverable extension descriptors without executing extension code. |
| `internal/planresolve` | Oracle-driven plan-content adjudicator. |
| `internal/reachdelta` | Categorizes what a policy change actually made REACHABLE: a newly permitted tool or tool prefix, a newly permitted egress host, a widened default posture, a removed explicit deny, a lost self-modify protection. |
| `internal/rsl` | Git Reference State Log: a forge-independent, append-only, hash-chained record of trunk ref transitions — the offline no-force-push proof. |
| `internal/verifierexposure` | Ranks the gameability of fak's own verification gates. |

#### G.5 — Scorecards, debt dispatchers, and RSI folds

| Package | Purpose |
| --- | --- |
| `internal/agentreadinessscore` | Grades the one thing the sibling cards do not: can an autonomous coding agent DISCOVER fak, WANT to adopt it, and do so easily — scored over the git-tracked tree on twenty-three mechanical KPIs. |
| `internal/antipattern` | The unifying registry for the agentic-dev anti-patterns whose common shape is "work that did not convert into progress": work REDONE that was already done, and work LANDED but connected to nothing. |
| `internal/brittleness` | The detector-and-capture for seams that "got lucky": process, commit, and test outcomes that worked only by timing, chance, or a symptom patch that did not hold — and the regressions those seams throw. |
| `internal/catchupscore` | The catch-up lens: not how much happened over a window, but how far BEHIND the dev system is right now at each level — intake, measurement, and the rest — as a number a human glances at and a gate ratchets on. |
| `internal/checkpointscore` | The deterministic checkpoint-readiness card: does each long-running process persist durable resumable state AND expose a witnessed status surface a peer can read without tailing its logs? |
| `internal/closureaudit` | The pure grader half of the issue-closure audit: binds commits to issue numbers from commit text, then grades each issue into exactly one witness bucket from the per-SHA verdicts the caller supplies. |
| `internal/conceptcatalog` | The typed catalog behind `fak concept`: the concept families with their roots and exclusions, plus the authoring, freshness, and separation folds over the disambiguation data file. |
| `internal/ctxknobs` | The manual-overlay COUNTER for the zero-knob automatic-context doctrine: it enumerates every surviving knob whose only purpose is context management and ratchets the count, so a new one cannot land silently. |
| `internal/ctxplans` | The context-plan-required lint: every context-touching verb or skill declares, as a structured directive co-located with the code, what enters the window, what pages out, and what warms. |
| `internal/envconfiglint` | The config-not-env ratchet: behavioral configuration must stay out of the environment (which is for secrets), enforced as a machine-checked gate on environment READS rather than by a human reading the diff. |
| `internal/findingsink` | A general sink seam for scorecard findings: a producer folds its debt into neutral Findings without knowing whether the sink is a dry-run terminal, a durable local ledger, or GitHub issues. |
| `internal/focusscore` | Grades whether the fleet as a WHOLE is converging on its live goal or fanning out — many objectives active at once, detours run past budget while their parents sit paused. The aggregate the per-objective fold cannot see. |
| `internal/headlesslint` | The sensor-side dual of `choicetriage`: scans a worker's final output for operator-directed notes ("do you want me to push?") that page a person who, in a headless run, is not there to answer. |
| `internal/hwgatelint` | The sensor for the "local machine is the compute boundary" regression. A laptop with no GPU is the CONTROL point, not the boundary; the work dispatches to the machine that can witness it. |
| `internal/issuehygiene` | The pure KPI core behind `fak score issue-hygiene`: how well the default GitHub issue surface is created and tagged, folded into one issue-hygiene-debt integer. |
| `internal/knobcensus` | The knob census over the whole user-facing surface: each knob classified as intent the system cannot infer versus a dial that should be automatic, so the context ratchet and the control ratchet read one inventory. |
| `internal/mcpfootprint` | Prices the always-sent MCP tool-schema floor — the fixed per-turn token tax every registered tool adds to every call, whether or not it is ever selected. Deterministic and offline, so it can be ratcheted down. |
| `internal/modedebt` | The consumer half of the mode-debt pair: reads the permission-dial scorecard and maps every HARD un-lifted dial onto the existing backlog bridge, adding no new issue-filing code of its own. |
| `internal/mutationefficacy` | A bounded, SOFT mutation-testing probe: it applies standard operator mutants and counts SURVIVORS — the one question coverage cannot ask, would the suite actually FAIL if the code were wrong? |
| `internal/orphanscan` | A syntactic detector for the "built but never wired up" smell: an unexported top-level function defined but referenced nowhere in its own package — the shape work takes when it was authored and then dropped. |
| `internal/promptlint` | The durable freshness monitor for the dispatch worker-issue prompts, whose executable claims (which verbs to run, which refusal tokens gate the commit) silently rot when a verb is renamed or a token retired. |
| `internal/qaprocessscore` | The QA-process card's KPI folds — the "is our test process honest?" signals, each a pure function over git and history facts, so the fold is deterministic and fixture-testable with no toolchain in the loop. |
| `internal/scdiff` | The shared diff-scoping seam for shift-left scorecards: given the ref the caller based its work on, exactly which paths changed — so a card can skip whole corpora instead of rescanning the tree after every edit. |
| `internal/sensecheck` | The "does this actually make sense?" side-car — a common-sense smell battery over claims the hard rungs already passed: a cache hit rate above 100%, a "fixed" commit over a visible non-zero exit, a test asserting a tautology. |
| `internal/seoaeoscore` | The discoverability measuring stick: will a reader — or an answer engine — find fak at all, and cite it correctly? Folded into an seo-debt integer, orthogonal to whether the docs are correct once found. |
| `internal/skillvalue` | The per-skill outcome-VALUE ledger: attributes the witnessed outcome delta of sessions that LOADED a skill against matched sessions of the same task class that did not. Staleness is not value. |
| `internal/mlpscore` | Grades the "first lovable cut" contract from committed, machine-checkable witness manifests. |
| `internal/unwiredscore` | The unwired-code card for "code complete but not wired into the default path": a package that compiles, carries its own tests, reads clean — and that nothing ever imports. |

#### G.6 — Issues, projects, and operator reporting

| Package | Purpose |
| --- | --- |
| `internal/choicetriage` | Decenters the human from a surfaced "choice" — most surfaced choices are fake and resolvable without a person. |
| `internal/dispatchaging` | The deterministic anti-starvation term the dispatch order was missing: among READY work, which unit a worker picks FIRST when raw priority alone would let a low-priority unit wait forever. |
| `internal/dispatchcache` | A content-addressed, in-memory TTL cache with an injectable clock for the routed-backlog inputs that successive dispatch ticks share. |
| `internal/issuededup` | The write-time near-duplicate gate shared by every issue producer: simhash embed plus top-K over the title and title+body axes, catching the paraphrase a producer's own seen-cache cannot. Advisory, never blocking. |
| `internal/issuestriage` | Folds one surfaced issue action — close a dormant question, mark an issue stale, review an under-labeled issue — into its decenter-the-human disposition, in one tested place instead of inline at every pane. |
| `internal/milestoneburndown` | The milestone SCHEDULE dimension: due dates, open/closed counts, and trailing closure velocity folded into ON_TRACK / AT_RISK / OVERDUE / NO_DUE_DATE / DONE, with a projected drain date compared against the due date. |
| `internal/operatorquestion` | Harness-agnostic operator-question normalization. |
| `internal/operatorresolve` | Evidence-first operator-question resolver. |
| `internal/projectcompletion` | Folds the board's project-work readouts into completion buckets and points per standard, so "how much of this project is actually done" becomes a derived number instead of a status opinion. |
| `internal/projectreport` | Folds a GitHub ProjectsV2 board into the same control-pane envelope the milestone report uses, so the board becomes an operator-visible dimension instead of a write-only sync target. |
| `internal/questionledger` | The deterministic labeling authority for the question-loop ledger: unambiguous ids, a closed category vocabulary, a closed status lifecycle, and no leaked host, path, or email — rules prose in a skill file cannot enforce. |
| `internal/spendrollup` | Builds the cross-account `fak spend` rollup and gate-fails any figure that forgets its valuation basis or its WITNESSED/OBSERVED provenance label — the conflation card's discipline applied to money. |
| `internal/steerpr` | Folds continuous-merge trunk commits into operator-legible PR-sized units banded by where operator attention is owed. |
| `internal/worklog` | The unified agent-work change feed: the commit, the diff-witnessed verdict, the lease epoch, and a later verdict flip folded into ONE append-only, cursor-drained feed a consumer tails by offset. |

#### G.7 — Context, cache, and token economics

| Package | Purpose |
| --- | --- |
| `internal/cacheprice` | The one source of truth for the provider prompt-cache price multipliers — the cost of a cached-prefix read or write relative to a base input token — so no layer re-declares the literals and drifts. |
| `internal/cachevalue` | Folds the persisted cache-savings ledger into per-session cache-efficiency metrics (hit rate, write amplification) and flags the churny sessions whose prefix keeps mutating. |
| `internal/cvregress` | Per-session cache-efficiency (hit% + write-amp) regression flagging over the cache-savings ledger. |
| `internal/computeadmit` | The one shared admission kernel over the compute partitioners — the compute-plane twin of the lane and region admitters, so a compute-region overflow refuses in the same closed vocabulary as a lane collision. |
| `internal/deadlineadmit` | A pure admission policy that orders pending work earliest-deadline-first and sheds degradation-eligible items it predicts will miss, so the survivors keep their SLO instead of everything missing together. |
| `internal/flowcredit` | The receiver-granted credit ledger for cross-node KV block transfer backpressure: a sender must reserve credit before transmitting, so a fast prefill node cannot overrun a slow decode node's KV-ingest buffers. |
| `internal/guideddecode` | A sound, byte-level compiler that constrains a model's decode to a valid tool-call envelope. This slice constrains the tool-NAME enum and the fixed skeleton around it; full argument-schema enforcement is a later slice. |
| `internal/kvbudget` | A pure, GPU-free calculator for the KV-cache VRAM budget of concurrent decode streams: the closed-form KV-bytes-per-token sizing math, with the hardware A/B left owed by a GPU node. |
| `internal/l3kv` | Durable L3 KV residency backend: `StageSpan`/`RestoreSpan` persist a demoted span by digest behind a durable manifest. |
| `internal/stepbaton` | The pre-resume step-advice stamp: the durable, cross-restart carrier of the managed-context decision captured while the trace is still live, so a resuming successor can read what the window pressure WAS. |
| `internal/stepbatoncapture` | The live-side producer of that stamp: it reads the gateway's managed-context report while the trace is alive and projects it into a durable stamp — kept out of `stepbaton` so that core never imports the gateway. |
| `internal/stripeload` | Fans a single logical read across N byte-identical mirrors of the same file, sized by relative bandwidth. It is a `ReaderAt`, so the model loader needs no changes to stripe across several NVMes or sources. |
| `internal/toon` | A general JSON/TOON codec whose correctness spine is a lossless, type-preserving round-trip. A uniform array of flat objects collapses to one header plus one line per row, so field names are not repeated per row. |
| `internal/turnkind` | Classifies the latest user turn from message STRUCTURE alone — which content-block types it carries, never their content — so a routine tool-result continuation is told apart from a fresh instruction. |

#### G.8 — Model, kernel, and hardware benchmarks

| Package | Purpose |
| --- | --- |
| `internal/benchauthority` | The typed, in-binary source of truth for the primary benchmark NUMBERS fak claims — the number, its baseline, its provenance, its retraction history — replacing a hand-typed table of run-on cells. |
| `internal/benchckpt` | The shared per-cell write-ahead checkpoint the compute-bench executors write through, so a crash at cell N does not discard the cells already measured. |
| `internal/deepseekv4kv` | A pure, weight-free block-accounting fixture for the DeepSeek V4 heterogeneous KV plane and its prefix-reuse policies. It reasons in normalized units, so unpublished head dimensions are never fabricated. |
| `internal/deepseekv4moe` | A pure, weight-free synthetic model of V4's all-MoE dispatch that compares naive per-expert scheduling against grouped/fused scheduling in work-units, never in fabricated milliseconds. |
| `internal/deploymanifest` | Defines the unified `fak.toml` all-in-one deployment manifest and its fail-closed loader — one reviewable declarative artifact in place of flag-soup, a policy file, environment variables, and service registration. |
| `internal/dsparity` | The pure, offline parity-harness SPECIFICATION for future DeepSeek-V4 native kernels: what a batch-invariant, bit-reproducible decode would have to prove before any such kernel is trusted. |
| `internal/glm52prefillsweep` | The GLM-5.2 pure-fak prefill-latency sweep driver: a prefill-dominant request at each prompt length, landing time-to-first-token and prefill throughput per length as discoverable benchmark-ledger artifacts. |
| `internal/macfit` | Models Apple unified-memory capacity for many concurrent agents. |
| `internal/modelaccept` | Evaluates versioned exact-model capability corpora without letting missing evidence or averages authorize a tier. |
| `internal/modelops` | Folds exact-model canary observations into capability-safe promotion / rollback / hold decisions. |
| `internal/roofline` | Folds a lab drive's run artifacts into one current-vs-target-vs-ceiling dashboard, joining the ceiling note that carries the targets with the run artifacts that carry the measurements. |
| `internal/zaitask` | Bounded non-streaming Z.AI task runner. |

#### G.9 — Code intelligence and big-doc paging

| Package | Purpose |
| --- | --- |
| `internal/agentsindex` | A view over `AGENTS.md` that parses it into a deterministic section model, so a worker holds a compact resident table of contents and faults in only the section its task needs. It changes no content — only the load path. |
| `internal/astquery` | Structural (AST-shape) search over Go source with metavariables: a pattern is Go with `$NAME` holes. A repeated hole must bind consistently, so `$X == $X` matches `a == a` but not `a == b` — which no regex can express. |
| `internal/atif` | Projects the redacted trajectory corpus onto ATIF, the Agent Trajectory Interchange Format, so a fak session round-trips into a portable artifact a standard eval or fine-tuning pipeline can consume. |
| `internal/codegraph` | A directed code knowledge-graph with breadth-first traversal: forward reachability ("what does this end up calling") and reverse reachability ("what breaks if I change this"), each as a shortest deterministic path. |
| `internal/codexlifecycle` | Folds a native Codex rollout transcript into an exactly-once task lifecycle keyed by turn id, and decodes the structured tool envelope into a closed outcome vocabulary so expected negatives are not counted as failures. |

#### G.10 — Trajectory, tool, and partner-runtime analytics

| Package | Purpose |
| --- | --- |
| `internal/evebridge` | The connection-security preflight for the fak/Eve bridge: a mechanical fold over Eve's connection manifest (auth posture, tool filters, approval policies, scopes) into typed pass/fail diagnostics, so a scoping mistake fails closed. |
| `internal/eveimport` | The read-only importer that folds saved Eve observability artifacts into fak's session-ledger row shape, consuming only framework-owned workflow tags — never free-form assistant text. |
| `internal/eveparity` | Runs a fixture Eve-shaped eval suite raw and fak-routed and proves the two arms agree — in particular that fak never silently downgrades a hard Eve gate FAILURE into a soft observation. |
| `internal/toolrollup` | Folds a corpus of individual tool-call records into a per-tool-TYPE rollup: call count, token totals and means, wall duration, error count and rate. |
| `internal/toolseq` | Turns ordered per-session tool-call sequences into a tool-transition graph and its most common contiguous variants. Transitions never span a session boundary, so unrelated sessions cannot manufacture an adjacency. |
| `internal/toolshape` | Fingerprints the SHAPE of one tool call's input and output — the redaction-safe structural record the analytics chain consumes, where the trajectory turn itself carries only opaque content digests. |
| `internal/tooltrend` | Folds a sequence of per-session tool-call buckets into a tool-mix and output-shape TREND: which tools and which response shapes are rising or falling across N sessions. |
| `internal/trajctlhook` | The impure call-site assembly that binds the pure trajectory-control turn-boundary fold to a running session's host evidence — whether a claimed commit SHA still resolves, the analyzed audit rows, the wall-clock stamp. |

### H. FAK 618 field map — choosing the right shipped entry point

These are the surfaces added after the last learning snapshot. Read each table as a route,
not a vocabulary list: begin with the operator question, use the public verb when one
exists, drop to a standalone command for its bounded artifact workflow, and edit the
internal leaf only when the reusable contract itself must change.

#### H.1 — Agent, fleet, evidence, and model verbs

| Verb | Use it when you need to… |
| --- | --- |
| `fak agent-queue` | Reconcile an agent-pool snapshot into explicit start or hold actions. |
| `fak agents` | Query live and historical sessions with constrained SQL, grouping, counts, or JSON. |
| `fak architecture` | Inspect dependency tiers, violations, fan-out, depth, and blast radius before changing a leaf. |
| `fak armbench` | Run provenance-locked paired benchmark arms from one immutable manifest. |
| `fak borrow-provenance` | Pin an external source and later verify its bytes against the recorded digest. |
| `fak breath` | Check the counted, mechanically enforceable half of the one-breath documentation contract. |
| `fak capabilities` | Find token, turn, cache, routing, session-control, and supporting-floor outcomes by intent. |
| `fak codex-resume` | Resume Codex sessions under bounded deadlines and report their rollout outcomes. |
| `fak component` | Check component contracts and workload coverage from a declared root. |
| `fak compute-trace` | Turn compute events into a bounded trace for placement and performance diagnosis. |
| `fak config` | Locate configuration guidance and audit a deployed posture against it. |
| `fak disambiguation` | Query, audit, or regenerate the canonical concept-separation index. |
| `fak dormancy` | Classify dormant loop work from the loop ledger instead of guessing from age alone. |
| `fak enroll` | Pin, inspect, or revoke a host's opt-in organization trust anchor. |
| `fak fanout` | Fold nightrun fan-out reuse receipts into a trend report. |
| `fak gitd` | Serve provenance-bearing, content-addressed git reads through the resident repo broker. |
| `fak goal` | Manage canonical goal state, bindings, evidence, lifecycle transitions, and execution topology. |
| `fak guard-goal-question` | Enforce the active-goal boundary before a guarded agent asks an operator question. |
| `fak harness` | Compose, inspect, resolve, and verify reusable agent-harness stacks. |
| `fak hostdiag` | Correlate privacy-safe host resource symptoms with fak work before prescribing recovery. |
| `fak launch` | Reversibly install, enable, inspect, or remove provider routing through fak. |
| `fak learning-observation` | Trace an observation through candidate, witness, and verdict rather than calling it learned early. |
| `fak lifecycle` | Inspect or control phase-aware capability lifecycle state. |
| `fak m` | Use the short alias for `fak manage` when wrapping a harness interactively. |
| `fak manage` | Put a harness behind the managed-agent door so tool calls can be denied, repaired, or quarantined. |
| `fak model-default` | Read the default model identity together with its dated evidence fold. |
| `fak model-observe` | Proxy, summarize, or verify model-performance and cache-transition observations. |
| `fak native-benchmarks` | List required fak-native witnesses and fail when an obligation is still missing. |
| `fak native-first-lint` | Catch prose that treats missing local hardware as a terminal native-inference blocker. |
| `fak native-performance` | Query, compare, or choose the next profile in the committed native optimization graph. |
| `fak org` | See organization-policy posture and which control channel owns each capability. |
| `fak progress` | Detect fleet stalls by comparing a recent window with its declared baseline. |
| `fak provider-cost` | Import, report, and reconcile provider-cost ledgers against registered sessions. |
| `fak quantbench` | Run the quantization benchmark contract or emit its self-test matrix. |
| `fak quantwatch` | Collect bounded arXiv and GitHub quantization updates, offline or live. |
| `fak schedule-held` | Evaluate held hardware-job schedules and measure admission-policy overhead. |
| `fak scratch-janitor` | Preview or remove abandoned session scratch while respecting age and resume guards. |
| `fak search` | Search the tracked repository corpus with bounded results and machine-readable output. |
| `fak shellprov` | Capture shell-command provenance so later evidence names what actually ran. |
| `fak speed-ab` | Replay a captured speed A/B manifest and emit its benchmark witness. |
| `fak stale-work` | Discover and rank stale repository work against open issues within a bounded scan. |
| `fak temp-artifacts` | Preview or quarantine aged temporary artifacts, then recheck before permanent reap. |
| `fak terminal-relief` | Measure terminal-host pressure and relaunch only restorable dashboards when armed. |
| `fak test-quality` | Score defects in test source against a shrinks-only baseline and emit repair candidates. |
| `fak token-profile` | Price forecast or observed token classes into dollars and scheduler-weight units. |
| `fak tool-width` | Fold tool-width observations and ratchet the rate of batched turns. |
| `fak trajectory` | Audit Claude or Codex logs into scrubbed JSONL and Markdown summaries. |
| `fak turnavoid` | Replay whole-model-turn avoidance traces with strict input and net-true attribution. |
| `fak turntax` | Measure the extra recovery turns a baseline loop fires against fak's one-shot path. |
| `fak value-chain` | Hold a value-chain manifest against observed artifacts and their evidence. |
| `fak watchdog-audit-run` | Run the watchdog's own liveness/productivity audit rather than trusting scheduled-task status. |
| `fak work-delivery` | Track, transition, and diagnose a work unit across declared delivery stages. |
| `fak workpattern` | Mine recurring work shapes from source or recorded trajectories. |
| `fak worktype` | Attribute session token spend and witnessed outcomes to classified work types. |

Start read-only: `fak architecture`, `fak agents`, `fak capabilities`, and the report
subcommands expose state; `fak enroll`, `fak launch`, and reap modes cross a mutation
boundary and should follow their preview/status path first.

#### H.2 — Bounded standalone commands

| Command | Artifact or boundary it owns |
| --- | --- |
| `cmd/amoprofpub` | Converts an AMOProf directory or archive into Confluence storage XHTML plus an attachment manifest. |
| `cmd/caveman-pairwise-judge` | Judges paired caveman benchmark outputs against an immutable source manifest and versioned receipt protocol. |
| `cmd/fak-dev` | Hosts contributor-only, shared-trunk-safe diagnostics and project-maintenance helpers outside the product CLI. |
| `cmd/fak-dos` | Applies journaled DOS decision changes through the writable host adapter; inspect/list before add or remove. |
| `cmd/fak-project-assets` | Syncs generated project assets and checks parity with their canonical sources. |
| `cmd/framevisibility` | Probes whether harness frames remain observable across the adapter boundary. |
| `cmd/kvdepth` | Measures reusable prefix depth per observation instead of collapsing reuse into one hit-rate scalar. |
| `cmd/managedinventory` | Generates or checks the managed-agent portability inventory without reading live credentials. |
| `cmd/modelperfobs` | Captures OpenAI-compatible timings, renders reports, and verifies cache-state benchmark receipts. |
| `cmd/nvidia-nemo-api-demo-audit` | Audits the NVIDIA NeMo API demo inputs and outputs into a reproducible, scrubbed receipt. |
| `cmd/pagescheck` | Checks documentation source, freshness, discoverability, and built publication artifacts. |
| `cmd/portability-adapter-selfcheck` | Exercises adapter registration or prints a skeleton without launching a real workload. |
| `cmd/portability-lab` | Runs the clean-room portability acceptance lab and writes its authoritative JSON report. |
| `cmd/portabilitycontract` | Explains, identities, or validates a packaged portability contract and its round trip. |
| `cmd/qwen38campaign` | Runs the frozen Qwen3.8 soak or oracle evidence campaign from declared config and corpus. |
| `cmd/streamcapture` | Records a real provider stream, scrubs it, and verifies the fixture before hermetic replay. |
| `cmd/supportwitness` | Applies a support witness to a capability graph and writes the promoted graph separately. |
| `cmd/testenv` | Runs a test command after removing credential-shaped environment variables. |
| `cmd/uxjourneyproxy` | Scores a deterministic user-journey corpus for proxy-path cognitive load. |
| `cmd/verbsdoc` | Renders the source-derived verb and refusal reference so documentation follows the registry. |

These binaries are narrow by design. If the same job has a `fak` verb, teach and automate
the verb; invoke `go run ./cmd/<name>` when reproducing the specific artifact workflow.

#### H.3 — Internal leaves behind the new surface

| Package | Contract to understand before editing it |
| --- | --- |
| `internal/archreport` | Derives a queryable architecture report from the tier/source registry. |
| `internal/cloudroute` | Detects request-signed cloud model routes without confusing provider routing with model identity. |
| `internal/codexresume` | Drives bounded Codex headless resumes to rollout-witnessed terminal outcomes. |
| `internal/corelockgate` | Owns the hard-self core-lock question so one boundary decides whether it may be opened. |
| `internal/customizationindex` | Indexes supported customization seams so operators can discover the owning guide and contract. |
| `internal/depthadmit` | Folds witnessed plan depth and admits continuation only while declared closure can still be reached. |
| `internal/dispatchdoa` | Detects workers that died before completing useful dispatch work. |
| `internal/docrender` | Renders repository Markdown into print-ready HTML and related publication artifacts. |
| `internal/fleetbus` | Carries typed fleet-control messages rather than treating terminal text as a control protocol. |
| `internal/fp4runtime` | Negotiates versioned FP4 and microscaling artifacts at the runtime boundary. |
| `internal/generationctl` | Coordinates live generation epochs, steering directives, and their acknowledgements. |
| `internal/gitbroker` | Centralizes guarded git execution and the resident content-addressed read broker. |
| `internal/gitdaily` | Runs the deduped daily lock-reap and safe object-database consolidation fold. |
| `internal/harnessmodelset` | Declares strict, role-indexed model requirements for a harness. |
| `internal/harnessmodelsetconformance` | Captures the end-to-end witness that a resolved role/model set actually conforms. |
| `internal/harnessserver` | Binds a harness to an externally owned, already-ready inference server. |
| `internal/hostdiag` | Correlates Windows resource warnings with privacy-safe fak workload facts. |
| `internal/httptrust` | Resolves one declared corporate CA bundle source for outbound HTTP clients. |
| `internal/humanctl` | Indexes operator-requested outcomes and the control surfaces that can satisfy them. |
| `internal/kvquantmeta` | Defines provider-neutral KV-cache quantization descriptors and compatibility rules. |
| `internal/leasequeue` | Queues region-admission waiters with explicit ownership instead of busy retry. |
| `internal/lightgapport` | Audits claimed portability swap points against committed CI witnesses. |
| `internal/managedocs` | Ratchets the canonical managed-agent documentation against its shipped surfaces. |
| `internal/mixedprecision` | Defines deterministic layerwise mixed-precision policy and accounting contracts. |
| `internal/modelinventory` | Normalizes model artifacts and runtime observations into one inventory. |
| `internal/modelperfobs` | Measures OpenAI-compatible requests and folds cache-state observations into reports. |
| `internal/modelsetlock` | Persists canonical model-set selections with stable identity and locking. |
| `internal/modelsetreceipt` | Independently attests what harness model set was resolved and launched. |
| `internal/modelsetresolve` | Deterministically binds harness roles to compatible model candidates. |
| `internal/ociartifact` | Implements the activation-neutral OCI 1.1 collection profile. |
| `internal/portabilityswitch` | Coordinates context changes with lifecycle state during a portability switch. |
| `internal/projectassets` | Resolves canonical project assets and proves generated copies remain in parity. |
| `internal/quantdetect` | Performs bounded, weight-free detection of quantization formats. |
| `internal/quantpolicy` | Evaluates explicit constraints over a proposed quantization choice. |
| `internal/quantprov` | Carries neutral provenance for quantized artifacts and their transformations. |
| `internal/scratchmark` | Detects source files whose leading contract marks them as disposable scratch. |
| `internal/serveradapter` | Renders configuration for supported external inference servers and probes readiness. |
| `internal/serverlifecycle` | Owns one local server instance from configuration through readiness and teardown. |
| `internal/serverproduct` | Defines the secret-free boundary between a server product description and its adapter. |
| `internal/sessionintent` | Represents provider-neutral session-level operator intent. |
| `internal/skilleffectiveness` | Measures whether loading a skill improved witnessed outcomes for matched work. |
| `internal/skillfootprint` | Prices the resident skill-description context floor. |
| `internal/streamrules` | Matches incremental provider output against regex rules without waiting for stream completion. |
| `internal/studydrift` | Detects when a pinned external study no longer matches the source revision it analyzed. |
| `internal/studymonitor` | Folds study provenance and drift checks into an operator-readable monitoring verdict. |
| `internal/taskvc` | Binds enabled fleet Scheduled Tasks to versioned installers or scrubbed captures. |
| `internal/tempartifact` | Inventories and conservatively reaps direct fak temporary artifacts. |
| `internal/tokenprofile` | Classifies forecast and observed tokens by economic and scheduling duty. |
| `internal/toolcallcontrol` | Applies deterministic pre-execution checks to proposed tool calls. |
| `internal/toolcatalog` | Separates executable tool registration from the catalog metadata agents discover. |
| `internal/ultracodebench` | Evaluates paired single-agent and fleet coding runs with accepted-effect accounting. |
| `internal/ultracodenegcontrol` | Evaluates predeclared negative controls so fleet gains cannot be self-attributed. |
| `internal/valuechain` | Verifies that promised value-chain stages resolve to observed, evidence-bearing artifacts. |

The recurring pattern is shell → pure fold → witness: for example,
`fak model-observe` or `cmd/modelperfobs` gathers facts, `internal/modelperfobs` folds
them, and the receipt makes the result replayable. Keep those responsibilities separate
when adding a new surface.

**Checkpoint:** Name the verb you would reach for to (a) prove the committed trunk builds
without trusting your working tree, (b) see which ready issues are starving, and (c) get
an offline per-task spend readout mid-run. (Answers: `fak validate` / `fak ci-preflight`,
`fak dispatch-aging`, `fak budget`.)

## Current shipped-surface map (August 2026)

Use this compact map after the core course when a current issue or receipt names a newer leaf. Each entry names the implementation surface and its operator-facing purpose; follow the linked package or command help before changing behavior.

- Desktop trust helpers: `cmd/localappcert` provisions the local app certificate contract, while `cmd/localapphelper` hosts the narrowly scoped desktop helper.
- Coordination and recovery: run `fak coordinate --help` for the coordinator surface and `fak watchdog-audit-health --help` for the watchdog health audit.
- Fleet economics: `internal/microfleeteconomics` computes bounded micro-fleet cost and value evidence.
- Model identity: `internal/modeldescriptor` owns normalized model descriptors rather than scattering model-name parsing.
- Placement accounting: `internal/placementtax` measures the incremental cost of placing compute and state across boundaries.
- Study pipeline: `internal/studyadjacency`, `internal/studyclass`, `internal/studylink`, `internal/studyprio`, and `internal/studytickets` turn external-study evidence into classified, linked, prioritized tickets.
- Baseline evidence: `internal/systembaseline` captures the host baseline used to interpret benchmark receipts.

```bash
fak coordinate --help
fak watchdog-audit-health --help
```

Expected result: both commands print their current flags and exit successfully; use package tests for internal-only leaves.
