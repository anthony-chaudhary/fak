---
title: "Talk + blog series — 'Treat the model like an untrusted program'"
description: "A talk outline and a 4-post blog-series outline for fak's systems-security framing of agent safety: the model is an untrusted program, the tool call is a syscall. Each post maps to an existing sourced explainer so the writing reuses material the repo already defends, and every number traces to BENCHMARK-AUTHORITY.md or a named doc. Extends the organic-discovery kit rather than duplicating it."
---

# Talk + blog series — "Treat the model like an untrusted program"

The long-form spine behind the launch kit. The short assets (`show-hn.md`,
`reddit-localllama.md`, `x-thread.md`, …) are paste-ready drops for the channels
where a project gets *discovered*; this doc is the talk you could give and the
essay series you could write once someone is curious enough to read 20 minutes.
It reuses sourced material the repo already defends — it adds no new claims.

**The one tagline, verbatim (consistent with [`README.md`](../../README.md) §"The
one move" and [`docs/index.md`](../index.md); see
[#372](https://github.com/anthony-chaudhary/fak/issues/372) for the tagline-drift
this avoids):**

> Treat the model like an untrusted program, and the tool call like a syscall.

That line is the whole talk. Everything below is unpacking it, and the unpacking
is fenced.

## The governing rule (same as the rest of the kit)

Lead with the fence — it's the hook, not the caveat (see
[`positioning-brief.md`](positioning-brief.md) §"Cross-channel through-line").
A systems-security room is the most overclaim-hostile audience there is; the
credibility *is* the honesty ledger. Three fences must survive every slide:

- The result **detector is ~100% evadable by design** — it is *not* the floor.
  The floor is two structural gates: a capability lock (the destructive lever was
  never wired up) and result quarantine (poison bytes never reach context).
  ([`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md))
- The reuse win is **self-host only**, and a few-fold vs a *tuned* warm-cache
  stack (the eye-catching multiples are vs the naive re-send-everything loop).
  An app that merely calls a frontier API gets the safety floor and none of the
  reuse savings. ([`docs/explainers/one-binary-one-surface.md`](../explainers/one-binary-one-surface.md))
- "kernel" / "syscall" is a **one-line intuition pump**, not ring-style isolation
  the project claims to have. The gate runs *in-process on the same call path* as
  the tool call — no privilege ring, no process boundary the model physically
  can't cross. The shipped thing (an in-process default-deny capability check) is
  what's defensible; never let the metaphor do work the engineering doesn't.
  ([`positioning-brief.md`](positioning-brief.md) AVOID #3)

---

## Part 1 — The talk outline

**Working title:** *Treat the model like an untrusted program: agent security as
kernel design.*
**Length:** ~30 min + Q&A. **Audience:** systems / security engineers who grade
mechanisms, not marketing (the same audience as Lobsters, r/netsec, a USENIX
Security or SREOPS room).

The arc below is built so the *strongest, most falsifiable* claim lands as a cold
open, and every later claim re-cites a fence before the number. There is no slide
in the deck a hostile questioner can dismiss by pointing at an unstated caveat.

### Slide 0 — Cold open: one terminal frame (60 seconds)

Run the proof live, no key, no model, no GPU:

```
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json \
  --tool refund_payment --args "{}"   # → DENY (POLICY_BLOCK)
go run ./cmd/fak preflight …          --tool search_kb     --args "{}"   # → ALLOW
```

**Spoken:** "Same machine, two tools. One is refused *by structure* — not by a
detector, not by a model in the loop. The other is allowed. That's the whole
talk." Verified command ([`AGENTS.md`](../../AGENTS.md) "60-second proof");
`--explain` prints the 8-rung ladder ending `DENY DEFAULT_DENY <- winner`
([`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md) §"The adjudicator is a chain").

### Slide 1 — The reframe (the tagline, unpacked once)

The model is *ring-3 userspace*. Everything it wants to do to the outside world —
call a tool, admit a result into its memory, reuse a cached answer — becomes a
**syscall** that passes through a kernel it does not control, adjudicated from
evidence it did not author. An injected instruction can *ask* for a destructive
action; it cannot *grant* it, because the model's say-so is never the
authorization.

Source / framing: [`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md)
§"The flip: the tool call is a syscall", and the threat-model answer in
[`docs/FAQ.md`](../FAQ.md) ("Why does fak treat the language model as an untrusted
program?").

> **On-slide fence (read it aloud):** "I'm using 'syscall' as an intuition pump.
> This is *not* a hardware privilege ring — the gate runs in-process on the same
> call path. What's real is an in-process default-deny check; the metaphor is the
> handle, not the mechanism." (positioning-brief AVOID #3.)

### Slide 2 — Why recognizers lose (the setup for the flip)

Most agent safety is a *recognizer* bolted on the outside: a pre-tool hook, a
sidecar policy service, an LLM judge. Two weaknesses, both load-bearing:

1. **The model can argue its way past a recognizer.** fak's own audit of its
   built-in detector measured it as **≈100% evadable** by a determined attacker.
   (Sourced: [`README.md`](../../README.md); the detector is in
   [`CLAIMS.md`](../../CLAIMS.md).)
2. **When the outside thing breaks, the call usually proceeds anyway** (fail-open),
   because failing closed would wedge the agent on every transient hiccup.

Source: [`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md)
§"The thing most systems do: recognize, from outside".

### Slide 3 — Flip 1: the floor is a lever that was never wired up

Move the "may this run?" check onto the *same call path* as the tool call, in one
address space, **default-deny**. Three things a recognizer-from-outside cannot
have (each pinned by a named test):

- **Nothing to talk past** — the model never addresses the gate; its call is
  *subject to* it (`TestNoOsExecOnHotPath`).
- **Default closed by construction** — anything not allow-listed resolves to
  `DEFAULT_DENY`; an empty policy permits *nothing* (`TestFoldDefaultDenyEmptyPolicy`).
- **Structural, not heuristic** — whether an irreversible tool runs is decided by
  whether its name is on a reviewable list, not by whether a model *recognized*
  the request as dangerous.

**The honest scope (say it on this slide, before anyone Googles it):** the floor
bounds tool *names* structurally; it does **not**, on its own, bound the
*arguments* of an allow-listed tool (`Bash{rm -rf /}` is allowed if `Bash` is).
Argument-scoped *capabilities* (path/host/amount as first-class constraints) are
on the roadmap, not shipped. Source:
[`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md)
§"Honest scope".

**The conjunctive bar (the close):** an attacker must beat **two independent
gates** — slip past the evadable screener *and* find an irreversible lever that
was deliberately never wired up. A normal filter is one gate.

### Slide 4 — Flip 2: the cache as a governance object (the skeptic-proof slide)

This is the slide a hostile room can't dismiss, because it is *falsifiable in an
afternoon against a public HuggingFace oracle*.

Every production prefix cache (vLLM, SGLang, OpenAI, Anthropic) is **prefix-only**:
reuse is a run from token 0, and a change at position *N* invalidates everything
after *N*. fak owns its KV cache as a kernel object, so it can do one thing no
shipped engine does: **remove a poisoned tool-result span from the *middle* of a
kept sequence and leave the cache bit-for-bit identical to a run that never saw
it.**

**The witness (the money frame):** `evict-vs-never: max|Δ| = 0` — every logit
matches to the last bit, not just the argmax. The negative control
(`poison-vs-never: max|Δ| > 0`) is non-vacuous, so the zero is a real erasure.

Reproduce live: `go run ./cmd/deletioncert -selfcheck` (≈1 s, offline, exits
non-zero on any violation). Source:
[`docs/explainers/addressable-kv-cache.md`](../explainers/addressable-kv-cache.md)
§"The thing fak does that no shipped engine does", and the causal-invalidation
witness `cmd/causalbench -selfcheck`
([`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) "Causal
invalidation-on-external-write", commit `0fc39aa`).

> **On-slide fence (read it aloud):** "`max|Δ|=0` is proven on a *synthetic*
> model whose numerics are separately oracle-checked against HuggingFace. The live
> `fak agent` HTTP loop doesn't drive this in-kernel engine yet — today's live
> path quarantines at the byte layer; attention-state eviction is the proven next
> rung. The deletion certificate is self-signed v1 and `EvictedCount` is
> self-reported." (addressable-kv-cache §"Honest scope".)

### Slide 5 — The honest 0/29 posture (the credibility slide)

The disarming slide. A 29-claim prior-art audit scored **0/29 novel**: every
individual primitive — capability-based security, KV caching, RadixAttention
prefix sharing, ed25519 receipts — is established prior art. **The contribution
is the *assembly***: one in-process gate where the tool call is the checkpoint,
fused with an addressable cache, in one binary.

Source: [`README.md`](../../README.md) ("A 29-claim prior-art audit scored 0/29
novel"), [`CLAIMS.md`](../../CLAIMS.md). **Spoken payoff:** "We're not claiming
we invented any of this. We're claiming we put it together in a shape nobody
shipped, and we can prove the hard part to the last bit."

### Slide 6 — What's measured, and what isn't (the fences slide)

Lead every number with its fence. Nothing here is a simulated number cited as
measured.

| Claim | Number | Source | Fence (say it with the number) |
|---|---|---|---|
| WebVoyager per-turn prefill elimination | **8.8× (1 worker) → 9.7× (8 workers)** net (A/C) on the real 643-task set | [`docs/webbench-baselines.md`](../webbench-baselines.md) (modeled geometry) | vs the **naive** re-prefill loop, **modeled** (closed-form geometry, no wall-clock). It is prefill-*token* elimination, not a wall-clock speedup over a tuned stack. The win is the structural per-turn turn-tax (A/B = 8.8×, worker-independent); cross-worker reuse (B/C) is only 1.00×–1.10×. |
| 50-turn × 5-agent reuse (README headline) | **4.1× vs tuned** warm-cache · 60.3× vs naive | [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) row "README headline" (commit `2bbda6f`) | The 60.3× is vs naive; **lead 4.1×**. Arm A's ~19 h is **modeled** (validated within ~0.4%), not run live. Reuse is **self-host only**. |
| RadixAttention reuse ladder | 4.58× → 6.95× live (135M → 1.5B); 7.50× token ceiling; **86.7%** hit rate | [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) "RadixAttention Model Ladder" (commit `92896a4`) | Hit rate is hardware-independent and inside SGLang's published 50–99% band; the comparison to SGLang is on **hit rate, not throughput**. |
| In-process adjudication cost | ~2.4 µs in-process vs ~6.9 ms spawned `fak hook` ≈ **2,800×** boundary tax | [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) "Pure-kernel latency" (commit `bcad56e`) | A **subsystem regression sentinel**, not a fleet-speed headline. Nobody runs a per-call process spawn in production, so the ratio is not a "fak is faster" claim — it's *why fail-closed is affordable*. |

> **The four numbers to never let on a slide unqualified:** the naive 8.8×–9.7×
> (always beside the tuned 4.1×), the ~60× / "agent city" projection (**DESIGN
> TARGET**, not measured), any power/energy/$ figure (**SIMULATED** — no power
> meter on the box), and "fak is faster" (it isn't —
> [`docs/explainers/one-binary-one-surface.md`](../explainers/one-binary-one-surface.md):
> "fak is not a faster token engine").

### Slide 7 — One binary is the whole surface (the deployment close)

fak fronts a real token engine (vLLM / SGLang / llama.cpp / a cloud provider); it
does not replace one. What it collapses into one ~13 MB static Go binary (zero
external deps, no `go.sum`) is the *other half of the stack*: the OpenAI /
Anthropic / MCP gateway, the capability floor, result quarantine, audit, and auth
— the parts a serving engine explicitly leaves to "the caller's responsibility."

The laptop story and the fleet story are the **same binary**: you add flags, you
don't graduate to a different system. Source:
[`docs/explainers/one-binary-one-surface.md`](../explainers/one-binary-one-surface.md).

> **On-slide fence:** "fak's own in-binary model is a correctness *reference*
> forward pass (proven bit-exact vs HuggingFace), not a production serving engine.
> For chat-quality serving, front vLLM / SGLang / llama.cpp / a cloud provider."

### Slide 8 — Close: lead with the fence

Restate the through-line: the project's entire credibility lives in its fences
and its zeros — `max|Δ|=0`, 0/29 novel, DENY-by-structure, detector-evadable-by-
design. The one move a hostile room can't counter is **make your own first
question the prosecution**: name the detector is evadable, the reuse is
self-host, the metaphor isn't ring isolation — *before* the room does. End on the
live `deletioncert -selfcheck` frame if the room is warm.

---

## Part 2 — The blog series (4 posts, each mapped to a sourced explainer)

The writing reuses material the repo already defends. Each post's source explainer
is linked so the author lifts sourced prose rather than re-deriving claims. No
post cites a simulated number as measured; the honesty ledger
([`CLAIMS.md`](../../CLAIMS.md)) is cited in every post.

**Through-line across the series:** the model is an untrusted program; every effect
it wants is a syscall through a kernel it doesn't control; the same boundary carries
security (which effects/results are allowed) and performance (do shared work once).
*(Used as a worked motif, not a headline — see positioning-brief AVOID #2: the
"same boundary" line survives as an essay arc, never as a launch tagline.)*

### Post 1 — "The lever was never wired up" (security floor)
**Maps to:** [`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md).
**Arc:** recognizer-from-outside loses (argue-past + fail-open) → the flip
(call-on-the-same-path, default-deny, structural not heuristic) → the two-gate
conjunctive bar → the honest scope (names bounded, arguments not yet).
**Carry the number:** the ~2,800× in-process-vs-spawned boundary tax, framed as
*why fail-closed is affordable*, not a speed brag.
**Close on:** the 60-second `preflight → DENY/ALLOW` proof, reproducible by the
reader.

### Post 2 — "Cut the poison out of the middle" (addressable KV cache)
**Maps to:** [`docs/explainers/addressable-kv-cache.md`](../explainers/addressable-kv-cache.md).
**Arc:** production reuse is *always* a prefix (the causal-attention mechanism) →
the precise, shipped claim is bit-exact **span removal** (`max|Δ|=0` vs
HF oracle) → why fak can when llama.cpp/vLLM/SGLang can't quite (kept pre-RoPE
`Kraw`, one clean re-rotation) → span removal as a *governance* primitive, not a
speed trick.
**Carry the number:** `max|Δ|=0` (evict-vs-never), with the non-vacuous
`max|Δ|>0` negative control.
**Fence hard:** proven on a synthetic model; live path quarantines at the byte
layer today; self-signed v1 cert / self-reported `EvictedCount`.

### Post 3 — "One binary, laptop to fleet" (operational surface)
**Maps to:** [`docs/explainers/one-binary-one-surface.md`](../explainers/one-binary-one-surface.md).
**Arc:** serving an agent safely is a *stack*, not a component (the engine gives
you one band; the rest you bolt on) → fak is that rest-of-stack collapsed into one
static Go binary → the laptop story and the fleet story are the same artifact.
**Carry the number:** ~13 MB binary, zero deps, no `go.sum`; the
vLLM/SGLang-vs-fak operational-surface table (lifted from the explainer).
**Fence hard:** not a faster token engine (vLLM/SGLang win tok/s); reuse is
self-host only; in-binary model is a correctness reference, not a production
server.

### Post 4 — "0/29 novel — and we lead with it" (the honesty ledger)
**Maps to:** [`CLAIMS.md`](../../CLAIMS.md) + [`README.md`](../../README.md)
(0/29-novel posture) and [`docs/explainers/fleet-benchmarks.md`](../explainers/fleet-benchmarks.md).
**Arc:** the contribution is the *assembly* (0/29 primitives novel) → every
capability carries exactly one machine-checked tag (`[SHIPPED]` /
`[SIMULATED]` / `[STUB]`) → the disciplined numbers (4.1× tuned headline,
WebVoyager 8.8×–9.7× modeled vs naive, RadixAttention 86.7% hit rate) → the design
targets honestly labeled (~60× / "agent city", simulated power/$).
**Carry the number:** 0/29 novel, and the SHIPPED-vs-SIMULATED discipline that
makes the measured numbers trustworthy.
**Close on:** the through-line — credibility *is* the fences; lead with them.

> **Series rule (from [`positioning-brief.md`](positioning-brief.md)):** never let
> the naive 8.8×–9.7× or the ~60× projection appear without its fence on the same
> line; lead **~1.5–4.1× vs tuned** wherever a multiplier appears; the
> "security-boundary == reuse-boundary" line is an essay arc, never a headline.

---

## Part 3 — Submission targets (where a systems-security-of-agents talk fits)

Each venue gets the angle that fits *its* audience; the security half is universal,
the perf half is self-host-niche and should only lead where noted. None of these
are AI-marketing venues — they grade mechanisms.

| Venue / format | Why it fits | The angle to lead with | fak half that plays |
|---|---|---|---|
| **USENIX Security** (talk / poster) | The reference venue for *boundaries*, not products. Graders live for "refused by structure, not by detection." | The two-gate reference-monitor argument + the fail-closed-default cost analysis (`policy-in-the-kernel` §"Why in-process is load-bearing"). Frame as a access-control / reference-monitor paper talk, not a product. | Security (universal). Drop perf entirely in the abstract. |
| **SREOPS / LISA / SREcon** (talk) | Operates agent *fleets*; feels the fail-open + multi-component pain personally. | "One binary is the governed-serving surface" (`one-binary-one-surface`): how many moving parts a safe agent fleet is, and who owns them. | Operational surface (universal) + self-host reuse as a cost lever. |
| **OSDI / SOSP** (talk, stretch) | Systems-design venue; the KV-as-kernel-object framing is genuinely a systems reframing. | Span-addressed, bit-exact cache mutation as a governance primitive (`addressable-kv-cache`); the kept-`Kraw` one-rotation exactness argument. | The KV mechanism + the assembly thesis (0/29 novel). |
| **DEF CON / Black Hat (briefings)** + **OffSecLive / sectorless CTF tracks** | The "lever was never wired up — refuses by structure" is a CTF-shaped claim; binary-exploitation audiences cover *boundaries*, not tutorials. | A "break my quarantine gate" framing: attacker controls the tool result, win = fire the destructive call. (See the LiveOverflow/Hammond/IppSec row in `positioning-brief.md`.) | Security (universal); the evadable-by-design admission is the credibility here. |
| **Simon Willison's "lethal trifecta" orbit** (guest essay / talk at an AI-eng meetup) | The dominant 2026 AI-security vocabulary; fak is the rare project that *closes* the trifecta by structure. | "Private data + untrusted content + exfiltration path = guaranteed exploit; two structural gates close it without detection." (See `landscape-research.md` "lethal trifecta" note.) | Security (universal). |
| **Papers We Love / reading-group meetups** (local chapters) | Idea-first, mechanism-curious; loves a falsifiable claim + a reproduce-in-an-afternoon proof. | The 0/29-novel honesty posture + the `max|Δ|=0` eviction reproduced live from a clean checkout. | The assembly thesis + the KV witness. |
| **Local Go / infrastructure meetups** (Go NYC, Berlin Go, etc.) | "Zero deps, no `go.sum`, one static binary, the policy is a file not a code edit" is a Go-engineering story. | The `Register*` extension seam (policy rungs as a chain, like LSM hooks) + the operational-surface contrast. | Operational surface + engineering craftsmanship. |
| **Dev.to / Hashnode / Lobsters (long-form essay)** | The durable, search-indexed surface; the asset that keeps delivering for months. | The Lobsters essay angle already drafted in [`lobsters-and-blog.md`](lobsters-and-blog.md) — *extend* it with the talk's KV and 0/29 sections rather than write fresh. | All four halves, fences intact. |

**The one rule across all of them (from [`landscape-research.md`](landscape-research.md)):**
disclose authorship in the first line, lead technical, and make your own first
comment/abstract the prosecution. The honesty ledger is the asset a competing
overclaim cannot copy.

---

## Provenance & fact-check (for the author; strip before pasting)

Every claim/number above traces to a committed source. No simulated number is
cited as measured; no projection appears without its label.

- **Tagline** ("Treat the model like an untrusted program, and the tool call like
  a syscall"): [`README.md`](../../README.md) §"The one move";
  [`docs/index.md`](../index.md). Consistent with
  [`positioning-brief.md`](positioning-brief.md) (no contradicting tagline —
  [#372](https://github.com/anthony-chaudhary/fak/issues/372)).
- **~100% evadable detector, two-gate floor, lever-never-wired-up:**
  [`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md);
  [`README.md`](../../README.md); [`CLAIMS.md`](../../CLAIMS.md).
- **`max|Δ|=0` mid-run eviction vs HF oracle:** synthetic-model witness;
  [`docs/explainers/addressable-kv-cache.md`](../explainers/addressable-kv-cache.md)
  §"The thing fak does that no shipped engine does"; reproduce `go run
  ./cmd/deletioncert -selfcheck`; causal witness
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) commit `0fc39aa`.
  Fences: self-signed v1 cert / self-reported `EvictedCount`; live path quarantines
  at byte layer; attention-state eviction = proven next rung.
- **0/29-novel audit:** [`README.md`](../../README.md) ("A 29-claim prior-art
  audit scored 0/29 novel"); [`CLAIMS.md`](../../CLAIMS.md).
- **WebVoyager 8.8×–9.7× (643 tasks, vs naive re-prefill):**
  [`docs/webbench-baselines.md`](../webbench-baselines.md) "REAL MEASUREMENTS".
  Net prefill-token elimination (A/C); the structural per-turn turn-tax (A/B) is
  8.8× worker-independent; cross-worker reuse (B/C) is 1.00×–1.10×.
- **50×5 reuse 4.1× vs tuned / 60.3× vs naive (arm A modeled):**
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) row "README headline"
  (commit `2bbda6f`).
- **RadixAttention 4.58×→6.95× / 7.50× token ceiling / 86.7% hit rate:**
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) "RadixAttention Model
  Ladder" (commit `92896a4`).
- **In-process ~2.4 µs vs ~6.9 ms (~2,800×) boundary tax:**
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) "Pure-kernel latency"
  (commit `bcad56e`); subsystem sentinel, not a fleet-speed headline.
- **~13 MB static binary / zero deps / no `go.sum`:**
  [`docs/explainers/one-binary-one-surface.md`](../explainers/one-binary-one-surface.md);
  [`go.mod`](../../go.mod).
- **Fences carried verbatim from the kit:** reuse self-host-only / read-heavy /
  ~1%-write-flips-negative; in-binary model = correctness reference, not
  production server; not a faster token engine; ~60× / "agent city" = DESIGN
  TARGET; power/energy/$ = SIMULATED; "kernel/syscall" = intuition pump, not
  ring isolation. Sources:
  [`positioning-brief.md`](positioning-brief.md),
  [`docs/launch/README.md`](README.md) §"The one rule that governs all of it".

*This doc ships words, not code. It extends the launch kit; it does not duplicate
it.*
