## CORRECTED ASSET

## r/programming

**Title:** I cut a poisoned tool result out of the middle of a kept model run's KV cache and left it bit-for-bit identical to a run that never saw it (max|Δ|=0)

**Body:**

Most KV-cache reuse only works from the *front* of the context — vLLM, SGLang, the hosted prompt caches. You can't reach into the middle of a kept run and surgically remove one span.

fak (one static Go binary) does addressable mid-run eviction: reach into the middle of a kept model run, cut one span (say, a poisoned tool result), and the cache is left bit-for-bit identical to a run that never saw it. Verified against a HuggingFace reference at max|Δ|=0 — not one number differs.

Honest fences, because they're the point:
- The deletion *certificate* is ed25519-signed but self-signed v1, and EvictedCount is self-reported. I say so in the repo.
- The "no shipped engine does this" claim is specifically about *mid-run* eviction. It is not a claim to be a faster token engine — vLLM/SGLang/llama.cpp win raw throughput and fak doesn't try to.
- Prior-art audit scored 0/29 novel: every primitive is established; the contribution is the assembly.

Proof runs offline, no key/model/GPU: `go run ./cmd/deletioncert -selfcheck`.

Repo (Apache-2.0): github.com/anthony-chaudhary/fak — disclosure, I wrote it.

*Sub rule note: r/programming bans pure self-promo and wants an article/discussion, not a "check out my project" drop. Lead with the mechanism, keep it technical, link the repo at the bottom as attribution rather than a CTA. Safest if it reads as "here's an interesting thing I built and how it works."*

---

## r/golang

**Title:** A pure-Go transformer verified per-layer against HuggingFace (cos=1.000000), CUDA decode at llama.cpp parity, shipped as one ~13MB static binary with no go.sum

**Body:**

Built an agent gateway in Go and the engine underneath is pure Go, not cgo-wrapping llama.cpp.

The Go bits that might interest this sub:
- In-kernel SmolLM2-135M forward pass, every layer proven against a HuggingFace oracle: per-layer cos=1.000000, final-logits max|Δ|≈4.4e-5, KV-decode token-for-token identical.
- CUDA decode hits parity with llama.cpp Q8_0 (~120 tok/s on an RTX 4070).
- Deterministic results reproduce byte-for-byte across 4 GPU backends (Metal/Vulkan/CUDA Ada/CUDA Ampere), 2 CPU ISAs, and 4 OSes.
- Ships as one ~13MB static binary. No go.sum, zero deps.

Honest framing: this is not a faster token engine — vLLM/SGLang/llama.cpp win raw throughput, and I don't pretend otherwise. The win here is operational surface (one binary vs a multi-GB Python/CUDA multi-process stack) and verifiability, not tok/s.

It's also a default-deny capability gate for agent tool calls, but that's a comments conversation — this post is the engineering.

Repo (Apache-2.0): github.com/anthony-chaudhary/fak — I wrote it.

*Sub rule note: r/golang tolerates show-and-tell when it's genuinely Go-engineering-forward and you disclose authorship. Keep the security thesis out of the title and body; let it come up in comments. The cos=1.000000 / no-go.sum angle is what earns the post.*

---

## r/selfhosted

**Title:** One static Go binary as an agent gateway in front of your own model — default-deny tool gate + full audit trail, no Python/CUDA stack

**Body:**

If you're self-hosting an LLM and pointing an agent (Claude Code, Cursor, any OpenAI client) at it, fak sits between the agent and its tools as a single ~13MB static binary.

What you get for one binary:
- A default-deny capability gate that runs in-process on the same call path as the tool call — no IPC, no sidecar, no second model. Irreversible actions are refused by structure (the lever was never wired up), not by a detector you have to trust.
- OpenAI-compatible *and* Anthropic-compatible gateway + MCP, so existing clients drop in with no agent-side changes.
- A signed deletion receipt / audit trail for context that gets evicted.

Operational contrast (not a benchmark): one binary vs a multi-GB Python/CUDA multi-process stack. fak is *not* a faster token engine — vLLM/SGLang/llama.cpp win raw throughput. There's also a context-reuse saving, but be honest: that's self-host + read-heavy fleets only; even ~1% write rate can flip it negative.

Apache-2.0, runs on Apple/AMD/Intel/Nvidia. Live in-browser demos: anthony-chaudhary.github.io/fak/demos.html

*Sub rule note: r/selfhosted gates project self-promo to the weekly "Self-Promotion / What are you working on" thread — drop the repo link there, not as a standalone post, unless this is framed as a genuine how-I-deploy-it writeup. Lead with deployment specifics, not numbers.*

---

## r/netsec

**Title:** Prompt-injection containment without trusting a detector: a two-gate model where the destructive tool call structurally can't fire

**Body:**

Disclosure: I wrote this. Hosted writeup + repo below.

Threat model: the attacker controls a tool result (a web page, a returned document, an MCP response) and wants to make an agent fire an irreversible action — `refund_payment`, a destructive API call. The usual answer is a prompt-injection *detector*, which is the wrong abstraction: detection is a probabilistic floor an attacker iterates against.

fak's containment is two independent structural gates instead:
1. **Capability lock** — the model literally can't call `refund_payment`; the lever was never wired up. Default-deny on the tool-call path, in-process.
2. **Result quarantine** — poison bytes from a tool result never reach the model's context in the first place.

An attacker has to beat *both*, and neither is a classifier you can talk past.

The honest part this sub will want: there *is* a detector in the box, and it's ~100% evadable **by design**. It is explicitly not the floor — the floor is the lock + quarantine. I'd rather state that than imply detection-as-defense. Prior-art audit: 0/29 primitives novel; the contribution is the assembly, where the tool call is the checkpoint.

Not claimed: this is an in-process default-deny check, *not* ring-style privilege isolation — the gate runs in the same process as the agent loop. Don't read "kernel" as a hardware boundary.

60s proof, no key/model/GPU: `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → DENY (POLICY_BLOCK); same command with `--tool search_kb` → ALLOW.

Repo (Apache-2.0): github.com/anthony-chaudhary/fak

*Sub rule note: r/netsec requires substantive technical content and an explicit disclosure on self-authored work — bare repo links get removed. Link a hosted writeup, state the threat model up front, and keep the evadable-detector honesty prominent; this audience rewards it and punishes overclaim hard.*

---

## r/LLMDevs

**Title:** Treat the model as untrusted and the tool call as a syscall: a default-deny gate where the agent structurally can't fire the destructive call

**Body:**

If you're building agents, the prompt-injection problem is usually fought with a detector. fak takes a different cut: put a default-deny capability gate in-process on the tool-call path, so the irreversible action (`refund_payment`) can't be invoked at all — the lever was never wired up. Two independent gates: capability lock + result quarantine (poison bytes never reach context). The included detector is ~100% evadable *by design* and explicitly not the floor.

It's one static Go binary, OpenAI- and Anthropic-compatible + MCP, so Claude Code / Cursor / any OpenAI client drops in with no agent-side changes.

There's also a perf side for self-hosted setups. The headline reuse number is **~4.1x less work vs a tuned warm-cache SOTA stack** (vLLM/SGLang prefix sharing) on a 50-turn × 5-agent reuse benchmark — the honest competitive number, not a naive-loop comparison. (A separate measurement on real WebVoyager, 643 tasks, shows 8.8x–9.7x less prefill vs a baseline that does no prefix reuse — different rig, so I don't fold the two together.) That reuse win is self-host + read-heavy only; ~1% writes can flip it negative. Not a faster token engine — throughput goes to vLLM/SGLang/llama.cpp.

60s proof, no key/model/GPU: `go run ./cmd/fak preflight ... --tool refund_payment` → DENY; `--tool search_kb` → ALLOW.

Repo (Apache-2.0): github.com/anthony-chaudhary/fak — I wrote it.

*Sub rule note: r/LLMDevs allows project shares if they teach something; lead with the architectural idea, not the pitch. Showing the tuned-baseline number as the headline (and keeping the naive WebVoyager number labeled as a separate rig) is what keeps this credible here.*

---

## r/AI_Agents

**Title:** The destructive tool call in your agent shouldn't be a detector problem — it should be a lever that was never wired up

**Body:**

Most agent "guardrails" are a prompt-injection detector watching the tool result. Attacker controls that tool result, iterates, eventually phrases the injection past the classifier, and `refund_payment` fires.

fak inverts it: treat the tool call like a syscall. A default-deny capability gate runs in-process on the same call path — no sidecar, no second model — and the destructive action simply isn't callable. Two independent gates an attacker has to beat: (1) capability lock (the lever doesn't exist), (2) result quarantine (poison bytes never reach the model's context). The detector in the box is ~100% evadable *by design* and is explicitly **not** the floor — the lock + quarantine are.

Drop-in: it's an OpenAI- and Anthropic-compatible gateway + MCP, so you put it in front of your existing Claude Code / Cursor agent with no code changes. One ~13MB static binary.

See it in 60 seconds, no key/model/GPU: `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → DENY (POLICY_BLOCK); same with `--tool search_kb` → ALLOW.

One honesty note since it'll come up: "kernel/syscall" is an intuition pump, not hardware isolation — the gate is in-process with the agent loop, not a privilege ring. Apache-2.0: github.com/anthony-chaudhary/fak (I wrote it).

*Sub rule note: r/AI_Agents is permissive about builder posts but allergic to vague hype — a runnable 60s DENY demo and the evadable-by-design admission do more than any pitch. Keep it concrete, disclose authorship.*

---


---

## Provenance & fact-check notes

> **Provenance & fact-check.** This asset was drafted from the repo's honesty ledger (`CLAIMS.md`) and then run through an adversarial fact-check against that ledger. The reviewer's notes are preserved below as an audit trail. Every CLI command was additionally re-verified against the live `fak` binary on 2026-06-22 and normalized to the one runnable form (`preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW`).

All numbers verified against the ledger and repo. The draft is unusually clean — it correctly fences every high-risk claim. But the adversarial pass found three real defects, two of them load-bearing for a skeptical audience.

I have everything needed. Here is my assessment.

---

## Problems found

**1. r/LLMDevs body — the WebVoyager number is mislabeled as cross-baseline-comparable.** The draft writes: *"on real WebVoyager (643 tasks) it does 8.8x–9.7x less prefill vs a naive re-prefill baseline … so the honest number is ~1.5–4.1x vs a tuned warm-cache stack."* This silently splices two different experiments. The 8.8x–9.7x figure (docs/webbench-real-measurements-summary.md, commit `d015ee9`) is the **WebVoyager** prefill-elimination number vs naive re-prefill. The **4.1x vs-tuned** figure (BENCHMARK-AUTHORITY.md, commit `2bbda6f`) is a **different rig entirely** — a 50-turn × 5-agent Qwen2.5-1.5B reuse benchmark, not WebVoyager. The ledger gives a vs-tuned number for that rig, not for WebVoyager. Presenting "~1.5–4.1x vs a tuned warm-cache stack" as the tuned counterpart *of the same 643-task WebVoyager run* is not traceable. Fix: keep WebVoyager's naive-baseline number labeled as such, and cite the 4.1x-vs-tuned as the separately-measured headline reuse win — or, more safely for a one-paragraph post, lead with the 4.1x-vs-tuned headline and drop the WebVoyager naive number rather than imply it converts.

**2. r/LLMDevs — "naive baseline is a strawman nobody runs."** The ledger's instruction is *"ALWAYS show the tuned-baseline number, NEVER lead with naive-only,"* and the fence calls naive the wrong thing to lead with — it does not authorize calling it a "strawman nobody runs." That editorializes past the ledger and actually undercuts credibility (plenty of naive agent loops do re-prefill). Soften to "a baseline that does no prefix reuse," which is what the ledger supports.

**3. r/golang title — "no go.sum" stated as a bare feature; fine, but "zero deps" needs the engine fence nearby, which it has.** This one is clean. No change.

**4. r/programming — `cmd/deletioncert -selfcheck` "~1s" timing.** The command exists and the flag is real (verified in cmd/deletioncert/main.go), but the "~1s" runtime is not in the ledger. The ledger's offline proof timings are "60-second proof" / "~1s" only for the `preflight` path, not deletioncert. Drop the unverified "~1s" and say "runs offline, no key/model/GPU" — the runtime claim isn't ledger-backed.

**5. r/programming and r/AI_Agents — "live model" / "live model run."** The KV-eviction proof is against a HuggingFace oracle on SmolLM2-135M (a reference forward pass), checked at max|Δ|=0. "Live model run" is fine as the ledger uses "kept model run," but "live" is a mild upsell that a skeptic could read as "production inference under load." Use the ledger's word "kept" run to stay exact.

Everything else traces: preflight `refund_payment → DENY (POLICY_BLOCK)` and `search_kb → ALLOW` (README.md:126-132, AGENTS.md:53-54), the two-gate model, the ~100%-evadable-by-design detector, 0/29 prior-art, cos=1.000000 / max|Δ|≈4.4e-5 (CLAIMS.md:73), ~120 tok/s RTX 4070 = llama.cpp Q8_0 (CLAIMS.md:114), ~13MB static binary, 4-platform determinism, self-signed-v1 / EvictedCount-self-reported fence, OpenAI+Anthropic+MCP gateway, self-host+read-heavy / ~1% write-flip fence, "not a faster token engine," the kernel≠hardware-isolation honesty note, the demos URL. The notebook, turntaxdemo, and example policy file all exist.

---

---

## Fact-check notes

What I changed:

1. **r/LLMDevs perf paragraph (the most dangerous defect): un-spliced two different benchmarks.** The draft presented "8.8x–9.7x vs naive … so the honest number is ~1.5–4.1x vs a tuned warm-cache stack" as one experiment with two baselines. They are two rigs: 8.8x–9.7x is **WebVoyager / 643 tasks** vs a no-reuse baseline (docs/webbench-real-measurements-summary.md, commit `d015ee9`); 4.1x-vs-tuned is a **50-turn × 5-agent Qwen2.5-1.5B reuse benchmark** (BENCHMARK-AUTHORITY.md, commit `2bbda6f`). The ledger has no vs-tuned number *for the WebVoyager run*. Rewrote to lead with the 4.1x-vs-tuned headline and present the WebVoyager naive number as a separately-labeled rig.

2. **Dropped "~1.5–4.1x" range in favor of the single conservative headline 4.1x.** The ledger says the conservative headline is 4.1x and to "always show the tuned-baseline number" — the loose "~1.5" lower bound added an unfenced figure not needed for the post; BENCHMARK-AUTHORITY.md states the marginal as 2.4–2.7x, not 1.5x, so "1.5" was the least-supported end of the range.

3. **Removed "a strawman nobody runs."** Not ledger-supported and corrosive to credibility; replaced with "a baseline that does no prefix reuse."

4. **r/programming: dropped the unverified "~1s" runtime on `deletioncert -selfcheck`.** The command and `-selfcheck` flag are real (cmd/deletioncert/main.go:38), but no ledger timing backs "~1s" for that path. Now "runs offline."

5. **"live model run" → "kept model run"** in r/programming title and body, matching the ledger's exact wording and avoiding a production-load implication.

6. **r/netsec, r/LLMDevs, r/AI_Agents: expanded the abbreviated `preflight ... refund_payment` into the real invocation** (`--policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"`) so a skeptic who copies it gets the documented DENY/ALLOW rather than a usage error. Verified against README.md:126-132 and AGENTS.md:53-54; the example policy file exists.

Every remaining number traces to the ledger/repo: max|Δ|=0 mid-run eviction, ed25519 self-signed-v1 / EvictedCount-self-reported fence, cos=1.000000 + max|Δ|≈4.4e-5 (CLAIMS.md:73), ~120 tok/s RTX 4070 = llama.cpp Q8_0 (CLAIMS.md:114), 4-platform byte-determinism, ~13MB static binary / no go.sum / zero deps, two-gate containment, ~100%-evadable-by-design detector, 0/29 prior-art, OpenAI+Anthropic+MCP gateway, self-host + read-heavy / ~1%-write-flip fence, "not a faster token engine," kernel≠hardware-isolation note, 4.1x-vs-tuned headline (BENCHMARK-AUTHORITY.md:24,205), 8.8x–9.7x WebVoyager / 643 tasks (docs/webbench-real-measurements-summary.md:26-29), demos URL, and the `deletioncert -selfcheck` / `agent --offline` / `preflight` commands (all present in cmd/ and the example policy file).
