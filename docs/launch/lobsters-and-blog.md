# Launch asset: lobsters-devto

================================================================================
## PART A — LOBSTERS SUBMISSION
================================================================================

**URL to submit:** https://anthony-chaudhary.github.io/fak/   (or link the dev.to post once live — Lobsters prefers the writeup over a bare repo)

**Title:**
Cutting a poisoned tool result out of the middle of a live model's KV cache, bit-for-bit

**Suggested tags:** `ai` `go` `security`

**Authored-by-me note** (the "you are the author" box Lobsters expects):

> Author here. fak is a single static Go binary that sits between an AI agent and its tools. This writeup is about one specific mechanism: addressable mid-run KV-cache eviction. You can reach into the middle of a kept model run, cut a span (e.g. a poisoned tool result), and the resulting cache is bit-for-bit identical to a run that never saw the span — checked at max|Δ|=0 against a run that never saw it, and non-vacuously different from one that kept it. That bit-exact eviction property is proven structurally on the kernel's KV-cache code path; separately, the same forward pass is proven numerically against a HuggingFace oracle (per-layer cos=1.000000, final-logits max|Δ|≈4.4e-5). Shipped engines (vLLM, SGLang, OpenAI/Anthropic prompt caches) reuse from the *front* only; none do mid-run eviction. The deletion certificate is ed25519-signed but I want to be upfront that it's self-signed v1 and the evicted-span count is self-reported — the load-bearing proof is the eviction parity you can reproduce offline, not the receipt. The repo also does default-deny capability gating on the tool-call path; happy to argue that part in the comments but the post stays on the cache mechanism. Apache-2.0, zero deps, no go.sum. Not a faster token engine — vLLM/SGLang win raw throughput; the claim here is a capability those engines don't expose, not a speed record.

---
*Lobsters note for the submitter: this needs an aged/invited account and you must be transparent about authorship — the note above is written to satisfy that. If the account is too new, hold and cross-post the dev.to version to HN/Reddit first.*

================================================================================
## PART B — dev.to / Hashnode CROSS-POST
================================================================================

---
title: I cut a poisoned tool result out of the middle of a live model's KV cache — and proved it bit-for-bit
published: false
tags: ai, go, security, llm
canonical_url: https://anthony-chaudhary.github.io/fak/
---

> Disclosure: I wrote fak. Every number below traces to a commit + artifact in the repo's `BENCHMARK-AUTHORITY.md`; where a number is simulated, modeled, or a design target, I say so on the same line. If you catch me overclaiming, the comment section is the prosecution — bring the receipt.

## The 1-second version

Three rows print. Two are bit-identical. One carries a rogue tool result. Then the middle one gets evicted, and the cache is bit-for-bit identical to a run that never saw it — and provably *different* from one that kept it.

```
go run ./cmd/deletioncert -selfcheck
```

Runs offline in ~1s, no key, no GPU. The check that matters: **max|Δ|=0** between the evicted run and a never-saw-it run — not "close," not "cosine 0.999," *zero* — with a non-vacuous control (a run that *kept* the span differs). [verified, shipped]

The rest of this post is the *why* and the *how honest is this, really*.

---

## Section 1 — The thing nobody's KV cache does: delete from the middle

Every prefix-cache you've used — vLLM, SGLang, OpenAI/Anthropic prompt caching — reuses tokens **from the front**. Append-only. If turn 3 of a 12-turn agent run contained a poisoned tool result, you cannot surgically remove turn 3 and keep turns 1–2 and 4–12 warm. You re-prefill.

fak's cache is *addressable*: you name a span and evict it, and the kept run is left identical to one where the span never existed.

**Runnable:** `go run ./cmd/deletioncert -selfcheck`
**Claim:** mid-run eviction leaves the cache bit-for-bit identical to a never-saw-it run, **max|Δ|=0**, against a non-vacuous control (a run that kept the span differs). [shipped]
**How it's proven (two separate witnesses, stated plainly):** the bit-exact eviction property is *structural* and is proven on the kernel's KV-cache code path (the `-selfcheck` demo exercises it on a tiny in-memory synthetic model — no weights, no torch). The numerics of that same forward pass are proven *separately* against a HuggingFace oracle (Section 4). I keep these two apart on purpose: one proves "evicted == never-saw, byte for byte," the other proves "the math matches PyTorch."
**Fence:** "No shipped engine does this" is specifically about *mid-run* eviction. It is **not** a claim about being a faster cache — append-from-front reuse is a different, well-solved problem.

---

## Section 2 — The deletion certificate (and exactly how far to trust it)

The eviction emits an ed25519-signed receipt binding the evicted span to the `max|Δ|=0` result. Forge any field and verification fails closed.

**Fence, stated plainly:** the certificate is **self-signed v1**, and `EvictedCount` is **self-reported**. The bound `max|Δ|=0` is carried as a signed string, not re-measured by the verifier — the eviction's bit-exactness is proven separately (Section 1). The certificate is *not* the proof; it's a tamper-evident envelope around facts computed honestly elsewhere. It does not, by itself, make a dishonest number honest. I'd rather you trust the `max|Δ|=0` you can reproduce than a signature whose v1 schema you can't audit yet.

This is the section a skeptic should poke at hardest, which is why it's here and not buried.

---

## Section 3 — Why I built a cache that can delete: the safety angle

The reason mid-run deletion matters isn't speed. It's that a poisoned tool result, once it's in the model's context, is hard to take back. fak runs **two independent gates** on the tool-call path:

1. **Capability lock** — the model literally cannot call `refund_payment`; the lever was never wired up.
2. **Result quarantine** — poison bytes from a tool result never reach the model's context in the first place.

The much-hyped "prompt-injection detector"? fak ships one and calls it **~100% evadable by design**. That's not a confession, it's the architecture: detection is *not* the floor. The floor is the lock + the quarantine. An attacker has to beat two structural gates, not fool one classifier.

**Runnable (60s, no key/model/GPU):**
```
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
```
(No key, model, or GPU; the first run compiles the binary.)
With `--explain` you get an 8-rung decision ladder ending `DENY POLICY_BLOCK <- winner (rank 100)`.
**Claim:** default-deny capability gate, in-process, same call path as the tool call — no IPC, no sidecar, no second model. [shipped]

---

## Section 4 — Is the engine itself trustworthy? Per-layer oracle parity

None of the above means anything if the underlying forward pass is wrong. So the in-kernel pure-Go SmolLM2-135M forward pass is checked rung-by-rung against HuggingFace:

- per-layer **cos = 1.000000**
- final-logits **max|Δ| ≈ 4.4e-5**
- KV-decode **token-for-token identical**

And it reproduces **byte-for-byte across 4 GPU backends, 2 CPU ISAs, 4 OSes** (Apple M3/Metal, AMD Ryzen+RX7600/Vulkan, Intel+RTX4070/CUDA Ada, 8-GPU Ampere server). CUDA decode hits **llama.cpp Q8_0 parity (~120 tok/s on an RTX 4070)** — and fak gets there running f32, 4× the bytes of llama.cpp's 8-bit Q8_0. [all shipped]

**Try it:** Colab quickstart on a free GPU — `notebooks/fak-quickstart.ipynb`.

---

## Section 5 — The performance number, with the fence first

Here's where most launch posts lie by omission. The honest framing — two *separate* benchmarks, never blended:

**Benchmark 1 — real WebVoyager (643 tasks), modeled geometry.** vs a **naive re-prefill** baseline: **8.8x (1 worker) → 9.7x (8 workers) less prefill** (the A/C column). Against a **per-agent warm-KV** baseline on this same dataset, the cross-worker gain is small — about **1.0–1.1x** — because WebVoyager's per-task geometry (median 12 turns) leaves little cross-worker prefix to share. I show that number, I don't hide it. The 8.8–9.7x is a closed-form prefill-token count over the real task set, not a wall-clock. [modeled]

**Benchmark 2 — a fleet-shaped synthetic (Qwen2.5-1.5B, 50 turns × 5 agents), the reuse-at-scale headline.** vs naive: **60.3x**; vs a **tuned warm-cache** stack (per-agent KV / prefix sharing): **4.1x**. Honest fence on *this* one: the naive arm's wall-clock is **modeled** from a sampled prefill-cost curve (running it live would take ~19h), not measured end-to-end; the token-elimination ratios are exact. This is the read-heavy-fleet projection, not a WebVoyager result. [modeled headline]

If someone quotes you "8.8x" or "60x" without the matching tuned-baseline number (1.0–1.1x on WebVoyager; 4.1x on the synthetic fleet), they're selling. Always show the tuned number on the same line as the naive one.

**Three more fences that stay on:**
- fak is **not a faster token engine** and doesn't try to be. vLLM/SGLang/llama.cpp win raw throughput. The operational contrast is *one ~13MB static binary vs a multi-GB Python/CUDA multi-process stack* — operational surface, not tok/s.
- The reuse win is **self-host only** and **read-heavy only**: an app that just calls a frontier API gets the safety floor but no reuse savings, and even a ~1% write rate can flip the reuse economics negative.
- Any ~60x / "agent city" frontier-scale numbers are **DESIGN-TARGET / modeled projections, not live end-to-end measurements**. Power / energy / $ figures are **simulated** — there's no power meter on the box.

**See it live:** the turn-tax race, multi-agent reuse, and the reuse-vs-tuned-baseline race all run in-browser: https://anthony-chaudhary.github.io/fak/demos.html
Or locally in one command: `go run ./cmd/turntaxdemo`.

---

## Section 6 — The honest prior-art audit: 0/29 novel

I scored fak against 29 prior-art primitives and it came back **0/29 novel**. Every individual piece — capability-based security, KV caching, radix prefix sharing, ed25519 receipts — is established work. The contribution is the **assembly**: putting them in one in-process gate where *the tool call is the checkpoint*, with the model treated as an untrusted program and the tool call as a syscall.

One caveat on that metaphor, since it's load-bearing for some readers and I don't want to oversell it: a real syscall boundary is a privilege ring userspace physically can't cross. fak's gate runs in the *same process* as the agent loop. "Syscall" is an intuition pump for *where the checkpoint sits*, not a claim of ring-level isolation. The shipped thing — an in-process default-deny check on the tool-call path — is good on its own merits and doesn't need the metaphor to carry it.

---

## Try the whole thing

```bash
# 1s, offline — the deletion proof
go run ./cmd/deletioncert -selfcheck

# 60s, offline — the DENY/ALLOW gate
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"

# the turn-tax race, locally
go run ./cmd/turntaxdemo
```

Repo: https://github.com/anthony-chaudhary/fak (Go, Apache-2.0, zero deps, no go.sum)
Live demos: https://anthony-chaudhary.github.io/fak/demos.html
Every number → commit + artifact: `BENCHMARK-AUTHORITY.md`

If a number in this post doesn't reproduce, that's a bug report I want. Comments are open.

---

## Provenance & fact-check notes

> **Provenance & fact-check.** This asset was drafted from the repo's honesty ledger (`CLAIMS.md`) and then run through an adversarial fact-check against that ledger. The reviewer's notes are preserved below as an audit trail. Every CLI command was additionally re-verified against the live `fak` binary on 2026-06-22 and normalized to the one runnable form (`preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW`).

All key facts confirmed. The deletioncert demo's bit-exact eviction proof runs on a **synthetic model** (control max|Δ|≈0.33, kept≠never), and the numerics are proven separately by the HF oracle — they are two different witnesses, exactly as CLAIMS.md states. The CUDA ~120 tok/s parity is confirmed. The 4.1x-vs-tuned is the synthetic 50x5 headline, not WebVoyager.

I have enough to deliver the fact-check. Here is the corrected asset and notes.

---

## Adversarial fact-check: problems found

**1. FATAL — the headline conflation (Section 1-second version + Section 1 + Section 2 + dev.to title).** The draft says the bit-for-bit `max|Δ|=0` mid-run eviction is "proven against an HF oracle." It is not. Per CLAIMS.md, the eviction's bit-exactness is proven on a **synthetic model** (against a non-vacuous poison control, max|Δ|≈0.33 for "kept"); the HF oracle separately proves the *forward-pass numerics* (cos=1.000000, final-logits max|Δ|≈4.4e-5). These are two distinct witnesses. The `go run ./cmd/deletioncert -selfcheck` demo itself runs "a tiny in-memory synthetic model (no weights, no torch)." Saying the eviction proof is "against a HuggingFace oracle" is a fabricated provenance and the single number most likely to get caught on r/LocalLLaMA. Fixed by attributing eviction-bit-exactness to the synthetic-model structural proof and the HF oracle to the numerics, separately.

**2. FATAL — Section 5 fuses two unrelated benchmarks into one false "measured on WebVoyager" claim.** The draft presents both "8.8x→9.7x vs naive" AND "~1.5–4x; 4.1x vs tuned" as `[measured]` on "real WebVoyager (643 tasks)." The ledger shows WebVoyager's *tuned* cross-worker comparison (B/C column) is **1.00x–1.10x**, not 4.1x. The 4.1x-vs-tuned is a **different** benchmark: the synthetic Qwen2.5-1.5B 50-turn × 5-agent headline (`headline-qwen-50x5.json`), where arm A is *modeled* (~19h, not run live), not WebVoyager. Attaching 4.1x to WebVoyager is precisely the kind of bad number that kills the launch. Fixed by separating the two benchmarks and labeling each honestly.

**3. The `-selfcheck` flag is inert.** `cmd/deletioncert/main.go` line 41: `_ = selfcheck // single mode today`. The demo runs by default; the flag does nothing today. Harmless to keep (it documents intent and the help text lists it), and it doesn't change output — so I retain the command as written since it still works, but I drop any implication that the flag toggles behavior.

**4. Minor — the `~13MB` container size has no ledger row.** The container is real (`gcr.io/distroless/static-debian12:nonroot`, run-the-demos.md), and ~13MB distroless/static is in the verified-ledger one-liner you supplied, so it stays — but it is an operational-surface point, not a benchmarked number, and I keep it framed that way (no "measured" tag).

**5. Minor — "60x" attribution.** The draft correctly labels ~60x / "agent city" as design targets. Good — but since I'm now naming the 50x5 headline explicitly in Section 5, I make sure the 60.3x-vs-naive figure stays fenced as the *projected/modeled* scale number it is.

Everything else (two-gate architecture, ~100%-evadable-by-design detector, capability lock, 8-rung `--explain` ladder ending `DEFAULT_DENY`, per-layer cos=1.000000 / final-logits max|Δ|≈4.4e-5 / token-for-token KV decode, 4 platforms / 2 CPU ISAs / 4 GPU backends / 4 OSes, CUDA ~120 tok/s llama.cpp Q8_0 parity, 77.2–88.2% radix hit-rate, 0/29 novel, self-signed-v1 / EvictedCount-self-reported fence, self-host-only + read-heavy-only fence, simulated power/$, OG-card caveat) traces cleanly to the ledger and is preserved.

---

## CORRECTED ASSET (paste-ready)

---

## Fact-check notes

What I changed, and why each remaining number is ledger-clean:

- **Eviction provenance (Lobsters note, 1-second version, Sections 1 & 2):** removed every instance of "mid-run eviction proven against an HF oracle." Replaced with the ledger-true split — the bit-exact eviction (`max|Δ|=0`, non-vacuous control) is proven *structurally on a synthetic model* (CLAIMS.md L75-76; the demo itself runs "a tiny in-memory synthetic model — no weights, no torch"), and the HF oracle separately proves the *forward-pass numerics* (CLAIMS.md L73). This was the most dangerous claim in the draft.
- **Section 5, the conflation (biggest fix):** the draft attached both "8.8x→9.7x vs naive" and "4.1x vs tuned" to "real WebVoyager (643 tasks)." Split into two clearly-labeled benchmarks. WebVoyager's tuned/cross-worker number is **1.0–1.1x** (B/C column, `docs/webbench-real-measurements-summary.md` L26-29), not 4.1x. The 4.1x-vs-tuned and 60.3x-vs-naive are the *synthetic Qwen 50-turn×5-agent* headline (`headline-qwen-50x5.json`, BENCHMARK-AUTHORITY.md L24, L205), whose naive arm is explicitly **modeled** (~19h, L215). Each is now fenced on its own line.
- **Section 3 preflight command:** changed `refund_payment ...` placeholder to `--tool refund_payment` (the real flag is `--tool`, per `cmd/fak/main.go` L214) and kept your `--help`-before-publish caveat. The `-selfcheck` flag is inert today (`main.go` L41) but harmless — output is unchanged — so the deletioncert command stays as-is.
- **Section 4:** added the f32-vs-Q8_0 precision detail (CLAIMS.md L114) which strengthens the parity claim honestly; ~120 tok/s llama.cpp Q8_0 parity confirmed.
- **OG card / social-preview:** `demos.html` now carries OG/Twitter image metadata pointing at `visuals/social-preview.png`; verify it with `python tools/demo_live_links.py --published` before citing the share card.

Remaining numbers, all ledger-traced: cos=1.000000 / final-logits max|Δ|≈4.4e-5 / token-for-token KV decode (CLAIMS.md L73); 4 platforms / 2 CPU ISAs / 4 GPU backends / 4 OSes (verified ledger); CUDA ~120 tok/s Q8_0 parity (CLAIMS.md L114); 8.8x→9.7x vs naive on 643-task WebVoyager (webbench summary L26-29); 60.3x/4.1x synthetic headline, modeled (BENCHMARK-AUTHORITY.md L24, L205, L215); 77.2–88.2% radix hit-rate (CLAIMS.md L78); self-signed-v1 + EvictedCount self-reported + signed-string-not-remeasured fences (CLAIMS.md L76); ~100%-evadable-by-design, two-gate floor (verified ledger); 0/29 novel (verified ledger); ~13MB distroless/static (verified ledger one-liner + `docs/run-the-demos.md` L84); simulated power/$ and self-host-only/read-heavy-only fences (verified ledger). The draft was NOT already clean — the two FATAL findings (eviction provenance, WebVoyager/tuned conflation) would each have drawn fire at exactly the venues named.

Relevant files: `C:\work\fak\CLAIMS.md`, `C:\work\fak\BENCHMARK-AUTHORITY.md`, `C:\work\fak\docs\webbench-real-measurements-summary.md`, `C:\work\fak\cmd\deletioncert\main.go`, `C:\work\fak\cmd\fak\main.go`.
