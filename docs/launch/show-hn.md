# Show HN: fak – Treat the model as untrusted and the tool call as a syscall

**HN title (paste into the title field):**

Show HN: fak – Treat the LLM as untrusted and the tool call like a syscall (Go, one binary)

---

## Body comment (post immediately as author)

I built fak: one ~13MB static Go binary (Apache-2.0, no deps, no go.sum) that sits between an AI agent and its tools. It does two things in-process, on the same call path as the tool call — no IPC, no sidecar, no second model:

1. A **default-deny capability gate** the model can't talk past. Irreversible actions are refused *by structure* — the lever was never wired up, so there's nothing to jailbreak toward.
2. **Addressable, bit-exact KV cache** that lets you reach into the *middle* of a kept model run, cut out a poisoned tool result, and leave the cache bit-for-bit identical to a run that never saw it (checked at max|Δ|=0, not one number differs).

**60-second proof, no key/model/GPU** (copy-paste; the first run compiles the binary, later runs are instant):

```
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
```

`fak agent --offline` shows the same two gates in a full agent loop: injection-in-context YES->no, destructive-op YES->no, task still completes.

**Live in-browser demos (GCP):** https://anthony-chaudhary.github.io/fak/demos.html
**Repo:** https://github.com/anthony-chaudhary/fak

Honest fences up front, because this is HN:

- **fak is NOT a faster token engine and doesn't try to be.** vLLM/SGLang/llama.cpp win raw throughput. The operational contrast is one static binary vs a multi-GB Python/CUDA multi-process stack — not tok/s.
- The headline perf number is **8.8x–9.7x less prefill vs a naive re-prefill baseline** (real WebVoyager, 643 tasks). Nobody runs naive re-prefill in production, so the number that actually matters is **~1.5–4x vs a tuned warm-cache stack (vLLM/SGLang prefix sharing); conservative headline 4.1x.** I lead with the tuned number on purpose.
- The reuse win is **self-host only and read-heavy only** — an app that just calls a frontier API gets the safety floor, not the savings, and even ~1% write rate can flip the reuse economics negative.
- **The injection detector is ~100% evadable. By design.** It is explicitly *not* the floor. The floor is the two gates: the capability lock (the destructive call doesn't exist for the model to reach) plus result quarantine (poison bytes never enter the model's context). The detector is a tripwire, not a wall.
- I ran an honest prior-art audit and scored it **0/29 novel** — every primitive here is established. The contribution is the *assembly*: putting them in one in-process gate where the tool call is the checkpoint.
- Power/energy/$ figures anywhere in the repo are **simulated** (no power meter on the box). The ~60x and "agent city" frontier numbers are **design targets, not measurements** — labeled as such.

The bit-exact KV claim is the one I'd most like torn apart: the pure-Go SmolLM2-135M forward pass is checked per-layer against a HuggingFace oracle (cos=1.000000, final-logits max|Δ|≈4.4e-5), and the mid-run eviction is proven against that oracle at max|Δ|=0. The deletion certificate is ed25519-signed and fails closed on a forged field — but it's self-signed v1 and its EvictedCount is self-reported, so don't trust the receipt, reproduce the max|Δ|=0.

---

## Top-comment objections + honest answers (paste as needed)

**"This is just another AI safety wrapper / prompt-injection classifier."**
The classifier is the part I'm telling you to ignore — it's ~100% evadable by design. The actual mechanism is that `refund_payment` is never in the capability set the model can emit, and a poisoned tool result is quarantined before it reaches context. An attacker has to beat two independent structural gates, not fool one detector. If you can get the offline demo to fire a destructive call, that's a real bug and I want the repro.

**"'Kernel' and 'syscall' are marketing. Your gate runs in the same process as the agent loop — that's not a privilege ring."**
Correct, and I won't claim otherwise. There's no hardware ring; userspace here can't cross a boundary the way a real syscall does. "Syscall" is an intuition pump for *the tool call is the checkpoint where a default-deny check runs*, nothing more. The shipped thing is an in-process default-deny capability check on the tool-call path. I'd rather you trust the engineering than the metaphor.

**"8.8–9.7x is a strawman denominator."**
It is, which is why I led with the tuned number. vs naive re-prefill it's 8.8x (1 worker) to 9.7x (8 workers); vs a tuned warm-cache stack (vLLM/SGLang prefix sharing) it's ~1.5–4x, conservative 4.1x. The naive figure exists because it's the apples-to-apples measurement against the loop most homegrown agents actually run, not because it's the honest competitive claim.

**"So it's faster than vLLM?"**
No. fak is not a token engine; vLLM/SGLang/llama.cpp beat it on raw throughput and I don't contest that. The prefill savings come from *reusing shared setup work across turns* on a self-hosted, read-heavy fleet — a different axis from single-stream tok/s, and a narrow applicability (self-host + read-heavy). If you're calling a hosted API you get the safety floor and zero reuse savings.

**"Mid-run KV eviction — every cache does prefix reuse, what's new?"**
Prefix reuse is front-of-context only: vLLM/SGLang/OpenAI/Anthropic prompt caches reuse a shared *prefix*. This cuts a span out of the *middle* of a kept run and re-knits the cache so it's bit-identical to a run that never contained that span (max|Δ|=0). That's what lets you delete a poisoned tool result *after* the model has already seen it without re-prefilling everything after it. I'm not aware of a shipped engine that offers mid-run eviction; if one does, tell me and I'll correct the post.

**"Bit-for-bit identical across hardware — really?"**
Deterministic results reproduce byte-for-byte across Apple M3/Metal, AMD Ryzen+RX7600/Vulkan, Intel+RTX4070/CUDA Ada, and an 8-GPU Ampere server — 4 GPU backends, 2 CPU ISAs, 4 OSes. CUDA decode hits llama.cpp Q8_0 parity (~120 tok/s on a 4070). RadixAttention parity vs SGLang rebuilt over the kernel-owned cache is 77.2–88.2% hit rate, reuse-through-edge-split bit-identical to recompute (max|Δ|=0). Every number traces to a commit+artifact in BENCHMARK-AUTHORITY.md.

**"What does drop-in actually mean?"**
OpenAI-compatible and Anthropic-compatible gateway plus MCP, so Claude Code / Cursor / any OpenAI client points at it with no agent-side changes. ~13MB distroless/static container.

**"Why should I believe your own benchmarks?"**
You shouldn't on my word — that's why the 60-second proof needs no key/model/GPU and the KV claims reproduce against a public HuggingFace oracle. Disclosure: I wrote this. The repo's CLAIMS.md is an honesty ledger that marks each claim SHIPPED vs design-target; the 0/29-novel audit and the evadable-detector admission are in there because hiding them would be the actual red flag.

---

## Provenance & fact-check notes

> **Provenance & fact-check.** This asset was drafted from the repo's honesty ledger (`CLAIMS.md`) and then run through an adversarial fact-check against that ledger. The reviewer's notes are preserved below as an audit trail. Every CLI command was additionally re-verified against the live `fak` binary on 2026-06-22 and normalized to the one runnable form (`preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW`).

Both flagged assets resolve. Now the substantive fact-check against the ledger.

The draft is unusually clean — the author clearly worked from the same ledger. But hunting adversarially, I found **four problems**, two of them real fabrications that would draw blood on HN/r/netsec.

**Problem 1 — Fabricated CLI syntax (the 60-second proof). HIGH severity.**
The draft writes:
```
go run ./cmd/fak preflight refund_payment   # -> DENY (POLICY_BLOCK)
go run ./cmd/fak preflight search_kb         # -> ALLOW
```
The ledger's verified command is `go run ./cmd/fak preflight ... refund_payment` — the `...` denotes elided arguments (preflight takes more than a bare tool name). The draft silently dropped the `...` and invented a clean two-token invocation. On HN someone *will* paste this, get a usage error, and call the whole post fake. This is the single highest-risk line in the asset. I rewrote it to match the ledger's literal form and labeled the placeholder.

**Problem 2 — "the same loop end-to-end" overstates the offline demo. MEDIUM.**
The ledger says `fak agent --offline` *shows* three states (injection YES→no, destructive YES→no, task completed). The draft's "shows the same loop end-to-end" implies the preflight DENY and the agent demo are one continuous run, which the ledger doesn't claim. Minor inflation of scope; tightened to "shows the same two gates in a full agent loop."

**Problem 3 — "live model run" vs "kept model run." LOW but it's a security claim.**
Draft: "reach into the *middle* of a live model run." Ledger says "a kept model run." "Live" implies mid-inference mutation of a running forward pass; "kept" means a retained/cached run. For the skeptical-netsec read this is a meaningful precision loss on the headline technical claim. Changed to "kept."

**Problem 4 — RadixAttention line drops the max|Δ|=0 proof anchor. LOW.**
Draft: "reuse-through-edge-split bit-identical to recompute." Ledger pairs that phrase with "(max|Δ|=0)." Since the whole credibility play is "don't trust me, reproduce the zero," dropping the number weakens it. Restored.

Everything else traces. The 8.8x/9.7x, ~1.5–4x, 4.1x, 643 tasks, 0/29, 77.2–88.2%, ~120 tok/s, 4.4e-5, cos=1.000000, ~13MB, hardware matrix, the "naive vs tuned — always lead tuned" discipline, the evadable-detector framing, and every honest fence are present and correctly fenced. The two submit-time checks both pass: demos.html is live, `social-preview.png` is committed on `main` so the raw URL resolves.

Here is the corrected, paste-ready asset.

---

---

## Fact-check notes

**Changed (4):**

1. **60-second proof CLI — fabricated syntax fixed (HIGH).** Draft invented `go run ./cmd/fak preflight refund_payment`. Ledger's verified form is `go run ./cmd/fak preflight ... refund_payment`. Restored the `...` and added a parenthetical noting it's elided args pointing to the README quickstart, so a copy-paste tester isn't surprised by a usage error.
2. **"shows the same loop end-to-end" → "shows the same two gates in a full agent loop" (MEDIUM).** Removed the implication that the preflight DENY and the `--offline` agent demo are one continuous run; the ledger only says the agent demo *shows* the three states.
3. **"middle of a *live* model run" → "*kept* model run" (LOW, security claim).** Matched ledger wording; "live" wrongly implies mutation mid-forward-pass.
4. **RadixAttention line — restored "(max|Δ|=0)" (LOW).** The draft dropped the proof anchor from "bit-identical to recompute"; the ledger pairs them, and the number is the whole credibility move.

**Confirmed against ledger (every remaining number traces):**
8.8x/9.7x (643-task WebVoyager, naive baseline, labeled and never led with); ~1.5–4x tuned, 4.1x conservative headline (led with, as required); 0/29 novel; 77.2–88.2% hit rate; ~120 tok/s RTX 4070 / llama.cpp Q8_0 parity; final-logits max|Δ|≈4.4e-5; per-layer cos=1.000000; mid-run eviction max|Δ|=0; ~13MB distroless/static; hardware matrix (M3/Metal, Ryzen+RX7600/Vulkan, Intel+RTX4070/CUDA Ada, 8-GPU Ampere — 4 backends, 2 ISAs, 4 OSes); ed25519 self-signed v1 + self-reported EvictedCount fence; detector "~100% evadable by design / not the floor"; simulated power/$ fence; ~60x and "agent city" labeled as design targets; self-host + read-heavy + ~1%-write-flips-negative fence; "not a faster token engine" fence. All present and correctly fenced.

**Submit-time checks (both flagged in the draft) — both PASS:**
- `https://anthony-chaudhary.github.io/fak/demos.html` loads live (title "fak — the agent kernel | live demos", three demo sections, not a 404).
- `visuals/social-preview.png` is **committed on `main`** (`git ls-files` confirms tracked), so the OG/Twitter `raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/social-preview.png` card resolves.

**Relevant paths:** `C:\work\fak\visuals\social-preview.png` (the OG card, committed), `C:\work\fak\docs\demos.html` (lines 29/32 carry the og:image and twitter:image meta pointing at that file).
