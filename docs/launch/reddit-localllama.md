Disclosure up front: I wrote this. It's Apache-2.0, github.com/anthony-chaudhary/fak. I'll put the parity tables and the honest baseline in the first comment so this post stays about the mechanism.

## What it is, for self-hosters

One static Go binary (~13MB, no go.sum, zero deps) that sits between an agent and its tools. It fronts your self-hosted engine (vLLM/llama.cpp/etc.), speaks both the OpenAI and Anthropic APIs + MCP, and runs **offline with no API key**. Claude Code / Cursor / any OpenAI client drops in with no agent-side changes.

It is **not** a faster token engine and doesn't try to be. vLLM/SGLang/llama.cpp win raw throughput — that's not the contest. The contrast is operational surface: one binary vs a multi-GB Python/CUDA multi-process stack. The two things this audience might actually care about are below.

## 1. Bit-exact mid-run KV eviction

The hook: you can reach into the **middle** of a kept model run, cut one span out of the KV cache — say a tool result that turned out to be prompt injection — and leave the cache **bit-for-bit identical to a run that never saw that span**. On the in-binary SmolLM2-135M run this is proven against a HuggingFace reference at `max|Δ|=0`. Not "close." Not one number differs.

As far as I can tell no shipped engine does this. vLLM, SGLang, OpenAI's and Anthropic's prompt caches all reuse from the **front** (prefix sharing). None of them evict an addressable span from the interior of a live run and prove the result. If I'm wrong about an engine that does, I genuinely want to know — that's half of why I'm posting here.

Honest fences, because they matter:
- The thing that *flags* poison (a detector) is **~100% evadable by design**. It is explicitly **not** the security floor. The floor is two structural gates: a capability lock (the destructive lever was never wired up, so the model can't call it) + result quarantine (poison bytes never reach context). An attacker has to beat both; the detector is a convenience, not a wall.
- The deletion **certificate** (ed25519-signed receipt binding the evicted span to `max|Δ|=0`) is self-signed v1, and its `EvictedCount` field is self-reported. The `max|Δ|=0` parity is the real claim; the cert is bookkeeping on top.

There's an offline demo of the deletion property — no GPU, no key, runs in ~1s:

    go run ./cmd/deletioncert -selfcheck

It runs the same Prefill/Step/Evict cache path the HF-verified model uses, on a tiny synthetic model (the deletion property is structural, not numeric). It prints three continuations — `never-saw`, `kept-secret`, `evicted` — proves `evicted == never-saw` at `max|Δ|=0` while `kept-secret` differs (so the span genuinely perturbs decode), then mints a deletion certificate, verifies it, and shows verification fail closed under three tamper attempts (forged cert field, inflated scope, rewritten journal).

## 2. The pure-Go forward pass is verified per-layer against HF

Because "I deleted from the KV cache correctly" only means something if the KV cache is correct in the first place. The in-binary SmolLM2-135M forward pass is proven rung-by-rung against a HuggingFace oracle: per-layer **cos=1.000000**, final-logits **max|Δ|≈4.4e-5**, KV-decode token-for-token identical. CUDA decode hits parity with **llama.cpp Q8_0 (~120 tok/s on an RTX 4070)**.

Deterministic results reproduce byte-for-byte across 4 hardware platforms (Apple M3/Metal, AMD Ryzen+RX7600/Vulkan, Intel+RTX4070/CUDA Ada, 8-GPU Ampere server), 2 CPU ISAs, 4 GPU backends, 4 OSes.

## On the performance number (the honest version)

You'll see "8.8–9.7x" somewhere in the repo. That is **less prefill vs a naive re-prefill baseline** on real WebVoyager (643 tasks) — and a naive re-prefill is a strawman nobody runs in production, so I won't lead with it. Against a **tuned warm-cache stack (vLLM/SGLang prefix sharing)** the honest gain is **~1.5–4x**, conservative headline **4.1x**.

And it only helps if you **self-host and your fleet is read-heavy** — an app that just calls a frontier API gets the safety floor, not the savings, and even ~1% writes can flip the reuse math negative. Any power/energy/$ numbers in the repo are **simulated** (no power meter on the box). The ~60x "frontier" numbers are design targets, not measurements — labeled as such.

## Try it

- Live in-browser demos (turn-tax race + context-reuse vs a tuned warm-cache baseline): https://anthony-chaudhary.github.io/fak/demos.html
- One-command local demo, no setup: `go run ./cmd/turntaxdemo`
- Colab quickstart (free GPU): `notebooks/fak-quickstart.ipynb` in the repo

The prior-art audit scored **0/29 novel** — every primitive here is established. The contribution is the assembly: one in-process gate where the tool call is the checkpoint. Happy to be torn apart on any of the above.

**First comment (the prosecution — post this yourself immediately):**

```markdown
Prosecuting my own claims so you don't have to dig:

**The parity table (KV reuse / eviction):**
- Mid-run eviction reconciled to a never-saw-it run: `max|Δ|=0` (SmolLM2-135M, vs HF oracle)
- SmolLM2-135M forward pass vs HF oracle: per-layer `cos=1.000000`, final logits `max|Δ|≈4.4e-5`
- RadixAttention parity rebuilt over the kernel-owned cache: 77.2–88.2% hit rate; reuse-through-edge-split bit-identical to recompute (`max|Δ|=0`). SGLang baseline numbers checked against the NeurIPS 2024 paper.

**The speedup, both denominators (don't quote the first alone):**
- vs naive re-prefill: 8.8x (1 worker) → 9.7x (8 workers) less prefill on real WebVoyager (643 tasks) — strawman baseline, stated only for completeness
- vs tuned warm-cache SOTA (vLLM/SGLang prefix sharing): **~1.5–4x, conservative headline 4.1x** — this is the real number

**Where it does nothing for you:** if you call a hosted API instead of serving the model, you get the capability lock + quarantine (security floor) and zero reuse savings. Reuse is a self-host, read-heavy-fleet win only.

Every number traces to a commit + artifact in BENCHMARK-AUTHORITY.md. If any of these don't reproduce on your box, that's a bug report I want.
```

---

**Fact-check notes**

What I changed:

1. **Fabricated demo output (critical).** The draft narrated `deletioncert -selfcheck` as "Three rows print; two are bit-identical and one carries a rogue token — then the rogue span is cut and the cache reconciles to the clean run." I ran it. The real output prints three token-id *continuations* (`never-saw`, `kept-secret`, `evicted`), asserts `evicted == never-saw (max|Δ|=0)` while `kept-secret` differs, then mints a certificate, verifies it, and rejects three tamper attempts. Rewrote the demo description to match actual output verbatim. A skeptic running the command would have caught the original — that alone could have sunk the post.

2. **HF-oracle conflation (critical).** The draft said the mid-run eviction was "Checked against a HuggingFace reference at max|Δ|=0" and pointed at the offline `-selfcheck` demo as the proof. The demo runs a **synthetic in-memory model** (no weights — the deletion property is structural); the HF-oracle parity belongs to the SmolLM2-135M run per the ledger ("proven against HF oracle") and the forward-pass claim. I attached the HF reference to the SmolLM2 run explicitly ("On the in-binary SmolLM2-135M run this is proven against a HuggingFace reference at max|Δ|=0") and made the demo paragraph say plainly it runs the same cache path on a tiny synthetic model. Carried the SmolLM2 qualifier into the first-comment parity table too.

3. **Operator-preamble bleed.** Removed the draft's opening line ("All assets verified... Writing the post.") and the closing operator note ("Both demo commands... verified present on disk — the demos.html OG card will not 404."). Those are out-of-band verification notes, not post copy. (I did independently confirm on disk: `cmd/deletioncert`, `cmd/turntaxdemo`, `notebooks/fak-quickstart.ipynb`, and `visuals/social-preview.png` all exist; `-selfcheck` is a real flag; no `go.sum`.)

4. **Engine name not in ledger.** Changed "fronts Ollama/vLLM/llama.cpp" to "fronts your self-hosted engine (vLLM/llama.cpp/etc.)" — Ollama isn't in the ledger, and the ledger's operational-surface fence (one binary vs a Python/CUDA stack) was missing from the "not a faster engine" paragraph, so I added it.

Every remaining number traces to the ledger and was confirmed in-repo: ~13MB / no go.sum / zero deps; SmolLM2-135M `cos=1.000000`, final-logits `max|Δ|≈4.4e-5`, KV-decode token-for-token identical; CUDA = llama.cpp Q8_0 ~120 tok/s on RTX 4070; mid-run eviction `max|Δ|=0` vs HF oracle; RadixAttention 77.2–88.2% hit rate, reuse-through-edge-split `max|Δ|=0`, SGLang vs NeurIPS 2024; WebVoyager 643 tasks, 8.8x (1 worker)→9.7x (8 workers) vs naive (confirmed in `docs/webbench-baselines.md`, `docs/FAQ.md`, `v0.27.0` release notes); tuned ~1.5–4x, conservative headline 4.1x (confirmed in `README.md`, `docs/HARDWARE-MATRIX.md`); 4 platforms / 2 CPU ISAs / 4 GPU backends / 4 OSes; ed25519 self-signed v1 cert with self-reported `EvictedCount`; detector ~100% evadable by design; 0/29 prior-art audit; simulated power/$ and ~60x labeled as design targets. The two structural fences the draft already had right — tuned-baseline-first and self-host/read-heavy-only — were preserved unchanged.


---

## Provenance & fact-check notes

> **Provenance & fact-check.** This asset was drafted from the repo's honesty ledger (`CLAIMS.md`) and then run through an adversarial fact-check against that ledger. The reviewer's notes are preserved below as an audit trail. Every CLI command was additionally re-verified against the live `fak` binary on 2026-06-22 and normalized to the one runnable form (`preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW`).

All headline numbers (643 tasks, 8.8-9.7x naive, 4.1x tuned, ~1.5-4x range) match the ledger. Now I have everything I need. Let me assess the draft.

The draft is mostly clean and honest — it leads with the tuned baseline, keeps every fence, labels projections. But there are **three real defects**, one of them launch-killing:

1. **FABRICATED DEMO OUTPUT (critical).** The draft says `go run ./cmd/deletioncert -selfcheck` prints "Three rows print; two are bit-identical and one carries a rogue token — then the rogue span is cut and the cache reconciles to the clean run." The real output prints three *token-id continuations* (`never-saw`, `kept-secret`, `evicted`), proves `evicted == never-saw (max|Δ|=0)` while `kept-secret` differs, then mints/verifies/tamper-rejects a certificate. A skeptic who runs the command sees output that doesn't match the narration — instant credibility loss.

2. **HF-oracle conflation (critical).** The draft says the mid-run eviction is "Checked against a HuggingFace reference at `max|Δ|=0`" and frames the `deletioncert` demo as the proof. Per the ledger and the demo source, the *eviction* property is proven against the HF oracle on the SmolLM2-135M run; the `-selfcheck` demo itself runs a **synthetic in-memory model** (the property is structural). The HF parity (`cos=1.000000`, `max|Δ|≈4.4e-5`) belongs to the forward-pass claim, which the draft already states separately. The fix: keep the eviction-vs-HF claim (it's in the ledger: "proven against HF oracle") but don't imply the *offline 1s demo* is the HF run.

3. **Asset-existence preamble bleed.** The draft opens and closes with operator-facing verification notes ("All assets verified...", "Both demo commands... are all verified present on disk"). Those are not part of the post and must not ship inside the asset.

One smaller issue: "fronts Ollama/vLLM/llama.cpp" — the ledger says OpenAI/Anthropic-compatible gateway + MCP and that it fronts self-hosted engines; Ollama isn't named in the ledger. I'll generalize to "your self-hosted engine (vLLM/llama.cpp/etc.)" to stay traceable.

Below is the corrected asset.

---

**Title:**
Cut a poisoned tool result out of the middle of a live KV cache, left it bit-for-bit identical to a run that never saw it (max|Δ|=0). One Go binary, no Python/CUDA, runs offline.

**Body:**
