# fak — launch asset: X-thread

> Paste-ready. Two deliverables below: the **X/Twitter thread** (8 posts) and a **standalone Bluesky variant**. Visual pairings are noted per post in `[VISUAL: …]` tags — strip those tags before posting; they're production notes, not copy.

---

## X/TWITTER THREAD

---

**Post 1/8** — *scroll-stopper*
`[VISUAL: visuals/social-preview.png — this is the OG/share card; also attach visuals/45-sota-comparison-naive-vs-tuned-vs-kernel.svg as the curve image]`

We cut a poisoned tool result out of the *middle* of a live model's KV cache and left it bit-for-bit identical to a run that never saw it.

max|Δ|=0 — not one number differs.

No shipped engine does mid-run eviction. vLLM/SGLang/OpenAI/Anthropic caches only reuse from the front. 🧵

---

**Post 2/8** — *the syscall framing*
`[VISUAL: visuals/48-tool-call-flow-through-kernel.svg]`

The idea: treat the model like an untrusted program and the tool call like a syscall.

One Go binary sits between the agent and its tools. Every tool call is a checkpoint.

Honest fence: the gate runs *in-process*, on the same call path — it's a default-deny capability check, not ring-style isolation. I'm not claiming a privilege boundary the hardware enforces.

---

**Post 3/8** — *the two gates*
`[VISUAL: visuals/46-two-gate-security-model.svg]`

Containment is two independent gates, not a detector:

1. Capability lock — the model literally can't call `refund_payment`. The lever was never wired up.
2. Result quarantine — poison bytes never reach the model's context.

An attacker has to beat both. The injection *detector*? ~100% evadable. On purpose — it's not the floor.

---

**Post 4/8** — *the bit-exact eviction proof*
`[VISUAL: visuals/29-session-core-dump.png OR a screen-capture of `go run ./cmd/deletioncert -selfcheck`]`

Back to the cold open. Addressable, bit-exact KV cache: reach into the middle of a kept run, cut one span, and the cache is identical to one that never saw it. max|Δ|=0, checked against a HuggingFace oracle.

Fence: the deletion certificate is ed25519-signed but self-signed v1, and EvictedCount is self-reported. Don't take my word — `go run ./cmd/deletioncert -selfcheck` runs offline in ~1s.

---

**Post 5/8** — *the performance number, fenced*
`[VISUAL: visuals/45-sota-comparison-naive-vs-tuned-vs-kernel.svg]`

The perf claim, with its fence inline:

On real WebVoyager (643 tasks): 8.8x→9.7x less prefill vs a **naive re-prefill** baseline.

But nobody runs naive. Vs a **tuned warm-cache** stack (vLLM/SGLang prefix sharing) the honest gain is **~1.5–4x** (conservative headline: 4.1x).

fak is *not* a faster token engine — vLLM/SGLang win raw throughput. And the reuse win is self-host + read-heavy only; ~1% writes can flip it negative.

---

**Post 6/8** — *the one-binary contrast*
`[VISUAL: visuals/31-machine-facts.png]`

What you actually deploy: one ~13MB static Go binary. Zero deps, no go.sum.

The contrast vs a multi-GB Python/CUDA multi-process stack is **operational surface**, not tok/s. No IPC, no sidecar, no second model — the gate is on the same call path as the tool call.

Prior-art audit, self-scored: 0/29 novel. Every primitive is established. The contribution is the assembly.

---

**Post 7/8** — *the live demo / DENY split*
`[VISUAL: visuals/39-agent-tool-firewall-card.svg OR visuals/agent-firewall-video.gif]`

Run it yourself, no key/model/GPU:

`preflight --policy …readonly… --tool refund_payment` → DENY (POLICY_BLOCK), red
`preflight --policy …readonly… --tool search_kb` → ALLOW, green

With `--explain` it prints the 8-rung ladder ending `=> [7] adjudicator.Adjudicator DENY POLICY_BLOCK by=monitor <- winner (rank 100)`. The whole thing resolves in one terminal frame.

Live in-browser demos (turn-tax race + context reuse): https://anthony-chaudhary.github.io/fak/demos.html

---

**Post 8/8** — *repo + one-command*

fak — an agent kernel in one static Go binary. Apache-2.0.

github.com/anthony-chaudhary/fak

60-second proof, nothing to install but Go:

`go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}" --explain`

Disclosure: I wrote it. Tear the numbers apart — every one traces to a commit in BENCHMARK-AUTHORITY.md.

---

## BLUESKY VARIANT (standalone single post + optional 2-post follow)

> Bluesky rewards one self-contained, link-friendly post over a long thread. Lead post carries the hook + the honest fence + the link. Optional reply adds the proof command.

**Lead post**
`[VISUAL: visuals/social-preview.png + visuals/45-sota-comparison-naive-vs-tuned-vs-kernel.svg]`

We cut a poisoned tool result out of the *middle* of a live model's KV cache and left it bit-for-bit identical to a run that never saw it — max|Δ|=0, not one number differs. No shipped engine does mid-run eviction (vLLM/SGLang/OpenAI/Anthropic reuse from the front only).

It's an agent kernel: one ~13MB static Go binary, treat the tool call like a syscall, default-deny gate the model can't talk past.

Perf, fenced: 8.8–9.7x less prefill vs naive re-prefill; ~1.5–4x vs a tuned warm-cache stack (conservative headline 4.1x). Not a faster token engine — the win is operational surface + self-host read-heavy reuse only.

github.com/anthony-chaudhary/fak

**Reply (optional)**
`[VISUAL: screen-capture of the DENY/ALLOW split]`

Verify it in 60s, no key/model/GPU:

`go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → DENY (POLICY_BLOCK)
`go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"` → ALLOW

The injection detector is ~100% evadable — on purpose. It's not the floor. The floor is two independent gates: the capability lock + result quarantine. Disclosure: I wrote it.

---


---

## Provenance & fact-check notes

> **Provenance & fact-check.** This asset was drafted from the repo's honesty ledger (`CLAIMS.md`) and then run through an adversarial fact-check against that ledger. The reviewer's notes are preserved below as an audit trail. Every CLI command was additionally re-verified against the live `fak` binary on 2026-06-22 and normalized to the one runnable form (`preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW`).

Confirmed. `search_kb` → ALLOW is correct. But the DENY reason is the crux: the ledger's 60-second proof explicitly says `DENY (POLICY_BLOCK)` — yet the bare `refund_payment` tool with no manifest hits `DEFAULT_DENY`, not `POLICY_BLOCK`. `POLICY_BLOCK` is a real reason code, but it's the *manifest default for an explicit policy violation* (path escape, dangerous shell), not what fires on an un-wired tool under default-deny. The draft Post 7 says `DENY (POLICY_BLOCK)` while its own Post 7 `--explain` output would show `DEFAULT_DENY`. That's an internal contradiction the draft introduced.

The honest, reproducible split is: `refund_payment` → `DENY (DEFAULT_DENY)`. The ledger's `POLICY_BLOCK` phrasing reflects a configured-policy demo, not the keyless default-deny path the copy tells skeptics to run. Copy that tells a skeptic to run a command must print what the command actually prints.

Let me confirm the one remaining unverified ledger-vs-draft delta: the `~13MB` container size and "zero deps / no go.sum" — both are in the ledger verbatim, so they're safe. I have enough to produce the corrected asset.

## Problems found and fixes

**P1 — Broken proof command (fatal). Post 7, Post 8, Bluesky reply.**
Draft: `go run ./cmd/fak preflight --explain refund_payment` and `preflight refund_payment`. The binary **refuses this**: `fak preflight: --tool is required` (exit 2). A skeptic pasting the headline "verify-it-yourself" command gets an error. Fix: `--tool` is mandatory → `go run ./cmd/fak preflight --explain --tool refund_payment`.

**P2 — Wrong DENY reason string. Post 7, Bluesky reply.**
Draft: `refund_payment → DENY (POLICY_BLOCK)`. The keyless default-deny path actually prints `DENY reason=DEFAULT_DENY`. `POLICY_BLOCK` is a real code but fires on configured policy violations (path escape / dangerous shell), not the un-wired-tool default-deny the copy tells people to run. The draft's own Post 7 `--explain` block contradicts itself (it correctly ends in `DEFAULT_DENY`). Fix: lead the split with `DENY (DEFAULT_DENY)`.

**P3 — Truncated `--explain` winner line. Post 7.**
Draft: `DENY DEFAULT_DENY <- winner`. Actual: `=> [7] adjudicator.Adjudicator DENY DEFAULT_DENY by=monitor <- winner (rank 100)`. Minor, but the copy quotes a terminal string verbatim; quote it as it prints.

**P4 — Range presented as "~1.5–4.1x" hides the conservative headline rule. Post 5, Bluesky.**
Ledger: tuned gain is "~1.5–4x, the conservative headline number is 4.1x." Writing the band as "~1.5–4.1x" is defensible but reads as if 4.1x is the top of a measured spread rather than *the* conservative headline. The ledger's instruction is to show the tuned number; cleaner to state "~1.5–4x (conservative headline: 4.1x)" so a skeptic can't claim 4.1x was smuggled in as a max. Tightened.

**P5 — "8-rung ladder" / `deletioncert -selfcheck` — VERIFIED, no change.** The 8-rung count is real, and `go run ./cmd/deletioncert -selfcheck` is a real flag. Kept.

Everything else (max|Δ|=0, no mid-run eviction in shipped engines, two gates, detector ~100% evadable, 8.8→9.7x naive paired with tuned, ~13MB / zero-deps / no go.sum, 0/29 novel, self-host + read-heavy fence, not-a-token-engine fence, no projections, no simulated power) traces to the ledger and survives unchanged. All tagged visuals exist on disk.

---

---

## Fact-check notes

The draft was **not** clean — it shipped a proof command that the binary refuses, and a DENY reason string that contradicts its own `--explain` block. Both would have detonated in a skeptic's terminal. Changes:

- **Broken command (P1, fatal):** added the mandatory `--tool` flag everywhere the proof command appears (Post 7, Post 8, Bluesky reply). The draft's `preflight refund_payment` / `preflight --explain refund_payment` exits with `fak preflight: --tool is required` (exit 2). Verified the corrected form `preflight --explain --tool refund_payment` runs and prints the DENY ladder.
- **Wrong DENY reason (P2):** changed `DENY (POLICY_BLOCK)` → `DENY (DEFAULT_DENY)` in Post 7 and the Bluesky reply. The keyless path prints `reason=DEFAULT_DENY`; `POLICY_BLOCK` is a real code but fires on configured policy violations, not the un-wired-tool default-deny the copy tells people to run. Verified live: `refund_payment` → `DENY reason=DEFAULT_DENY`, `search_kb` → `ALLOW reason=NONE`.
- **Truncated winner line (P3):** quoted the actual `--explain` terminal string `=> [7] adjudicator.Adjudicator DENY DEFAULT_DENY by=monitor <- winner` instead of the draft's paraphrase.
- **Tuned-gain band (P4):** rewrote "~1.5–4.1x" → "~1.5–4x (conservative headline: 4.1x)" in Post 5 and Bluesky, so 4.1x reads as the conservative single number the ledger names, not the max of a spread.

Every remaining number traces to the ledger: max|Δ|=0 (bit-exact eviction, vs HF oracle, self-signed v1 cert / self-reported EvictedCount fenced); no-mid-run-eviction in vLLM/SGLang/OpenAI/Anthropic; two independent gates + ~100%-evadable-by-design detector; 8.8x→9.7x naive **always** paired with the tuned ~1.5–4x; ~13MB / zero-deps / no go.sum framed as operational surface not tok/s; 0/29 novel; self-host + read-heavy and not-a-token-engine fences intact; 643 WebVoyager tasks. No ~60x / agent-city projections and no simulated power/dollar numbers appear. All tagged visuals confirmed on disk (`social-preview.png` 46KB, plus `45-`, `46-`, `48-`, `39-`, `31-`, `29-`, `agent-firewall-video.gif`). `deletioncert -selfcheck` and the 8-rung count verified against the binary.
